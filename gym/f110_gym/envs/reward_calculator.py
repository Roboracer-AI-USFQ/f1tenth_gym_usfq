import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
from numba import njit


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(
    point: np.ndarray, trajectory: np.ndarray
) -> Tuple[np.ndarray, float, float, int]:
    """
    Return the nearest point along the given piecewise linear trajectory.

    Args:
        point: size 2 numpy array [x, y]
        trajectory: Nx2 matrix of (x,y) trajectory waypoints

    Returns:
        nearest_point: closest point on trajectory
        min_distance: distance to closest point
        t: parameter along segment
        min_dist_segment: index of closest segment
    """
    # Validate inputs
    if trajectory.shape[0] < 2:
        if trajectory.shape[0] == 1:
            dist = np.sqrt(np.sum((point - trajectory[0]) ** 2))
            return trajectory[0], dist, 0.0, 0
        else:
            return point, 1000.0, 0.0, 0
    
    # Clean input data
    point = np.array([np.nan_to_num(point[0], nan=0.0), np.nan_to_num(point[1], nan=0.0)])
    
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    
    # Prevent division by zero
    epsilon = 1e-8
    l2s = np.maximum(l2s, epsilon)

    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])

    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0

    projections = trajectory[:-1, :] + (t * diffs.T).T

    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dist_squared = np.sum(temp * temp)
        dists[i] = np.sqrt(max(0.0, dist_squared))

    # Ensure we have valid distances
    if np.all(np.isnan(dists)) or np.all(np.isinf(dists)):
        fallback_dist = np.sqrt(np.sum((point - trajectory[0]) ** 2))
        return trajectory[0], fallback_dist, 0.0, 0

    min_dist_segment = np.argmin(dists)
    
    # Clean output
    result_point = projections[min_dist_segment]
    result_dist = dists[min_dist_segment]
    result_t = t[min_dist_segment]
    
    # Final safety check
    if np.isnan(result_dist) or np.isinf(result_dist):
        result_dist = 1000.0
    if np.isnan(result_t) or np.isinf(result_t):
        result_t = 0.0
    
    return (
        result_point,
        result_dist,
        result_t,
        min_dist_segment,
    )


class RewardCalculator:
    """
    Modular reward calculator for F1TENTH RL training
    """

    def __init__(
        self,
        waypoints_path: Optional[str] = None,
        reward_config: Optional[Dict] = None,
        track_length: float = 100.0,
    ):
        """
        Initialize the reward calculator

        Args:
            waypoints_path: Path to waypoints file for centerline calculation
            reward_config: Dictionary with reward weights and parameters
            track_length: Total length of the track for progress calculation
        """

        # Default reward configuration
        default_config = {
            "weights": {
                "progress": 0.4,  # subir valor
                "speed": 0.25,
                "centerline": 0.15,
                "smoothness": 0.1,
                "inactivity": 0.1,
            },
            "penalties": {"collision": -50.0},  # no se puede salir
            "parameters": {
                "target_speed": 8.0,  # m/s, capaz no es muy buena idea, mejor una funcion sigmoid para impulsar mas velocidad
                "speed_tolerance": 2.0,
                "centerline_tolerance": 1.0,  # meters
                "inactivity_threshold": 0.5,  # m/s minimum speed
                "action_smoothness_window": 5,
            },
        }

        self.config = reward_config if reward_config else default_config
        self.track_length = track_length

        # Load waypoints if provided
        self.waypoints = None
        if waypoints_path:
            try:
                self.waypoints = np.loadtxt(waypoints_path, delimiter=";", skiprows=3)
                # Extract x, y coordinates (assuming columns 1, 2)
                if self.waypoints.shape[1] > 2:
                    self.waypoints = self.waypoints[:, 1:3]
                print(f"Loaded {len(self.waypoints)} waypoints from {waypoints_path}")
            except Exception as e:
                print(f"Could not load waypoints from {waypoints_path}: {e}")
                self.waypoints = None

        # Initialize state tracking
        self.reset_episode()

    def reset_episode(self):
        """Reset episode-specific tracking variables"""
        self.last_progress = 0.0
        self.last_position = np.array([0.0, 0.0])
        self.last_action = np.array([0.0, 0.0])
        self.action_history = []
        self.cumulative_distance = 0.0
        self.step_count = 0

    def calculate_progress_reward(self, position: np.ndarray) -> float:
        """
        Calculate reward based on progress along the track

        Args:
            position: Current [x, y] position

        Returns:
            Progress reward
        """
        # Clean input position
        position = np.nan_to_num(position, nan=0.0, posinf=1000.0, neginf=-1000.0)
        
        if self.waypoints is None:
            # Fallback: use euclidean distance from start
            progress = np.linalg.norm(position - self.last_position)
            progress = np.nan_to_num(progress, nan=0.0, posinf=10.0, neginf=0.0)
            self.last_position = position.copy()
            return max(0.0, progress * 2.0)  # Scale factor

        # Calculate progress using waypoints
        _, _, t, segment_idx = nearest_point_on_trajectory(position, self.waypoints)
        
        # Clean values from trajectory function
        t = np.nan_to_num(t, nan=0.0)
        segment_idx = max(0, min(segment_idx, len(self.waypoints) - 1))

        # Current progress as percentage of track completion
        current_progress = (segment_idx + t) / max(1, len(self.waypoints))
        current_progress = np.clip(current_progress, 0.0, 1.0)

        # Handle track wrapping
        progress_delta = current_progress - self.last_progress
        if progress_delta < -0.5:  # Wrapped around
            progress_delta += 1.0
        elif progress_delta > 0.5:  # Went backwards significantly
            progress_delta -= 1.0

        self.last_progress = current_progress

        # Reward forward progress, penalize backward movement
        result = max(0, progress_delta * self.track_length)
        return np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=0.0)

    def calculate_speed_reward(self, velocity: float) -> float:
        """
        Calculate reward based on optimal speed maintenance

        Args:
            velocity: Current longitudinal velocity

        Returns:
            Speed reward
        """
        # Clean input velocity
        velocity = np.nan_to_num(velocity, nan=0.0, posinf=20.0, neginf=-20.0)
        
        target_speed = self.config["parameters"]["target_speed"]
        speed_tolerance = self.config["parameters"]["speed_tolerance"]

        speed_error = abs(velocity - target_speed)

        if speed_error <= speed_tolerance:
            # Reward optimal speed maintenance
            result = 5.0 * (1.0 - speed_error / max(speed_tolerance, 1e-6))
            return np.clip(result, 0.0, 5.0)
        else:
            # Penalize being too far from optimal speed using sigmoid
            # Sigmoid parameters: output range [-3, 8], center at target_speed
            min_reward = -3.0
            max_reward = 8.0
            k = -1.0  # negative so reward decreases as error increases
            x = velocity
            x0 = target_speed  # center the sigmoid at target speed
            
            # Clamp exponential input to prevent overflow
            exp_input = -k * (x - x0)
            exp_input = np.clip(exp_input, -50, 50)  # Prevent exp overflow
            
            try:
                exp_val = np.exp(exp_input)
                if np.isnan(exp_val) or np.isinf(exp_val):
                    exp_val = 1.0
                
                sigmoid = (max_reward - min_reward) / (1.0 + exp_val) + min_reward
                sigmoid = np.nan_to_num(sigmoid, nan=0.0, posinf=max_reward, neginf=min_reward)
                
                return np.clip(sigmoid, min_reward, max_reward)
            except:
                # Fallback if sigmoid calculation fails
                return 0.0

    def calculate_centerline_penalty(self, position: np.ndarray) -> float:
        """
        Calculate penalty for deviating from centerline

        Args:
            position: Current [x, y] position

        Returns:
            Centerline deviation penalty (negative)
        """
        if self.waypoints is None:
            return 0.0  # No penalty if no centerline defined

        # Clean input position
        position = np.nan_to_num(position, nan=0.0, posinf=1000.0, neginf=-1000.0)
        
        _, distance, _, _ = nearest_point_on_trajectory(position, self.waypoints)
        
        # Clean distance value
        distance = np.nan_to_num(distance, nan=1000.0, posinf=1000.0, neginf=0.0)
        distance = max(0.0, distance)  # Ensure non-negative
        
        tolerance = self.config["parameters"]["centerline_tolerance"]

        if distance <= tolerance:
            return 0.0  # No penalty within tolerance
        else:
            # Quadratic penalty for deviation
            excess_deviation = distance - tolerance
            penalty = -3.0 * (excess_deviation**2)
            
            # Clamp penalty to reasonable range
            penalty = max(-100.0, penalty)
            return np.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=-100.0)

    def calculate_action_smoothness_reward(self, action: np.ndarray) -> float:
        """
        Calculate reward for smooth action transitions

        Args:
            action: Current action [steering_velocity, longitudinal_velocity]

        Returns:
            Smoothness reward
        """
        # Clean input action
        action = np.nan_to_num(action, nan=0.0, posinf=10.0, neginf=-10.0)
        
        if len(self.action_history) == 0:
            self.action_history.append(action.copy())
            self.last_action = action.copy()
            return 0.0

        # Calculate action change magnitude
        action_diff = action - self.last_action
        action_change = np.linalg.norm(action_diff)
        
        # Clean the calculated change
        action_change = np.nan_to_num(action_change, nan=0.0, posinf=10.0, neginf=0.0)
        action_change = max(0.0, action_change)  # Ensure non-negative

        # Add to history
        self.action_history.append(action.copy())
        window_size = self.config["parameters"]["action_smoothness_window"]
        if len(self.action_history) > window_size:
            self.action_history.pop(0)

        self.last_action = action.copy()

        # Reward smooth transitions, penalize jerky movements
        if action_change < 0.5:  # Smooth transition threshold
            reward = 1.0 - 2.0 * action_change
            return np.clip(reward, -2.0, 1.0)
        else:
            penalty = -2.0 * action_change
            return max(-10.0, penalty)  # Clamp extreme penalties

    def calculate_inactivity_penalty(self, velocity: float) -> float:
        """
        Calculate penalty for staying stationary or moving too slowly

        Args:
            velocity: Current longitudinal velocity

        Returns:
            Inactivity penalty (negative)
        """
        # Clean input velocity
        velocity = np.nan_to_num(velocity, nan=0.0, posinf=20.0, neginf=-20.0)
        
        min_speed = self.config["parameters"]["inactivity_threshold"]

        if abs(velocity) < min_speed:
            # Penalize inactivity - stronger penalty the longer stationary
            penalty = -5.0 * (min_speed - abs(velocity))
            return max(-10.0, penalty)  # Clamp extreme penalties
        else:
            return 0.0

    def calculate_collision_penalty(self, collision: bool) -> float:
        """
        Calculate penalty for collisions

        Args:
            collision: Whether collision occurred

        Returns:
            Collision penalty (large negative value)
        """
        return self.config["penalties"]["collision"] if collision else 0.0

    def calculate_total_reward(
        self, obs: Dict, action: np.ndarray, done: bool, info: Dict = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate the total reward combining all components

        Args:
            obs: Environment observation dictionary
            action: Action taken [steering_velocity, longitudinal_velocity]
            done: Whether episode is done
            info: Additional info dictionary

        Returns:
            total_reward: Combined reward value
            reward_components: Dictionary with individual reward components
        """
        try:
            # Extract state information with safety checks
            position = np.array([
                np.nan_to_num(obs.get("poses_x", [0.0])[0], nan=0.0),
                np.nan_to_num(obs.get("poses_y", [0.0])[0], nan=0.0)
            ])
            velocity = np.nan_to_num(obs.get("linear_vels_x", [0.0])[0], nan=0.0)
            collision = obs.get("collisions", [False])[0] if "collisions" in obs else False

            # Clean action input
            action = np.nan_to_num(action, nan=0.0, posinf=10.0, neginf=-10.0)

            # Calculate individual reward components
            progress_reward = self.calculate_progress_reward(position)
            speed_reward = self.calculate_speed_reward(velocity)
            centerline_penalty = self.calculate_centerline_penalty(position)
            smoothness_reward = self.calculate_action_smoothness_reward(action)
            inactivity_penalty = self.calculate_inactivity_penalty(velocity)
            collision_penalty = self.calculate_collision_penalty(collision)

            # Clean all reward components
            progress_reward = np.nan_to_num(progress_reward, nan=0.0, posinf=10.0, neginf=-10.0)
            speed_reward = np.nan_to_num(speed_reward, nan=0.0, posinf=10.0, neginf=-10.0)
            centerline_penalty = np.nan_to_num(centerline_penalty, nan=0.0, posinf=0.0, neginf=-100.0)
            smoothness_reward = np.nan_to_num(smoothness_reward, nan=0.0, posinf=10.0, neginf=-10.0)
            inactivity_penalty = np.nan_to_num(inactivity_penalty, nan=0.0, posinf=0.0, neginf=-10.0)
            collision_penalty = np.nan_to_num(collision_penalty, nan=0.0, posinf=0.0, neginf=-100.0)

            # Get weights from configuration
            weights = self.config["weights"]

            # Calculate weighted total reward
            total_reward = (
                weights["progress"] * progress_reward
                + weights["speed"] * speed_reward
                + weights["centerline"] * centerline_penalty
                + weights["smoothness"] * smoothness_reward
                + weights["inactivity"] * inactivity_penalty
                + collision_penalty  # Collision penalty not weighted (should be large)
            )

            # Base survival reward
            if not done:
                total_reward += 0.1

            # Final safety check on total reward
            total_reward = np.nan_to_num(total_reward, nan=0.0, posinf=100.0, neginf=-100.0)
            total_reward = np.clip(total_reward, -100.0, 100.0)

            # Store individual components for logging
            reward_components = {
                "progress": float(progress_reward),
                "speed": float(speed_reward),
                "centerline": float(centerline_penalty),
                "smoothness": float(smoothness_reward),
                "inactivity": float(inactivity_penalty),
                "collision": float(collision_penalty),
                "total": float(total_reward),
            }

            self.step_count += 1

            return float(total_reward), reward_components
        
        except Exception as e:
            # Fallback reward in case of any calculation error
            print(f"Error in reward calculation: {e}")
            fallback_reward = 0.1 if not done else -10.0
            fallback_components = {
                "progress": 0.0,
                "speed": 0.0,
                "centerline": 0.0,
                "smoothness": 0.0,
                "inactivity": 0.0,
                "collision": -10.0 if done else 0.0,
                "total": fallback_reward,
            }
            return fallback_reward, fallback_components

    def update_config(self, new_config: Dict):
        """Update reward configuration during training"""
        if "weights" in new_config:
            self.config["weights"].update(new_config["weights"])
        if "parameters" in new_config:
            self.config["parameters"].update(new_config["parameters"])
        if "penalties" in new_config:
            self.config["penalties"].update(new_config["penalties"])
