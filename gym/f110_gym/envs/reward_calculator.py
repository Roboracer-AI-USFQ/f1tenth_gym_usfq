import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional
from numba import njit


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point: np.ndarray, trajectory: np.ndarray) -> Tuple[np.ndarray, float, float, int]:
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
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0]**2 + diffs[:, 1]**2
    
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    
    projections = trajectory[:-1, :] + (t * diffs.T).T
    
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


class RewardCalculator:
    """
    Modular reward calculator for F1TENTH RL training
    """
    
    def __init__(self, 
                 waypoints_path: Optional[str] = None,
                 reward_config: Optional[Dict] = None,
                 track_length: float = 100.0):
        """
        Initialize the reward calculator
        
        Args:
            waypoints_path: Path to waypoints file for centerline calculation
            reward_config: Dictionary with reward weights and parameters
            track_length: Total length of the track for progress calculation
        """
        
        # Default reward configuration
        default_config = {
            'weights': {
                'progress': 0.4,
                'speed': 0.25,
                'centerline': 0.15,
                'smoothness': 0.1,
                'inactivity': 0.1
            },
            'penalties': {
                'collision': -100.0,
                'off_track': -50.0
            },
            'parameters': {
                'target_speed': 8.0,  # m/s
                'speed_tolerance': 2.0,
                'centerline_tolerance': 1.0,  # meters
                'inactivity_threshold': 0.5,  # m/s minimum speed
                'action_smoothness_window': 5
            }
        }
        
        self.config = reward_config if reward_config else default_config
        self.track_length = track_length
        
        # Load waypoints if provided
        self.waypoints = None
        if waypoints_path:
            try:
                self.waypoints = np.loadtxt(waypoints_path, delimiter=';', skiprows=3)
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
        if self.waypoints is None:
            # Fallback: use euclidean distance from start
            progress = np.linalg.norm(position - self.last_position)
            self.last_position = position.copy()
            return progress * 2.0  # Scale factor
        
        # Calculate progress using waypoints
        _, _, t, segment_idx = nearest_point_on_trajectory(position, self.waypoints)
        
        # Current progress as percentage of track completion
        current_progress = (segment_idx + t) / len(self.waypoints)
        
        # Handle track wrapping
        progress_delta = current_progress - self.last_progress
        if progress_delta < -0.5:  # Wrapped around
            progress_delta += 1.0
        elif progress_delta > 0.5:  # Went backwards significantly
            progress_delta -= 1.0
            
        self.last_progress = current_progress
        
        # Reward forward progress, penalize backward movement
        return max(0, progress_delta * self.track_length)
    
    def calculate_speed_reward(self, velocity: float) -> float:
        """
        Calculate reward based on optimal speed maintenance
        
        Args:
            velocity: Current longitudinal velocity
            
        Returns:
            Speed reward
        """
        target_speed = self.config['parameters']['target_speed']
        speed_tolerance = self.config['parameters']['speed_tolerance']
        
        speed_error = abs(velocity - target_speed)
        
        if speed_error <= speed_tolerance:
            # Reward optimal speed maintenance
            return 5.0 * (1.0 - speed_error / speed_tolerance)
        else:
            # Penalize being too far from optimal speed
            return -2.0 * (speed_error - speed_tolerance)
    
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
        
        _, distance, _, _ = nearest_point_on_trajectory(position, self.waypoints)
        tolerance = self.config['parameters']['centerline_tolerance']
        
        if distance <= tolerance:
            return 0.0  # No penalty within tolerance
        else:
            # Quadratic penalty for deviation
            excess_deviation = distance - tolerance
            return -3.0 * (excess_deviation ** 2)
    
    def calculate_action_smoothness_reward(self, action: np.ndarray) -> float:
        """
        Calculate reward for smooth action transitions
        
        Args:
            action: Current action [steering_velocity, longitudinal_velocity]
            
        Returns:
            Smoothness reward
        """
        if len(self.action_history) == 0:
            self.action_history.append(action.copy())
            return 0.0
        
        # Calculate action change magnitude
        action_change = np.linalg.norm(action - self.last_action)
        
        # Add to history
        self.action_history.append(action.copy())
        window_size = self.config['parameters']['action_smoothness_window']
        if len(self.action_history) > window_size:
            self.action_history.pop(0)
        
        self.last_action = action.copy()
        
        # Reward smooth transitions, penalize jerky movements
        if action_change < 0.5:  # Smooth transition threshold
            return 1.0 - 2.0 * action_change
        else:
            return -2.0 * action_change
    
    def calculate_inactivity_penalty(self, velocity: float) -> float:
        """
        Calculate penalty for staying stationary or moving too slowly
        
        Args:
            velocity: Current longitudinal velocity
            
        Returns:
            Inactivity penalty (negative)
        """
        min_speed = self.config['parameters']['inactivity_threshold']
        
        if abs(velocity) < min_speed:
            # Penalize inactivity - stronger penalty the longer stationary
            return -5.0 * (min_speed - abs(velocity))
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
        return self.config['penalties']['collision'] if collision else 0.0
    
    def calculate_total_reward(self, 
                             obs: Dict,
                             action: np.ndarray,
                             done: bool,
                             info: Dict = None) -> Tuple[float, Dict[str, float]]:
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
        # Extract state information
        position = np.array([obs['poses_x'][0], obs['poses_y'][0]])
        velocity = obs['linear_vels_x'][0]
        collision = obs['collisions'][0] if 'collisions' in obs else False
        
        # Calculate individual reward components
        progress_reward = self.calculate_progress_reward(position)
        speed_reward = self.calculate_speed_reward(velocity)
        centerline_penalty = self.calculate_centerline_penalty(position)
        smoothness_reward = self.calculate_action_smoothness_reward(action)
        inactivity_penalty = self.calculate_inactivity_penalty(velocity)
        collision_penalty = self.calculate_collision_penalty(collision)
        
        # Get weights from configuration
        weights = self.config['weights']
        
        # Calculate weighted total reward
        total_reward = (
            weights['progress'] * progress_reward +
            weights['speed'] * speed_reward +
            weights['centerline'] * centerline_penalty +
            weights['smoothness'] * smoothness_reward +
            weights['inactivity'] * inactivity_penalty +
            collision_penalty  # Collision penalty not weighted (should be large)
        )
        
        # Base survival reward
        if not done:
            total_reward += 0.1
        
        # Store individual components for logging
        reward_components = {
            'progress': progress_reward,
            'speed': speed_reward,
            'centerline': centerline_penalty,
            'smoothness': smoothness_reward,
            'inactivity': inactivity_penalty,
            'collision': collision_penalty,
            'total': total_reward
        }
        
        self.step_count += 1
        
        return total_reward, reward_components
    
    def update_config(self, new_config: Dict):
        """Update reward configuration during training"""
        if 'weights' in new_config:
            self.config['weights'].update(new_config['weights'])
        if 'parameters' in new_config:
            self.config['parameters'].update(new_config['parameters'])
        if 'penalties' in new_config:
            self.config['penalties'].update(new_config['penalties'])