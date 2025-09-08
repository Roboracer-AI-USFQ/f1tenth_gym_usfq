import torch
import numpy as np

class Base_Agent:
    def __init__(self, env, replay_buffer):
        self.env = env
        self.replay_buffer = replay_buffer
        
    def _do_random_actions(self, initial_steps: int) -> None:
        # Sample random actions depending on the type of action space
        actions = np.random.uniform(low=np.array([self.env.params["sv_min"], self.env.params["v_min"]]), high=[self.env.params["sv_max"], self.env.params["v_max"]], size=(initial_steps, 2))

        for i in range(initial_steps):
            state, reward, terminated, info = self.env.reset(np.array([[0.7, 0, 1.37]]))  # Reset env and get initial state
            state = np.concatenate([
            state['scans'][0].flatten(),
            np.array(state['poses_x']),
            np.array(state['poses_y']),
            np.array(state['poses_theta']),
            np.array(state['linear_vels_x']),
            np.array(state['linear_vels_y']),
            np.array(state['ang_vels_z'])
            ])
            terminated, truncated = False, False  
            while not terminated and not truncated:  
                action = np.array([actions[i]])  # While rollouts the target actor is used
                #print(action)
                next_state, reward, terminated, info = self.env.step(action)
                # Print the robot position 
                print("Robot position: x =", next_state['poses_x'][0], ", y =", next_state['poses_y'][0], ", theta =", next_state['poses_theta'][0])
                next_state = np.concatenate([
                    next_state['scans'][0].flatten(),
                    np.array(next_state['poses_x']),
                    np.array(next_state['poses_y']),
                    np.array(next_state['poses_theta']),
                    np.array(next_state['linear_vels_x']),
                    np.array(next_state['linear_vels_y']),
                    np.array(next_state['ang_vels_z'])
                ])
                self.replay_buffer.add(state, action, reward, next_state, terminated, truncated)
                state = next_state
                # print(state)
                
    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    