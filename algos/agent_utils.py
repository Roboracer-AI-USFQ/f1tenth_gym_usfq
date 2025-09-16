import torch
import numpy as np


class Base_Agent:
    def __init__(self, env, replay_buffer):
        self.env = env
        self.replay_buffer = replay_buffer

    def _do_random_actions(self, initial_steps: int) -> None:
        # Sample random actions depending on the type of action space
        # Access underlying environment if wrapped
        underlying_env = self.env
        while hasattr(underlying_env, "env"):
            underlying_env = underlying_env.env

        for i in range(initial_steps):
            state, info = (
                self.env.reset()
            )  # Reset env and get initial state (gymnasium format)
            state = np.concatenate(
                [
                    state["scans"][0].flatten(),
                    np.array(state["poses_x"]),
                    np.array(state["poses_y"]),
                    np.array(state["poses_theta"]),
                    np.array(state["linear_vels_x"]),
                    np.array(state["linear_vels_y"]),
                    np.array(state["ang_vels_z"]),
                ]
            )
            terminated, truncated = False, False
            while not terminated and not truncated:
                action = np.random.uniform(
                    low=np.array(
                        [
                            underlying_env.params["sv_min"],
                            underlying_env.params["v_min"],
                        ]
                    ),
                    high=[
                        underlying_env.params["sv_max"],
                        underlying_env.params["v_max"],
                    ],
                    size=(1, 2),
                ).astype(np.float32)

                next_state, reward, terminated, truncated, info = self.env.step(
                    action
                )  # gymnasium format
                next_state = np.concatenate(
                    [
                        next_state["scans"][0].flatten(),
                        np.array(next_state["poses_x"]),
                        np.array(next_state["poses_y"]),
                        np.array(next_state["poses_theta"]),
                        np.array(next_state["linear_vels_x"]),
                        np.array(next_state["linear_vels_y"]),
                        np.array(next_state["ang_vels_z"]),
                    ]
                )
                self.replay_buffer.add(
                    state, action[0], reward, next_state, terminated, truncated
                )
                state = next_state
        print(f"Filled replay buffer with {initial_steps} episodes of random actions.")

    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
