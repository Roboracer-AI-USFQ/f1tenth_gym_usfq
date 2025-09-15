try:
    from gymnasium.envs.registration import register
except ImportError:
    from gym.envs.registration import register

register(
	id='f110-v0',
	entry_point='f110_gym.envs:F110Env',
	)
