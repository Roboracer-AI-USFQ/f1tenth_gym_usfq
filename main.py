from dotenv import load_dotenv
load_dotenv()
from algos.CrossQ import CrossQSAC_Agent
from f110_gym.envs.base_classes import Integrator
from utils.buffers import SimpleBuffer
from argparse import Namespace
try:
    import gymnasium as gym
except ImportError:
    import gym
import yaml
import wandb


if "__main__" == __name__:
    with open('examples/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    replay_buffer = SimpleBuffer(max_size=int(10e6), batch_size=256, gamma=0.99, n_steps=1, seed=0)

    env = gym.make('f110_gym:f110-v0', 
                   map=conf.map_path, 
                   map_ext=conf.map_ext, 
                   num_agents=1, 
                   timestep=0.01, 
                   integrator=Integrator.RK4,
                   reward_config='./config/reward_config.yaml',
                   waypoints_path='./examples/example_waypoints.csv')
    agent = CrossQSAC_Agent(env, replay_buffer=replay_buffer, use_wandb=False, state_dim=1086, action_dim=2)
    batch_size = 256
    rollout_eps = 1
    total_steps = 20000
    save_freq = 2000000
    
    agent.train(batch_size, rollout_eps, total_steps, save_freq)
