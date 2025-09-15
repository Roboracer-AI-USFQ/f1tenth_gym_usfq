from algos.CrossQ import CrossQSAC_Agent
from networks.encoders import CNNEncoder
from f110_gym.envs.base_classes import Integrator
from utils.buffers import SimpleBuffer
from argparse import Namespace
import gym
import yaml
import wandb

if "__main__" == __name__:

    with open('examples/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    replay_buffer = SimpleBuffer(max_size=int(10e6), batch_size=256, gamma=0.99, n_steps=1, seed=0)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)
    
    cnn_encoder = CNNEncoder(
        input_dim=conf.encoder.input_dim,  # (channels, width, height)
        num_layers=conf.encoder.num_layers,
        hidden_size=conf.encoder.hidden_size,
        history_length=conf.encoder.history_length,
        concat_action=conf.encoder.concat_action,
        dropout=conf.encoder.dropout,
    )
    
    agent = CrossQSAC_Agent(env, replay_buffer=replay_buffer, use_wandb=True, 
                            state_dim=1086, action_dim=2, encoder=cnn_encoder)
    
    batch_size = 256
    rollout_eps = 1
    total_steps = 20000
    save_freq = 2000000
    
    agent.train(batch_size, rollout_eps, total_steps, save_freq)
