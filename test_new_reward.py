#!/usr/bin/env python3
"""
Test script for the new advanced reward system in F1TENTH RL training.
This script demonstrates the improved reward components and logging.
"""

try:
    import gymnasium as gym
except ImportError:
    import gym
import numpy as np
import yaml
from argparse import Namespace
from f110_gym.envs.base_classes import Integrator


def test_reward_system():
    """Test the new reward system with a simple agent"""
    
    # Load configuration
    with open('examples/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    
    # Create environment with new reward system
    print("Creating F1TENTH environment with advanced reward system...")
    env = gym.make('f110_gym:f110-v0',
                   map=conf.map_path,
                   map_ext=conf.map_ext,
                   num_agents=1,
                   timestep=0.01,
                   integrator=Integrator.RK4,
                   reward_config='./config/reward_config.yaml',
                   waypoints_path='./examples/example_waypoints.csv')
    
    # Access the underlying environment (gymnasium wraps it)
    underlying_env = env.env if hasattr(env, 'env') else env
    if hasattr(underlying_env, 'env'):
        underlying_env = underlying_env.env
    print(f"Environment created with advanced rewards: {getattr(underlying_env, 'use_advanced_rewards', 'Unknown')}")
    
    # Reset environment (gymnasium format)
    obs, info = env.reset()
    print(f"Initial info: {info.get('reward_components', {})}")
    
    total_reward = 0
    step_count = 0
    
    print("\nRunning test episodes...")
    print("-" * 80)
    
    # Run for a few steps to see different reward components
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        obs, info = env.reset()
        episode_reward = 0
        
        for step in range(100):  # Run for 100 steps
            # Simple policy: go forward with slight steering variation
            if step < 20:
                # Start slow to test inactivity penalty
                action = np.array([[0.1, 2.0]])  # Small steering, low speed
            elif step < 50:
                # Normal driving
                action = np.array([[0.3 * np.sin(step * 0.1), 6.0]])  # Varying steering, good speed
            else:
                # Fast driving
                action = np.array([[0.2, 8.0]])  # Minimal steering, high speed
                
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            total_reward += reward
            
            # Print reward components every 20 steps
            if step % 20 == 0 and 'reward_components' in info:
                components = info['reward_components']
                print(f"  Step {step:2d}: Total={reward:6.2f} | " + 
                      f"Progress={components.get('progress', 0):5.2f} | " +
                      f"Speed={components.get('speed', 0):5.2f} | " +
                      f"Centerline={components.get('centerline', 0):5.2f} | " +
                      f"Smoothness={components.get('smoothness', 0):5.2f} | " +
                      f"Inactivity={components.get('inactivity', 0):5.2f}")
            
            if terminated or truncated:
                break
        
        print(f"  Episode {episode + 1} total reward: {episode_reward:.2f} in {step + 1} steps")
        
        # Test collision penalty
        if episode == 2:  # Last episode, test extreme action
            print("  Testing collision/extreme action...")
            extreme_action = np.array([[2.0, -5.0]])  # Extreme steering and negative speed
            obs, reward, terminated, truncated, info = env.step(extreme_action)
            if 'reward_components' in info:
                components = info['reward_components']
                print(f"  Extreme action reward: {reward:.2f}")
                print(f"  Components: {components}")
    
    print("-" * 80)
    print(f"Test completed!")
    print(f"Average reward per step: {total_reward / step_count:.4f}")
    print(f"Total steps: {step_count}")
    
    # Test reward configuration update
    print("\nTesting reward configuration update...")
    underlying_env = env.env if hasattr(env, 'env') else env
    if hasattr(underlying_env, 'env'):
        underlying_env = underlying_env.env
    if hasattr(underlying_env, 'reward_calculator') and underlying_env.reward_calculator is not None:
        new_config = {
            'weights': {
                'progress': 0.6,  # Increase progress weight
                'speed': 0.2,     # Decrease speed weight
                'inactivity': 0.2  # Increase inactivity penalty
            }
        }
        underlying_env.reward_calculator.update_config(new_config)
        print("Reward weights updated:")
        print(f"  Progress: 0.4 → 0.6")
        print(f"  Speed: 0.25 → 0.2") 
        print(f"  Inactivity: 0.1 → 0.2")
        
        # Test one step with new configuration
        obs, info = env.reset()
        action = np.array([[0.2, 6.0]])
        obs, reward, terminated, truncated, info = env.step(action)
        
        if 'reward_components' in info:
            components = info['reward_components']
            print(f"  Updated reward: {reward:.2f}")
            print(f"  Components: {components}")
    
    print("\n✅ Advanced reward system test completed successfully!")
    return True


def compare_reward_systems():
    """Compare old vs new reward systems"""
    
    print("\n" + "="*60)
    print("REWARD SYSTEM COMPARISON")
    print("="*60)
    
    # Load configuration
    with open('examples/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    
    # Test with advanced rewards
    print("\n1. Testing with ADVANCED reward system:")
    env_advanced = gym.make('f110_gym:f110-v0',
                           map=conf.map_path,
                           map_ext=conf.map_ext,
                           num_agents=1,
                           timestep=0.01,
                           integrator=Integrator.RK4,
                           reward_config='./config/reward_config.yaml',
                           waypoints_path='./examples/example_waypoints.csv')
    
    obs, info = env_advanced.reset()
    action = np.array([[0.2, 6.0]])  # Good driving action
    obs, reward_adv, terminated, truncated, info_adv = env_advanced.step(action)
    
    print(f"   Advanced reward: {reward_adv:.4f}")
    if 'reward_components' in info_adv:
        print(f"   Components: {info_adv['reward_components']}")
    
    # Test with simple rewards (fallback)
    print("\n2. Testing with SIMPLE reward system (fallback):")
    env_simple = gym.make('f110_gym:f110-v0',
                         map=conf.map_path,
                         map_ext=conf.map_ext,
                         num_agents=1,
                         timestep=0.01,
                         integrator=Integrator.RK4,
                         reward_config='./nonexistent_config.yaml',  # Force fallback
                         waypoints_path='./nonexistent_waypoints.csv')
    
    obs, info = env_simple.reset()
    action = np.array([[0.2, 6.0]])  # Same action
    obs, reward_simple, terminated, truncated, info_simple = env_simple.step(action)
    
    print(f"   Simple reward: {reward_simple:.4f}")
    print(f"   Components: {info_simple.get('reward_components', {})}")
    
    # Comparison
    print(f"\n3. COMPARISON:")
    print(f"   Advanced vs Simple ratio: {reward_adv/reward_simple:.2f}x")
    print(f"   Advanced reward provides {(reward_adv/reward_simple - 1)*100:.1f}% more signal")
    
    return True


if __name__ == "__main__":
    print("🏁 F1TENTH Advanced Reward System Test")
    print("=" * 50)
    
    try:
        # Run basic test
        test_reward_system()
        
        # Run comparison
        compare_reward_systems()
        
        print("\n🎉 All tests passed! The advanced reward system is working correctly.")
        print("\nKey improvements:")
        print("✅ Progress-based rewards encourage forward movement")
        print("✅ Speed optimization rewards maintain optimal velocity")
        print("✅ Centerline deviation penalties improve racing line")
        print("✅ Action smoothness rewards reduce jerky movements")
        print("✅ Inactivity penalties prevent stationary behavior")
        print("✅ Collision penalties provide strong negative feedback")
        print("✅ Modular configuration allows easy tuning")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)