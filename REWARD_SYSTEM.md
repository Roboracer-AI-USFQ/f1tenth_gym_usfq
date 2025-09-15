# 🏁 F1TENTH Advanced Reward System

## Overview

This project implements a comprehensive reward system for F1TENTH reinforcement learning training, replacing the simple timestep-based reward with a multi-component system that encourages better racing behavior.

## 🚀 Key Features

- **Progress Rewards**: Encourages forward movement along the racing line
- **Speed Optimization**: Rewards maintaining optimal racing velocity
- **Centerline Tracking**: Penalizes deviation from the ideal racing line
- **Action Smoothness**: Rewards smooth driving, penalizes jerky movements
- **Inactivity Prevention**: Strong penalties for staying stationary or moving too slowly
- **Collision Avoidance**: Severe penalties for crashes
- **Configurable Weights**: Easy tuning of reward components
- **Comprehensive Logging**: Detailed component tracking for training analysis

## 📁 Files Added/Modified

### New Files
```
gym/f110_gym/envs/reward_calculator.py  # Core reward calculation logic
config/reward_config.yaml              # Reward configuration file
test_new_reward.py                     # Test script for reward system
REWARD_SYSTEM.md                       # This documentation
```

### Modified Files
```
gym/f110_gym/envs/f110_env.py          # Integrated RewardCalculator
main.py                               # Updated to use new reward system
algos/CrossQ.py                       # Added reward component logging
```

## 🔧 Installation & Setup

1. **Install dependencies** (if not already installed):
   ```bash
   pip install pyyaml numba
   ```

2. **Test the reward system**:
   ```bash
   python test_new_reward.py
   ```

3. **Run training with new rewards**:
   ```bash
   python main.py
   ```

## 🎯 Reward Components

### 1. Progress Reward (Weight: 0.4)
- **Purpose**: Encourages forward movement along the track
- **Calculation**: Based on distance covered along waypoint trajectory
- **Range**: 0 to ~10 per step (depending on speed and track)

### 2. Speed Reward (Weight: 0.25) 
- **Purpose**: Maintains optimal racing velocity
- **Target Speed**: 8.0 m/s (configurable)
- **Tolerance**: ±2.0 m/s (configurable)
- **Range**: -10 to +5 per step

### 3. Centerline Penalty (Weight: 0.15)
- **Purpose**: Keeps vehicle on optimal racing line
- **Tolerance**: 1.0 m deviation (configurable)
- **Calculation**: Quadratic penalty beyond tolerance
- **Range**: -∞ to 0 (penalty only)

### 4. Action Smoothness (Weight: 0.1)
- **Purpose**: Encourages smooth, realistic driving
- **Calculation**: Based on action change magnitude
- **Range**: -10 to +1 per step

### 5. Inactivity Penalty (Weight: 0.1)
- **Purpose**: **Prevents staying stationary or moving too slowly**
- **Threshold**: 0.5 m/s minimum speed (configurable)
- **Calculation**: Linear penalty below threshold
- **Range**: -25 to 0 per step

### 6. Collision Penalty (Not weighted)
- **Purpose**: Strong negative feedback for crashes
- **Value**: -100 (configurable)
- **Triggers**: On collision detection

## ⚙️ Configuration

### Reward Weights
Edit `config/reward_config.yaml` to adjust component weights:

```yaml
reward_system:
  weights:
    progress: 0.4      # Forward movement
    speed: 0.25        # Optimal velocity
    centerline: 0.15   # Racing line adherence
    smoothness: 0.1    # Smooth actions
    inactivity: 0.1    # Anti-stationary penalty
```

### Parameters
Fine-tune reward calculation parameters:

```yaml
parameters:
  target_speed: 8.0              # Optimal speed (m/s)
  speed_tolerance: 2.0           # Speed tolerance (m/s)
  centerline_tolerance: 1.0      # Max centerline deviation (m)
  inactivity_threshold: 0.5      # Min speed to avoid penalty (m/s)
  action_smoothness_window: 5    # Actions to consider for smoothness
```

### Penalties
Configure large penalty values:

```yaml
penalties:
  collision: -100.0    # Collision penalty
  off_track: -50.0     # Off-track penalty (future use)
```

## 🎓 Curriculum Learning

The configuration file includes three training phases for progressive difficulty:

### Phase 1: Basic Movement
- **Focus**: Avoid collisions, basic movement
- **Speed Target**: 4.0 m/s
- **Progress Weight**: 0.5 (higher)
- **Inactivity Weight**: 0.15 (higher)

### Phase 2: Speed & Centerline
- **Focus**: Introduce speed and line requirements
- **Speed Target**: 6.0 m/s
- **Balanced weights**

### Phase 3: Full Racing
- **Focus**: Optimal racing performance
- **Speed Target**: 8.0 m/s  
- **Full reward system enabled**

## 📊 Logging & Monitoring

### Weights & Biases Integration
The system logs detailed reward components to W&B:

```python
wandb.log({
    "reward/total": total_reward,
    "reward/progress": progress_component,
    "reward/speed": speed_component,
    "reward/centerline": centerline_component,
    "reward/smoothness": smoothness_component,
    "reward/inactivity": inactivity_component,
    "reward/collision": collision_component
})
```

### Console Output
Monitor training progress with detailed component breakdowns:

```
Step 20: Total=  2.45 | Progress= 1.20 | Speed= 4.50 | Centerline= 0.00 | Smoothness= 0.75 | Inactivity= 0.00
```

## 🔄 Dynamic Configuration

Update reward weights during training:

```python
new_config = {
    'weights': {
        'progress': 0.6,    # Increase progress importance
        'inactivity': 0.2   # Increase inactivity penalty
    }
}
env.reward_calculator.update_config(new_config)
```

## 🧪 Testing

### Basic Test
```bash
python test_new_reward.py
```

### Reward Comparison
The test script compares old vs new reward systems:
- **Old System**: Simple timestep (0.01)
- **New System**: Multi-component rewards (typically 2-10x higher signal)

### Expected Output
```
🏁 F1TENTH Advanced Reward System Test
==================================================
Creating F1TENTH environment with advanced reward system...
Advanced reward system initialized
Loaded 786 waypoints from ./examples/example_waypoints.csv
Environment created with advanced rewards: True

Running test episodes...
--------------------------------------------------------------------------------

Episode 1:
  Step  0: Total=  2.15 | Progress= 0.00 | Speed=-0.50 | Centerline= 0.00 | Smoothness= 0.00 | Inactivity=-2.25
  Step 20: Total=  3.20 | Progress= 1.80 | Speed= 3.40 | Centerline= 0.00 | Smoothness= 0.90 | Inactivity= 0.00
  ...
```

## 🎯 Training Improvements

### Expected Benefits
- **Faster Convergence**: Rich reward signal accelerates learning
- **Better Racing Lines**: Centerline tracking improves lap times
- **Smoother Control**: Action smoothness reduces jerky behavior
- **Collision Avoidance**: Strong penalties improve safety
- **Speed Optimization**: Balanced speed rewards optimize performance
- **No Stationary Behavior**: Inactivity penalties prevent getting stuck

### Recommended Hyperparameters
- **Learning Rate**: 3e-4 to 1e-3 (higher due to richer signal)
- **Batch Size**: 256-512
- **Replay Buffer**: 1M+ experiences
- **Training Frequency**: Every episode or every few steps

## 🔍 Troubleshooting

### Common Issues

1. **No waypoints loaded**
   - Check file path in configuration
   - Ensure waypoints file exists and is formatted correctly
   - Verify CSV delimiter and column structure

2. **Reward system not initializing**
   - Check YAML configuration file syntax
   - Verify all required dependencies are installed
   - Review error messages for missing components

3. **Poor training performance**
   - Adjust reward weights in configuration
   - Start with Phase 1 parameters for curriculum learning
   - Monitor individual reward components for imbalances

4. **Extreme reward values**
   - Check for NaN or infinite values in observations
   - Verify waypoint trajectory is reasonable
   - Adjust penalty values if too harsh

### Debug Mode
Enable detailed logging by setting environment variable:
```bash
export F1TENTH_DEBUG=1
python main.py
```

## 🔮 Future Enhancements

### Planned Features
- [ ] **Adaptive Rewards**: Dynamic weight adjustment based on training progress
- [ ] **Multi-Agent Rewards**: Competitive/collaborative multi-agent training
- [ ] **Track-Specific Tuning**: Automatic parameter adjustment per track
- [ ] **Safety Constraints**: Hard constraints for real-world deployment
- [ ] **Curriculum Automation**: Automatic progression through training phases
- [ ] **ROS2 Integration**: Bridge for real robot deployment

### Advanced Components
- [ ] **Tire Model Rewards**: Realistic tire usage and grip limits
- [ ] **Energy Efficiency**: Battery usage optimization
- [ ] **Overtaking Rewards**: Multi-agent racing strategies
- [ ] **Weather Adaptivity**: Variable conditions training

## 📚 References

- [F1TENTH Competition](https://f1tenth.org/)
- [CrossQ Paper](https://arxiv.org/abs/1902.05605)
- [SAC Algorithm](https://arxiv.org/abs/1801.01290)
- [Reward Shaping Theory](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-reward-component`)
3. Make changes and test thoroughly
4. Update documentation
5. Submit pull request with detailed description

## 📄 License

This project maintains the same license as the original F1TENTH Gym environment.

---

**Happy Racing! 🏎️💨**