# Real-Time Visualization Guide

This project includes comprehensive visualization tools for watching the RL agent learn and interact with the titration environment in real-time.

## Features

### 1. **Environment Render Method**
The `WeakAcidIndicatorEnv` now includes a full `render()` method with three modes:

- **`"human"`**: Displays matplotlib figure (non-blocking)
- **`"matplotlib"`**: Returns matplotlib figure object
- **`"rgb_array"`**: Returns RGB array for video recording

**Usage:**
```python
from src.titration_env import WeakAcidIndicatorEnv

env = WeakAcidIndicatorEnv()
obs, _ = env.reset()

# Take some steps
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        break

# Render the episode
fig = env.render(mode="matplotlib")
# fig.show() to display, or save with fig.savefig()
```

The render shows:
- **Left plot**: Full titration curve with start/current markers, target pH line, and equivalence point
- **Right plot**: Large indicator color circle with current state information

### 2. **Live Episode Viewer** (`watch_training.py`)

Watch episodes play out step-by-step in real-time, perfect for understanding agent behavior.

**Basic usage:**
```bash
cd env
source .venv/bin/activate
python watch_training.py
```

**With trained model:**
```bash
python watch_training.py --model models/ppo_weak_acid_indicator.zip
```

**Options:**
- `--model PATH`: Path to trained model (default: random policy)
- `--episodes N`: Number of episodes to watch (default: 1)
- `--delay SECONDS`: Delay between steps (default: 0.1)
- `--save`: Save episode visualizations to `episode_visualizations/`
- `--deterministic`: Use deterministic policy (default: True)

**Example:**
```bash
# Watch 3 episodes with 0.2s delay, saving visualizations
python watch_training.py --model models/ppo_weak_acid_indicator.zip \
                         --episodes 3 \
                         --delay 0.2 \
                         --save
```

The viewer shows:
- **Live titration curve** updating in real-time
- **Current indicator color** with state information
- **Step-by-step action selection** and rewards
- **Progress toward target pH 7.0**

### 3. **Training Progress Callback** (`training_callback.py`)

Automatically saves training visualizations during RL training, showing how the agent improves over time.

**Features:**
- Saves training progress plots every N episodes
- Tracks episode rewards, lengths, and statistics
- Creates comprehensive training dashboards

**Automatically integrated** into `train_rl.py` - no additional setup needed!

Visualizations are saved to `training_visualizations/` and include:
- Episode reward trends with moving averages
- Episode length trends
- Reward distribution histograms
- Training statistics summary

### 4. **Training with Visualization**

The training script now automatically includes visualization:

```bash
cd env
source .venv/bin/activate
python train_rl.py
```

During training:
- Progress visualizations saved every 500 episodes
- All plots saved to `training_visualizations/` directory
- Console output shows when visualizations are saved

## Visualization Outputs

### Episode Visualizations
- **Location**: `episode_visualizations/episode_XXX.png`
- **Content**: Full episode trajectory with indicator color
- **Use case**: Understanding single episode behavior

### Training Progress
- **Location**: `training_visualizations/training_progress_epXXXXXX.png`
- **Content**: Multi-panel dashboard of training metrics
- **Use case**: Tracking learning progress over time

## Portfolio Presentation Tips

For your portfolio submission to the AI RL lab:

1. **Create a demo video**: Use `watch_training.py` to record episodes showing:
   - Random policy (poor performance)
   - Early training (learning)
   - Fully trained policy (optimal behavior)

2. **Training progression**: Show training visualizations demonstrating:
   - Reward improvement over time
   - Episode length optimization
   - Convergence to target pH 7.0

3. **Key metrics to highlight**:
   - Final pH accuracy (should reach ~7.0)
   - Base volume precision (should reach ~49.7 mL)
   - Reward improvement trajectory
   - Sample efficiency

4. **Visual storytelling**:
   - Start with random policy visualization
   - Show training progress over episodes
   - End with trained policy successfully reaching pH 7.0

## Technical Details

### Render Implementation
- Uses matplotlib with non-blocking display
- Supports multiple render modes for flexibility
- Efficient RGB array generation for video recording
- Professional styling with clear labels and legends

### Performance
- Render calls are optimized for real-time viewing
- Training callbacks use efficient plotting
- Visualizations don't significantly slow down training

### Dependencies
All visualization features use only standard dependencies:
- `matplotlib` (already in requirements.txt)
- No additional packages needed

## Example Workflow

```bash
# 1. Train the agent (with automatic visualization)
python train_rl.py

# 2. Watch a few random episodes to understand the task
python watch_training.py --episodes 3 --delay 0.15

# 3. Watch trained agent perform
python watch_training.py \
    --model models/ppo_weak_acid_indicator.zip \
    --episodes 5 \
    --save \
    --delay 0.1

# 4. Review training progress
ls training_visualizations/
```

## Integration with Existing Code

- **All existing functionality preserved**
- `train_rl.py` works exactly as before (with added visualization)
- `visualize_policy.py` unchanged
- Environment API unchanged
- All backward compatible

The visualization is **additive** - it enhances the project without breaking anything.

