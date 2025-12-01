# Live Agent Animation Guide

## Overview

The React app now includes a **live animation mode** that shows the RL agent's performance step-by-step, with:
- Real-time titration curve building
- Animated indicator color changes
- Step-by-step action visualization
- Reward tracking
- Professional portfolio-quality presentation

## Quick Start

### 1. Export Episode Data

From the `env` directory:

```bash
cd env
source .venv/bin/activate

# Export a random episode (for testing)
python export_episode.py --random --output ../web/public/episode_data.json

# Export a trained agent episode
python export_episode.py --model models/ppo_weak_acid_indicator.zip --output ../web/public/episode_data.json
```

### 2. Start React App

```bash
cd web
npm run dev
```

### 3. View Animation

1. Open the app in your browser (usually `http://localhost:5173/`)
2. Click **"Live Agent Animation"** button at the top
3. Watch the agent's performance unfold step-by-step!

## Features

### Live Animation
- **Auto-play**: Automatically steps through the episode
- **Play/Pause**: Control playback
- **Reset/End**: Jump to start or end
- **Speed control**: Adjust animation speed
- **Trajectory toggle**: Show full path or just current point

### Visualizations
- **Titration curve**: SVG-based, smooth animations
- **Indicator color**: Large animated circle with color transitions
- **Real-time metrics**: pH, volume, rewards, actions
- **Progress bar**: Visual progress through episode
- **Summary**: Final results when episode completes

### Data Display
- Current pH and base volume
- Action taken at each step
- Step and total rewards
- Distance to target pH
- V/Veq ratio

## Portfolio Presentation

This animation is perfect for:
- **Demos**: Show how the agent learns and performs
- **Presentations**: Visual storytelling of RL performance
- **Portfolio**: Impressive interactive visualization
- **Interviews**: Demonstrate understanding of RL + visualization

## Export Options

```bash
# Random policy
python export_episode.py --random --output ../web/public/episode_data.json

# Trained model (deterministic)
python export_episode.py --model models/ppo_weak_acid_indicator.zip --output ../web/public/episode_data.json

# Trained model (stochastic - more interesting)
python export_episode.py --model models/ppo_weak_acid_indicator.zip --output ../web/public/episode_data.json --no-deterministic

# Custom output path
python export_episode.py --model models/ppo_weak_acid_indicator.zip --output my_episode.json
```

## Technical Details

### Data Format

The exported JSON contains:
- `steps`: Array of step data (action, pH, volume, reward, etc.)
- `initial_state`: Starting conditions
- `target_pH`: Goal pH (7.0)
- `Veq_ml`: Equivalence volume
- `summary`: Episode statistics

### Animation System

- **React hooks**: `useState`, `useEffect` for state management
- **SVG rendering**: Scalable vector graphics for smooth curves
- **CSS animations**: Smooth transitions and effects
- **Auto-play**: Interval-based step progression

### Performance

- Efficient rendering with React
- Smooth 60fps animations
- Responsive design (works on mobile)
- Lightweight JSON data format

## Use Cases

1. **Training Progress**: Export episodes at different training stages
2. **Comparison**: Export random vs trained episodes
3. **Analysis**: Study agent behavior patterns
4. **Presentation**: Impressive visual demos
5. **Debugging**: Visualize what the agent is learning

## Tips

- Export multiple episodes to compare performance
- Use deterministic mode for consistent demos
- Use stochastic mode to show exploration
- Adjust animation speed for presentations
- Show trajectory for full context, hide for cleaner view

---

**This is portfolio-grade visualization that will impress any AI/RL lab!**

