# Chem RL Indicator

A reinforcement learning environment for automated weak acid-base titration with partial observability. The agent learns to perform titration using only visual feedback from a pH indicator, without access to true pH values.

## Overview

This project implements a Gymnasium-compatible RL environment that simulates weak acid-strong base titration. The agent receives only RGB color values from a simulated pH indicator and must learn to control base addition to reach a target pH of 7.0. The environment includes realistic chemistry calculations, physical constraints (50mL burette capacity), and partial observability challenges.

## Key Features

### Chemistry Implementation

- Accurate weak acid-strong base titration calculations using Henderson-Hasselbalch equation
- Three distinct chemical regimes: buffer region, equivalence point, and excess base
- Realistic pH indicator model with continuous color transitions (yellow → green → blue)
- Neutral band enhancement around pH 7.0 ± 0.15 for visual guidance

### Reinforcement Learning

- PPO algorithm with optimized hyperparameters for stable learning
- Asymmetric reward shaping to prevent overshooting the target pH
- Reliability-based early stopping when agent achieves ≥90% success rate
- Extended training configuration: 10M timesteps with 16 parallel environments
- Network architecture: [512, 512, 256] with Tanh activation

### Visualization

- Interactive React dashboard for real-time visualization
- Live agent animation with step-by-step playback
- Training progress plots with automatic visualization
- Policy comparison tools (random vs trained)

## Project Structure

```
chem-rl-indicator/
├── env/                          # Python RL environment & training
│   ├── src/
│   │   └── titration_env.py     # Core environment (Gymnasium-compatible)
│   ├── train_rl.py               # PPO training script
│   ├── visualize_policy.py       # Random vs trained comparison
│   ├── export_episode.py         # Export episode data for React app
│   ├── training_callback.py      # Custom callbacks (visualization, early stopping)
│   ├── requirements.txt          # Python dependencies
│   └── models/                   # Trained PPO models
│
├── web/                          # React frontend
│   ├── src/
│   │   ├── App.tsx               # Main React component
│   │   ├── EpisodeAnimation.tsx  # Live agent dashboard
│   │   └── chem.ts               # Chemistry calculations (TypeScript)
│   └── public/
│       └── episode_data.json     # Exported episode data
│
└── README.md                     # This file
```

## Installation

### Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- (Optional) TensorBoard for training visualization

### Python Environment

```bash
cd env
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### React Frontend

```bash
cd web
npm install
```

## Usage

### Training the Agent

```bash
cd env
source .venv/bin/activate
python train_rl.py
```

Training configuration:
- Timesteps: 10,000,000 (or until reliability target reached)
- Parallel environments: 16
- Expected time: ~30-40 minutes
- Early stopping: Training stops automatically when ≥90% of evaluation episodes achieve pH 6.9-7.05

The trained model is saved to `env/models/ppo_weak_acid_indicator.zip`.

### Visualizing Trained Policy

```bash
python visualize_policy.py
```

This generates `titration_policy_comparison.png` showing random vs trained agent trajectories.

### Exporting Episode for Web Dashboard

```bash
python export_episode.py \
    --model models/ppo_weak_acid_indicator.zip \
    --output ../web/public/episode_data.json
```

### Launching React Dashboard

```bash
cd web
npm run dev
```

Open `http://localhost:5173/` and switch to "Live Agent Animation" mode to see the trained agent in action.

## Environment Details

### Observation Space

The agent never sees true pH. Instead, it receives:

```
[R, G, B, Vb_over_Veq, step_norm]
```

Where:
- `[R, G, B]`: Simulated indicator color (0-1 range)
  - Yellow (acidic, pH 2-6)
  - Green (neutral band, pH 6.85-7.15)
  - Blue (basic, pH 7.2+)
- `Vb_over_Veq`: Base volume relative to equivalence volume
- `step_norm`: Normalized step count (0-1)

### Action Space

Discrete actions:
- `0-5`: Add base in increments of 0.1, 0.2, 0.5, 1.0, 2.0, 3.0 mL
- `6`: Stop titration

### Reward Structure

Multi-component reward system:

1. **Exponential closeness reward**: `50.0 * exp(-|pH - 7.0| / 0.8)`
2. **Anti-overshoot penalties**: Heavy penalties for pH > 7.0 (up to -500 for severe overshoot)
3. **Progress bonuses**: `+5.0 * progress` when moving closer to target
4. **pH zone bonuses**: 
   - Buffer region (pH 3-6): +1.0
   - Target zone (pH 6.5-7.0): +5.0
   - Sweet spot (pH 6.9-7.0): +20.0
5. **Stopping rewards**: Up to +500 for stopping within 0.02 of pH 7.0

### Constraints

- **Burette capacity**: Maximum 50mL base can be added
- **Episode length**: Maximum 200 steps
- **Initial state**: Always starts at pH ~2.88 (weak acid, no base added)

## Training Configuration

### PPO Hyperparameters

```
Network: [512, 512, 256] with Tanh activation
n_steps: 4096
batch_size: 1024
n_epochs: 15
gamma: 0.998
learning_rate: 2e-4
ent_coef: 0.05
clip_range: 0.15
```

### Early Stopping

Training uses a reliability-based early stopping callback:
- Evaluates agent every 50,000 timesteps
- Runs 32 evaluation episodes
- Stops when ≥90% success rate (pH 6.9-7.05) achieved for 3 consecutive evaluations
- Maximum training: 10M timesteps

## Results

After training, the agent learns an adaptive titration strategy:

1. **Early phase (pH 2-5.5)**: Uses large steps (2-3 mL) for rapid progress
2. **Mid phase (pH 5.5-6.5)**: Slows down, uses medium steps (0.5-1.0 mL)
3. **Late phase (pH 6.5-7.0)**: Uses tiny steps (0.1-0.2 mL) to precisely hit target

**Performance:**
- Success rate: ≥90% (pH 6.9-7.05)
- Average episode length: ~100-150 steps
- Final pH: Typically 6.95-7.05
- Base volume: ~49-50 mL (near optimal)

## Technical Details

### Partial Observability

The agent operates under partial observability: it never sees true pH, only indicator color. This forces the agent to:
- Learn color-to-pH mapping implicitly
- Use volume ratio and step count as additional context
- Develop robust strategies that work despite uncertainty

### Reward Shaping Philosophy

The reward function was iteratively refined to address overshooting:
- Initial version: Simple distance penalty → agent overshot
- Iteration 1: Added anti-overshoot penalties → still overshot
- Final version: Asymmetric penalties (pH > 7.0 heavily penalized) → reliable performance

### Realistic Constraints

The environment includes realistic physical constraints:
- 50mL burette limit: Agent must work within finite titrant
- Step size granularity: Realistic laboratory step sizes (0.1mL minimum)
- Maximum steps: Prevents infinite episodes

## Documentation

Additional documentation available:

- [CHEMISTRY_EXPLANATION.md](CHEMISTRY_EXPLANATION.md): Detailed chemistry behind the titration model
- [env/VISUALIZATION.md](env/VISUALIZATION.md): Visualization tools and usage

## Dependencies

### Python
- gymnasium
- numpy
- stable-baselines3
- matplotlib

### JavaScript
- React 18+
- TypeScript
- Vite

## License

This project is open source and available for educational and research purposes.
