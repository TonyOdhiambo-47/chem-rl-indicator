# Chem RL Indicator

> A reinforcement learning environment for automated weak acid-base titration with partial observability

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18+-61dafb.svg)](https://reactjs.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-0081a5.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.0+-9b59b6.svg)](https://stable-baselines3.readthedocs.io/)

A complete RL project that trains an agent to perform automated titration using only visual feedback from a pH indicator. The agent learns to adaptively control base addition, using larger steps when far from the target and smaller steps when approaching pH 7.0, all while operating under realistic constraints (50mL burette limit) and partial observability (color-only observations).

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Environment Details](#environment-details)
- [Training Configuration](#training-configuration)
- [Visualization](#visualization)
- [Results](#results)
- [Technical Highlights](#technical-highlights)
- [Documentation](#documentation)

---

## Overview

This project demonstrates **environment design for reinforcement learning** with a focus on:

- **Realistic physics**: Implements actual weak acid-strong base titration chemistry (Henderson-Hasselbalch, weak base hydrolysis, excess base calculations)
- **Partial observability**: Agent only sees indicator color (RGB), not true pH
- **Adaptive control**: Agent learns to use large steps early (pH 2-5.5) and small steps near target (pH 6.5-7.0)
- **Realistic constraints**: 50mL burette capacity, maximum 200 steps per episode
- **Professional training**: 10M timesteps with reliability-based early stopping (stops when ≥90% success rate achieved)

The agent learns a sophisticated titration strategy purely from trial-and-error, receiving only color feedback and reward signals.

---

## Key Features

### Chemistry & Physics
- **Accurate titration equations**: Weak acid buffer region, equivalence point, excess base calculations
- **Realistic indicator model**: Continuous color transition (yellow → green → blue) with neutral band at pH 7.0 ± 0.15
- **Physical constraints**: 50mL burette limit, realistic step sizes (0.1mL to 3.0mL)

### Reinforcement Learning
- **PPO algorithm**: State-of-the-art policy gradient method with optimized hyperparameters
- **Asymmetric reward shaping**: Heavy penalties for overshooting (pH > 7.0), large bonuses for hitting target
- **Reliability-based early stopping**: Training automatically stops when agent achieves ≥90% success rate (pH 6.9-7.05)
- **Portfolio-grade configuration**: 10M timesteps, [512, 512, 256] network, extensive exploration

### Visualization & UI
- **Interactive React dashboard**: Real-time visualization of agent's titration trajectory
- **Live agent animation**: Step-by-step playback of trained policy with color, pH, and reward tracking
- **Training progress plots**: Automatic visualization of learning curves and episode statistics
- **Comparison tools**: Side-by-side random vs trained policy visualization

---

## Project Structure

```
chem-rl-indicator/
├── env/                          # Python RL environment & training
│   ├── src/
│   │   └── titration_env.py     # Core environment (Gymnasium-compatible)
│   ├── train_rl.py               # PPO training script
│   ├── visualize_policy.py       # Random vs trained comparison
│   ├── watch_training.py         # Live episode visualization
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

---

## Installation

### Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- (Optional) TensorBoard for training visualization

### Python Environment (RL)

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

---

## Quick Start

### 1. Train the Agent

```bash
cd env
source .venv/bin/activate
python train_rl.py
```

**Training Details:**
- **Timesteps**: 10,000,000 (or until reliability target reached)
- **Parallel environments**: 16
- **Expected time**: ~30-40 minutes
- **Early stopping**: Training stops automatically when ≥90% of evaluation episodes achieve pH 6.9-7.05

The trained model is saved to `env/models/ppo_weak_acid_indicator.zip`.

### 2. Visualize Trained Policy

```bash
python visualize_policy.py
```

This generates `titration_policy_comparison.png` showing random vs trained agent trajectories.

### 3. Export Episode for Web Dashboard

```bash
python export_episode.py \
    --model models/ppo_weak_acid_indicator.zip \
    --output ../web/public/episode_data.json
```

### 4. Launch React Dashboard

```bash
cd web
npm run dev
```

Open `http://localhost:5173/` and switch to **"Live Agent Animation"** mode to see the trained agent in action.

---

## Environment Details

### Observation Space

The agent **never sees true pH**. Instead, it receives:

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

**Multi-component reward system:**

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

---

## Training Configuration

### PPO Hyperparameters

```python
Network: [512, 512, 256] with Tanh activation
n_steps: 4096
batch_size: 1024
n_epochs: 15
gamma: 0.998
learning_rate: 2e-4
ent_coef: 0.05 (high exploration)
clip_range: 0.15
```

### Early Stopping

Training uses a **reliability-based early stopping callback**:
- Evaluates agent every 50,000 timesteps
- Runs 32 evaluation episodes
- Stops when ≥90% success rate (pH 6.9-7.05) achieved for 3 consecutive evaluations
- Maximum training: 10M timesteps

---

## Visualization

### React Dashboard

The web dashboard provides:
- **Live titration curve**: Real-time pH vs volume plot
- **Indicator color display**: Large circle showing current color
- **Current state metrics**: pH, base added, distance to target
- **Action & rewards**: Step-by-step agent decisions
- **Progress tracking**: Episode completion percentage

### Training Visualizations

During training, progress plots are automatically saved to `env/training_visualizations/`:
- Episode reward curves
- Episode length distributions
- Training statistics

---

## Results

After training, the agent learns an **adaptive titration strategy**:

1. **Early phase (pH 2-5.5)**: Uses large steps (2-3 mL) for rapid progress
2. **Mid phase (pH 5.5-6.5)**: Slows down, uses medium steps (0.5-1.0 mL)
3. **Late phase (pH 6.5-7.0)**: Uses tiny steps (0.1-0.2 mL) to precisely hit target

**Performance:**
- Success rate: ≥90% (pH 6.9-7.05)
- Average episode length: ~100-150 steps
- Final pH: Typically 6.95-7.05
- Base volume: ~49-50 mL (near optimal)

---

## Technical Highlights

### Partial Observability

The agent operates under **partial observability**: it never sees true pH, only indicator color. This forces the agent to:
- Learn color-to-pH mapping implicitly
- Use volume ratio and step count as additional context
- Develop robust strategies that work despite uncertainty

### Reward Shaping Philosophy

The reward function was iteratively refined to address overshooting:
- **Initial version**: Simple distance penalty → agent overshot
- **Iteration 1**: Added anti-overshoot penalties → still overshot
- **Final version**: Extreme asymmetric penalties (pH > 7.0 heavily penalized) → reliable performance

### Realistic Constraints

The environment includes realistic physical constraints:
- **50mL burette limit**: Agent must work within finite titrant
- **Step size granularity**: Realistic laboratory step sizes (0.1mL minimum)
- **Maximum steps**: Prevents infinite episodes

---

## Documentation

Additional documentation available:

- **[CHEMISTRY_EXPLANATION.md](CHEMISTRY_EXPLANATION.md)**: Detailed chemistry behind the titration model
- **[env/PORTFOLIO_CONFIG.md](env/PORTFOLIO_CONFIG.md)**: Portfolio-grade training configuration details
- **[env/ROBUST_TRAINING.md](env/ROBUST_TRAINING.md)**: Robust training methodology
- **[env/ANTI_OVERSHOOT_CONFIG.md](env/ANTI_OVERSHOOT_CONFIG.md)**: Anti-overshoot reward design

---

## Why This Project?

This project demonstrates several important RL concepts:

1. **Environment design**: How to structure state, action, and reward spaces
2. **Partial observability**: Learning from incomplete information
3. **Reward shaping**: Iterative refinement to achieve desired behavior
4. **Realistic constraints**: Incorporating physical limitations
5. **Visualization**: Building tools to understand agent behavior

Perfect for:
- **RL portfolio projects**: Shows environment design and training expertise
- **Chemistry education**: Interactive visualization of titration concepts
- **Research**: Foundation for more complex chemistry RL problems

---

## Contributing

This is a portfolio project, but suggestions and improvements are welcome! Key areas for extension:

- Randomized starting conditions (different initial pH)
- Multiple acid types (different pKa values)
- Noise in observations (realistic sensor uncertainty)
- Multi-step titration (polyprotic acids)

---

## License

This project is open source and available for educational and research purposes.

---

## Acknowledgments

Built with:
- [Gymnasium](https://gymnasium.farama.org/) - RL environment standard
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - PPO implementation
- [React](https://reactjs.org/) - Frontend framework
- [Vite](https://vitejs.dev/) - Build tool
