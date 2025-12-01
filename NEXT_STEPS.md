# Next Steps Guide

## Current Status
- Environment is set up and working
- All packages are installed
- Code is ready to run

---

## Step 1: Train the RL Agent (Optional)

Train a PPO agent to learn the titration task:

```bash
cd env
source .venv/bin/activate
python train_rl.py
```

**What this does:**
- Trains a PPO agent for ~200,000 timesteps
- Takes 5-10 minutes depending on your machine
- Saves model to `env/models/ppo_weak_acid_indicator.zip`

**Note:** You can stop training early with `Ctrl+C` if needed. The model saves periodically.

---

## Step 2: Visualize Policy Comparison

Compare random vs trained agent behavior:

```bash
cd env
source .venv/bin/activate
python visualize_policy.py
```

**What this does:**
- Runs one episode with a random policy
- Runs one episode with the trained PPO policy (if model exists)
- Creates `titration_policy_comparison.png` showing pH curves

**Expected result:** The trained agent should approach pH 7.0 more accurately than random.

---

## Step 3: Launch React Visualization App

See the indicator color change in real-time:

```bash
cd web
npm install  # Only needed first time
npm run dev
```

Then open the URL shown (usually `http://localhost:5173/`)

**What you can do:**
- Adjust acid/base concentrations and pKa
- Drag the slider to add base volume
- Watch the indicator color transition: yellow → green (neutral) → blue
- See pH calculated in real-time

---

## Step 4: Test the Environment Interactively

Quick test to see the environment in action:

```bash
cd env
source .venv/bin/activate
python -c "
from src.titration_env import WeakAcidIndicatorEnv
import numpy as np

env = WeakAcidIndicatorEnv()
obs, _ = env.reset()
print(f'Initial pH: {env._get_pH():.2f}')
print(f'Initial color (RGB): {obs[:3]}')

# Take a few random actions
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f'Step {i+1}: Action={action}, pH={info[\"pH\"]:.2f}, Reward={reward:.3f}')
    if terminated or truncated:
        break
"
```

---

## Step 5: Commit to Git (When Ready)

```bash
cd "/Users/tonyodhiambo/Desktop/Comp Chem/chem-rl-indicator"
git init
git add .
git commit -m "Initial commit: titration env + indicator visualizer"
```

---

## Recommended Order

1. **First:** Launch React app (`npm run dev`) - quickest way to see it working
2. **Then:** Train the agent (if you want to see RL in action)
3. **Finally:** Visualize and compare policies

---

## Tips

- The linter warnings in your IDE are just cache issues - the code runs fine
- Training can be interrupted and resumed
- The React app updates in real-time as you move the slider
- Try different pKa values to see how it affects the titration curve

