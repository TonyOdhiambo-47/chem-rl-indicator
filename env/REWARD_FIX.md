# Reward Function & Training Improvements

## Key Finding
**pH 7.0 occurs at 49.72 mL** (99.4% of the 50 mL equivalence point)
- The agent was stopping at 31.90 mL (pH 5.01) - way too early!
- Need to continue adding base until ~49.7 mL to reach pH 7.0

## Fixes Applied

### 1. **Exponential Reward Function**
Changed from linear to exponential reward:
- **Old:** `reward = 1.0 - (dist/7.0)` → At pH 5.01: reward = 0.71
- **New:** `reward = 10.0 * exp(-dist/1.5)` → At pH 5.01: reward = 2.05, At pH 7.0: reward = 10.0

This creates a **strong gradient** toward pH 7.0.

### 2. **Progress Bonus**
Added reward for making progress toward target:
```python
progress = last_dist - current_dist
reward += 2.0 * progress  # Bonus for getting closer
```

### 3. **Much Larger Stopping Bonuses**
- pH within 0.05: **+20.0 bonus** (was +2.0)
- pH within 0.1: **+10.0 bonus** (was +1.0)
- pH within 0.2: **+5.0 bonus** (was +0.5)
- Stopping far: **-5.0 penalty** (was -0.5)

### 4. **Training Parameters**
- Max steps: 40 → **60** (more room to explore)
- Step sizes: Added **2.0 mL** option for faster progress
- Training timesteps: 200k → **300k** (more training)
- Added exploration bonus (`ent_coef=0.01`)

## Expected Results After Retraining

- **Random policy:** Still random, but should explore more (up to 60 steps)
- **Trained policy:** Should reach pH ~7.0 at Vb ~49.7 mL
- **Total reward:** Should be much higher (10-30 range for good episodes)

## Next Steps

1. **Retrain with new reward function:**
   ```bash
   cd env
   source .venv/bin/activate
   python train_rl.py
   ```

2. **Re-run visualization:**
   ```bash
   python visualize_policy.py
   ```

3. **Expected:**
   - Trained agent reaches pH 6.9-7.1
   - Vb around 49-50 mL
   - Much better than previous pH 5.01

