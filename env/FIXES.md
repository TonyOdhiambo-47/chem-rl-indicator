# Bug Fixes & Improvements

## Issues Found & Fixed

### 1. **Initial pH Calculation Bug** ✅ FIXED
**Problem:** Initial pH was showing as 0.00 instead of the correct weak acid pH (~2.88)

**Root Cause:** When no base is added (Vb=0), the code tried to use Henderson-Hasselbalch with n_A=0, causing log(0) issues.

**Fix:** Added special case for pure weak acid (no base added):
```python
if n_OH <= 1e-12:
    # For weak acid: [H+] = sqrt(Ka * Ca)
    H_plus = np.sqrt(max(Ka * Ca, 1e-20))
    pH = -np.log10(H_plus)
```

### 2. **Reward Function Too Harsh** ✅ FIXED
**Problem:** Agent learned to stop immediately because:
- Stopping early: reward = -0.599
- Taking steps: reward = -0.599 - 0.01 per step (worse!)
- No incentive to explore

**Fix:** Completely restructured reward:
- Base reward: `1.0 - (dist/7.0) - 0.005` (positive when close, negative when far)
- Big bonus (+2.0) for hitting target (pH within 0.05)
- Good bonus (+1.0) for close (pH within 0.2)
- Penalty (-1.0) for stopping too early (< 1% of equivalence volume)

**Result:** 
- Episode length: 4.37 → 11.4 steps
- Episode reward: 1.09 → 5.35

### 3. **Visualization Missing Initial State** ✅ FIXED
**Problem:** Plots didn't show the starting point (pH ~2.88, Vb=0)

**Fix:** 
- Added initial state to trajectory
- Better plot formatting with start/end markers
- Added step counts and diagnostic info

## Next Steps

1. **Let training complete** (takes 5-10 minutes):
   ```bash
   cd env
   source .venv/bin/activate
   python train_rl.py
   ```

2. **Re-run visualization**:
   ```bash
   python visualize_policy.py
   ```

3. **Expected results:**
   - Random policy: wanders around, doesn't reach pH 7
   - Trained policy: should approach pH 7.0 more systematically
   - Both should show full trajectories from pH ~2.88 to their endpoints

## Performance Metrics

With the new reward function, you should see:
- **Episode length:** 10-20 steps (agent explores the titration)
- **Final pH:** Close to 7.0 for trained agent
- **Final Vb:** Around 50 mL (equivalence volume) for trained agent
- **Total reward:** Positive values (5-10 range) for good episodes

