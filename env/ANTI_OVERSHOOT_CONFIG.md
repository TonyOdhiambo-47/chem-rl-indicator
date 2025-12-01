# Anti-Overshoot Training Configuration

## Problem: Agent Overshooting pH 7.0

The agent was overshooting the target pH 7.0, going to pH 8+ instead of stopping at the optimal point.

## Solution: Asymmetric Reward + Optimized Training

### 1. **Asymmetric Reward Function**

**Key Insight:** Overshooting is MUCH worse than undershooting!

**Penalties for Overshooting:**
- pH > 8.0: **-100 reward** (MASSIVE penalty)
- pH > 7.5: **-50 reward** (Very heavy penalty)
- pH > 7.2: **-20 reward** (Heavy penalty)
- pH > 7.1: **-10 reward** (Moderate penalty)
- pH 7.0-7.1: **-2 reward** (Small penalty)

**Bonuses for Target Zone:**
- pH 6.9-7.0: **+20 reward** (Sweet spot - highest reward)
- pH 6.5-7.0: **+5 reward** (Target zone)
- pH 3-6: **+1 reward** (Buffer region)

**Why Asymmetric:**
- Going above pH 7.0 means you've passed the optimal point
- Can't "undo" overshooting (already added too much base)
- Must learn to stop BEFORE reaching pH 7.0
- Undershooting is recoverable (can add more base)

### 2. **Progress Penalties**

**Moving Away from Target:**
- If moving away from pH 7.0: reduced progress bonus
- If moving further ABOVE pH 7.0: extra -5 penalty
- Strongly discourages continuing after overshooting

### 3. **Volume-Based Anti-Overshoot**

**Past Equivalence Penalties:**
- If V/Veq > 1.0 AND pH > 7.0: **-15 penalty**
- Continuing past equivalence when already overshot is heavily penalized
- Forces agent to stop before equivalence

### 4. **Stopping Rewards (Asymmetric)**

**Stopping at/below target (GOOD):**
- pH within 0.02: **+150 bonus**
- pH within 0.05: **+75 bonus**
- pH within 0.1: **+40 bonus**

**Stopping above target (BAD):**
- pH > 8.0: **-150 penalty**
- pH > 7.5: **-75 penalty**
- pH > 7.2: **-40 penalty**
- pH > 7.1: **-20 penalty**

---

## Optimized Hyperparameters

### Network Architecture
- **Size:** [512, 512, 256] (was [256, 256, 128])
- **Why:** Larger network = more capacity to learn complex stopping strategy
- **Activation:** Tanh (was ReLU)
- **Why:** Tanh provides smoother gradients, better for precise control

### Training Parameters
- **Timesteps:** 10,000,000 (was 5M) - **~30-40 minutes**
- **Batch size:** 512 (was 256) - More stable gradients
- **Epochs:** 15 (was 10) - More optimization per rollout
- **Gamma:** 0.998 (was 0.995) - Longer-term planning
- **Learning rate:** 2e-4 (was 3e-4) - More stable, less overshooting
- **Entropy:** 0.05 (was 0.03) - Even more exploration
- **Clip range:** 0.15 (was 0.2) - Tighter updates, more stable

### Why These Changes Help

1. **Larger Network:**
   - Can learn subtle patterns in color changes
   - Better at recognizing "almost there" signals
   - More capacity for complex stopping strategy

2. **Tanh Activation:**
   - Smoother gradients than ReLU
   - Better for fine-grained control
   - Less likely to overshoot due to activation spikes

3. **Higher Gamma (0.998):**
   - Agent thinks more long-term
   - Considers consequences of overshooting
   - Plans ahead better

4. **Lower Learning Rate:**
   - More stable updates
   - Less likely to make drastic changes
   - Better fine-tuning near target

5. **More Epochs (15):**
   - Better optimization per rollout
   - Learns from mistakes more thoroughly
   - More stable convergence

6. **Tighter Clip Range (0.15):**
   - Prevents large policy updates
   - More stable learning
   - Less likely to overshoot due to policy jumps

---

## Expected Learning Progression

### Early Training (0-2M steps):
- Many overshooting mistakes
- Heavy penalties teach agent: "don't go above 7.0"
- Episode rewards: -50 to 50 (many penalties)

### Mid Training (2-6M steps):
- Starting to learn stopping strategy
- Fewer overshoots, more undershoots
- Episode rewards: 50-150

### Late Training (6-10M steps):
- Consistent stopping at pH 6.9-7.0
- Rarely overshoots
- Episode rewards: 150-250+

---

## Success Criteria

After 10M timesteps, agent should:

1. **Never overshoot:**
   - Final pH: 6.9-7.0 (never above 7.1)
   - Success rate: >90% within target range

2. **Optimal stopping:**
   - Base volume: 49-50 mL
   - Stops just before equivalence
   - Uses smaller steps near target

3. **Robust behavior:**
   - Consistent across episodes
   - Handles edge cases
   - Learns from mistakes

---

## Running Training

```bash
cd env
source .venv/bin/activate
python train_rl.py
```

**Training Time:** ~30-40 minutes

**What to Watch:**
- Episode rewards increasing
- Overshooting episodes decreasing
- Final pH converging to 6.9-7.0 range
- Visualizations showing learning curve

---

## Why This Works

### Asymmetric Penalties
- Creates strong gradient away from overshooting
- Agent learns: "stopping at 6.9 is better than 7.1"
- Heavy penalties make overshooting very costly

### Extended Training
- 10M timesteps = extensive exploration
- Sees many overshooting mistakes
- Learns robust stopping strategy

### Optimized Hyperparameters
- Larger network = more capacity
- Tanh = smoother control
- Higher gamma = better planning
- Lower LR = more stable

---

This configuration should produce a model that **consistently stops at pH 6.9-7.0 without overshooting!**

