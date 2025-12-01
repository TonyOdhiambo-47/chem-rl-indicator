# Chemistry Behind the Titration Model

## Overview

This project implements a **realistic weak acid-strong base titration** using actual analytical chemistry equations. The model calculates pH at every point in the titration curve using three distinct chemical regimes.

---

## Chemical Background

### The Reaction

We're titrating a **monoprotic weak acid (HA)** with a **strong base (NaOH)**:

```
HA(aq) + OH⁻(aq) → A⁻(aq) + H₂O(l)
```

**Example:** Acetic acid (CH₃COOH) with sodium hydroxide (NaOH)
- HA = CH₃COOH (weak acid, pKa = 4.76)
- A⁻ = CH₃COO⁻ (conjugate base)
- NaOH = strong base (completely dissociates)

### Key Constants

- **Ka** = acid dissociation constant = 10^(-pKa)
- **Kw** = water autoionization constant = 10⁻¹⁴ (at 25°C)
- **Kb** = base dissociation constant = Kw/Ka (for conjugate base)

---

## Three Chemical Regimes

The pH calculation switches between three distinct regions based on how much base has been added:

### 1. **Before Equivalence: Buffer Region**

**Condition:** `n_OH < n_HA0` (less base added than initial acid)

**Chemistry:**
- We have a **buffer solution**: mixture of weak acid (HA) and its conjugate base (A⁻)
- The base converts HA → A⁻: `HA + OH⁻ → A⁻ + H₂O`
- Moles of A⁻ = moles of OH⁻ added
- Moles of HA remaining = initial HA - OH⁻ added

**pH Calculation: Henderson-Hasselbalch Equation**

```python
n_A = n_OH              # moles of conjugate base
n_HA = n_HA0 - n_OH     # moles of weak acid remaining
ratio = n_A / n_HA
pH = pKa + log₁₀(ratio)
```

**Why this works:**
- Buffer solutions resist pH changes
- pH depends on the **ratio** of [A⁻]/[HA], not absolute concentrations
- When ratio = 1 (equal amounts), pH = pKa

**Special Case: Pure Weak Acid (Vb = 0)**
```python
# No base added yet
[H⁺] = √(Ka × Ca)      # From Ka = [H⁺][A⁻]/[HA] ≈ [H⁺]²/Ca
pH = -log₁₀([H⁺])
```

**Example (Acetic acid, pKa=4.76, Ca=0.1M):**
- Initial pH = -log₁₀(√(1.74×10⁻⁵ × 0.1)) ≈ **2.88**

---

### 2. **At Equivalence Point**

**Condition:** `n_OH ≈ n_HA0` (exactly enough base to neutralize all acid)

**Chemistry:**
- All HA has been converted to A⁻
- Solution now contains only the **conjugate base (A⁻)** in water
- A⁻ acts as a **weak base**: `A⁻ + H₂O ⇌ HA + OH⁻`
- pH will be **above 7** (basic) because A⁻ hydrolyzes water

**pH Calculation: Weak Base Hydrolysis**

```python
C_A = n_HA0 / Vtot      # Concentration of A⁻
Kb = Kw / Ka           # Base dissociation constant
[OH⁻] = √(Kb × C_A)    # From Kb = [OH⁻]²/C_A (approximation)
pOH = -log₁₀([OH⁻])
pH = 14 - pOH
```

**Why pH > 7:**
- The conjugate base of a weak acid is a weak base
- It pulls H⁺ from water, leaving OH⁻
- For acetic acid (pKa=4.76): pH at equivalence ≈ **8.73**

---

### 3. **After Equivalence: Excess Strong Base**

**Condition:** `n_OH > n_HA0` (more base added than needed)

**Chemistry:**
- All acid has been neutralized
- Excess strong base (NaOH) remains
- Strong base completely dissociates: `NaOH → Na⁺ + OH⁻`
- pH is determined by excess [OH⁻]

**pH Calculation: Strong Base**

```python
n_excess = n_OH - n_HA0    # Excess moles of OH⁻
[OH⁻] = n_excess / Vtot     # Concentration of excess OH⁻
pOH = -log₁₀([OH⁻])
pH = 14 - pOH
```

**Why this works:**
- Strong base dominates pH calculation
- Excess OH⁻ directly determines pOH
- pH rises rapidly as more base is added

**Example:**
- If we add 55 mL of 0.1M NaOH to 50 mL of 0.1M acetic acid:
- Excess = (55-50) × 0.1 = 0.5 mmol in 105 mL
- [OH⁻] = 0.5/105 = 0.00476 M
- pOH = 2.32, pH = **11.68**

---

## 🎨 pH Indicator Model

### Chemical Principle

**Universal indicators** are weak acids/bases that change color based on pH:

```
HIn (acid form, yellow) ⇌ In⁻ (base form, blue) + H⁺
```

The equilibrium is governed by:
```
Ka_ind = [H⁺][In⁻] / [HIn]
```

### Color Calculation

**1. Acid-Base Equilibrium Fraction**

```python
# Fraction of base form (In⁻)
f_base = 1 / (1 + 10^(pKa_ind - pH))
f_acid = 1 - f_base
```

**Why:**
- When pH = pKa_ind: f_base = 0.5 (equal amounts, transition point)
- When pH << pKa_ind: f_base → 0 (mostly acid form, yellow)
- When pH >> pKa_ind: f_base → 1 (mostly base form, blue)

**2. Color Blending**

```python
# Blend acid and base colors based on fraction
color = f_acid × yellow + f_base × blue
```

**3. Neutral Band Enhancement**

```python
# Extra green color near pH 7.0 ± 0.15
if |pH - 7.0| < 0.15:
    # Mix in green color, strongest at pH = 7.0
    w = 1 - |pH - 7.0| / 0.15
    color = (1-w) × acid/base_mix + w × green
```

**Why a neutral band?**
- Real indicators often show a "neutral" color (green) around pH 7
- This helps the agent identify when it's close to the target
- Creates a visual "sweet spot" that guides learning

---

## The Titration Curve

### Shape Characteristics

1. **Initial (Vb = 0):** pH ≈ 2.88 (pure weak acid)
2. **Buffer Region (0 < Vb < Veq):** Gradual pH increase
   - Steepest near equivalence (buffer capacity decreases)
   - pH = pKa when Vb = Veq/2 (half-equivalence point)
3. **Equivalence (Vb = Veq):** pH ≈ 8.73 (steep rise)
4. **After Equivalence (Vb > Veq):** pH continues rising, less steep

### Why pH 7.0 is Before Equivalence

For weak acid-strong base titrations:
- **Equivalence point** (all acid neutralized): pH ≈ 8.73
- **Target pH 7.0** occurs in the **buffer region**, before equivalence
- This happens at ~99.4% of equivalence volume (49.72 mL out of 50 mL)

**The agent must learn:**
- Stop **before** equivalence
- Recognize the color transition to green (neutral band)
- Use small steps near the target for precision

---

## Real-World Accuracy

### What's Realistic

- **Accurate equations:** Uses actual analytical chemistry formulas  
- **Correct regimes:** Properly handles all three titration regions  
- **Realistic pKa:** Uses acetic acid (pKa=4.76) as default  
- **Indicator behavior:** Models real indicator color transitions  
- **Volume calculations:** Proper mole and concentration math  

### Simplifications

- **Activity coefficients:** Assumes ideal solutions (activity = concentration)  
- **Temperature:** Fixed at 25°C (Kw = 10⁻¹⁴)  
- **Ionic strength:** Doesn't account for salt effects  
- **Dilution:** Assumes perfect mixing (no local concentration gradients)  

**For RL purposes, these simplifications are fine:**
- The physics are correct enough to be realistic
- The agent learns meaningful strategies
- The environment is deterministic and reproducible

---

## Why This Matters for RL

### 1. **Realistic Partial Observability**

The agent only sees **color**, not pH directly. This mirrors:
- Real lab scenarios (visual indicators)
- Sensor limitations in robotics
- Real-world POMDPs

### 2. **Non-Linear Dynamics**

The pH curve is **non-linear**:
- Small volume changes near equivalence cause large pH jumps
- Agent must learn to use smaller steps near target
- Tests ability to handle complex state spaces

### 3. **Multi-Stage Learning**

The agent must learn:
- **Exploration phase:** Large steps to get to buffer region
- **Precision phase:** Small steps near target
- **Stopping decision:** When to stop (action 3)

### 4. **Reward Shaping Reflects Chemistry**

Our reward zones align with chemical regions:
- Buffer region bonus (pH 3-6)
- Target zone bonus (pH 6.5-7.5)
- Sweet spot bonus (pH 6.9-7.1)
- Equivalence volume guidance

---

## References

These are standard equations from:
- **Analytical Chemistry** textbooks (Skoog, Harris, etc.)
- **General Chemistry** courses (acid-base equilibria)
- **Physical Chemistry** (thermodynamics of solutions)

**Key Equations:**
- Henderson-Hasselbalch: `pH = pKa + log([A⁻]/[HA])`
- Weak acid: `[H⁺] = √(Ka × Ca)`
- Weak base: `[OH⁻] = √(Kb × Cb)`
- Water: `Kw = [H⁺][OH⁻] = 10⁻¹⁴`

---

## Educational Value

This project demonstrates:
1. **Domain expertise:** Real chemistry knowledge
2. **Mathematical modeling:** Translating equations to code
3. **RL application:** Using ML to solve real problems
4. **Interdisciplinary thinking:** Chemistry + CS + ML

**Perfect for showing:**
- You understand both the domain (chemistry) and the method (RL)
- You can build realistic simulations
- You think about real-world applications
- You have strong fundamentals across disciplines

---

This is **real chemistry**, not a toy problem. The equations are what you'd use in an actual analytical chemistry lab!

