# Chem RL Indicator

A small but realistic **chemistry + reinforcement learning + visualization** project.

- **Physics**: weak acid titrated with strong base (real titration equations).
- **Observation**: a **simulated pH indicator color** (continuous RGB with a neutral band around pH 7.00 ± ε), not raw pH.
- **Environment**: a Gymnasium-compatible `WeakAcidIndicatorEnv` for RL agents.
- **Frontend**: a React app that visualizes pH and indicator color as you drag a slider for base volume.

This doubles as:

- A clean demo of **environment design for RL** (Mechanize-style), and
- A nice chem / visualization project.

---

## Repo structure

```text
chem-rl-indicator/
  env/
    src/
      titration_env.py         # physics + RL env
    train_rl.py                # PPO training script
    visualize_policy.py        # random vs trained comparison & plots
    requirements.txt
  web/
    src/
      chem.ts                  # titration + indicator math in TS
      App.tsx                  # React UI
    package.json
    ...
```

---

## Python environment (RL side)

```bash
cd env
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run random vs trained agent

1. Train PPO:

```bash
python train_rl.py
```

This trains a PPO agent on the `WeakAcidIndicatorEnv` and saves it to `env/models/ppo_weak_acid_indicator.zip`.

2. Compare random vs trained:

```bash
python visualize_policy.py
```

This script:

* runs one random episode and one PPO episode,
* plots **pH vs base volume** for each,
* saves `titration_policy_comparison.png`.

---

## Environment: WeakAcidIndicatorEnv

The environment simulates titration of a monoprotic weak acid HA (e.g. acetic acid) with a strong base (NaOH):

* Before equivalence: buffer region, pH via Henderson–Hasselbalch.
* At equivalence: solution of A⁻, pH via weak base hydrolysis.
* After equivalence: excess strong base, pH from excess [OH⁻].

The agent **never sees pH directly**. Instead, the observation is:

```text
[R, G, B, Vb_over_Veq, step_norm]
```

Where:

* `[R, G, B]` is a simulated indicator color, computed from pH:

  * acid form: yellow
  * base form: blue
  * smooth neutral band (green) around pH 7.00 ± 0.15
* `Vb_over_Veq` is the base volume relative to equivalence volume,
* `step_norm` is the current step / max_steps.

Actions:

```text
0: add small volume of base
1: add medium volume
2: add large volume
3: stop
```

Reward:

```text
reward = -|pH - pH_target| / 7  - 0.01  (per step)
+ bonus if the agent stops close to target pH
```

This makes it a **partially observable RL environment**:
the true state is (pH, volumes, moles), but the agent only sees a color and some meta info.

---

## React visualization (frontend)

```bash
cd web
npm install
npm run dev
```

Then open the printed URL (usually `http://localhost:5173/`).

The app lets you:

* set acid volume/concentration, base concentration, and pKa,
* slide base volume Vb,
* see:

  * the **computed pH**,
  * Vb/Veq,
  * a large circle colored via the same indicator model used in the RL env.

The indicator:

* transitions yellow → green → blue as pH crosses around 7,
* uses a **neutral band** where the color is strongly green near 7.00.

---

## Why this is interesting

* It uses **real chemistry** (not arbitrary math): the same equations you'd learn in an analytical chem or general chem course.
* It separates:

  * **World physics** (titration equations, pH),
  * **Sensor model** (indicator color),
  * **Agent** (RL policy).
* It's a clean example of **environment design for RL**, with:

  * clear state / action / reward choices,
  * partial observability via color,
  * and visual tools (React UI + matplotlib) to understand the dynamics.

This is exactly the kind of project you can walk through in an interview if someone asks:

> "Tell me about an RL environment you've designed and how you thought about its structure."

