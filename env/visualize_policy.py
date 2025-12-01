# env/visualize_policy.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from src.titration_env import WeakAcidIndicatorEnv


def run_episode(env, model=None, deterministic=True, max_steps=40):
    obs, _ = env.reset()
    traj = {
        "Vb_ml": [],
        "pH": [],
        "color": [],
        "action": [],
        "reward": [],
    }

    for _ in range(max_steps):
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)

        Vb_ml, pH, rgb = env.history[-1]  # last recorded state
        traj["Vb_ml"].append(Vb_ml)
        traj["pH"].append(pH)
        traj["color"].append(rgb)
        traj["action"].append(int(action))
        traj["reward"].append(float(reward))

        if terminated or truncated:
            break

    return traj


def plot_titration_curves(random_traj, rl_traj, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    # Random
    ax[0].plot(random_traj["Vb_ml"], random_traj["pH"], marker="o")
    ax[0].set_title("Random policy")
    ax[0].set_xlabel("Base added Vb (mL)")
    ax[0].set_ylabel("pH")

    # RL
    ax[1].plot(rl_traj["Vb_ml"], rl_traj["pH"], marker="o")
    ax[1].set_title("Trained PPO policy")
    ax[1].set_xlabel("Base added Vb (mL)")

    for a in ax:
        a.axhline(7.0, color="gray", linestyle="--", alpha=0.6)
        a.grid(alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def main():
    root = Path(__file__).resolve().parent
    models_dir = root / "models"
    model_path = models_dir / "ppo_weak_acid_indicator.zip"

    env = WeakAcidIndicatorEnv()

    # Random policy trajectory
    random_traj = run_episode(env, model=None)

    # Load trained model if available
    if model_path.exists():
        model = PPO.load(model_path)
        rl_traj = run_episode(env, model=model)
    else:
        print(f"Warning: {model_path} not found. Using random for both.")
        model = None
        rl_traj = run_episode(env, model=None)

    fig_path = root / "titration_policy_comparison.png"
    plot_titration_curves(random_traj, rl_traj, save_path=fig_path)


if __name__ == "__main__":
    main()

