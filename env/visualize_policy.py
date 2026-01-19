# env/visualize_policy.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from src.titration_env import WeakAcidIndicatorEnv


def run_episode(env, model=None, deterministic=True, max_steps=200):
    obs, _ = env.reset()
    traj = {
        "Vb_ml": [],
        "pH": [],
        "color": [],
        "action": [],
        "reward": [],
    }

    # Include initial state
    Vb_ml, pH, rgb = env.history[0]
    traj["Vb_ml"].append(Vb_ml)
    traj["pH"].append(pH)
    traj["color"].append(rgb)
    traj["action"].append(-1)  # No action for initial state
    traj["reward"].append(0.0)

    for step in range(max_steps):
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
    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Random
    ax[0].plot(random_traj["Vb_ml"], random_traj["pH"], marker="o", linewidth=2, markersize=6, label="Trajectory")
    ax[0].scatter([random_traj["Vb_ml"][0]], [random_traj["pH"][0]], 
                  color="green", s=100, zorder=5, label="Start")
    ax[0].scatter([random_traj["Vb_ml"][-1]], [random_traj["pH"][-1]], 
                  color="red", s=100, zorder=5, label="End")
    ax[0].set_title(f"Random policy ({len(random_traj['Vb_ml'])} steps)")
    ax[0].set_xlabel("Base added Vb (mL)")
    ax[0].set_ylabel("pH")
    ax[0].legend()

    # RL
    ax[1].plot(rl_traj["Vb_ml"], rl_traj["pH"], marker="o", linewidth=2, markersize=6, label="Trajectory")
    ax[1].scatter([rl_traj["Vb_ml"][0]], [rl_traj["pH"][0]], 
                  color="green", s=100, zorder=5, label="Start")
    ax[1].scatter([rl_traj["Vb_ml"][-1]], [rl_traj["pH"][-1]], 
                  color="red", s=100, zorder=5, label="End")
    ax[1].set_title(f"Trained PPO policy ({len(rl_traj['Vb_ml'])} steps)")
    ax[1].set_xlabel("Base added Vb (mL)")
    ax[1].legend()

    for a in ax:
        a.axhline(7.0, color="gray", linestyle="--", alpha=0.6, label="Target pH")
        a.grid(alpha=0.2)
        a.set_ylim(0, 14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def main():
    root = Path(__file__).resolve().parent
    models_dir = root / "models"
    model_path = models_dir / "ppo_weak_acid_indicator.zip"

    # Use same env parameters as training
    env = WeakAcidIndicatorEnv(
        max_steps=200,
        step_sizes_ml=(0.1, 0.2, 0.5, 1.0, 2.0, 3.0),
        max_burette_ml=50.0,
    )
    print(f"Environment: Veq = {env.Veq_L*1000:.2f} mL, Initial pH = {env._get_pH():.2f}")
    print(f"Note: pH 7.0 occurs BEFORE equivalence (at equivalence, pH ≈ 8.73)")

    # Random policy trajectory
    print("\nRunning random policy...")
    random_traj = run_episode(env, model=None)
    print(f"Random: {len(random_traj['Vb_ml'])} steps, Final pH = {random_traj['pH'][-1]:.2f}, "
          f"Final Vb = {random_traj['Vb_ml'][-1]:.2f} mL, Total reward = {sum(random_traj['reward']):.2f}")

    # Load trained model if available
    if model_path.exists():
        print(f"\nLoading trained model from {model_path}...")
        model = PPO.load(model_path)
        rl_traj = run_episode(env, model=model)
        print(f"Trained: {len(rl_traj['Vb_ml'])} steps, Final pH = {rl_traj['pH'][-1]:.2f}, "
              f"Final Vb = {rl_traj['Vb_ml'][-1]:.2f} mL, Total reward = {sum(rl_traj['reward']):.2f}")
    else:
        print(f"\nWarning: {model_path} not found. Using random for both.")
        model = None
        rl_traj = run_episode(env, model=None)

    fig_path = root / "titration_policy_comparison.png"
    plot_titration_curves(random_traj, rl_traj, save_path=fig_path)


if __name__ == "__main__":
    main()

