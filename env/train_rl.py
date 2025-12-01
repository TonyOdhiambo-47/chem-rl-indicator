# env/train_rl.py

import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.titration_env import WeakAcidIndicatorEnv


def make_env():
    return WeakAcidIndicatorEnv(
        Va_ml=50.0,
        Ca=0.1,
        Cb=0.1,
        pKa=4.76,
        pH_target=7.0,
        max_steps=40,
        step_sizes_ml=(0.1, 0.5, 1.0),
        pKa_ind=7.0,
        neutral_band=0.15,
    )


def main():
    root = Path(__file__).resolve().parent
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)

    # Vectorized env for parallel rollout (8 copies)
    env = make_vec_env(make_env, n_envs=8)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(root / "tb_logs"),
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
    )

    total_timesteps = 200_000
    model.learn(total_timesteps=total_timesteps)

    save_path = models_dir / "ppo_weak_acid_indicator"
    model.save(save_path)
    print(f"Saved trained model to: {save_path}")


if __name__ == "__main__":
    main()

