# env/train_rl.py

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from src.titration_env import WeakAcidIndicatorEnv
from training_callback import EpisodeVisualizationCallback, ReliabilityEarlyStopCallback


def make_env():
    return WeakAcidIndicatorEnv(
        Va_ml=50.0,
        Ca=0.1,
        Cb=0.1,
        pKa=4.76,
        pH_target=7.0,
        max_steps=200,
        step_sizes_ml=(0.1, 0.2, 0.5, 1.0, 2.0, 3.0),
        pKa_ind=7.0,
        neutral_band=0.15,
        max_burette_ml=50.0,  # Standard burette capacity
    )


def main():
    root = Path(__file__).resolve().parent
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)

    # Vectorized env for parallel rollout (16 copies for faster training)
    vec_env = make_vec_env(make_env, n_envs=16)
    
    # Single env for visualization + evaluation callbacks
    single_env = make_env()
    eval_env = make_env()

    # Check if tensorboard is available
    try:
        import tensorboard  # type: ignore
        tb_log = str(root / "tb_logs")
    except ImportError:
        tb_log = None
        print("TensorBoard not installed. Training without tensorboard logging.")

    # Setup callbacks: training visualization + reliability-based early stopping
    vis_callback = EpisodeVisualizationCallback(
        env=single_env,
        log_dir=str(root / "training_visualizations"),
        save_freq=250,
        verbose=1,
    )
    reliability_callback = ReliabilityEarlyStopCallback(
        eval_env=eval_env,
        eval_freq=50_000,         # Evaluate every 50k steps
        n_eval_episodes=32,       # Number of eval episodes per check
        pH_low=6.9,
        pH_high=7.05,
        success_threshold=0.9,    # 90% of eval episodes must succeed
        patience=3,               # 3 consecutive good evals to stop
        verbose=1,
    )
    callbacks = CallbackList([vis_callback, reliability_callback])

    # Hyperparameters tuned to prevent overshooting
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=tb_log,
        n_steps=4096,
        batch_size=1024,
        n_epochs=15,
        gamma=0.998,
        learning_rate=2e-4,
        ent_coef=0.05,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.15,
        policy_kwargs=dict(
            net_arch=[512, 512, 256],
            activation_fn=torch.nn.Tanh,
        ),
    )

    # Extended training for robust convergence
    # With 16 parallel envs: ~10M timesteps ≈ 30-40 minutes
    total_timesteps = 10_000_000
    print(f"\n{'='*70}")
    print("Training Configuration")
    print(f"{'='*70}")
    print(f"Environment: 200 max steps, 7 action options, 50mL burette limit")
    print(f"Training: 10,000,000 timesteps with 16 parallel environments")
    print(f"Network: [512, 512, 256] architecture with Tanh activation")
    print(f"Exploration: Entropy coefficient 0.05")
    print(f"Learning rate: 2e-4 with 15 epochs per rollout")
    print(f"Reward: Asymmetric anti-overshoot penalties (pH>7.0 heavily penalized)")
    print(f"{'='*70}")
    print(f"Anti-overshoot penalties:")
    print(f"  - pH > 8.0: -500 reward penalty")
    print(f"  - pH > 7.5: -200 reward penalty")
    print(f"  - pH > 7.2: -100 reward penalty")
    print(f"  - pH > 7.1: -50 reward penalty")
    print(f"{'='*70}")
    print(f"Visualizations will be saved to: {root / 'training_visualizations'}")
    print(f"Expected training time: ~30-40 minutes (depending on hardware)")
    print(f"{'='*70}\n")
    
    # Check if progress bar dependencies are available
    try:
        import tqdm
        import rich
        use_progress_bar = True
    except ImportError:
        use_progress_bar = False
        print("Note: tqdm/rich not installed. Training without progress bar.")
        print("Install with: pip install stable-baselines3[extra]")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=use_progress_bar,
    )

    save_path = models_dir / "ppo_weak_acid_indicator"
    model.save(save_path)
    print(f"Saved trained model to: {save_path}")


if __name__ == "__main__":
    main()

