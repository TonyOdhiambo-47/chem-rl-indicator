"""
Training callback for real-time visualization during RL training.

This callback periodically visualizes episodes during training to show
how the agent's policy improves over time.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeVisualizationCallback(BaseCallback):
    """
    Callback that visualizes episodes during training.
    
    Saves episode visualizations to a directory and optionally
    displays them in real-time.
    """
    
    def __init__(
        self,
        env,
        log_dir: str = None,
        save_freq: int = 1000,  # Save every N episodes
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env = env
        self.log_dir = Path(log_dir) if log_dir else Path("training_visualizations")
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.save_freq = save_freq
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.final_pHs = []
        self.final_Vbs = []
        
    def _on_step(self) -> bool:
        """Called after each step."""
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout (collection of steps)."""
        # Get info from the last episode in the vectorized env
        infos = self.locals.get("infos", [])
        if infos:
            # Extract episode info from the last environment
            for info in infos:
                if "episode" in info:
                    episode_info = info["episode"]
                    self.episode_count += 1
                    self.episode_rewards.append(episode_info["r"])
                    self.episode_lengths.append(episode_info["l"])
                    
                    # Try to get final pH and Vb from the environment
                    # This requires accessing the env's internal state
                    # For now, we'll track what we can from the callback
                    
                    # Save visualization periodically
                    if self.episode_count % self.save_freq == 0:
                        self._save_training_progress()
        
        return True
    
    def _save_training_progress(self):
        """Save training progress visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Episode rewards over time
        ax1 = axes[0, 0]
        if len(self.episode_rewards) > 0:
            ax1.plot(self.episode_rewards, alpha=0.6, linewidth=1, label='Episode Rewards')
            # Moving average
            if len(self.episode_rewards) > 10:
                window = min(50, len(self.episode_rewards) // 10)
                moving_avg = np.convolve(
                    self.episode_rewards, 
                    np.ones(window)/window, 
                    mode='valid'
                )
                ax1.plot(range(window-1, len(self.episode_rewards)), 
                        moving_avg, 'r-', linewidth=2, label='Moving Average')
        ax1.set_xlabel('Episode', fontweight='bold')
        ax1.set_ylabel('Episode Reward', fontweight='bold')
        ax1.set_title('Training Progress: Episode Rewards', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        if len(self.episode_rewards) > 10:
            ax1.legend()
        
        # Plot 2: Episode lengths
        ax2 = axes[0, 1]
        if len(self.episode_lengths) > 0:
            ax2.plot(self.episode_lengths, alpha=0.6, linewidth=1, color='green', label='Episode Lengths')
            if len(self.episode_lengths) > 10:
                window = min(50, len(self.episode_lengths) // 10)
                moving_avg = np.convolve(
                    self.episode_lengths, 
                    np.ones(window)/window, 
                    mode='valid'
                )
                ax2.plot(range(window-1, len(self.episode_lengths)), 
                        moving_avg, 'r-', linewidth=2, label='Moving Average')
        ax2.set_xlabel('Episode', fontweight='bold')
        ax2.set_ylabel('Episode Length (steps)', fontweight='bold')
        ax2.set_title('Training Progress: Episode Lengths', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        if len(self.episode_lengths) > 10:
            ax2.legend()
        
        # Plot 3: Reward distribution
        ax3 = axes[1, 0]
        if len(self.episode_rewards) > 0:
            ax3.hist(self.episode_rewards, bins=30, alpha=0.7, edgecolor='black')
            ax3.axvline(np.mean(self.episode_rewards), color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.episode_rewards):.2f}')
        ax3.set_xlabel('Episode Reward', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Reward Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        stats_text = f"""
Training Statistics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Episodes: {self.episode_count}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Reward Statistics:
  Mean: {np.mean(self.episode_rewards) if self.episode_rewards else 0:.2f}
  Std:  {np.std(self.episode_rewards) if self.episode_rewards else 0:.2f}
  Min:  {np.min(self.episode_rewards) if self.episode_rewards else 0:.2f}
  Max:  {np.max(self.episode_rewards) if self.episode_rewards else 0:.2f}
  Latest: {self.episode_rewards[-1] if self.episode_rewards else 0:.2f}

Episode Length Statistics:
  Mean: {np.mean(self.episode_lengths) if self.episode_lengths else 0:.1f}
  Std:  {np.std(self.episode_lengths) if self.episode_lengths else 0:.1f}
  Min:  {np.min(self.episode_lengths) if self.episode_lengths else 0:.0f}
  Max:  {np.max(self.episode_lengths) if self.episode_lengths else 0:.0f}
        """.strip()
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes,
                fontsize=11, va='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle(f'Training Progress at Episode {self.episode_count}', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        save_path = self.log_dir / f"training_progress_ep{self.episode_count:06d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.verbose > 0:
            print(f"Saved training visualization to {save_path}")


class ReliabilityEarlyStopCallback(BaseCallback):
    """
    Early stopping based on *pH reliability* instead of raw reward.

    We periodically run evaluation episodes and measure how often the
    final pH is within a target band [pH_low, pH_high]. Training stops
    once the success rate exceeds a threshold for a number of consecutive
    evaluations.

    Stops training once the agent reliably reaches pH 6.9–7.05.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int = 50_000,
        n_eval_episodes: int = 32,
        pH_low: float = 6.9,
        pH_high: float = 7.05,
        success_threshold: float = 0.9,
        patience: int = 3,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.pH_low = pH_low
        self.pH_high = pH_high
        self.success_threshold = success_threshold
        self.patience = patience
        self._consecutive_good = 0
        self._last_success_rate: Optional[float] = None

    def _on_step(self) -> bool:
        # Only evaluate every eval_freq timesteps
        if self.n_calls % self.eval_freq != 0:
            return True

        success_rate = self._evaluate_policy()
        self._last_success_rate = success_rate

        if self.verbose > 0:
            print(
                f"[ReliabilityEarlyStop] step={self.num_timesteps:,} "
                f"success_rate={success_rate*100:.1f}% "
                f"(target ≥ {self.success_threshold*100:.1f}% "
                f"for {self.patience} evals)"
            )

        if success_rate >= self.success_threshold:
            self._consecutive_good += 1
        else:
            self._consecutive_good = 0

        if self._consecutive_good >= self.patience:
            if self.verbose > 0:
                print(
                    f"[ReliabilityEarlyStop] Achieved stable success rate "
                    f"{success_rate*100:.1f}% in pH band "
                    f"[{self.pH_low}, {self.pH_high}] for "
                    f"{self.patience} evaluations. Stopping training."
                )
            # Returning False signals SB3 to stop training
            return False

        return True

    def _evaluate_policy(self) -> float:
        """
        Run n_eval_episodes using the current policy and compute the fraction
        of episodes that end with final pH in the target band.
        """
        successes = 0
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            truncated = False
            final_pH = None

            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, truncated, info = self.eval_env.step(action)
                # Track last pH from info
                if "pH" in info:
                    final_pH = float(info["pH"])

            if final_pH is not None and self.pH_low <= final_pH <= self.pH_high:
                successes += 1

        return successes / float(self.n_eval_episodes)

