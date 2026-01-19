"""
Live episode viewer for watching the agent learn in real-time.

This script runs episodes and displays them visually, showing how
the agent's policy evolves during training.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from src.titration_env import WeakAcidIndicatorEnv


def watch_episode(env, model=None, deterministic=True, delay=0.1, save_path=None):
    """
    Run and visualize a single episode step-by-step.
    
    Args:
        env: The environment
        model: Trained model (None for random)
        deterministic: Whether to use deterministic policy
        delay: Delay between steps (seconds)
        save_path: Path to save final visualization
    """
    obs, _ = env.reset()
    
    # Create figure for live updates
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.ion()  # Interactive mode
    
    Vb_history = []
    pH_history = []
    color_history = []
    reward_history = []
    action_history = []
    
    step = 0
    total_reward = 0.0
    
    while True:
        # Get action
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
        
        # Convert numpy array to int if needed (model.predict can return array)
        if isinstance(action, np.ndarray):
            action = int(action.item())
        else:
            action = int(action)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record state
        Vb_ml, pH, rgb = env.history[-1]
        Vb_history.append(Vb_ml)
        pH_history.append(pH)
        color_history.append(rgb)
        reward_history.append(reward)
        action_history.append(action)
        total_reward += reward
        step += 1
        
        # Update visualization
        axes[0].clear()
        axes[1].clear()
        
        # Left plot: Titration curve
        if len(Vb_history) > 1:
            axes[0].plot(Vb_history, pH_history, 'o-', linewidth=2.5, 
                       markersize=8, color='steelblue', zorder=2, label='Trajectory')
        axes[0].scatter([Vb_history[0]], [pH_history[0]], color='green', 
                       s=150, zorder=5, label='Start', marker='s')
        axes[0].scatter([Vb_history[-1]], [pH_history[-1]], color='red', 
                       s=200, zorder=5, label='Current', marker='*', edgecolors='black', linewidths=2)
        axes[0].axhline(7.0, color='gray', linestyle='--', linewidth=2, 
                       alpha=0.7, label='Target pH 7.0', zorder=1)
        axes[0].axvline(env.Veq_L * 1000, color='orange', linestyle=':', 
                       linewidth=2, alpha=0.7, label=f'Equivalence ({env.Veq_L*1000:.1f} mL)', zorder=1)
        axes[0].set_xlabel('Base Volume Added (mL)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('pH', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Live Episode - Step {step} | Reward: {total_reward:.2f}', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='best', fontsize=9)
        axes[0].set_ylim(0, 14)
        axes[0].set_xlim(-1, max(Vb_history) * 1.1 if Vb_history else 60)
        
        # Right plot: Current state
        from matplotlib.patches import Circle
        current_color = color_history[-1]
        circle = Circle((0.5, 0.5), 0.35, transform=axes[1].transAxes, 
                        facecolor=current_color, edgecolor='black', linewidth=3)
        axes[1].add_patch(circle)
        
        action_names = {0: 'Small', 1: 'Medium', 2: 'Large', 3: 'Stop'}
        info_text = f"""
Current State
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pH: {pH_history[-1]:.2f}
Base Added: {Vb_history[-1]:.2f} mL
V/Veq: {Vb_history[-1] / (env.Veq_L * 1000):.2%}
Step: {step}/{env.max_steps}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Action: {action} ({action_names.get(action, 'Unknown')})
Reward: {reward:.3f}
Total Reward: {total_reward:.2f}
Distance to Target: {abs(pH_history[-1] - 7.0):.2f}
        """.strip()
        
        axes[1].text(0.5, 0.15, info_text, transform=axes[1].transAxes,
                    fontsize=10, ha='center', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[1].axis('off')
        axes[1].set_title('Indicator Color & Info', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        
        if terminated or truncated:
            break
        
        time.sleep(delay)
    
    plt.ioff()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved episode visualization to {save_path}")
    
    return {
        'Vb_ml': Vb_history,
        'pH': pH_history,
        'color': color_history,
        'reward': reward_history,
        'action': action_history,
        'total_reward': total_reward,
        'steps': step,
    }


def main():
    """Main function to watch episodes."""
    import argparse
    parser = argparse.ArgumentParser(description='Watch RL agent episodes in real-time')
    parser.add_argument('--model', type=str, default=None, 
                       help='Path to trained model (None for random)')
    parser.add_argument('--episodes', type=int, default=1, 
                       help='Number of episodes to watch')
    parser.add_argument('--delay', type=float, default=0.1, 
                       help='Delay between steps (seconds)')
    parser.add_argument('--save', action='store_true', 
                       help='Save episode visualizations')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic policy')
    
    args = parser.parse_args()
    
    env = WeakAcidIndicatorEnv(
        max_steps=200,
        step_sizes_ml=(0.1, 0.2, 0.5, 1.0, 2.0, 3.0),
        max_burette_ml=50.0,
    )
    
    model = None
    if args.model:
        model_path = Path(args.model)
        if model_path.exists():
            print(f"Loading model from {model_path}...")
            model = PPO.load(model_path)
        else:
            print(f"Warning: Model not found at {model_path}, using random policy")
    
    save_dir = Path("episode_visualizations")
    if args.save:
        save_dir.mkdir(exist_ok=True)
    
    for episode in range(args.episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{args.episodes}")
        print(f"{'='*60}")
        
        save_path = save_dir / f"episode_{episode+1:03d}.png" if args.save else None
        
        traj = watch_episode(
            env, 
            model=model, 
            deterministic=args.deterministic,
            delay=args.delay,
            save_path=save_path
        )
        
        print(f"\nEpisode Summary:")
        print(f"  Steps: {traj['steps']}")
        print(f"  Final pH: {traj['pH'][-1]:.2f}")
        print(f"  Final Vb: {traj['Vb_ml'][-1]:.2f} mL")
        print(f"  Distance to target: {abs(traj['pH'][-1] - 7.0):.2f}")
        print(f"  Total reward: {traj['total_reward']:.2f}")
        
        if episode < args.episodes - 1:
            input("\nPress Enter to continue to next episode...")
    
    print("\n" + "="*60)
    print("Done watching episodes!")
    print("="*60)


if __name__ == "__main__":
    main()

