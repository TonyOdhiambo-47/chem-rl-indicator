"""
Export episode trajectory data for React visualization.

This script runs an episode (random or trained) and exports the trajectory
as JSON that can be loaded into the React app for live animation.
"""

import json
from pathlib import Path
import argparse

import numpy as np
from stable_baselines3 import PPO

from src.titration_env import WeakAcidIndicatorEnv


def run_and_export_episode(model_path=None, deterministic=True, output_path="episode_data.json"):
    """
    Run an episode and export trajectory data.
    
    Args:
        model_path: Path to trained model (None for random)
        deterministic: Use deterministic policy
        output_path: Where to save JSON data
    """
    env = WeakAcidIndicatorEnv(
        max_steps=200,
        step_sizes_ml=(0.1, 0.2, 0.5, 1.0, 2.0, 3.0),
        max_burette_ml=50.0,  # Realistic burette limit
    )
    
    # Load model if provided
    model = None
    if model_path and Path(model_path).exists():
        print(f"Loading model from {model_path}...")
        model = PPO.load(model_path)
    
    # Run episode
    obs, _ = env.reset()
    trajectory = {
        "steps": [],
        "initial_state": {
            "Vb_ml": float(env.Vb_L * 1000.0),
            "pH": float(env._get_pH()),
            "color": env.history[0][2].tolist() if isinstance(env.history[0][2], np.ndarray) else list(env.history[0][2]),
        },
        "target_pH": float(env.pH_target),
        "Veq_ml": float(env.Veq_L * 1000.0),
        "step_sizes_ml": env.step_sizes_ml.tolist(),
        "action_names": ["0.1mL", "0.2mL", "0.5mL", "1.0mL", "2.0mL", "3.0mL", "Stop"],
    }
    
    step = 0
    total_reward = 0.0
    
    while True:
        # Get action
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
            if isinstance(action, np.ndarray):
                action = int(action.item())
            else:
                action = int(action)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record step
        Vb_ml, pH, rgb = env.history[-1]
        step_data = {
            "step": step + 1,
            "action": int(action),
            "action_name": trajectory["action_names"][action],
            "Vb_ml": float(Vb_ml),
            "pH": float(pH),
            "color": rgb.tolist() if isinstance(rgb, np.ndarray) else list(rgb),
            "reward": float(reward),
            "total_reward": float(total_reward + reward),
            "distance_to_target": float(abs(pH - env.pH_target)),
            "V_over_Veq": float(Vb_ml / (env.Veq_L * 1000.0)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
        trajectory["steps"].append(step_data)
        
        total_reward += reward
        step += 1
        
        if terminated or truncated:
            break
    
    # Add summary
    trajectory["summary"] = {
        "total_steps": step,
        "final_pH": float(trajectory["steps"][-1]["pH"]),
        "final_Vb_ml": float(trajectory["steps"][-1]["Vb_ml"]),
        "total_reward": float(total_reward),
        "final_distance": float(trajectory["steps"][-1]["distance_to_target"]),
        "success": trajectory["steps"][-1]["distance_to_target"] < 0.1,
    }
    
    # Export to JSON
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        json.dump(trajectory, f, indent=2)
    
    print(f"\n✅ Episode exported to {output_path}")
    print(f"   Steps: {step}")
    print(f"   Final pH: {trajectory['summary']['final_pH']:.2f}")
    print(f"   Final Vb: {trajectory['summary']['final_Vb_ml']:.2f} mL")
    print(f"   Total reward: {trajectory['summary']['total_reward']:.2f}")
    print(f"   Success: {trajectory['summary']['success']}")
    
    return trajectory


def main():
    parser = argparse.ArgumentParser(description='Export episode data for React visualization')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--output', type=str, default='episode_data.json', help='Output JSON file')
    parser.add_argument('--deterministic', action='store_true', default=True, help='Use deterministic policy')
    parser.add_argument('--no-deterministic', dest='deterministic', action='store_false', help='Use stochastic policy')
    parser.add_argument('--random', action='store_true', help='Use random policy (overrides model)')
    
    args = parser.parse_args()
    
    model_path = None if args.random else args.model
    run_and_export_episode(
        model_path=model_path,
        deterministic=args.deterministic,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

