
import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Parse arguments first to get webrtc setting
parser = argparse.ArgumentParser(description="Train PPO with YAML configuration")
parser.add_argument(
    "--config", 
    type=str, 
    required=True,
    help="Name of config file in configs folder (without extension)"
)
parser.add_argument(
    "--override", 
    type=str, 
    nargs="*",
    help="Override config values (format: key=value)"
)
parser.add_argument(
    "--webrtc", 
    action="store_true",
    help="Enable WebRTC streaming for remote visualization"
)

# Parse only the webrtc argument first (don't exit on missing required args)
args, unknown = parser.parse_known_args()

# Set environment variable based on webrtc argument
if args.webrtc:
    os.environ["ISAAC_WEBRTC"] = "1"
    print("[INFO] WebRTC enabled")
else:
    os.environ["ISAAC_WEBRTC"] = "0"
    print("[INFO] WebRTC disabled")

# Now import PPO (after environment variable is set)
# This is due t an isaac sim need to create SimulatedApp before importing ther isaacsim modules
from cleanrl_new.ppo import Args, main as ppo_main


def load_config(config_path: str) -> Args:
    """Load configuration from YAML file into Args object."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create Args object with defaults
    args = Args()
    
    # Handle env_kwargs separately
    env_kwargs = config_dict.pop('env_kwargs', None)
    
    # Update with values from YAML
    for key, value in config_dict.items():
        if hasattr(args, key):
            setattr(args, key, value)
        else:
            print(f"Warning: Unknown config key '{key}' will be ignored")
    
    # Set env_kwargs if provided, otherwise it will remain None (default)
    if env_kwargs is not None:
        args.env_kwargs = env_kwargs
        print(f"Loaded env_kwargs: {env_kwargs}")
    else:
        print("No env_kwargs provided, using defaults")
    
    return args


def get_config_path(config_name: str) -> str:
    """Get the full path to a config file in the configs folder."""
    # Remove any file extension if provided
    config_name = Path(config_name).stem
    
    # Look for the config file in the configs folder
    config_path = Path("configs") / f"{config_name}.yml"
    
    if not config_path.exists():
        # Also try .yaml extension
        config_path = Path("configs") / f"{config_name}.yaml"
    
    return str(config_path)


def get_experiment_name(config_name: str) -> str:
    """Extract experiment name from config name."""
    # Remove any file extension if provided
    return Path(config_name).stem


def main():
    # Re-parse arguments properly (including the unknown ones)
    args = parser.parse_args()
    
    # Get the full path to the config file
    config_path = get_config_path(args.config)
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found")
        print(f"Available configs in configs folder:")
        configs_dir = Path("configs")
        if configs_dir.exists():
            for config_file in configs_dir.glob("*.yml"):
                print(f"  - {config_file.stem}")
            for config_file in configs_dir.glob("*.yaml"):
                print(f"  - {config_file.stem}")
        sys.exit(1)
    
    # Load configuration
    print(f"Loading configuration from {config_path}")
    ppo_args = load_config(config_path)
    
    # Set experiment name from config name
    experiment_name = get_experiment_name(args.config)
    ppo_args.exp_name = experiment_name
    
    # Apply overrides if provided
    #TODO: Add override for env_kwargs
    if args.override:
        for override in args.override:
            if "=" not in override:
                print(f"Warning: Invalid override format '{override}', skipping")
                continue
            
            key, value = override.split("=", 1)
            if hasattr(ppo_args, key):
                # Try to convert value to appropriate type
                current_value = getattr(ppo_args, key)
                if isinstance(current_value, bool):
                    setattr(ppo_args, key, value.lower() in ('true', '1', 'yes'))
                elif isinstance(current_value, int):
                    setattr(ppo_args, key, int(value))
                elif isinstance(current_value, float):
                    setattr(ppo_args, key, float(value))
                else:
                    setattr(ppo_args, key, value)
                print(f"Override: {key} = {value}")
            else:
                print(f"Warning: Unknown config key '{key}' in override")
    
    # Print configuration summary
    print("\n" + "="*50)
    print("PPO Training Configuration")
    print("="*50)
    print(f"Experiment name: {ppo_args.exp_name}")
    print(f"Total timesteps: {ppo_args.total_timesteps:,}")
    print(f"Learning rate: {ppo_args.learning_rate}")
    print(f"Number of environments: {ppo_args.num_envs}")
    print(f"Seed: {ppo_args.seed}")
    
    # Print environment kwargs if present
    if ppo_args.env_kwargs:
        print(f"Environment kwargs: {ppo_args.env_kwargs}")
    
    # Import torch here to avoid import issues
    import torch
    print(f"Device: {'CUDA' if ppo_args.cuda and torch.cuda.is_available() else 'CPU'}")
    print("="*50 + "\n")
    
    # Run PPO training
    ppo_main(ppo_args)


if __name__ == "__main__":
    main()