# utils/config_parser.py
import os
import yaml

def load_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    # Test the config parser
    cfg = load_config("configs/experiment_config.yaml")
    print(cfg)
