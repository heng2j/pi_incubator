# utils/config_parser.py
"""
Configuration parser for loading YAML config files.
"""
import os
import yaml
from pi_incubator.utils.logging_utils import get_logger

# Create module-specific logger
logger = get_logger("config_parser")

def load_config(config_path: str):
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file {config_path} not found.")
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    
    logger.debug(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    logger.debug(f"Loaded configuration with {len(config)} keys")
    return config

if __name__ == "__main__":
    # Test the config parser
    cfg = load_config("configs/experiment_config.yaml")
    logger.info(f"Loaded config: {cfg}")
