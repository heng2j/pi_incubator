#!/usr/bin/env python
"""
Omni RL Pi Incubator – PPO Training Script (Modular Version)
------------------------------------------------------------
This script loads experiment configuration from a YAML file and trains a PPO policy
using TorchRL. It follows a modular architecture while keeping the main experimental
flow in a single script for clarity and debugging.
"""

import os
import torch
import argparse
import multiprocessing
import logging

# Import modular components from the pi_incubator package
from pi_incubator.utils.config_parser import load_config
from pi_incubator.environments.gym_env import make_env, initialize_env_stats
from pi_incubator.models.networks import create_actor_critic_networks
from pi_incubator.collectors.collectors import create_collector
from pi_incubator.utils.loggers import create_logger, create_recorder
from pi_incubator.algorithms.ppo import create_ppo_components, create_ppo_trainer, save_policy
from pi_incubator.utils.logging_utils import setup_logger

def train_ppo(config, logger):
    """
    Train a PPO agent with the given configuration.
    
    This function contains the main experimental flow and orchestrates the components
    to provide a clear, sequential view of the PPO training process.
    """
    # =============================================
    # 1. Set up the environment
    # =============================================
    logger.info(f"Creating environment: {config['env_name']}")
    env = make_env(config, device)
    initialize_env_stats(env, num_iter=1000)
    
    # =============================================
    # 2. Create actor-critic networks
    # =============================================
    logger.info("Building policy and value networks")
    policy_module, value_module = create_actor_critic_networks(env, config, device)
    
    # Initialize networks with a dummy forward pass
    policy_module(env.reset())
    value_module(env.reset())
    
    # =============================================
    # 3. Create data collector
    # =============================================
    logger.info("Setting up data collection")
    collector = create_collector(env, policy_module, config, device)
    
    # =============================================
    # 4. Set up PPO components (buffer, loss, optimizer)
    # =============================================
    logger.info("Creating PPO training components")
    ppo_components = create_ppo_components(policy_module, value_module, config, device)
    loss_module = ppo_components["loss_module"]
    optimizer = ppo_components["optimizer"]
    
    # =============================================
    # 5. Create logger
    # =============================================
    logger.info("Setting up experiment logging")
    experiment_logger = create_logger(config, "tuned_PPO")
    
    # =============================================
    # 6. Create and start trainer
    # =============================================
    logger.info("Creating trainer")
    trainer = create_ppo_trainer(collector, loss_module, optimizer, experiment_logger, config)
    
    # Optional: Create a recorder for visualizing agent behavior
    if config.get("use_recorder", False):
        test_env = make_env(config, device)
        recorder = create_recorder(trainer, test_env, policy_module, config)
        logger.info("Recorder created for agent visualization")
    
    # =============================================
    # 7. Train the agent
    # =============================================
    logger.info(f"Starting training for {config['total_frames']} frames")
    trainer.train()
    
    # =============================================
    # 8. Save the trained policy
    # =============================================
    policy_path = save_policy(policy_module, config.get("policy_path", "ppo_policy.pth"))
    logger.info(f"Saved policy to {policy_path}")
    
    return {
        "policy_module": policy_module,
        "value_module": value_module,
        "env": env,
        "trainer": trainer
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent with TorchRL")
    parser.add_argument("-c", "--config", type=str, 
                        default="configs/experiment_config_baseline.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Optional log file path")
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logger("pi_incubator.train", level=log_level, log_file=args.log_file)
    
    # Determine device (CPU if no CUDA or if using fork mode on Mac)
    is_fork = multiprocessing.get_start_method() == "fork"
    device = torch.device(0) if torch.cuda.is_available() and not is_fork else torch.device("cpu")
    
    # Load config
    config = load_config(config_path=args.config)
    
    # Override device if specified in config
    if "device" in config:
        device = torch.device(config["device"])
    
    # Set MuJoCo rendering backend if on CPU (macOS)
    if device.type == "cpu":
        os.environ["MUJOCO_GL"] = "glfw"
    
    logger.info(f"Using device: {device}")
    
    # Run training
    train_results = train_ppo(config, logger)
