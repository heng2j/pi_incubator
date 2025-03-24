"""
Logging utilities for reinforcement learning experiments.
"""
import os
from torchrl.record.loggers.csv import CSVLogger
from torchrl.record.loggers.tensorboard import TensorboardLogger
from torchrl.record.loggers.wandb import WandbLogger
from torchrl.record.loggers.mlflow import MLFlowLogger
from torchrl.trainers import Recorder
from torchrl.envs import ExplorationType

def create_logger(config, experiment_name=None):
    """
    Create a logger based on configuration.
    
    Args:
        config: Configuration dictionary
        experiment_name: Name of the experiment
        
    Returns:
        A logger instance
    """
    logger_type = config.get("logger", "tensorboard")
    exp_name = experiment_name or config.get("exp_name", "experiment")
    
    if logger_type == "tensorboard":
        log_dir = config.get("log_dir", "./tb_logs")
        os.makedirs(log_dir, exist_ok=True)
        return TensorboardLogger(exp_name=exp_name, log_dir=log_dir)
    
    elif logger_type == "wandb":
        project = config.get("wandb_project", "RL_Experiment")
        entity = config.get("wandb_entity", None)
        log_dir = config.get("wandb_log_dir", "./wandb_logs")
        os.makedirs(log_dir, exist_ok=True)
        
        return WandbLogger(
            exp_name=exp_name, 
            project=project,
            entity=entity,
            log_dir=log_dir,
            offline=config.get("wandb_offline", False)
        )
    
    elif logger_type == "csv":
        log_dir = config.get("log_dir", "./csv_logs")
        os.makedirs(log_dir, exist_ok=True)
        return CSVLogger(exp_name=exp_name, log_dir=log_dir)
    
    elif logger_type == "mlflow":
        tracking_uri = config.get("mlflow_tracking_uri", None)
        return MLFlowLogger(exp_name=exp_name, tracking_uri=tracking_uri)
    
    else:
        # Default to tensorboard
        log_dir = config.get("log_dir", "./tb_logs")
        os.makedirs(log_dir, exist_ok=True)
        return TensorboardLogger(exp_name=exp_name, log_dir=log_dir)

def create_recorder(trainer, environment, policy_module, config):
    """
    Create a recorder for visualizing agent behavior.
    
    Args:
        trainer: The trainer instance
        environment: The environment to record
        policy_module: The policy module
        config: Configuration dictionary
        
    Returns:
        A recorder instance
    """
    record_interval = config.get("record_interval", 100)
    record_frames = config.get("record_frames", 1000)
    frame_skip = config.get("frame_skip", 1)
    
    recorder = Recorder(
        record_interval=record_interval,
        record_frames=record_frames,
        frame_skip=frame_skip,
        policy_exploration=policy_module,
        environment=environment,
        exploration_type=ExplorationType.DETERMINISTIC,
        log_keys=[("next", "reward")],
        out_keys={("next", "reward"): "rewards"},
        log_pbar=True,
    )
    
    recorder.register(trainer)
    return recorder 