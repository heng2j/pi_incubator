"""
Data collectors for reinforcement learning.
"""
import multiprocessing
from torchrl.collectors import SyncDataCollector, MultiaSyncDataCollector
from torchrl.envs import ExplorationType
from torchrl.data import MultiStep
from pi_incubator.utils.logging_utils import get_logger

# Create a module-specific logger
logger = get_logger("collectors")

def create_collector(env, policy_module, config, device):
    """
    Create an appropriate data collector based on config and environment.
    
    Args:
        env: The environment to collect data from
        policy_module: The policy module to use for data collection
        config: Configuration dictionary
        device: Torch device
        
    Returns:
        A data collector instance
    """
    frames_per_batch = config.get("frames_per_batch", 1000)
    total_frames = config.get("total_frames", 10000)
    num_collectors = config.get("num_collectors", 1)
    
    # Check if fork is the multiprocessing method (important for Mac/Unix systems)
    is_fork = multiprocessing.get_start_method() == "fork"
    
    # Additional collector params
    n_steps = config.get("n_steps", 5)
    gamma = config.get("gamma", 0.99)
    
    # Use a SyncDataCollector if multiprocessing fork is allowed; else use MultiaSyncDataCollector
    if is_fork:
        collector_cls = SyncDataCollector
        env_arg = env
        logger.debug("Using SyncDataCollector")
    else:
        collector_cls = MultiaSyncDataCollector
        env_arg = [env] * num_collectors
        logger.info(f"Using MultiaSyncDataCollector with {num_collectors} collectors")

    collector = collector_cls(
        env_arg,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
        storing_device=device,
        exploration_type=ExplorationType.RANDOM,
        postproc=MultiStep(gamma=gamma, n_steps=n_steps),
    )
    
    logger.debug(f"Created collector with {frames_per_batch} frames per batch, {total_frames} total frames")
    return collector 