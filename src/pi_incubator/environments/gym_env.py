"""
Environment setup module for Gymnasium environments.
"""
import torch
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    ParallelEnv,
    EnvCreator,
)
from torchrl.envs.libs.gym import GymEnv

def make_env(config, device, parallel=False, num_workers=None):
    """
    Create a Gymnasium environment with appropriate transformations.
    
    Args:
        config: Configuration dictionary containing environment parameters
        device: Torch device to place the environment on
        parallel: Whether to create a parallel environment
        num_workers: Number of parallel workers (if parallel is True)
    
    Returns:
        A TransformedEnv instance
    """
    env_name = config["env_name"]
    frame_skip = config.get("frame_skip", 1)
    mp_context = config.get("mp_context", "fork")
    
    if num_workers is None and "num_workers" in config:
        num_workers = config.get("num_workers", 1)
    
    if parallel:
        def maker():
            return GymEnv(
                env_name=env_name,
                device=device,
                frame_skip=frame_skip
            )
        base_env = ParallelEnv(
            num_workers,
            EnvCreator(maker),
            serial_for_single=True,
            mp_start_method=mp_context,
        )
    else:
        base_env = GymEnv(
            env_name,
            device=device,
            frame_skip=frame_skip
        )
    
    # Apply common transformations
    env = TransformedEnv(
        base_env,
        Compose(
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    
    return env

def initialize_env_stats(env, num_iter=1000, reduce_dim=0, cat_dim=0):
    """Initialize observation normalization statistics."""
    if hasattr(env, "transform") and len(env.transform) > 0:
        transform = env.transform[0]
        if isinstance(transform, ObservationNorm):
            transform.init_stats(num_iter=num_iter, reduce_dim=reduce_dim, cat_dim=cat_dim)
    return env 