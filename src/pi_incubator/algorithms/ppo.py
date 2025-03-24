"""
PPO-specific training components.
"""
import torch
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.trainers import Trainer

def create_ppo_components(policy_module, value_module, config, device=None):
    """
    Create PPO-specific components: buffer, advantage module, loss, optimizer.
    
    Args:
        policy_module: The policy (actor) module
        value_module: The value (critic) module
        config: Configuration dictionary
        device: Torch device
        
    Returns:
        Dictionary containing replay_buffer, advantage_module, loss_module, and optimizer
    """
    # Build Replay Buffer
    frames_per_batch = config.get("frames_per_batch", 1000)
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )
    
    # Build Loss and Advantage Modules
    gamma = config.get("gamma", 0.99)
    lmbda = config.get("lmbda", 0.95)
    clip_epsilon = config.get("clip_epsilon", 0.2)
    entropy_eps = config.get("entropy_eps", 1e-4)
    
    advantage_module = GAE(
        gamma=gamma,
        lmbda=lmbda,
        value_network=value_module,
        average_gae=True,
    )
    
    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )
    
    # Build Optimizer
    lr = config.get("lr", 3e-4)
    beta1 = config.get("beta1", 0.9)
    beta2 = config.get("beta2", 0.999)
    
    optimizer = torch.optim.Adam(
        loss_module.parameters(),
        betas=(beta1, beta2),
        lr=lr
    )
    
    return {
        "replay_buffer": replay_buffer,
        "advantage_module": advantage_module,
        "loss_module": loss_module,
        "optimizer": optimizer
    }

def create_ppo_trainer(collector, loss_module, optimizer, logger, config):
    """
    Create a trainer for PPO.
    
    Args:
        collector: Data collector
        loss_module: Loss module
        optimizer: Optimizer
        logger: Logger
        config: Configuration dictionary
        
    Returns:
        A trainer instance
    """
    total_frames = config.get("total_frames", 10000)
    frame_skip = config.get("frame_skip", 1)
    n_optim = config.get("n_optim", 10)
    
    trainer = Trainer(
        collector=collector,
        total_frames=total_frames,
        frame_skip=frame_skip,
        loss_module=loss_module,
        optimizer=optimizer,
        optim_steps_per_batch=n_optim,
        logger=logger,
    )
    
    return trainer

def save_policy(policy_module, path="ppo_policy.pth"):
    """Save the trained policy."""
    torch.save(policy_module.state_dict(), path)
    return path 