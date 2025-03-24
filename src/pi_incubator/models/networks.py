"""
Network architectures for PPO and other RL algorithms.
"""
import torch
from torch import nn
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator

def create_mlp(input_dim, hidden_dims, output_dim, activation=nn.ReLU, output_activation=None):
    """Create a multi-layer perceptron with the specified architecture."""
    layers = []
    prev_dim = input_dim
    
    # Hidden layers
    for dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, dim))
        layers.append(activation())
        prev_dim = dim
    
    # Output layer
    layers.append(nn.Linear(prev_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation())
    
    return nn.Sequential(*layers)

def create_actor_critic_networks(env, config, device):
    """Create actor and critic networks for PPO."""
    # Actor network
    if "hidden_dims" in config:
        hidden_dims = config["hidden_dims"]
    else:
        hidden_dims = [config["num_cells"]] * 3  # Default: 3 hidden layers of size num_cells
    
    # Actor network using Lazy layers for flexibility
    actor_net = nn.Sequential(
        nn.LazyLinear(config["num_cells"], device=device),
        nn.ReLU(),
        nn.LazyLinear(config["num_cells"], device=device),
        nn.ReLU(),
        nn.LazyLinear(config["num_cells"], device=device),
        nn.ReLU(),
        nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
        NormalParamExtractor(),
    )
    
    # Create policy module
    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )
    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
        },
        return_log_prob=True,
    )
    
    # Value network
    value_net = nn.Sequential(
        nn.LazyLinear(config["num_cells"], device=device),
        nn.ReLU(),
        nn.LazyLinear(config["num_cells"], device=device),
        nn.ReLU(),
        nn.LazyLinear(config["num_cells"], device=device),
        nn.ReLU(),
        nn.LazyLinear(1, device=device),
    )
    value_module = ValueOperator(module=value_net, in_keys=["observation"])
    
    return policy_module, value_module 