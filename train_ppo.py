#!/usr/bin/env python
"""
Omni RL Training Incubator – PPO Training Script
-------------------------------------------------
This script loads experiment configuration from a YAML file and trains a PPO policy
using TorchRL. It builds the environment, actor-critic modules, collector, loss,
optimizer, and trainer from configuration values.
"""

import os
import torch
from torch import nn
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
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
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.collectors import SyncDataCollector, MultiaSyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record.loggers.csv import CSVLogger
from torchrl.record.loggers.tensorboard import TensorboardLogger
from torchrl.record.loggers.wandb import WandbLogger
from torchrl.record.loggers.mlflow import MLFlowLogger
from torchrl.trainers import (
    LogReward,
    Recorder,
    ReplayBufferTrainer,
    Trainer,
    UpdateWeights,
)
from torchrl.envs import ExplorationType
from torchrl.data import MultiStep

import multiprocessing
import argparse

# -----------------------
# Load configuration
# -----------------------
# from utils.config_parser import load_config
import yaml



# -----------------------
# Helper: Create the Environment
# -----------------------
def make_env(env_name, parallel=False, num_workers=1):
    if parallel:
        def maker():
            return GymEnv(
                env_name=env_name,
                device=device,
                frame_skip=config["frame_skip"]
            )
        base_env = ParallelEnv(
            num_workers,
            EnvCreator(maker),
            serial_for_single=True,
            mp_start_method=config.get("mp_context", "fork"),
        )
    else:
        base_env = GymEnv(
            env_name,
            device=device,
            frame_skip=config["frame_skip"]
        )
    env = TransformedEnv(
        base_env,
        Compose(
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )
    return env



# -----------------------
# Helper: Create Actor-Critic Modules
# -----------------------
def make_actor_critic_modules(env):
    actor_net = nn.Sequential(
        nn.LazyLinear(config["num_cells"], device=device),
        # nn.Tanh(),
        nn.ReLU(),
        nn.LazyLinear(config["num_cells"], device=device),
        # nn.Tanh(),
        nn.ReLU(),
        nn.LazyLinear(config["num_cells"], device=device),
        # nn.Tanh(),
        nn.ReLU(),
        nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
        NormalParamExtractor(),
    )
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
    
    value_net = nn.Sequential(
        nn.LazyLinear(config["num_cells"], device=device),
        # nn.Tanh(),
        nn.ReLU(),
        nn.LazyLinear(config["num_cells"], device=device),
        # nn.Tanh(),
        nn.ReLU(),
        nn.LazyLinear(config["num_cells"], device=device),
        # nn.Tanh(),
        nn.ReLU(),
        nn.LazyLinear(1, device=device),
    )
    value_module = ValueOperator(module=value_net, in_keys=["observation"])
    
    return policy_module, value_module



def rl_incubator(config):

    # Create environment and initialize observation normalization stats
    env = make_env(config["env_name"], parallel=False)
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)


    policy_module, value_module = make_actor_critic_modules(env)
    # Run dummy forward passes to initialize modules.
    policy_module(env.reset())
    value_module(env.reset())

    # -----------------------
    # Build Collector
    # -----------------------
    # Use a SyncDataCollector if multiprocessing fork is allowed; else use MultiaSyncDataCollector.
    if is_fork:
        collector_cls = SyncDataCollector
        env_arg = env
    else:
        collector_cls = MultiaSyncDataCollector
        env_arg = [env] * config["num_collectors"]
        print("Using MultiaSyncDataCollector")

    collector = collector_cls(
        env_arg,
        policy_module,
        frames_per_batch=config["frames_per_batch"],
        total_frames=config["total_frames"],
        split_trajs=False,
        device=device,
        storing_device=device,
        exploration_type=ExplorationType.RANDOM,
        postproc=MultiStep(gamma=config["gamma"], n_steps=5),
    )

    # 4. Define Collector
    # collector = SyncDataCollector(
    #     env,
    #     policy_module,
    #     frames_per_batch=config["frames_per_batch"],
    #     total_frames=config["total_frames"],
    #     split_trajs=False,
    #     device=device,
    # )

    # -----------------------
    # Build Replay Buffer
    # -----------------------
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(config["frames_per_batch"]),
        sampler=SamplerWithoutReplacement(),
    )

    # -----------------------
    # Build Loss and Advantage Modules
    # -----------------------
    advantage_module = GAE(
        gamma=config["gamma"],
        lmbda=config["lmbda"],
        value_network=value_module,
        average_gae=True,
    )

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=config["clip_epsilon"],
        entropy_bonus=bool(config["entropy_eps"]),
        entropy_coef=config["entropy_eps"],
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    # -----------------------
    # Build Optimizer
    # -----------------------
    optimizer = torch.optim.Adam(
        loss_module.parameters(),
        betas=(config["beta1"], config["beta2"]),
        lr=config["lr"]
    )


    # -----------------------
    # Build Logger
    # -----------------------

    # import wandb

    # wandb_run = wandb.init(
    #     entity="geoff",
    #     project="TunedPPO",
    #     name="TunedPPO_experiment_demo",
    # )

    # wandb_logger = WandbLogger(exp_name="tuned_PPO", project="TunedPPO_experiment_demo", offline=True, log_dir="./wnb_logs") #  save_dir="./wnb_artifacts",

    tensorboard_logger = TensorboardLogger(exp_name="tuned_PPO", log_dir="./tb_logs")


    # -----------------------
    # Build Trainer
    # -----------------------
    trainer = Trainer(
        collector=collector,
        total_frames=config["total_frames"],
        frame_skip=config["frame_skip"],
        loss_module=loss_module,
        optimizer=optimizer,
        optim_steps_per_batch=config["n_optim"],
        logger=tensorboard_logger #wandb_logger,
    )




    # test_env = make_env(config["env_name"], parallel=False)


    # recorder = Recorder(
    # record_interval=100,  # log every 100 optimization steps
    # record_frames=1000,  # maximum number of frames in the record
    # frame_skip=1,
    # policy_exploration=policy_module,
    # environment=test_env,
    # exploration_type=ExplorationType.DETERMINISTIC,
    # log_keys=[("next", "reward")],
    # out_keys={("next", "reward"): "rewards"},
    # log_pbar=True,
    # )
    # recorder.register(trainer)


    trainer.train()

    
    # Save trained policy
    torch.save(policy_module.state_dict(), "ppo_policy.pth")


# -----------------------
# Run Training
# -----------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="configs/experiment_config_baseline.yaml")

    args = parser.parse_args()



    def load_config(config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    # config = load_config("rl_incubator/configs/experiment_config.yaml")

    # Determine device (this example forces CPU if no CUDA or if using fork mode on Mac)
    is_fork = multiprocessing.get_start_method() == "fork"
    device = torch.device(0) if torch.cuda.is_available() and not is_fork else torch.device("cpu")

    config=load_config(config_path=args.config)

    # Override device if specified in config (e.g., "cpu" or "cuda:0")
    if "device" in config:
        device = torch.device(config["device"])

    # Set MuJoCo rendering backend if on CPU (macOS)
    if device.type == "cpu":
        os.environ["MUJOCO_GL"] = "glfw"


    # Run training
    rl_incubator(config)
