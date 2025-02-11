import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.collectors import SyncDataCollector, MultiaSyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data import  MultiStep

from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
    EnvCreator,
    ExplorationType,
    ParallelEnv,
    RewardScaling,
)
from torchrl.envs.transforms import (
    CatFrames,
    Compose,
    GrayScale,
    ObservationNorm,
    Resize,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.trainers import Trainer
import multiprocessing



# Set device
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)


# Set environment variables for MuJoCo rendering on macOS
import os
# if not torch.cuda.is_available():
if device.type == "cpu":
    os.environ["MUJOCO_GL"] = "glfw"

# 1. Define Hyperparameters
device = "cpu"
num_cells = 256
frame_skip = 1
lr = 3e-4
frames_per_batch = 1000 // frame_skip
total_frames = 5000 // frame_skip  #50_000 // frame_skip
sub_batch_size = 64
num_epochs = 10
clip_epsilon = 0.2
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4
n_optim = 8  # Optimization steps per batch collected (UPD)
log_interval = 500
mp_context = "fork"  # "spawn" or "fork" for MacOS
env_name="InvertedDoublePendulum-v4"


num_workers = 2  # 8
num_collectors = 2  # 4


# 2. Define Environment
# base_env = GymEnv("InvertedDoublePendulum-v4", device=device, frame_skip=frame_skip)
# env = TransformedEnv(
#     base_env,
#     Compose(
#         ObservationNorm(in_keys=["observation"]),
#         DoubleToFloat(),
#         StepCounter(),
#     ),
# )


def make_env(
    env_name="InvertedDoublePendulum-v4",
    parallel=False,
    obs_norm_sd=None,
    num_workers=1,
):
    if obs_norm_sd is None:
        obs_norm_sd = {"standard_normal": True}
    if parallel:

        def maker():
            return GymEnv(
                env_name=env_name,
                # from_pixels=True,
                # pixels_only=True,
                device=device,
                frame_skip=frame_skip
            )

        base_env = ParallelEnv(
            num_workers,
            EnvCreator(maker),
            # Don't create a sub-process if we have only one worker
            serial_for_single=True,
            mp_start_method=mp_context,
        )
    else:
        base_env = GymEnv(
            env_name,
            # from_pixels=True,
            # pixels_only=True,
            device=device,
            frame_skip=frame_skip
        )

   
    # NOTE: Here is another use case using image-based observations
    # env = TransformedEnv(
    #     base_env,
    #     Compose(
    #         StepCounter(),  # to count the steps of each trajectory
    #         ToTensorImage(),
    #         RewardScaling(loc=0.0, scale=0.1),
    #         GrayScale(),
    #         Resize(64, 64),
    #         CatFrames(4, in_keys=["pixels"], dim=-3),
    #         ObservationNorm(in_keys=["pixels"], **obs_norm_sd),
    #     ),
    # )

    env = TransformedEnv(
        base_env,
        Compose(
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
        ),
    )

    return env



env = make_env(env_name, parallel=False)
env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)


# 3. Define Actor-Critic Model

def make_actor_critic_modules(env):

    actor_net = nn.Sequential(
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
    nn.LazyLinear(num_cells, device=device),
    nn.Tanh(),
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
            "min": env.action_spec.space.low,
            "max": env.action_spec.space.high,
        },
        return_log_prob=True,
    )

    value_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(1, device=device),
    )
    value_module = ValueOperator(module=value_net, in_keys=["observation"])

    return policy_module, value_module


policy_module, value_module = make_actor_critic_modules(env)


policy_module(env.reset())
value_module(env.reset())


# 4. Define Collector
collector = SyncDataCollector(
    env,
    policy_module,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)


# def get_norm_stats():
#     test_env = make_env()
#     test_env.transform[-1].init_stats(
#         num_iter=1000, cat_dim=0, reduce_dim=[-1, -2, -4], keep_dims=(-1, -2)
#     )
#     obs_norm_sd = test_env.transform[-1].state_dict()
#     # let's check that normalizing constants have a size of ``[C, 1, 1]`` where
#     # ``C=4`` (because of :class:`~torchrl.envs.CatFrames`).
#     print("state dict of the observation norm:", obs_norm_sd)
#     test_env.close()
#     del test_env
#     return obs_norm_sd


stats = None # get_norm_stats()


def get_collector(
    stats,
    num_collectors,
    actor_explore,
    frames_per_batch,
    total_frames,
    device,
):
    # We can't use nested child processes with mp_start_method="fork"
    if is_fork:
        cls = SyncDataCollector
        env_arg = make_env(parallel=True, obs_norm_sd=stats, num_workers=num_workers)
    else:
        cls = MultiaSyncDataCollector
        env_arg = [
            make_env(parallel=True, obs_norm_sd=stats, num_workers=num_workers)
        ] * num_collectors

    data_collector = cls(
        env_arg,
        policy=actor_explore,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
        storing_device=device,
        # this is the default behavior: the collector runs in ``"random"`` (or explorative) mode
        exploration_type=ExplorationType.RANDOM,
        postproc=MultiStep(gamma=gamma, n_steps=5),
    )
    return data_collector





# collector = get_collector(stats=stats, num_collectors=num_collectors, actor_explore=policy_module, frames_per_batch=frames_per_batch, total_frames=total_frames, device=device)





# 5. Define Replay Buffer
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)


# 6. Define Loss
advantage_module = GAE(
    gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True
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





# 7. Define Optimizer
optimizer = torch.optim.Adam(loss_module.parameters(), betas=(0.9, 0.999), lr=lr)

# 8. Define Trainer
trainer = Trainer(
    collector=collector,
    total_frames=total_frames,
    frame_skip=frame_skip,
    loss_module=loss_module,
    optimizer=optimizer,
    optim_steps_per_batch=n_optim,
)

# 9. Train PPO
trainer.train()

# Save trained model
torch.save(policy_module.state_dict(), "ppo_policy.pth")
