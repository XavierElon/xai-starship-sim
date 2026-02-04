"""Utilities for GPU-parallel SAC training with RocketLanderWarp."""
import os
import sys

import torch
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn, optim
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import Compose, TransformedEnv
from torchrl.envs.transforms import InitTracker, RewardSum, StepCounter
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ====================================================================
# Environment utils
# -----------------


def env_maker(cfg, curriculum_height=None, num_envs=None):
    """Create a RocketLanderWarp environment (already a TorchRL EnvBase)."""
    from env.rocket_landing_warp import RocketLanderWarp

    device = cfg.network.device or "cuda"

    # Build reward weight kwargs
    rw = {}
    if hasattr(cfg.env, "reward_weights"):
        rw_cfg = cfg.env.reward_weights
        rw_map = {
            "position": "w_position",
            "orientation": "w_orientation",
            "velocity": "w_velocity",
            "angular_velocity": "w_angular_velocity",
            "distance": "w_distance",
            "success_bonus": "w_success",
            "crash_penalty": "w_crash",
            "tip_over_penalty": "w_tipover",
        }
        for cfg_key, env_key in rw_map.items():
            if hasattr(rw_cfg, cfg_key):
                rw[env_key] = getattr(rw_cfg, cfg_key)

    # Build termination kwargs
    term_kwargs = {}
    if hasattr(cfg.env, "termination"):
        term_cfg = cfg.env.termination
        if hasattr(term_cfg, "max_distance"):
            term_kwargs["max_distance"] = term_cfg.max_distance
        if hasattr(term_cfg, "max_angle"):
            term_kwargs["max_angle"] = term_cfg.max_angle

    # Rocket design
    rocket_design = "v0"
    if hasattr(cfg.env, "rocket") and hasattr(cfg.env.rocket, "design"):
        rocket_design = cfg.env.rocket.design

    env = RocketLanderWarp(
        num_envs=num_envs if num_envs is not None else cfg.env.num_envs,
        rocket_design=rocket_design,
        device=device,
        max_episode_steps=cfg.env.max_episode_steps,
        starting_height=curriculum_height or 50.0,
        **rw,
        **term_kwargs,
    )
    return env


def apply_env_transforms(env, max_episode_steps):
    transformed_env = TransformedEnv(
        env,
        Compose(
            StepCounter(max_steps=max_episode_steps),
            InitTracker(),
            RewardSum(),
        ),
    )
    return transformed_env


def make_environment(cfg, logger=None, curriculum_height=None):
    """Make environments for training and evaluation.

    The Warp env is already batched, so we don't need ParallelEnv.
    Train uses cfg.env.num_envs; eval uses a single env for clean episode metrics.
    """
    train_env = env_maker(cfg, curriculum_height=curriculum_height)
    train_env = apply_env_transforms(train_env, cfg.env.max_episode_steps)

    eval_env = env_maker(cfg, curriculum_height=curriculum_height, num_envs=1)
    eval_env = apply_env_transforms(eval_env, cfg.env.max_episode_steps)

    return train_env, eval_env


# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, train_env, actor_model_explore, total_frames=None, init_random_frames=None):
    """Make collector for GPU environments."""
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=init_random_frames if init_random_frames is not None else cfg.collector.init_random_frames,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=total_frames if total_frames is not None else cfg.collector.total_frames,
        device=cfg.collector.device,
    )
    collector.set_seed(cfg.env.seed)
    return collector


def make_replay_buffer(
    batch_size,
    prb=False,
    buffer_size=100000,
    device="cuda",
    prefetch=3,
):
    if prb:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyTensorStorage(
                buffer_size,
                device=device,
            ),
            batch_size=batch_size,
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyTensorStorage(
                buffer_size,
                device=device,
            ),
            batch_size=batch_size,
        )
    return replay_buffer


# ====================================================================
# Model
# -----


def make_sac_agent(cfg, train_env, eval_env, device):
    """Make SAC agent."""
    in_keys = ["observation"]
    action_spec = train_env.action_spec
    if train_env.batch_size:
        action_spec = action_spec[(0,) * len(train_env.batch_size)]
    actor_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": 2 * action_spec.shape[-1],
        "activation_class": get_activation(cfg),
    }

    actor_net = MLP(**actor_net_kwargs)

    dist_class = TanhNormal
    dist_kwargs = {
        "low": action_spec.space.low,
        "high": action_spec.space.high,
        "tanh_loc": False,
    }

    actor_extractor = NormalParamExtractor(
        scale_mapping=f"biased_softplus_{cfg.network.default_policy_scale}",
        scale_lb=cfg.network.scale_lb,
    )
    actor_net = nn.Sequential(actor_net, actor_extractor)

    in_keys_actor = in_keys
    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=["loc", "scale"],
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    # Define Critic Network
    qvalue_net_kwargs = {
        "num_cells": cfg.network.hidden_sizes,
        "out_features": 1,
        "activation_class": get_activation(cfg),
    }

    qvalue_net = MLP(**qvalue_net_kwargs)

    qvalue = ValueOperator(
        in_keys=["action"] + in_keys,
        module=qvalue_net,
    )

    model = nn.ModuleList([actor, qvalue]).to(device)

    # init nets
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = eval_env.fake_tensordict()
        td = td.to(device)
        for net in model:
            net(td)
    return model, model[0]


# ====================================================================
# SAC Loss
# ---------


def make_loss_module(cfg, model):
    """Make loss module and target network updater."""
    loss_module = SACLoss(
        actor_network=model[0],
        qvalue_network=model[1],
        num_qvalue_nets=2,
        loss_function=cfg.optim.loss_function,
        delay_actor=False,
        delay_qvalue=True,
        alpha_init=cfg.optim.alpha_init,
    )
    loss_module.make_value_estimator(gamma=cfg.optim.gamma)

    target_net_updater = SoftUpdate(loss_module, eps=cfg.optim.target_update_polyak)
    return loss_module, target_net_updater


def make_sac_optimizer(cfg, loss_module):
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
    actor_params = list(loss_module.actor_network_params.flatten_keys().values())

    optimizer_actor = optim.Adam(
        actor_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    optimizer_critic = optim.Adam(
        critic_params,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        eps=cfg.optim.adam_eps,
    )
    optimizer_alpha = optim.Adam(
        [loss_module.log_alpha],
        lr=3.0e-4,
    )
    return optimizer_actor, optimizer_critic, optimizer_alpha


# ====================================================================
# General utils
# ---------


def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)


def get_activation(cfg):
    if cfg.network.activation == "relu":
        return nn.ReLU
    elif cfg.network.activation == "tanh":
        return nn.Tanh
    elif cfg.network.activation == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError
