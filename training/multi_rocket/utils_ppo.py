"""Utilities for GPU-parallel PPO training with RocketLanderWarp."""
import os
import sys

import torch
import torch.nn
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.envs import ExplorationType
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Reuse env utilities from SAC setup
from utils import env_maker, apply_env_transforms, make_environment, log_metrics, get_activation  # noqa: E402

from torchrl.envs import Compose, TransformedEnv
from torchrl.envs.transforms import StepCounter, InitTracker, RewardSum, DoubleToFloat


def make_render_env(cfg):
    """Create a CPU Gymnasium env with pixel rendering for video logging."""
    from env.rocket_landing import RocketLander
    from torchrl.envs import GymWrapper

    rocket_design = "v0"
    if hasattr(cfg.env, "rocket") and hasattr(cfg.env.rocket, "design"):
        rocket_design = cfg.env.rocket.design

    rocket_env = RocketLander(
        rocket_design=rocket_design,
        render_mode="rgb_array",
        width=256,
        height=256,
    )
    env = GymWrapper(rocket_env, device="cpu", from_pixels=True)
    env = TransformedEnv(
        env,
        Compose(
            StepCounter(max_steps=cfg.env.max_episode_steps),
            InitTracker(),
            DoubleToFloat(),
            RewardSum(),
        ),
    )
    return env


# ====================================================================
# PPO Model
# ---------


def make_ppo_models(cfg, train_env, device):
    """Build PPO actor and critic networks.

    Actor: MLP → AddStateIndependentNormalScale → ProbabilisticActor (returns log_prob)
    Critic: MLP → ValueOperator (predicts state_value)
    """
    input_shape = train_env.observation_spec["observation"].shape
    action_spec = train_env.action_spec
    if train_env.batch_size:
        action_spec = action_spec[(0,) * len(train_env.batch_size)]

    num_outputs = action_spec.shape[-1]
    activation_class = get_activation(cfg)
    hidden_sizes = cfg.network.hidden_sizes

    # --- Actor ---
    policy_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=activation_class,
        out_features=num_outputs,
        num_cells=hidden_sizes,
        device=device,
    )

    # Orthogonal init (PPO standard)
    for layer in policy_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    policy_mlp = torch.nn.Sequential(
        policy_mlp,
        AddStateIndependentNormalScale(num_outputs, scale_lb=1e-8).to(device),
    )

    policy_module = ProbabilisticActor(
        TensorDictModule(
            module=policy_mlp,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        spec=action_spec,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
            "tanh_loc": False,
        },
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # --- Critic ---
    value_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=activation_class,
        out_features=1,
        num_cells=hidden_sizes,
        device=device,
    )

    # Orthogonal init with small scale for value head
    for layer in value_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    value_module = ValueOperator(
        value_mlp,
        in_keys=["observation"],
    )

    return policy_module, value_module
