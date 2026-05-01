"""Utilities for PPO training (MuJoCo Warp GPU or Gymnasium MuJoCo CPU)."""
import os
import sys
from functools import partial

import torch
import torch.nn
from tensordict.nn import AddStateIndependentNormalScale, TensorDictModule
from torchrl.envs import Compose, ExplorationType, GymWrapper, SerialEnv, TransformedEnv
from torchrl.envs.transforms import StepCounter, InitTracker, RewardSum, DoubleToFloat
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator

try:
    from torchrl.envs.utils import default_info_dict_reader
except ImportError:  # pragma: no cover
    from torchrl.envs import default_info_dict_reader

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ====================================================================
# General utils
# ====================================================================


def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)


def get_activation(cfg):
    if cfg.network.activation == "relu":
        return torch.nn.ReLU
    elif cfg.network.activation == "tanh":
        return torch.nn.Tanh
    elif cfg.network.activation == "leaky_relu":
        return torch.nn.LeakyReLU
    else:
        raise NotImplementedError(f"Unknown activation: {cfg.network.activation}")


# ====================================================================
# Environment utils
# ====================================================================


def _env_backend(cfg):
    return getattr(cfg.env, "backend", "warp")


def _rocket_lander_kwargs_from_cfg(cfg):
    """Keyword args for env.rocket_landing.RocketLander (Gymnasium path)."""
    rocket_design = "v0"
    if hasattr(cfg.env, "rocket") and hasattr(cfg.env.rocket, "design"):
        rocket_design = cfg.env.rocket.design

    kwargs = {
        "rocket_design": rocket_design,
        "max_episode_length": cfg.env.max_episode_steps,
        "frame_skip": 5,
        "verbose": 0,
    }

    rw = {}
    if hasattr(cfg.env, "reward_weights"):
        rw_cfg = cfg.env.reward_weights
        for key in ("distance", "velocity", "upright", "angular", "success", "crash", "tipover"):
            if hasattr(rw_cfg, key):
                rw[key] = float(getattr(rw_cfg, key))
    if rw:
        kwargs["reward_weights"] = rw

    if hasattr(cfg.env, "termination"):
        term_cfg = cfg.env.termination
        if hasattr(term_cfg, "max_distance"):
            kwargs["max_distance"] = term_cfg.max_distance
        if hasattr(term_cfg, "max_angle"):
            kwargs["max_angle"] = term_cfg.max_angle

    return kwargs


def _make_rocket_gym_instance(lander_kwargs, curriculum_height):
    from env.rocket_landing import RocketLander

    env = RocketLander(**lander_kwargs)
    if curriculum_height is not None:
        env.set_curriculum_height(float(curriculum_height))
    gym_rl = GymWrapper(env, device="cpu")
    gym_rl.set_info_dict_reader(default_info_dict_reader(["crash_report"]))
    return gym_rl


def env_maker_gym(cfg, curriculum_height=None, num_envs=None):
    """Batched Gymnasium RocketLander via TorchRL SerialEnv (CPU MuJoCo)."""
    n = int(num_envs if num_envs is not None else cfg.env.num_envs)
    max_gym = int(getattr(cfg.env, "max_gym_envs", 128))
    if n > max_gym:
        raise ValueError(
            f"env.backend=gym uses one MuJoCo instance per worker; env.num_envs={n} exceeds "
            f"env.max_gym_envs={max_gym}. Lower env.num_envs and collector.frames_per_batch, "
            f"or raise env.max_gym_envs if you accept the memory/CPU cost."
        )

    policy_device = cfg.network.device or "cpu"
    if policy_device in ("", None):
        policy_device = "cpu"
    policy_device = torch.device(policy_device)

    lander_kwargs = _rocket_lander_kwargs_from_cfg(cfg)
    factory = partial(
        _make_rocket_gym_instance,
        dict(lander_kwargs),
        curriculum_height,
    )

    # SerialEnv steps all workers in-process; cast tensordicts to policy device for the learner.
    return SerialEnv(
        num_workers=n,
        create_env_fn=factory,
        device=policy_device,
    )


def env_maker(cfg, curriculum_height=None, num_envs=None):
    """Create training env: RocketLanderWarp (GPU) or Serial Gym RocketLander (CPU)."""
    if _env_backend(cfg) == "gym":
        return env_maker_gym(cfg, curriculum_height=curriculum_height, num_envs=num_envs)

    from env.rocket_landing_warp import RocketLanderWarp

    device = cfg.network.device or "cuda"

    rw = {}
    if hasattr(cfg.env, "reward_weights"):
        rw_cfg = cfg.env.reward_weights
        rw_map = {
            "distance": "w_distance",
            "velocity": "w_velocity",
            "upright": "w_upright",
            "angular": "w_angular",
            "success": "w_success",
            "crash": "w_crash",
            "tipover": "w_tipover",
            "time_penalty": "w_time_penalty",
        }
        for cfg_key, env_key in rw_map.items():
            if hasattr(rw_cfg, cfg_key):
                rw[env_key] = getattr(rw_cfg, cfg_key)

    term_kwargs = {}
    if hasattr(cfg.env, "termination"):
        term_cfg = cfg.env.termination
        if hasattr(term_cfg, "max_distance"):
            term_kwargs["max_distance"] = term_cfg.max_distance
        if hasattr(term_cfg, "max_angle"):
            term_kwargs["max_angle"] = term_cfg.max_angle
        if hasattr(term_cfg, "crash_velocity"):
            term_kwargs["crash_velocity"] = term_cfg.crash_velocity

    rocket_design = "v0"
    if hasattr(cfg.env, "rocket") and hasattr(cfg.env.rocket, "design"):
        rocket_design = cfg.env.rocket.design

    reset_noise_kwargs = {}
    if hasattr(cfg.env, "reset_noise"):
        rn_cfg = cfg.env.reset_noise
        if hasattr(rn_cfg, "pos"):
            reset_noise_kwargs["reset_pos_noise"] = rn_cfg.pos
        if hasattr(rn_cfg, "vel"):
            reset_noise_kwargs["reset_vel_noise"] = rn_cfg.vel
        if hasattr(rn_cfg, "ang"):
            reset_noise_kwargs["reset_ang_noise"] = rn_cfg.ang
        if hasattr(rn_cfg, "angvel"):
            reset_noise_kwargs["reset_angvel_noise"] = rn_cfg.angvel

    vel_penalty_kwargs = {}
    if hasattr(cfg.env, "velocity_penalty"):
        vp_cfg = cfg.env.velocity_penalty
        if hasattr(vp_cfg, "gate_scale"):
            vel_penalty_kwargs["vel_gate_scale"] = vp_cfg.gate_scale
        if hasattr(vp_cfg, "coeff"):
            vel_penalty_kwargs["vel_penalty_coeff"] = vp_cfg.coeff

    env = RocketLanderWarp(
        num_envs=num_envs if num_envs is not None else cfg.env.num_envs,
        rocket_design=rocket_design,
        device=device,
        max_episode_steps=cfg.env.max_episode_steps,
        starting_height=curriculum_height or 50.0,
        **rw,
        **term_kwargs,
        **reset_noise_kwargs,
        **vel_penalty_kwargs,
    )
    return env


def apply_env_transforms(env, max_episode_steps, double_to_float=False):
    parts = []
    if double_to_float:
        parts.append(DoubleToFloat())
    parts.extend(
        [
            StepCounter(max_steps=max_episode_steps),
            InitTracker(),
            RewardSum(),
        ]
    )
    transformed_env = TransformedEnv(env, Compose(*parts))
    return transformed_env


def make_environment(cfg, logger=None, curriculum_height=None):
    """Make environments for training and evaluation."""
    use_d2f = _env_backend(cfg) == "gym"
    train_env = env_maker(cfg, curriculum_height=curriculum_height)
    train_env = apply_env_transforms(
        train_env, cfg.env.max_episode_steps, double_to_float=use_d2f
    )

    eval_env = env_maker(cfg, curriculum_height=curriculum_height, num_envs=1)
    eval_env = apply_env_transforms(
        eval_env, cfg.env.max_episode_steps, double_to_float=use_d2f
    )

    return train_env, eval_env


def make_render_env(cfg):
    """Create a CPU Gymnasium env for eval/video rollouts.

    Uses vector observations (same 12-dim state as training) so the trained actor
    can drive the rocket. RGB frames are not exposed here; W&B video logs only
    when a pixel observation key exists on the rollout.
    """
    from env.rocket_landing import RocketLander

    rocket_design = "v0"
    if hasattr(cfg.env, "rocket") and hasattr(cfg.env.rocket, "design"):
        rocket_design = cfg.env.rocket.design

    rocket_env = RocketLander(
        rocket_design=rocket_design,
        render_mode="rgb_array",
        width=256,
        height=256,
    )
    env = GymWrapper(rocket_env, device="cpu", from_pixels=False)
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
