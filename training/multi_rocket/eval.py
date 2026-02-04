"""Evaluate a warp-trained SAC checkpoint.

Can evaluate using either:
- CPU RocketLander env (for video rendering)
- GPU RocketLanderWarp env (for fast metrics)

Usage:
    python training/multi_rocket/eval.py --checkpoint training/multi_rocket/checkpoints/final.pt
    python training/multi_rocket/eval.py --checkpoint training/multi_rocket/checkpoints/final.pt --gpu --episodes 10
    python training/multi_rocket/eval.py --checkpoint training/multi_rocket/checkpoints/final.pt --resolution 720 --episodes 5
"""

import argparse
import os
import sys

import imageio
import numpy as np
import torch
from torch import nn
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs import GymWrapper, TransformedEnv, Compose
from torchrl.envs.transforms import (
    DoubleToFloat,
    InitTracker,
    RewardSum,
    StepCounter,
    VecNorm,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from env.rocket_landing import RocketLander

CRASH_REPORT_NAMES = {
    0: "ongoing",
    1: "success",
    2: "crash",
    3: "roll_over",
    4: "pitch_over",
    5: "out_of_bounds",
}


def build_cpu_env(rocket_design, resolution, curriculum_height=None, max_episode_steps=1000):
    """Create a CPU environment for video rendering."""
    os.chdir(ROOT_DIR)
    rocket_env = RocketLander(
        rocket_design=rocket_design,
        render_mode="rgb_array",
        width=resolution,
        height=resolution,
    )
    if curriculum_height is not None:
        rocket_env.set_curriculum_height(curriculum_height)

    env = GymWrapper(rocket_env, device="cpu", from_pixels=True)
    env = TransformedEnv(
        env,
        Compose(
            StepCounter(max_steps=max_episode_steps),
            InitTracker(),
            DoubleToFloat(),
            RewardSum(),
            VecNorm(),
        ),
    )
    return env


def build_actor(env, hidden_sizes, activation, scale_mapping, scale_lb, device):
    """Reconstruct the actor network matching the training architecture."""
    action_spec = env.action_spec

    act_cls = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}[activation]

    actor_net = MLP(
        num_cells=hidden_sizes,
        out_features=2 * action_spec.shape[-1],
        activation_class=act_cls,
    )
    actor_extractor = NormalParamExtractor(
        scale_mapping=scale_mapping,
        scale_lb=scale_lb,
    )
    actor_net = nn.Sequential(actor_net, actor_extractor)

    actor_module = TensorDictModule(
        actor_net,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        module=actor_module,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": action_spec.space.low,
            "high": action_spec.space.high,
            "tanh_loc": False,
        },
        default_interaction_type=InteractionType.RANDOM,
        return_log_prob=False,
    )

    qvalue_net = MLP(
        num_cells=hidden_sizes,
        out_features=1,
        activation_class=act_cls,
    )
    qvalue = ValueOperator(
        in_keys=["action", "observation"],
        module=qvalue_net,
    )

    model = nn.ModuleList([actor, qvalue]).to(device)

    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = env.fake_tensordict().to(device)
        for net in model:
            net(td)

    return model


def run_eval_episode(env, actor, device, max_steps=1000):
    """Run a single deterministic evaluation episode."""
    frames = []
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        td = env.rollout(
            max_steps,
            actor,
            auto_cast_to_device=True,
            break_when_any_done=True,
        )

    if "pixels" in td.keys():
        pixels = td["pixels"]
        if pixels.ndim == 5:
            pixels = pixels[0]
        frames = pixels.cpu().numpy().astype(np.uint8)

    reward = td["next", "reward"].sum().item()
    episode_length = td.batch_size[-1] if td.batch_size else len(frames)

    final_obs = td["next", "observation"][-1]
    if final_obs.dim() > 1:
        final_obs = final_obs[0]
    pos = final_obs[:3].cpu().numpy()
    vel = final_obs[6:9].cpu().numpy()
    roll_deg = float(final_obs[3].cpu())
    pitch_deg = float(final_obs[4].cpu())
    horizontal_dist = float(np.sqrt(pos[0] ** 2 + pos[1] ** 2))

    terminated = td["next", "terminated"][-1].item()
    if not terminated:
        outcome = "truncated"
    elif pos[2] < 0.5:
        outcome = "crash"
    elif abs(roll_deg) > 70:
        outcome = "roll_over"
    elif abs(pitch_deg) > 70:
        outcome = "pitch_over"
    elif horizontal_dist > 20:
        outcome = "out_of_bounds"
    else:
        outcome = "success"

    metrics = {
        "reward": reward,
        "episode_length": episode_length,
        "outcome": outcome,
        "final_horizontal_error": horizontal_dist,
        "final_velocity": float(np.linalg.norm(vel)),
        "final_height": float(pos[2]),
    }
    return frames, metrics


def save_video(frames, path, fps=40):
    """Save frames as an MP4 video."""
    if len(frames) == 0:
        print(f"  No frames to save for {path}")
        return
    writer = imageio.get_writer(path, fps=fps, quality=8, codec="libx264")
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved: {path} ({len(frames)} frames, {size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate warp-trained SAC checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--resolution", type=int, default=720, help="Video resolution (default: 720)")
    parser.add_argument("--fps", type=int, default=40, help="Video FPS (default: 40)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to render")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: videos/)")
    parser.add_argument("--rocket-design", type=str, default=None, help="Override rocket design (v0/v1/v2)")
    parser.add_argument("--height", type=float, default=None, help="Override starting height")
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[256, 256], help="Network hidden sizes")
    parser.add_argument("--activation", type=str, default="relu", help="Activation function")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Extract config from checkpoint if available
    ckpt_config = checkpoint.get("config", {})
    rocket_design = args.rocket_design
    if rocket_design is None:
        rocket_design = ckpt_config.get("env", {}).get("rocket", {}).get("design", "v0")
    curriculum_height = args.height
    if curriculum_height is None:
        curriculum_height = checkpoint.get("curriculum_height", None)

    hidden_sizes = args.hidden_sizes
    activation = args.activation
    net_cfg = ckpt_config.get("network", {})
    if net_cfg:
        hidden_sizes = net_cfg.get("hidden_sizes", hidden_sizes)
        activation = net_cfg.get("activation", activation)

    scale_mapping = f"biased_softplus_{net_cfg.get('default_policy_scale', 1.0)}"
    scale_lb = net_cfg.get("scale_lb", 0.1)

    print(f"  Rocket design: {rocket_design}")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    print(f"  Network: {hidden_sizes}, {activation}")
    if curriculum_height is not None:
        print(f"  Starting height: {curriculum_height}m")

    # Build CPU environment for video rendering
    env = build_cpu_env(
        rocket_design=rocket_design,
        resolution=args.resolution,
        curriculum_height=curriculum_height,
        max_episode_steps=args.max_steps,
    )

    model = build_actor(env, hidden_sizes, activation, scale_mapping, scale_lb, device)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    actor = model[0]

    frames_trained = checkpoint.get("collected_frames", "unknown")
    print(f"  Trained for {frames_trained} frames")

    # Output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(ROOT_DIR, "videos")
    os.makedirs(output_dir, exist_ok=True)

    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]

    # Run episodes
    print(f"\nRunning {args.episodes} evaluation episodes...")
    all_metrics = []
    for ep in range(args.episodes):
        print(f"\nEpisode {ep + 1}/{args.episodes}:")
        frames, metrics = run_eval_episode(env, actor, device, args.max_steps)
        all_metrics.append(metrics)

        print(f"  Outcome: {metrics['outcome']}")
        print(f"  Reward: {metrics['reward']:.1f}")
        print(f"  Length: {metrics['episode_length']} steps")
        print(f"  Final height: {metrics['final_height']:.2f}m")
        print(f"  Final h-error: {metrics['final_horizontal_error']:.2f}m")
        print(f"  Final velocity: {metrics['final_velocity']:.2f}m/s")

        video_path = os.path.join(output_dir, f"{ckpt_name}_ep{ep}.mp4")
        save_video(frames, video_path, fps=args.fps)

    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    outcomes = [m["outcome"] for m in all_metrics]
    for outcome in set(outcomes):
        count = outcomes.count(outcome)
        print(f"  {outcome}: {count}/{len(outcomes)}")
    avg_reward = np.mean([m["reward"] for m in all_metrics])
    print(f"  Avg reward: {avg_reward:.1f}")
    print(f"  Videos saved to: {output_dir}/")

    env.close()


if __name__ == "__main__":
    main()
