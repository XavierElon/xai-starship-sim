# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""SAC Example.

This is a simple self-contained example of a SAC training script.

It supports state environments like MuJoCo.

The helper functions are coded in the utils.py associated with this script.
"""
import os
import time
from collections import defaultdict

import hydra

# Save original working directory before Hydra changes it
_ORIGINAL_CWD = os.getcwd()

import numpy as np
import torch
import torch.cuda
import tqdm
import wandb
from tensordict import TensorDict
from torchrl._utils import logger as torchrl_logger
from torchrl.envs.utils import ExplorationType, set_exploration_type

from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    log_metrics,
    make_collector,
    make_environment,
    make_loss_module,
    make_replay_buffer,
    make_sac_agent,
    make_sac_optimizer,
)


# Crash report codes to names
CRASH_REPORT_NAMES = {
    0: "ongoing",
    1: "success",
    2: "crash",
    3: "roll_over",
    4: "pitch_over",
    5: "out_of_bounds",
}


def aggregate_crash_stats(crash_reports):
    """Aggregate crash report statistics from a batch of episodes."""
    stats = defaultdict(int)
    for code in crash_reports:
        name = CRASH_REPORT_NAMES.get(int(code), "unknown")
        stats[name] += 1
    return dict(stats)


def aggregate_reward_components(info_dicts):
    """Aggregate reward component statistics from info dictionaries."""
    component_sums = defaultdict(float)
    count = 0

    for info in info_dicts:
        if "reward_components" in info:
            components = info["reward_components"]
            for key, value in components.items():
                component_sums[key] += value
            count += 1

    if count == 0:
        return {}

    return {key: value / count for key, value in component_sums.items()}


def aggregate_episode_metrics(info_dicts):
    """Aggregate episode metrics from completed episodes."""
    metrics = defaultdict(list)

    for info in info_dicts:
        if "episode_metrics" in info:
            for key, value in info["episode_metrics"].items():
                metrics[key].append(value)

    return {key: np.mean(values) for key, values in metrics.items() if values}


@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821
    # Reset working directory to project root - Hydra changes it which breaks wandb video logging
    project_root = os.path.dirname(_ORIGINAL_CWD) if _ORIGINAL_CWD.endswith('sac') else _ORIGINAL_CWD
    os.chdir(project_root)

    device = cfg.network.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    device = torch.device(device)

    # Create logger
    exp_name = generate_exp_name("SAC", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="sac_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Get logging config options
    log_reward_components = getattr(cfg.logger, "log_reward_components", True)
    log_crash_breakdown = getattr(cfg.logger, "log_crash_breakdown", True)

    # Create environments
    train_env, eval_env = make_environment(cfg, logger=logger)

    # Create agent
    model, exploration_policy = make_sac_agent(cfg, train_env, eval_env, device)

    # Create SAC loss
    loss_module, target_net_updater = make_loss_module(cfg, model)

    # Create off-policy collector
    collector = make_collector(cfg, train_env, exploration_policy)

    # Create replay buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        scratch_dir=cfg.replay_buffer.scratch_dir,
        device="cpu",
    )

    # Create optimizers
    (
        optimizer_actor,
        optimizer_critic,
        optimizer_alpha,
    ) = make_sac_optimizer(cfg, loss_module)

    # Main loop
    start_time = time.time()
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    init_random_frames = cfg.collector.init_random_frames
    num_updates = int(
        cfg.collector.env_per_collector
        * cfg.collector.frames_per_batch
        * cfg.optim.utd_ratio
    )
    prb = cfg.replay_buffer.prb
    eval_iter = cfg.logger.eval_iter
    frames_per_batch = cfg.collector.frames_per_batch
    eval_rollout_steps = cfg.env.max_episode_steps

    # Tracking for crash statistics
    batch_crash_reports = []
    batch_reward_components = []
    batch_episode_metrics = []

    sampling_start = time.time()
    for i, tensordict in enumerate(collector):
        sampling_time = time.time() - sampling_start

        # Update weights of the inference policy
        collector.update_policy_weights_()

        pbar.update(tensordict.numel())

        tensordict = tensordict.reshape(-1)
        current_frames = tensordict.numel()
        # Add to replay buffer
        replay_buffer.extend(tensordict.cpu())
        collected_frames += current_frames

        # Collect crash reports and info from this batch
        if "crash_report" in tensordict.keys():
            # Get crash reports for done episodes
            done_mask = tensordict["next", "done"].squeeze(-1)
            if done_mask.any():
                crash_codes = tensordict["crash_report"][done_mask]
                batch_crash_reports.extend(crash_codes.tolist())

        # Optimization steps
        training_start = time.time()
        if collected_frames >= init_random_frames:
            losses = TensorDict({}, batch_size=[num_updates])
            for j in range(num_updates):
                # Sample from replay buffer
                sampled_tensordict = replay_buffer.sample()
                if sampled_tensordict.device != device:
                    sampled_tensordict = sampled_tensordict.to(
                        device, non_blocking=True
                    )
                else:
                    sampled_tensordict = sampled_tensordict.clone()

                # Compute loss
                loss_td = loss_module(sampled_tensordict)

                actor_loss = loss_td["loss_actor"]
                q_loss = loss_td["loss_qvalue"]
                alpha_loss = loss_td["loss_alpha"]

                # Update actor
                optimizer_actor.zero_grad()
                actor_loss.backward()
                optimizer_actor.step()

                # Update critic
                optimizer_critic.zero_grad()
                q_loss.backward()
                optimizer_critic.step()

                # Update alpha
                optimizer_alpha.zero_grad()
                alpha_loss.backward()
                optimizer_alpha.step()

                losses[j] = loss_td.select(
                    "loss_actor", "loss_qvalue", "loss_alpha"
                ).detach()

                # Update qnet_target params
                target_net_updater.step()

                # Update priority
                if prb:
                    replay_buffer.update_priority(sampled_tensordict)

        training_time = time.time() - training_start
        episode_end = (
            tensordict["next", "done"]
            if tensordict["next", "done"].any()
            else tensordict["next", "truncated"]
        )
        episode_rewards = tensordict["next", "episode_reward"][episode_end]

        # Logging
        metrics_to_log = {}
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][episode_end]
            metrics_to_log["train/reward"] = episode_rewards.mean().item()
            metrics_to_log["train/episode_length"] = episode_length.sum().item() / len(
                episode_length
            )

            # Log crash breakdown
            if log_crash_breakdown and batch_crash_reports:
                crash_stats = aggregate_crash_stats(batch_crash_reports)
                total_episodes = sum(crash_stats.values())
                for crash_type, count in crash_stats.items():
                    metrics_to_log[f"train/crash_{crash_type}"] = count
                    metrics_to_log[f"train/crash_{crash_type}_rate"] = (
                        count / total_episodes if total_episodes > 0 else 0
                    )
                batch_crash_reports = []  # Reset for next batch

        if collected_frames >= init_random_frames:
            metrics_to_log["train/q_loss"] = losses.get("loss_qvalue").mean().item()
            metrics_to_log["train/actor_loss"] = losses.get("loss_actor").mean().item()
            metrics_to_log["train/alpha_loss"] = losses.get("loss_alpha").mean().item()
            metrics_to_log["train/alpha"] = loss_td["alpha"].item()
            metrics_to_log["train/entropy"] = loss_td["entropy"].item()
            metrics_to_log["train/sampling_time"] = sampling_time
            metrics_to_log["train/training_time"] = training_time

        # Evaluation
        eval_video = None
        if abs(collected_frames % eval_iter) < frames_per_batch:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                eval_start = time.time()
                eval_rollout = eval_env.rollout(
                    eval_rollout_steps,
                    model[0],
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_time = time.time() - eval_start
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                # batch_size is [num_envs, timesteps] for rollout
                eval_episode_length = eval_rollout.batch_size[-1]  # Get timesteps dimension
                metrics_to_log["eval/reward"] = eval_reward
                metrics_to_log["eval/time"] = eval_time
                metrics_to_log["eval/episode_length"] = eval_episode_length

                # Prepare video for logging
                eval_video = None
                if cfg.logger.video and "pixels" in eval_rollout.keys():
                    pixels = eval_rollout["pixels"]
                    # pixels shape: [B, T, H, W, C] where B=num_envs, T=timesteps
                    if pixels.ndim == 5:
                        pixels = pixels[0]  # Take first env -> [T, H, W, C]
                    eval_video = pixels

                # Log evaluation episode metrics if available
                if "crash_report" in eval_rollout.keys():
                    # Get final crash report
                    final_crash = eval_rollout["crash_report"][-1].item()
                    crash_name = CRASH_REPORT_NAMES.get(int(final_crash), "unknown")
                    metrics_to_log["eval/outcome"] = final_crash
                    metrics_to_log[f"eval/outcome_{crash_name}"] = 1

                # Calculate eval episode metrics
                final_obs = eval_rollout["next", "observation"][-1]
                if final_obs.dim() > 1:
                    final_obs = final_obs[0]  # Take first env if batched

                # Extract position and velocity from observation
                # Observation: [pos_x, pos_y, pos_z, roll, pitch, yaw, vel_x, vel_y, vel_z, ...]
                pos = final_obs[:3].cpu().numpy()
                vel = final_obs[6:9].cpu().numpy()

                metrics_to_log["eval/final_horizontal_error"] = float(
                    np.sqrt(pos[0] ** 2 + pos[1] ** 2)
                )
                metrics_to_log["eval/final_velocity"] = float(np.linalg.norm(vel))

                # Calculate total thrust used in eval episode
                if "action" in eval_rollout.keys():
                    actions = eval_rollout["action"].cpu().numpy()
                    total_thrust = float(np.sum(np.abs(actions)))
                    metrics_to_log["eval/total_thrust"] = total_thrust

        if logger is not None:
            log_metrics(logger, metrics_to_log, collected_frames)
            # Log video directly with wandb - bypass TorchRL logger
            if eval_video is not None:
                vid = eval_video.cpu().numpy()
                vid = np.transpose(vid, (0, 3, 1, 2))  # [T,H,W,C] -> [T,C,H,W]
                vid = vid.astype(np.uint8)
                wandb.log({"eval/video": wandb.Video(vid, fps=20, format="mp4")})
        sampling_start = time.time()

    collector.shutdown()
    if not eval_env.is_closed:
        eval_env.close()
    if not train_env.is_closed:
        train_env.close()
    end_time = time.time()
    execution_time = end_time - start_time
    torchrl_logger.info(f"Training took {execution_time:.2f} seconds to finish")

    # Finish wandb run to ensure all media is synced
    if logger is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
