#!/usr/bin/env python
"""Hyperparameter sweep for PPO training.

Runs each parameter configuration with multiple seeds and groups them in W&B
for easy comparison. Based on LOGBOOK TODO for improving seed consistency.
"""
import itertools
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class SweepConfig:
    """Configuration for a single sweep run."""
    name: str
    overrides: dict
    seeds: list[int]


# Baseline config (current settings)
BASELINE = {
    "loss.entropy_coeff": 0.01,
    "loss.gae_lambda": 0.95,
    "loss.mini_batch_size": 4096,
    "loss.ppo_epochs": 4,
    "loss.anneal_clip_epsilon": False,
}

# Sweep configurations - each tests one hypothesis from LOGBOOK
SWEEP_CONFIGS = [
    # Baseline
    SweepConfig(
        name="baseline",
        overrides={},
        seeds=[0, 1, 2],
    ),
    # Higher entropy to prevent early policy collapse
    SweepConfig(
        name="entropy_0.02",
        overrides={"loss.entropy_coeff": 0.02},
        seeds=[0, 1, 2],
    ),
    SweepConfig(
        name="entropy_0.05",
        overrides={"loss.entropy_coeff": 0.05},
        seeds=[0, 1, 2],
    ),
    # Higher GAE lambda for longer credit assignment
    SweepConfig(
        name="gae_0.97",
        overrides={"loss.gae_lambda": 0.97},
        seeds=[0, 1, 2],
    ),
    # Larger mini-batch for more stable gradients
    SweepConfig(
        name="minibatch_8192",
        overrides={"loss.mini_batch_size": 8192},
        seeds=[0, 1, 2],
    ),
    # More PPO epochs (with same clip epsilon)
    SweepConfig(
        name="epochs_8",
        overrides={"loss.ppo_epochs": 8},
        seeds=[0, 1, 2],
    ),
    # Clip epsilon annealing
    SweepConfig(
        name="clip_anneal",
        overrides={"loss.anneal_clip_epsilon": True},
        seeds=[0, 1, 2],
    ),
    # Combined: higher entropy + GAE (two most likely to help)
    SweepConfig(
        name="entropy_0.02_gae_0.97",
        overrides={
            "loss.entropy_coeff": 0.02,
            "loss.gae_lambda": 0.97,
        },
        seeds=[0, 1, 2],
    ),
]

# Follow-up sweep: combine clip_anneal with best performers
SWEEP_CONFIGS_V2 = [
    # Clip anneal + entropy 0.02
    SweepConfig(
        name="clip_anneal_entropy_0.02",
        overrides={
            "loss.anneal_clip_epsilon": True,
            "loss.entropy_coeff": 0.02,
        },
        seeds=[0, 1, 2],
    ),
    # Clip anneal + GAE 0.97
    SweepConfig(
        name="clip_anneal_gae_0.97",
        overrides={
            "loss.anneal_clip_epsilon": True,
            "loss.gae_lambda": 0.97,
        },
        seeds=[0, 1, 2],
    ),
    # Clip anneal + entropy 0.02 + GAE 0.97 (full combo)
    SweepConfig(
        name="clip_anneal_entropy_0.02_gae_0.97",
        overrides={
            "loss.anneal_clip_epsilon": True,
            "loss.entropy_coeff": 0.02,
            "loss.gae_lambda": 0.97,
        },
        seeds=[0, 1, 2],
    ),
    # Clip anneal + entropy 0.05 + GAE 0.97
    SweepConfig(
        name="clip_anneal_entropy_0.05_gae_0.97",
        overrides={
            "loss.anneal_clip_epsilon": True,
            "loss.entropy_coeff": 0.05,
            "loss.gae_lambda": 0.97,
        },
        seeds=[0, 1, 2],
    ),
]


def run_training(config: SweepConfig, seed: int, dry_run: bool = False) -> int:
    """Run a single training with the given config and seed."""
    # Build command
    cmd = ["uv", "run", "python", "train_ppo.py"]

    # Set seed
    cmd.append(f"env.seed={seed}")

    # Set group name for W&B (groups runs by hyperparam config)
    group_name = f"sweep_{config.name}"
    cmd.append(f"logger.group_name={group_name}")

    # Set experiment name (includes seed for uniqueness)
    exp_name = f"{config.name}_seed{seed}"
    cmd.append(f"logger.exp_name={exp_name}")

    # Add config overrides
    for key, value in config.overrides.items():
        if isinstance(value, bool):
            cmd.append(f"{key}={str(value).lower()}")
        else:
            cmd.append(f"{key}={value}")

    print(f"\n{'='*60}")
    print(f"Running: {config.name} (seed={seed})")
    print(f"Group: {group_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    if dry_run:
        return 0

    result = subprocess.run(cmd, cwd="/home/sebastian/Documents/smaller_projects/spaceX/training/multi_rocket")
    return result.returncode


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PPO Hyperparameter Sweep")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--config", type=str, help="Run only specific config (by name)")
    parser.add_argument("--seed", type=int, help="Run only specific seed")
    parser.add_argument("--list", action="store_true", help="List all configs and exit")
    parser.add_argument("--v2", action="store_true", help="Run follow-up sweep (clip_anneal combinations)")
    args = parser.parse_args()

    # Select sweep config set
    sweep_configs = SWEEP_CONFIGS_V2 if args.v2 else SWEEP_CONFIGS

    if args.list:
        print("Available sweep configurations:" + (" (v2)" if args.v2 else ""))
        for cfg in sweep_configs:
            print(f"  - {cfg.name}: {cfg.overrides or '(baseline)'}")
        return

    # Filter configs if requested
    configs = sweep_configs
    if args.config:
        configs = [c for c in configs if c.name == args.config]
        if not configs:
            print(f"Error: Config '{args.config}' not found")
            sys.exit(1)

    # Calculate total runs
    total_runs = sum(
        len([s for s in cfg.seeds if args.seed is None or s == args.seed])
        for cfg in configs
    )
    print(f"Total runs: {total_runs}")

    # Run sweep
    run_idx = 0
    failed = []
    for cfg in configs:
        seeds = cfg.seeds if args.seed is None else [args.seed]
        seeds = [s for s in seeds if s in cfg.seeds]

        for seed in seeds:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] Starting {cfg.name} seed={seed}")

            ret = run_training(cfg, seed, dry_run=args.dry_run)
            if ret != 0:
                failed.append((cfg.name, seed))
                print(f"WARNING: {cfg.name} seed={seed} failed with code {ret}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Sweep complete: {total_runs - len(failed)}/{total_runs} succeeded")
    if failed:
        print("Failed runs:")
        for name, seed in failed:
            print(f"  - {name} seed={seed}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
