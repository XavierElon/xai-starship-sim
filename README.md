# SpaceX Rocket Landing Simulator

A SpaceX Falcon 9-style rocket landing simulator using MuJoCo physics and Soft Actor-Critic (SAC) reinforcement learning.

![Rocket Landing](rocket_preview.png)

## Overview

Train an RL agent to land a rocket vertically on a target pad, similar to SpaceX Falcon 9 landings. The environment features:

- **Realistic physics**: MuJoCo simulation with 6-DOF free joint dynamics
- **Multiple rocket designs**: Simple cylinder (v0), two-leg (v1), and stable tripod (v2)
- **GPU-accelerated training**: MuJoCo Warp backend for ~10,000x speedup with 4096 parallel envs
- **Curriculum learning**: Progressive height stages (5m -> 10m -> 20m -> 35m -> 50m)
- **Domain randomization**: Configurable mass, thrust, gravity, and initial conditions
- **Detailed metrics**: Crash breakdowns, reward components, and video logging to W&B

See [LOGBOOK.md](LOGBOOK.md) for the full development history, experiment results, and design decisions.

## Setup

```bash
uv sync
source .venv/bin/activate
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export MUJOCO_GL=egl  # headless rendering
```

## Training

### Single Rocket (CPU)

```bash
# Standard training
cd training/single_rocket && python train.py

# With curriculum learning
cd training/single_rocket && python train.py env.curriculum.enabled=true

# Override parameters
cd training/single_rocket && python train.py collector.total_frames=500000 optim.lr=1e-4
```

Checkpoints are saved to `training/single_rocket/checkpoints/`.

### Multi Rocket (GPU, MuJoCo Warp)

```bash
# GPU-parallel training (4096 environments)
cd training/multi_rocket && python train.py

# Override number of environments
cd training/multi_rocket && python train.py env.num_envs=8192
```

Checkpoints are saved to `training/multi_rocket/checkpoints/`.

## Evaluation

```bash
# Render high-quality videos from a single-rocket checkpoint
python training/single_rocket/eval.py --checkpoint training/single_rocket/checkpoints/final.pt --resolution 720 --episodes 5

# Evaluate a warp-trained checkpoint (renders via CPU env)
python training/multi_rocket/eval.py --checkpoint training/multi_rocket/checkpoints/final.pt --resolution 720 --episodes 5

# Multi-rocket visualization (100 rockets landing simultaneously)
python env/multi_rocket_viz.py --checkpoint training/single_rocket/checkpoints/final.pt --num-rockets 100 --resolution 1080
```

## GPU Environment (MuJoCo Warp)

For massively parallel training using MuJoCo Warp on NVIDIA GPUs:

```bash
pip install mujoco-warp
python env/rocket_landing_warp.py  # benchmark: ~1.2M steps/sec on RTX 4060
```

## Project Structure

```
training/
  single_rocket/              # CPU-based SAC training (single env)
    train.py                  # Training script (Hydra + TorchRL)
    eval.py                   # Checkpoint evaluation with video rendering
    utils.py                  # Environment/model creation utilities
    config.yaml               # Hydra configuration
    checkpoints/              # Saved checkpoints
  multi_rocket/               # GPU-parallel SAC training (MuJoCo Warp)
    train.py                  # GPU training script
    eval.py                   # Checkpoint evaluation
    utils.py                  # Warp env utilities
    config.yaml               # GPU training configuration
    checkpoints/              # Saved checkpoints
env/                          # Shared environments
  rocket_landing.py           # Gymnasium MuJoCo environment (CPU)
  rocket_landing_warp.py      # MuJoCo Warp environment (GPU, batched)
  multi_rocket_viz.py         # Multi-rocket visualization
  config.py                   # Configuration dataclasses
  rewards.py                  # Modular reward calculator
  xml_files/                  # MuJoCo XML model files
rocket_designs/               # Design documentation and screenshots
LOGBOOK.md                    # Development log and experiment history
```
