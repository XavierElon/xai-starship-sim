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

```bash
# Standard training (CPU)
cd sac && python sac.py

# With curriculum learning
cd sac && python sac.py env.curriculum.enabled=true

# Override parameters
cd sac && python sac.py collector.total_frames=500000 optim.lr=1e-4
```

Checkpoints are saved to `checkpoints/` at each evaluation step and at the end of training.

## Evaluation

```bash
# Render high-quality videos from a checkpoint
python sac/eval.py --checkpoint checkpoints/final.pt --resolution 720 --episodes 5

# Multi-rocket visualization (100 rockets landing simultaneously)
python env/multi_rocket_viz.py --checkpoint checkpoints/final.pt --num-rockets 100 --resolution 1080
```

## GPU Environment (MuJoCo Warp)

For massively parallel training using MuJoCo Warp on NVIDIA GPUs:

```bash
pip install mujoco-warp
python env/rocket_landing_warp.py  # benchmark: ~1.2M steps/sec on RTX 4060
```

## Project Structure

```
sac/                        # SAC training
  sac.py                    # Training script (Hydra + TorchRL)
  eval.py                   # Checkpoint evaluation with video rendering
  utils.py                  # Environment/model creation utilities
  config.yaml               # Hydra configuration
env/                        # Environments
  rocket_landing.py         # Gymnasium MuJoCo environment (CPU)
  rocket_landing_warp.py    # MuJoCo Warp environment (GPU, batched)
  multi_rocket_viz.py       # Multi-rocket visualization
  config.py                 # Configuration dataclasses
  rewards.py                # Modular reward calculator
  xml_files/                # MuJoCo XML model files
rocket_designs/             # Design documentation and screenshots
LOGBOOK.md                  # Development log and experiment history
```
