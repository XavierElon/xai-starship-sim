# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpaceX rocket landing simulator using MuJoCo physics engine and Soft Actor-Critic (SAC) reinforcement learning. The goal is to train a policy to land a rocket vertically on a target pad, similar to SpaceX Falcon 9 landings.

## Commands

**Setup with uv:**
```bash
uv sync
source .venv/bin/activate
```

**Environment variables for MuJoCo rendering:**
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export MUJOCO_GL=egl  # headless rendering
```

**Run training:**
```bash
cd sac && python sac.py
```

**Override config parameters:**
```bash
cd sac && python sac.py collector.total_frames=500000 optim.lr=1e-4
```

**Test environment standalone:**
```bash
python env/rocket_landing.py
```

## Architecture

### MuJoCo Model (`env/xml_files/single_rocket_test.xml`)
- Rocket: 10kg cylinder (0.1m radius, 1m height) with free joint (6 DOF)
- Starting height: 50m
- Three actuators at thruster site:
  - `thrust_x/y`: lateral control (gear=25, range [-1,1])
  - `thrust_z`: main engine (gear=200, range [0,1])
- Target: 1m radius pad at origin with red cross marker

### Environment (`env/rocket_landing.py`)
- 13-dim observation: `[pos(3), roll, pitch, yaw, vel(3), angular_vel(3), distance]`
- 3-dim action: `[thrust_x, thrust_y, thrust_z]`
- Episode ends on: success (reaches target state), crash (z<0.5), roll/pitch > 70°
- Reward combines: position, orientation, velocity, angular velocity, distance components

### SAC Training (`sac/`)
- Uses TorchRL library with Hydra config
- Actor/Critic: 2-layer MLP (256, 256) with ReLU
- Logs to Weights & Biases
- Config in `sac/config.yaml`

## Key Parameters (config.yaml)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `collector.total_frames` | 1M | Total training steps |
| `collector.init_random_frames` | 25k | Random exploration before training |
| `optim.gamma` | 0.99 | Discount factor |
| `optim.lr` | 3e-4 | Learning rate |
| `env.max_episode_steps` | 1000 | Max steps per episode |
| `logger.eval_iter` | 20000 | Frames between evaluations |

## Planned Improvements

1. **Curriculum learning**: Start training at 5m height, progressively increase to 10m, 20m, etc. using transfer learning
2. **Rocket model**: Update to match SpaceX Falcon 9 with two landing legs (2D environment focus)
