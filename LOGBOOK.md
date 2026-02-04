# Project Logbook

Development log for the SpaceX rocket landing simulator. Tracks design decisions, experiments, and results chronologically.

---

## Phase 1: Environment Foundation

### Initial Setup
Built a MuJoCo-based rocket landing environment inspired by SpaceX Falcon 9 booster landings. The core challenge: train an RL agent to control a rocket's three thrusters (lateral X/Y + main vertical) to land upright on a target pad from 50m altitude.

**Environment specs:**
- 13-dim observation: position, orientation (roll/pitch/yaw), velocity, angular velocity, distance to target
- 3-dim continuous action: thrust_x [-1,1], thrust_y [-1,1], thrust_z [0,1]
- Physics: MuJoCo with 0.005s timestep, frame_skip=5 (40Hz control rate)
- Termination: crash (z<0.5m), tip over (roll/pitch > 70deg), out of bounds (>20m horizontal)

### Rocket Design v0 — Simple Cylinder
Starting point: a 10kg cylinder with a free joint at 50m height. No legs, lands on its flat bottom.

![v0](rocket_designs/screenshots/design_v0.png)

- Target landed height: 1.0m
- Simple enough to debug physics and reward shaping
- Thrust-to-weight ratio ~2.0 (can hover at ~50% throttle)

### Rocket Design v1 — Two Landing Legs
Added two landing legs on the X-axis for visual clarity. Quickly discovered this was inherently unstable — two legs can't provide a stable base.

![v1](rocket_designs/screenshots/design_v1.png)

- Target landed height: 1.85m
- **Deprecated**: two-point contact is unstable, rocket tips over even when stationary

### Rocket Design v2 — Tripod (Current)
Three legs at 120deg spacing. Stable tripod configuration, realistic Falcon 9-style proportions with nose cone, interstage band, and engine section.

![v2](rocket_designs/screenshots/design_v2.png)

- Total mass: 9.71kg (distributed across body sections and legs)
- Target landed height: 1.93m
- Stable at rest — the agent just needs to get there

---

## Phase 2: Training Pipeline

### SAC with TorchRL
Chose Soft Actor-Critic for continuous control. Built on TorchRL with Hydra config.

**Architecture:**
- Actor/Critic: 2-layer MLP (256, 256) with ReLU
- Learning rate: 3e-4, batch size: 256, gamma: 0.99
- Replay buffer: 1M transitions
- 25k random exploration frames before training

### Reward Shaping
Modular `RewardCalculator` with weighted components:
- Position: penalize horizontal drift + vertical distance from target
- Orientation: penalize deviation from upright
- Velocity: penalize high speeds
- Angular velocity: penalize rotation
- Distance: penalize distance to target
- Terminal bonuses: +100 success, -50 crash, -30 tip-over

### Video Logging
Added W&B video logging for eval episodes. Initially struggled with:
- Hydra changing the working directory (broke file paths)
- TorchRL's VideoRecorder not syncing properly with W&B
- Settled on direct `wandb.Video()` calls bypassing TorchRL's logger

### Domain Randomization
Added configurable randomization (disabled by default, tight ranges for stable training):
- Mass: +/-5%, Thrust: +/-5%, Gravity: 9.7-9.9 m/s^2
- Initial height: +/-4m, velocity: +/-0.5 m/s, orientation: +/-2deg

---

## Phase 3: Curriculum Learning

### Motivation
Training from 50m is hard — the rocket falls for a long time before any meaningful reward signal. Starting from lower heights gives denser rewards and faster initial learning.

### Implementation
`CurriculumScheduler` in the training loop:
- Height stages: 5m -> 10m -> 20m -> 35m -> 50m
- Advancement: 70% success rate over a rolling window of 100 episodes
- On advancement: save checkpoint, recreate environments at new height, keep replay buffer
- No fresh random exploration after stage 1 (policy already knows basics)

### Config
```yaml
curriculum:
  enabled: true
  heights: [5.0, 10.0, 20.0, 35.0, 50.0]
  success_threshold: 0.7
  window_size: 100
```

### First Training Run (1M frames, curriculum enabled)
- Ran for ~980k frames on CPU before interruption
- Stayed on 5m stage the entire time — didn't reach 70% success threshold
- The v0 cylinder design may be too hard to land precisely (no legs for stability)
- **Next step**: try with v2 design, tune reward weights, or lower success threshold

---

## Phase 4: Evaluation & Checkpointing

### Checkpoint Saving
Added automatic checkpoint saving during training:
- Saves `checkpoints/eval_{frames}.pt` at every evaluation interval
- Saves `checkpoints/final.pt` at end of training
- Each checkpoint contains: model weights, frame count, full Hydra config

### eval.py
Standalone evaluation script that loads any checkpoint and renders high-quality videos:
```bash
python sac/eval.py --checkpoint checkpoints/final.pt --resolution 720 --episodes 5
```
- 720x720 default resolution (vs 256x256 during training)
- H.264 codec at 40fps matching env physics rate
- Reads config from checkpoint (rocket design, network architecture)
- Prints per-episode metrics (outcome, reward, final error, velocity)

---

## Phase 5: GPU-Accelerated Parallel Simulation

### Motivation
CPU training is slow: ~120 steps/sec with a single environment. At 1M frames, that's ~2 hours just for environment stepping. We need thousands of parallel environments on GPU.

### MuJoCo Warp Integration
Integrated [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) (beta) — a GPU-optimized MuJoCo implementation by Google DeepMind + NVIDIA, built on NVIDIA Warp.

**`env/rocket_landing_warp.py`** — TorchRL `EnvBase` wrapping MuJoCo Warp:
- Steps N parallel worlds on GPU via `nworld` parameter
- Zero-copy data transfer between Warp and PyTorch via `wp.to_torch()`/`wp.from_torch()`
- Observation, reward, and termination computed in batched PyTorch ops
- CUDA graph capture for maximum throughput
- Auto-reset of terminated environments

**Benchmark (RTX 4060 Laptop GPU, 4096 parallel envs):**

| Setup | Throughput | Speedup |
|-------|-----------|---------|
| CPU, 1 env | ~120 steps/sec | 1x |
| **GPU, 4096 envs** | **1,176,178 steps/sec** | **~10,000x** |

### Multi-Rocket Visualization
Created `env/multi_rocket_viz.py` for cinematic multi-rocket renders:
- Programmatically generates a MuJoCo XML with N rockets on a grid
- Each rocket has its own landing pad, actuators, and free joint
- Runs a trained policy on all rockets simultaneously using CPU MuJoCo
- Renders high-quality video from a cinematic camera angle

```bash
python env/multi_rocket_viz.py --checkpoint checkpoints/eval_980000.pt --num-rockets 100 --resolution 1080
```

![100 rockets descending](videos/multi_100_preview.png)

---

## Phase 6: Repository Reorganization

### Training Directory Structure
Reorganized the project from a flat `sac/` directory into dedicated training setups under `training/`:

```
training/
  single_rocket/    # CPU-based, single env (was sac/)
    train.py        # Renamed from sac.py
    eval.py, utils.py, config.yaml
    checkpoints/    # Moved from top-level checkpoints/
  multi_rocket/     # GPU-parallel, MuJoCo Warp (NEW)
    train.py, eval.py, utils.py, config.yaml
    checkpoints/
```

**Why:** Each training setup (CPU single-env vs GPU 4096-env) has different configs, utilities, and env wrappers. Keeping them self-contained makes it easy to version, test, and iterate on each independently. The shared `env/` directory remains unchanged — both setups import from it.

**Multi-rocket training setup:** The new `training/multi_rocket/` wires up `RocketLanderWarp` (the GPU batched env from Phase 5) to the SAC training loop. Key differences from single_rocket:
- No `GymWrapper` needed — `RocketLanderWarp` is already a TorchRL `EnvBase`
- No `ParallelEnv` — the Warp env handles batching internally
- Larger batch size (1024), higher UTD ratio (4.0) for GPU throughput
- No video logging (Warp env doesn't render pixels)
- Eval uses CPU `RocketLander` for video rendering of GPU-trained policies

---

## Phase 7: GPU-Native Training & PPO

### SAC GPU Replay Buffer Fix
The SAC multi_rocket training was transferring data CPU↔GPU every step: `replay_buffer.extend(tensordict.cpu())` moved collected data to CPU for `LazyMemmapStorage`, then `sampled.to(device)` moved it back to GPU for training. This dominated wall time.

**Fix:** Switched to `LazyTensorStorage` with `device="cuda"` — the replay buffer now lives entirely in VRAM. Reduced buffer size from 1M to 100K to fit in GPU memory (100K transitions × 13-dim obs is ~5MB, well within budget).

### PPO for Massive Parallelism
Added PPO training setup alongside SAC. PPO is on-policy and a better fit for high-throughput parallel envs — it uses all collected data immediately rather than storing it in a replay buffer.

**Architecture:**
- Actor: MLP(256,256) with Tanh + `AddStateIndependentNormalScale` (state-independent std)
- Critic: MLP(256,256) with Tanh → `ValueOperator` (state value, not Q-value)
- Orthogonal weight initialization (PPO standard)
- GAE(γ=0.99, λ=0.95) for advantage estimation

**Training loop:**
- Collect 2048 transitions per batch (8 steps × 256 envs)
- Compute GAE advantages on GPU (no CPU transfer)
- 4 PPO epochs per batch with mini-batch size 256
- `SamplerWithoutReplacement` for mini-batch shuffling
- Gradient clipping at 0.5

**Config:** `training/multi_rocket/config_ppo.yaml`, run with:
```bash
cd training/multi_rocket && python train_ppo.py
```

---

## TODO: PPO Hyperparameter Tuning (seed sensitivity)

5-seed sweep (seeds 1-5, 50M frames each) showed only 2/5 seeds converge to ~900 reward.
The other 3 plateau around 600 (gets partway down but doesn't land). Seed 42 (default) works reliably.

**Tune these to improve convergence consistency:**
- [ ] **Entropy coefficient**: Currently 0.01 — try 0.02-0.05 to prevent early policy collapse
- [ ] **LR annealing**: Currently linear decay to 0 — try cosine schedule or slower warmup
- [ ] **Clip epsilon annealing**: Currently disabled — try enabling to tighten updates late in training
- [ ] **GAE lambda**: Currently 0.95 — try 0.97 for longer-horizon credit assignment
- [ ] **Mini-batch size**: Currently 4096 — try larger (8192) for more stable gradients
- [ ] **PPO epochs**: Currently 4 — try 8-10 with smaller clip epsilon
- [ ] **Total frames**: 50M may not be enough for slow seeds — try 100M

W&B group: `ppo_v2_5seeds` (seed 5 and seed 2 solved, seeds 1/3/4 stuck at ~600)

---

## Future Improvements

- [x] **PPO for massive parallelism**: Added PPO training setup in `training/multi_rocket/` (Phase 7)
- [ ] **Curriculum on GPU**: Implement curriculum height changes in the Warp environment
- [ ] **Sim-to-real considerations**: Widen domain randomization after initial convergence
- [ ] **Thrust vectoring**: Add gimbal actuators for more realistic engine control
- [ ] **Multi-stage landing**: Boostback burn -> entry burn -> landing burn sequence
- [ ] **Wind disturbances**: Add random lateral forces during descent
