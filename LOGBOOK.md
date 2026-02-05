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

## PPO Hyperparameter Tuning (seed sensitivity)

5-seed sweep (seeds 1-5, 50M frames each) showed only 2/5 seeds converge to ~900 reward.
The other 3 plateau around 600 (gets partway down but doesn't land). Seed 42 (default) works reliably.

**Tune these to improve convergence consistency:**
- [x] **Entropy coefficient**: Currently 0.01 — try 0.02-0.05 to prevent early policy collapse
- [ ] **LR annealing**: Currently linear decay to 0 — try cosine schedule or slower warmup
- [x] **Clip epsilon annealing**: Currently disabled — try enabling to tighten updates late in training
- [x] **GAE lambda**: Currently 0.95 — try 0.97 for longer-horizon credit assignment
- [x] **Mini-batch size**: Currently 4096 — try larger (8192) for more stable gradients
- [x] **PPO epochs**: Currently 4 — try 8-10 with smaller clip epsilon
- [ ] **Total frames**: 50M may not be enough for slow seeds — try 100M

### Sweep v1 Results (2026-02-05)

Ran 8 configs × 3 seeds = 24 runs. W&B project: `SpaceX-Landing`, groups: `sweep_*`

| Config | Mean | Std | Min | Max | Notes |
|--------|------|-----|-----|-----|-------|
| **entropy_0.02_gae_0.97** | **796.0** | 137.7 | 603.2 | 916.0 | Best mean |
| gae_0.97 | 795.9 | 152.3 | 580.9 | 912.3 | |
| entropy_0.05 | 792.1 | 142.4 | 590.7 | 895.2 | |
| entropy_0.02 | 765.2 | 201.4 | 480.8 | 918.5 | High variance |
| **clip_anneal** | 746.4 | **110.5** | **641.9** | 899.3 | **Most consistent** |
| minibatch_8192 | 723.9 | 126.3 | 582.2 | 888.9 | |
| baseline | 690.8 | 154.0 | 544.8 | 903.8 | |
| epochs_8 | 590.4 | 249.6 | 289.4 | 900.6 | **Avoid** - hurts training |

**Key findings:**
1. `clip_anneal` has best consistency (lowest std=110.5, highest min=641.9)
2. `entropy_0.02_gae_0.97` has best mean (796.0)
3. `epochs_8` makes things worse — too many PPO epochs hurts convergence
4. GAE 0.97 and entropy changes both help independently

### Sweep v2: Combining clip_anneal with best performers (2026-02-05)

Testing whether clip_anneal's consistency combines with entropy/GAE's performance gains.

```bash
cd training/multi_rocket && uv run python sweep_ppo.py --v2
```

| Config | Seeds | Mean | Std | Min | Max | Notes |
|--------|-------|------|-----|-----|-----|-------|
| **clip_anneal_gae_0.97** | 5 | **716.0** | 155.3 | 568.0 | 910.7 | **Best config** |
| clip_anneal_entropy_0.05_gae_0.97 | 3 | 765.1 | 108.4 | 637.2 | 902.1 | Good balance |
| clip_anneal_entropy_0.02_gae_0.97 | 3 | 729.5 | 129.8 | 635.7 | 913.0 | |
| clip_anneal_entropy_0.02 | 3 | 636.9 | **14.5** | 616.9 | 651.1 | Stuck - too consistent! |

**5-seed validation of `clip_anneal_gae_0.97`:**
- 3-seed mean was 804.1, 5-seed mean is 716.0 (seeds 3&4 underperformed)
- Still beats baseline (690.8 mean)
- Convergence rate: ~3/5 seeds reach >800 vs baseline's ~2/5
- Seed sensitivity not fully solved, but improved

**Key findings:**
1. **Best config: `clip_anneal_gae_0.97`** — Improves over baseline but seed sensitivity remains
2. `clip_anneal_entropy_0.02` gets stuck at ~637 — entropy suppression + clip annealing prevents late exploration
3. GAE 0.97 is the key ingredient — helps across all combinations
4. Adding entropy to clip_anneal hurts more than helps

### Recommended Config

### Reset Noise Experiment (2026-02-05)

**Problem:** Seed sensitivity remained even with tuned hyperparameters. Root cause: `reset_noise=0.01` meant all episodes started in nearly identical states (centered, upright, stationary). Policy overfits to narrow initial conditions.

**Solution:** Increased reset noise significantly:
| Parameter | Old | New |
|-----------|-----|-----|
| xy position | ±0.01m | ±3.0m |
| velocity | ±0.01 m/s | ±3.0 m/s |
| angular | ±0.01 rad | ±0.15 rad (~8°) |
| angular vel | ±0.01 rad/s | ±0.3 rad/s |

**Results (W&B group: `reset_noise_v1`):**
| Config | Seeds | Mean | Std | Min | Max |
|--------|-------|------|-----|-----|-----|
| **With reset noise** | 3 | **832.4** | **17.6** | **807.7** | 847.6 |
| Without (clip_anneal_gae_0.97) | 5 | 716.0 | 155.3 | 568.0 | 910.7 |
| Baseline | 3 | 690.8 | 154.0 | 544.8 | 903.8 |

All 3 seeds: [841.9, 847.6, 807.7] — **seed sensitivity solved!**

**Key insight:** Forcing the policy to handle varied initial conditions improves both generalization AND consistency. The std dropped from 155 → 18 (9x improvement).

### Final Recommended Config

Updated in `config_ppo.yaml`:
```yaml
loss:
  anneal_clip_epsilon: true
  gae_lambda: 0.97

env:
  reset_noise:
    pos: 3.0
    vel: 3.0
    ang: 0.15
    angvel: 0.3
```

### Sweep Script

Run hyperparameter sweep with 3 seeds per config:
```bash
cd training/multi_rocket && uv run python sweep_ppo.py      # v1 (original 8 configs)
cd training/multi_rocket && uv run python sweep_ppo.py --v2  # v2 (clip_anneal combos)
```

**Options:**
- `--dry-run`: Print commands without running
- `--list`: Show all sweep configs
- `--config <name>`: Run only one config (e.g., `--config entropy_0.02`)
- `--seed <n>`: Run only one seed
- `--v2`: Run follow-up sweep (clip_anneal combinations)

---

## Future Improvements

- [x] **PPO for massive parallelism**: Added PPO training setup in `training/multi_rocket/` (Phase 7)
- [x] **Hyperparameter tuning**: `clip_anneal_gae_0.97` + reset noise (mean=832, std=18)
- [x] **Seed sensitivity solved**: Reset noise forces generalization (std: 155 → 18)
- [ ] **Curriculum on GPU**: Implement curriculum height changes in the Warp environment
- [ ] **Sim-to-real considerations**: Mass/thrust randomization after initial convergence
- [ ] **Thrust vectoring**: Add gimbal actuators for more realistic engine control
- [ ] **Multi-stage landing**: Boostback burn -> entry burn -> landing burn sequence
- [ ] **Wind disturbances**: Add random lateral forces during descent

---

## TODO (2026-02-06)

- [ ] **Run training with tightened success condition**: Success zone changed from `target_height + 0.5` to `target_height + 0.15`. This should make the rocket actually land instead of hovering 0.5m above ground.
  ```bash
  cd training/multi_rocket && uv run python train_ppo.py env.seed=0 logger.group_name=demo_model_tight logger.exp_name=demo_tight_seed0
  ```
- [ ] **Update LOGBOOK with training speed comparison**: Document PPO multi-env training speed (~75k it/s, 11 min for 50M frames)
