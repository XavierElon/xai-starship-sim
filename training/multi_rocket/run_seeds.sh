#!/bin/bash
# Run PPO training with 5 seeds for convergence plot
# Usage: bash run_seeds.sh

set -e

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export MUJOCO_GL=egl

GROUP="ppo_v2_5seeds"

for SEED in 1 2 3 4 5; do
    echo "=========================================="
    echo "Starting training with seed=$SEED"
    echo "=========================================="
    python train_ppo.py env.seed=$SEED logger.group_name=$GROUP logger.exp_name=ppo_seed_${SEED}
    echo "Seed $SEED complete."
    echo ""
done

echo "All 5 seeds complete!"
