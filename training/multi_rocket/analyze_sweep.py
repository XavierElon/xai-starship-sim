#!/usr/bin/env python
"""Analyze sweep results from W&B."""
import wandb
import numpy as np
from collections import defaultdict

api = wandb.Api()

# Get all runs from the sweep
runs = api.runs('sebastian-dittert/SpaceX-Landing', filters={'group': {'$regex': '^sweep_'}})

# Collect results
results = defaultdict(list)

for run in runs:
    if run.state != 'finished':
        continue
    group = run.group
    # Get final eval reward from summary (more reliable)
    summary = run.summary
    if 'eval/reward' in summary:
        final_reward = summary['eval/reward']
        results[group].append(final_reward)

# Print summary
print('Config                        | Seeds | Mean    | Std     | Min     | Max')
print('-' * 75)
for group in sorted(results.keys()):
    rewards = results[group]
    if len(rewards) > 0:
        mean = np.mean(rewards)
        std = np.std(rewards)
        min_r = np.min(rewards)
        max_r = np.max(rewards)
        print(f'{group:30} | {len(rewards):5} | {mean:7.1f} | {std:7.1f} | {min_r:7.1f} | {max_r:7.1f}')
