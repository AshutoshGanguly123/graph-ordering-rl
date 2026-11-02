# Launch notes (HPC)

Working directory for heavy artifacts (not in git):

`/l/users/ashutosh.ganguly/graph_ordering/compute_matched_control_20261102/`

```bash
# Example: continue teacher RL for seed 202611020 (4 GPUs, same as E007)
cd /l/users/ashutosh.ganguly/github_public/graph-ordering-rl
python experiments/compute_matched_control/prepare_configs.py --seed 202611020 --arm teacher_continued_rl
# Then point rl_train / runner at the generated config under the HPC workdir.
```

`prepare_configs.py` writes immutable JSON configs per (seed, arm). Training uses existing `src/rl_train.py` with `initial_checkpoint` set to the teacher for arm A, or student SFT checkpoint for arm C.
