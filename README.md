# graph-ordering-rl

Learned policies for **minimum-cost graph vertex ordering** via a teacher → SFT → RLOO pipeline on dense swap environments.

**Author:** Ashutosh Ganguly — AI researcher @ MBZUAI.

## Claim

A same-architecture student trained from an audited teacher (best-of-16 trajectory prefixes) then RLOO-refined can **match or slightly beat** the teacher on fresh graphs under fixed swap budgets — with transparent selection counts and paired win rates.

## Method

1. **Teacher data:** E006 teacher emits best-of-16 prefixes on fresh graphs.
2. **SFT:** new student learns selected actions from scratch (config-locked).
3. **RL:** exact-cost RLOO treatment (`rl_config.json`).
4. **Eval:** fresh 1,000-graph test, best-of-16 rollouts @ 1024 / 2048 swaps vs teacher and classical heuristics.

Code: `src/`. Locked configs: `config.json`, `rl_config.json`. Review pack: `review/`.

## Metrics (E007 / policy_teacher_v2)

From `review/` and `metrics/*_final_summary.json` (pipeline completed 22 Oct 2026):

| Policy | Cost @ 1024 swaps | Cost @ 2048 swaps |
|--------|------------------:|------------------:|
| **New SFT+RL** | **2286.495** | **2279.087** |
| E006 learned teacher | 2297.885 | 2288.940 |
| New SFT (pre-RL) | 2301.088 | 2292.434 |
| Spectral / BFS / DFS | ~2396 / 2452 / 2443 | (single ordering) |

Paired vs teacher @1024: **+0.50%** relative, wins **810/1000**. SFT updates 10k (selected 6500); RL updates 1k (selected 1000).

## Setup

```bash
# create a Python env with the experiment deps (torch, etc.)
pip install -r requirements.txt  # add pins for your cluster
python src/test_experiment.py
# Full pipeline is HPC-oriented; see HANDOFF.md / PRD.md for stage gates.
# python src/launch.py
```

`deployment.example.json` shows resource fields with local paths redacted.

## Limitations / non-claims

- Checkpoints, teacher trajectory shards, and full `runs/` are **not** published (~5GB+); summaries + review figures only.
- Gains vs teacher are modest; heuristics remain much worse — this is not a solved combinatorial optimum claim.
- Results depend on fixed seeds, swap budgets, and the audited 12k-graph teacher manifest.
- Launcher assumes reservation/GPU occupancy checks on shared HPC — not a cloud AutoScale trainer.

## License

MIT. Document any third-party env code separately if vendored later.
