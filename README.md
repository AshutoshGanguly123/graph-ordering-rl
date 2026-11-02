# graph-ordering-rl

Learned policies for **minimum-cost graph vertex ordering** via a locked teacher → SFT → dense RLOO pipeline.

**Author:** Ashutosh Ganguly — AI researcher @ MBZUAI.

## Results (E007, @ 1,024 swaps)

Lower mean full symmetric recursive cost is better.

| Policy | Mean cost |
|--------|----------:|
| **SFT + RL (student)** | **2286.5** |
| Learned teacher (E006) | 2297.9 |
| SFT only (pre-RL) | 2301.1 |
| Spectral bisection | 2396.2 |
| DFS | 2442.7 |
| BFS | 2452.2 |
| Initial ordering | ~2707.6 |

Learned policies use best-of-16 rollouts; BFS/DFS/spectral are single orderings (not an equal-compute bakeoff). Sources: [`review/`](review/), [`metrics/`](metrics/).

## Method

1. Teacher emits best-of-16 trajectory prefixes on fresh graphs.
2. Fresh student SFT on selected actions (10k updates; selected step 6,500).
3. Dense RLOO (`rl_config.json`; 1k updates).
4. Eval on 1,000 held-out graphs @ 256/512/1024/2048 swap budgets.

Code: [`src/`](src/). Configs: `config.json`, `rl_config.json`. Design notes: `PRD.md`.

## Setup

```bash
pip install -r requirements.txt
python src/test_experiment.py
```

## Limitations

- Single training seed so far.
- Classical heuristics are weak relative to a matched wall-clock local-search baseline (planned).
- Learned vs classical comparisons are not equal attempt budgets (best-of-16 vs one shot).
- Checkpoints stay on HPC; this repo ships code, configs, and audited summaries.

## Ongoing work

Compute-matched continued-teacher RL vs fresh SFT→RL (multi-seed) is in progress under [`experiments/compute_matched_control/`](experiments/compute_matched_control/).

Broader program: teaching Transformers executable programs over graphs, toward a study of **capabilities vs incapacities** of Transformers on graph-structured computation. Related reading (links only): [`reference_research_papers/`](reference_research_papers/).

## License

MIT.
