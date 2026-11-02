# graph-ordering-rl

Learned policies for **minimum-cost graph vertex ordering** via a locked teacher → SFT → dense RLOO pipeline.

**Author:** Ashutosh Ganguly — AI researcher @ MBZUAI.

## Claim

On a fresh 1,000-graph test set (best-of-16 rollouts), the SFT+RL student **beats classical orderings by a wide margin** and **wins the majority of paired comparisons** against a strong learned teacher—while the incremental gain over that teacher is intentionally the hard control.

## Headline results (E007, @ 1,024 swaps)

Lower mean full symmetric recursive cost is better. Learned policies: best-of-16. Classical: single ordering.

| Policy | Mean cost | vs policy | Paired wins |
|--------|----------:|----------:|------------:|
| **SFT + RL (ours)** | **2286.5** | — | — |
| Spectral bisection | 2396.2 | **−4.6%** | **879 / 1000** |
| DFS | 2442.7 | **−6.4%** | **~993–1000 / 1000** |
| BFS | 2452.2 | **−6.8%** | **1000 / 1000** |
| Initial ordering | ~2707.6 | **−15.6%** | — |

### Hard control (same architecture, matched eval)

| Comparison @ 1024 | Relative gain | Wins |
|-------------------|--------------:|-----:|
| vs previous learned teacher (E006) | **+0.50%** (95% CI 0.46–0.53) | **810 / 1000** |
| vs frozen SFT (pre-RL) | **+0.63%** | **890 / 1000** |
| vs teacher @ 2048 swaps | **+0.43%** | **811 / 1000** |

**Read this correctly:** the ~0.5% teacher delta is *small and real* under a paired protocol. The **~5–7% vs spectral/BFS/DFS** and **~15% vs initial** are the practical impact numbers. Both belong in the story.

Sources: [`review/`](review/) (`comparison.csv`, `audited_results.json`, figures) and [`metrics/`](metrics/).

## Method

1. **Teacher data** — audited E006 teacher emits best-of-16 trajectory prefixes on fresh graphs.
2. **SFT** — new same-architecture student learns selected actions (10k updates; selected step 6,500).
3. **RL** — exact-cost dense RLOO (`rl_config.json`; 1k updates, all selected).
4. **Eval** — 1,000 held-out graphs, best-of-16 @ 256/512/1024/2048 swap budgets vs teacher, SFT, and classical heuristics.

Code: [`src/`](src/). Locked configs: `config.json`, `rl_config.json`. Design record: `PRD.md`, `HANDOFF.md`.

## Setup

```bash
pip install -r requirements.txt
python src/test_experiment.py
# Full HPC pipeline (reservations / GPU checks): see HANDOFF.md
# python src/launch.py
```

`deployment.example.json` has local paths redacted.

## Limitations (honest)

- **One training seed**; CIs condition on this model and test distribution.
- No equal-budget **continued-RL-on-teacher** control yet — so we do *not* claim student retrain beats “just keep RLing the teacher.”
- Not an equal wall-clock or decoder-speed bakeoff vs local search.
- Checkpoints / trajectory shards stay on HPC; this repo ships code, configs, audited summaries, and figures.

## RE3 roadmap (next experiments)

1. Matched-compute continued-RL on the teacher vs student retrain.
2. Multi-seed training; separate model variance from graph-sampling uncertainty.
3. Solution quality vs wall-clock including strong local-search baselines.
4. Publish a small eval checkpoint + one-command reproduce path.



## In-flight RE3 experiment (compute-matched control)

Decisive follow-up: **continued teacher RL vs fresh SFT→RL** at matched 1000-update budgets across **3 seeds**. Spec and harness: [`experiments/compute_matched_control/`](experiments/compute_matched_control/). Results will land in that folder’s `results/` when HPC runs complete — E007 numbers above remain the audited public evidence until then.

## Reference papers

See [`reference_research_papers/`](reference_research_papers/) (PPO, RLOO, combinatorial / graph RL).

## License

MIT.
