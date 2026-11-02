# E007 — Learned teacher, student v2, then matched RL fine-tuning

Date: 21 October 2026. Completion reviewed 22 October 2026: SFT completed 10,000
updates (selected 6,500), then RL completed 1,000 updates (selected 1,000).
The full pipeline and final fresh 1,000-graph evaluation completed successfully.
RL resumed on <gpu-node> cards 0–3 after an earlier free-GPU resource halt.
At 1,024 swaps, best of 16: new policy 2286.495 vs old learned teacher 2297.885;
mean paired improvement 0.4959%, wins 810/1,000. See review_2026_10_22/README.md.
CPU tests, all four-rank CUDA smoke stages and
100 full-size demonstration audits passed. User authorized building
and starting this complete experiment. Folder: experiment__policy_teacher_v2__2026_10_21__1.
The question is whether the improved SFT+RL policy can teach a fresh same-capacity
student that improves further after the same RL treatment. Improvement is not
assumed: distilling successful searches can amortize their behaviour, but may lose
quality or overfit. Preserve all earlier experiments unchanged.

## Frozen teacher and demonstrations

- E006 selected RL update 950, parent SFT update 9,500; freeze parameters and eval mode.
- Exact checkpoint SHA256: 8c4ee8085621251a2a7ce04434d329beacc584de34bfe487e542724bd3a7d382.
- Same 64 U + 64 V undirected sparse bipartite four-community distribution;
  expected degree 4, mixing uniform 0.1–0.3. One fixed random legal start per graph.
- Fresh 10,000 SFT training / 1,000 validation / 1,000 test graphs, seeds 409M,
  410M and 411M. These exclude all prior training/validation/test seed ranges.
- Teacher: 16 independent sampled rollouts × 1,024 swaps per graph, countdown zero.
  Retain the rollout with lowest best-encountered exact cost, ties by rollout index.
  Store its chronological prefix ending at the FIRST occurrence of that best cost.
  This is one demonstrated trajectory per graph, not randomized alternative paths.
  Preserve temporary regressions and all preceding actions; no action shuffling,
  shortcut swaps, greedy filtering, synthesized paths or STOP labels.
- Each sample includes original graph/start, every pre-action permutation, next
  action, exact initial and per-action costs, incumbent/initial fractions, selected
  final order, winning rollout index and ending step. Also save all sampled actions,
  costs/incumbents, RNG seed and frozen model hash as compact block-level evidence.
  Zero-action cases are retained and reported; they contribute no CE target.
- Generate on four GPUs in eight-graph/rank batches. Independent deterministic
  seeds per split/block make committed shards resumable without sampler drift.
- Audit the first 100 training examples: legal partition swaps, edge preservation,
  every selected-prefix cost, first-best endpoint, exact vectorized vs literal
  recursion, and agreement with the winning best-of-16 teacher outcome. A small
  teacher/SFT/RL CUDA smoke precedes the full-size data audit and generation.

## Student v2 SFT — matched to E004

- Initialize FROM SCRATCH, not teacher weights. Same unchanged 3-layer width-64
  GNN, pair head 128, 63,169 parameters, 4,032 legal swap actions. Countdown zero.
- Graph-uniform then within-trajectory-state-uniform sampling; cross-entropy on
  current graph -> next demonstrated swap, no other objective or input head.
- Four synchronized GPUs, global batch 512 (128/rank), Adam LR 3e-4, gradient-norm
  clipping 1, bf16. 10,000 updates; validation/checkpoint every 500, select by
  best-of-16 recursive cost at 256 swaps. Budgets 16/64/128/256/512 as in E004.
- Validation includes graph-uniform CE and autonomous search. Frozen E006 teacher
  reference uses zero countdown, matching its actual training/inference setup.
  Teacher trajectories are never training inputs beyond ordinary graph context.
- Validation/test teacher pseudo-labels are for held-out evaluation, not fitted.
  The selected SFT model is committed before any RL initialization.

## Student v2 RL — matched to E006

- Fresh 1,000 RL training / 256 validation / 1,000 test graphs, seeds 412M, 413M,
  414M, independent of SFT and all prior datasets.
- Initialize the validation-selected student-v2 SFT model with a fresh Adam.
  The pipeline binds its actual checkpoint step/hash AFTER SFT completes; both
  are written into the RL run's resolved config and identity and checked on resume.
- Unchanged dense RLOO: 1,000 updates × 1,024 swaps, four sampled rollouts/graph;
  eight graphs/rank on four GPUs, 131,072 action samples/update. Recompute stored
  states in 512-state/rank microbatches and apply one on-policy update per batch.
- LR 3e-5, gradient-norm clipping 1, bf16. No critic, GAE, PPO clipping, entropy/KL,
  STOP or auxiliary objective. Same features and permanently zero countdown.
- C_t = exact full symmetric row-tree cost, B_t = best cost encountered.
  r_t=(C_(t-1)-C_t)/C_0; add (C_T-B_T)/C_0 to final reward. Gamma 1. Per-step
  reward-to-go baseline is the mean return of the other three trajectories.
  Total reward remains (C_0-B_T)/C_0. Potential shaping alone does not guarantee
  improved long-range credit assignment; this repeats E006's tested formulation.
- Validate every 50; select best-of-four cost at 1,024, including unmodified SFT
  initialization as a valid candidate. Same checkpoint and exact RNG recovery.
- Final fresh test: selected SFT+RL v2, its frozen SFT-v2 initialization AND the
  old E006 frozen teacher on IDENTICAL graphs/starts, best of 16 at 256/512/1024/2048.
  Also report best-of-four and first rollout. No selection on final test.
  Spectral bisection, BFS and DFS are one-ordering structural references.

## Metrics and comparisons

1. Autonomous exact recursive cost versus budget: held-out ordering quality.
2. Paired gap and wins against frozen learned teacher, with graph-level interval:
   whether this second cycle improves on the previous policy, not only on its
   own SFT initialization. Intervals are conditional on one model seed.
3. SFT validation cross-entropy, then RL validation improvement versus frozen
   student: distinguish imitation fit from useful autonomous optimization.

Log model/data/source identity, timings, memory, entropy, gradients, logp replay
error and zero-action/advantage rates. No claim about actual graph-generation
latency, temporal forecasting, conditioning, generalization to larger graphs or
causal mechanisms. This experiment remains the static ordering task.

## Finite pipeline, resources and handoff

CPU objective/compiler/recovery tests -> tiny GPU teacher data + SFT smoke + RL
smoke -> first 100 full-size demonstrations audited -> remaining teacher data ->
first ten real SFT updates/checkpoint -> SFT resume through 10,000 -> fresh RL data
-> first ten real RL updates/checkpoint -> RL resume through 1,000 -> final tests.
Real ten-update gates count toward their respective full budgets.

Estimated wall time on four 32 GB Ada cards: teacher scoring roughly one hour,
SFT roughly two hours, RL about 2.5–3 hours, plus setup/evaluation overhead.
These are estimates based on earlier experiments, to be replaced by measured
stage times. No four-card guarantee: inspect eligibility immediately before use.

Only unreserved, physically empty (0 MiB, 0%, no compute process) cards on IDLE or
MIXED gpu-xx nodes. Direct named tmux, no Slurm launch/reservation, agents or
background monitor. Node preference is not an allocation; never stop other jobs.

Remote root: ${REPO_ROOT}.
Runtime: <local-venv>/bin/python.
All code/data/results on /l; source/config locked after deployment. Phase JSON,
atomic shard commits, rank RNG snapshots and checkpoint pointers support restart.
Prune old optimizer checkpoints where documented, retain best models and provenance.
Credentials/helper/runbook never uploaded. Monitor only when user asks, apart
from checks necessary to complete the authorized launch. If resources are
unavailable, preserve the complete runnable experiment and report the exact blocker.
