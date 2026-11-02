# E007 continuation

Status: COMPLETED, reviewed 22 October 2026. SFT 10,000 updates, selected 6,500;
RL 1,000 updates, selected 1,000. Full pipeline complete, final fresh test done.
No training is pending. Completion artifacts and figures: review_2026_10_22/.
User explicitly authorized
full teacher-data -> SFT 10,000 updates -> RL 1,000 × 1,024 sequence.
Read the PRD and both configs. No agents, scheduler or background monitor.

Directory: experiment__policy_teacher_v2__2026_10_21__1.
Remote: ${REPO_ROOT}.
Runtime: <local-venv>/bin/python.
Named tmux: <tmux-session>.

Teacher is E006 best_model.pt, RL update 950, SHA256
8c4ee8085621251a2a7ce04434d329beacc584de34bfe487e542724bd3a7d382.
New student initializes from scratch; the RL phase initializes from selected SFT-v2.
New seed ranges 409M–414M; prior test pools never used for training.
Launched on <gpu-node>, cards 0–3 (RTX 5000 Ada Generation, 32 GB each), after checking
zero GPU reservations, zero physical memory/utilization and no compute processes.
CPU objective/compiler/recovery/SFT/RL cycle tests passed locally and remotely.
Four-rank CUDA teacher/SFT/RL smoke tests passed. All 100 full-size demonstration
audits passed before completing the remaining teacher data. Full SFT began after
the complete data manifest was committed, and RL follows without further prompts.

Read `artifacts/pipeline.json` for the active stage and the corresponding stage log.
Use `tmux capture-pane -pt <tmux-session>` only when checking a launch
or when the user asks. Resume a stopped pipeline on an eligible compute node using
`python3 launch.py --resume`; source/config hashes must match deployment.json.
Never launch a duplicate or modify scientific files after deployment.

Launch evidence: deployment.json plus artifacts/launch_evidence.tar.gz (a launch-time
snapshot, not live status), data/sft/audit.json, and the phase exit JSONs.
Observed 1,777 MiB and 100% GPU utilization on each of cards 0–3 during the full
teacher audit. Further progress is checked only when the user asks. Initial ETA
for the whole cycle is approximately 6–8 hours, based on prior SFT/RL runs and
early generation throughput; not a completion guarantee.

## Manual status review, 21 October 2026

The launch-time paragraphs above describe launch evidence, not current progress.
Full teacher-data stage exited 0 in 3,922.43 seconds (~65 minutes), with 10k/1k/1k
demonstrations complete. Main SFT resumed the ten-update gate and reached 4,500;
the last committed checkpoint at inspection was 4,000, with validation at 4,500
still in progress. All four ranks active on <gpu-node> cards 0,2,3,4; cards were
reselected between stages. Other processes now share physical cards 0 and 2;
do not stop or modify anyone else's work. No pipeline_error.json was present.

Latest complete 1,000-graph validation, step 4,000, best of 16:
- CE: 8.354604 at initialization -> 6.058136.
- 256 swaps: student 2340.884 vs frozen E006 teacher 2333.954.
- 512 swaps: student 2312.326 vs frozen E006 teacher 2305.900 (~0.279% higher).
This is validation evidence before RL, not final-test superiority. Teacher here
means the learned E006 policy, not spectral bisection. SFT batches ~0.3–0.5 s;
validation adds substantial wall time. Rough remaining estimate 4–5 h (SFT plus
RL and evaluation), sensitive to shared GPU load. Check only on user request.
Evidence snapshot: artifacts/status_review_20261021/ (copied stage JSONs,
progress, status, two training metrics, and validation summaries through 4,000).

## SFT completion and RL resource recovery

The later SFT final summary confirms 10,000 updates and selected step 6,500.
Fresh SFT test: 1,000 graphs, best of 16, recursive cost at 256 swaps 2344.757
(old E006 policy at the same budget 2343.038); at 512 swaps 2318.790 versus
2315.449, approximately 0.1443% higher mean cost. Test CE 5.971275. The separate
teacher_calls field (2300.430) is the demonstration outcome at 1,024 swaps and
must not be described as the teacher's matched 256/512-swap cost.

After RL data generation, resource inspection failed with "Four physically
empty, unreserved >=30GB cards required"; no RL updates had started. The prior
pipeline.json still described completed rl_data, so that file alone was not
evidence of an active run. No old experiment runner/tmux was active on any of the
four nodes when checked. Four empty, unreserved cards 0–3 were found on <gpu-node>;
the standard guard and immutable source/config check passed on resume. Archived
the old pipeline_error.json as pipeline_error_before_rl_resume_<timestamp>.json
instead of deleting the failure evidence. Same named tmux; completed stages skip.
Selected SFT-v2 checkpoint 6,500 initializes RL, not the old teacher or step 9,500.
No scientific source/config edits, new scheduler, agent or monitor.

Resume verification: rl_profile completed exit 0 in 113.32 s including baseline
evaluation and startup. Checkpoint 10 SHA256
cb28042ba8cb123d83382fc616b0c485361604b8a2960c0f0b23ec7d9da10795.
Main rl_train resumed from it and emitted update 11: 8.17 s, 131,072 actions,
finite gradients/loss, log-probability replay absolute error 5.04e-8, peak allocated
5.47 GB/GPU. During the gate, all four cards showed 100% utilization and 8,691 MiB
physical memory. Initial fresh validation best-of-four at 1,024 is 2308.667969;
post-training validation is first at step 50, so no RL improvement claim yet.
Rough remaining estimate 2.5–3 h including validation/final evaluation, contingent
on resources. Check progress only when the user asks. Local evidence snapshot:
artifacts/status_review_after_sft_20261021/; earlier review remains unchanged.

## Completion review — 22 October 2026

Pipeline complete and rl_train exit 0; selected RL update 1,000 by the 256-graph
validation best-of-four cost at 1,024 swaps: 2308.667969 -> 2294.652344.
All 1,000 metric steps, 21 validation summaries, 12,000 learned-policy test records
(three policies × 1,000 graphs × four budgets) and 1,000 heuristic rows audited.
Source/config hashes match deployed identity; RL initialized selected new SFT
6,500; old E006 teacher checkpoint hash remains unchanged. Finite losses and
gradients throughout; maximum mean absolute logp replay error 6.74e-7.

Fresh test (seed 414M), same graphs/starts, best of 16:
- 1,024 swaps: new SFT+RL 2286.495; frozen SFT-v2 2301.088; old learned E006
  teacher 2297.885. Mean paired improvement vs teacher 0.4959%, graph-level
  approximate 95% interval 0.4631–0.5288%, wins 810/1,000.
- 2,048 swaps: new 2279.087; frozen 2292.434; old teacher 2288.940. Paired
  improvement vs teacher 0.4305%, wins 811/1,000.
- Spectral bisection 2396.249, BFS 2452.158, DFS 2442.682 (one ordering each).
  At 1,024 swaps new policy beats BFS and DFS on all 1,000 graphs; spectral on 879.
- RL training including periodic validation 8668.45 s (~2.41 h); final three
  model evaluations 1884.68 s (~31.4 min). Median update 7.844 s, peak allocated
  5.47 GB/GPU. Selected RL model SHA256:
  70d5381f7749f6585122c620622f8cfb76a266df7465070d7b371c56274d49e6.

Interpretation: this second teacher-student-RL cycle improved held-out ordering
quality modestly. One training seed; graph-level intervals do not measure training
seed variance. No continued-old-teacher-RL control, so gains cannot be attributed
specifically to distillation rather than additional RL. No actual BiGG decoder
latency/downstream-generation claim. No further runs or monitors started.
