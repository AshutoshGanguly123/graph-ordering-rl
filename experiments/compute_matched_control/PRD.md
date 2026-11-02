# PRD — Compute-matched control: continued teacher RL vs fresh SFT→RL

## Decisive question

Does training a **fresh student** (SFT→dense RLOO) improve test cost more than spending the **same RL update budget** continuing RLOO on the **existing teacher**, under identical evaluation?

## Locked protocol

| Item | Value |
|------|-------|
| Graph family | Same as E007 (`side=64`, train/val/test seeds from configs) |
| RL algorithm | Dense RLOO, group=4, horizon=1024, LR=3e-5, bf16 (from `rl_config.json`) |
| Matched budget | **1000 RL updates** per seed (same as E007 RL phase) |
| Seeds | **3** independent training seeds: `202611020`, `202611021`, `202611022` |
| Arms | (A) continued teacher RL from E006/E007 teacher ckpt; (B) SFT-only freeze; (C) SFT→RL student (reference E007 + reseed); (D) local-search swap hill-climb under matched wall-clock of one RL eval |
| Eval | Fresh 1000 test graphs; best-of-16 @ 256/512/1024/2048; report mean cost + paired wins + seed mean±std |
| Primary metric | Mean cost @ 1024 swaps; secondary wall-clock to reach that eval |
| Non-goals this ITER | Full distribution-shift suite (document as follow-on) |

## Deliverables

1. Per-seed `results/seed_*/summary.json`
2. Aggregated `results/aggregate.json` + quality-vs-wallclock plot data CSV
3. README update only after aggregates exist (no speculative numbers)

## Status

Harness + configs committed. Full GPU training launches from HPC experiment root (see `launch_notes.md`). Until training completes, public README cites E007 audited numbers and links this PRD as the ownership experiment in flight.
