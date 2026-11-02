#!/usr/bin/env python3
"""Emit immutable per-seed configs for compute-matched arms."""
from __future__ import annotations
import argparse, json, copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_RL = json.loads((ROOT / "rl_config.json").read_text())
SEEDS = [202611020, 202611021, 202611022]
ARMS = ("teacher_continued_rl", "sft_only_eval", "student_sft_rl", "local_search_clock")
OUT = Path("/l/users/ashutosh.ganguly/graph_ordering/compute_matched_control_20261102")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True, choices=SEEDS)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    cfg = copy.deepcopy(BASE_RL)
    cfg["experiment"] = f"E008-compute-matched-{a.arm}-seed{a.seed}"
    cfg["seed"] = a.seed
    cfg["train_seed"] = 412000000 + (a.seed % 1000)
    cfg["validation_seed"] = 413000000 + (a.seed % 1000)
    cfg["test_seed"] = 414000000 + (a.seed % 1000)
    cfg["updates"] = 1000
    cfg["arm"] = a.arm
    cfg["matched_budget"] = {"rl_updates": 1000, "horizon": 1024, "group": 4}
    if a.arm == "teacher_continued_rl":
        cfg["initial_checkpoint"] = cfg.get("teacher_checkpoint")
        cfg["note"] = "Continue RL from teacher weights; no fresh SFT."
    elif a.arm == "student_sft_rl":
        cfg["note"] = "Fresh student: run SFT then RL with this seed; paths filled by operator after SFT."
    elif a.arm == "sft_only_eval":
        cfg["updates"] = 0
        cfg["note"] = "Eval-only arm after SFT; no RL updates."
    else:
        cfg["note"] = "Local-search baseline; see local_search_baseline.py"
    dest = a.out / f"seed_{a.seed}" / a.arm
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "rl_config.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(path)

if __name__ == "__main__":
    main()
