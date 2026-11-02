#!/usr/bin/env python3
"""Greedy random-swap hill-climb under a wall-clock budget (CPU reference)."""
from __future__ import annotations
import argparse, json, time, random
from pathlib import Path

def hill_climb(n: int, seconds: float, seed: int) -> dict:
    rng = random.Random(seed)
    order = list(range(n))
    # Placeholder cost: inversion-like proxy so the script is runnable without GPU graphs.
    def cost(o):
        return sum(1 for i in range(n) for j in range(i+1, n) if o[i] > o[j])
    best = cost(order)
    t0 = time.time(); steps = 0
    while time.time() - t0 < seconds:
        i, j = rng.randrange(n), rng.randrange(n)
        order[i], order[j] = order[j], order[i]
        c = cost(order)
        if c <= best:
            best = c
        else:
            order[i], order[j] = order[j], order[i]
        steps += 1
    return {"n": n, "seconds": seconds, "steps": steps, "best_proxy_cost": best, "seed": seed}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = hill_climb(a.n, a.seconds, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(result)

if __name__ == "__main__":
    main()
