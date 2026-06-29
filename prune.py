"""Select prune targets from a scores JSON and write a mask plan.

Reads the per-unit scores produced by `pruning.py --out scores.json`, picks
the lowest-scoring fraction of each granularity (heads / neurons / layers),
and writes a mask plan JSON that `eval_drift.py` applies by zero-out.

Cut = MASK (zero the unit's contribution via hooks), NOT architecture
surgery. Equivalent to removal for measuring feature drift; doesn't shrink
on-disk params.

Selection: pick units with the LOWEST score (least important). For neurons,
the dead-neuron tie (many identical low scores) is selected first — those are
the safest cuts.

Usage
-----
    python prune.py --scores scores.json --out mask.json \
        --head-frac 0.05 --neuron-frac 0.20 --layer-frac 0.0

mask.json:
    { "meta": {n_heads, intermediate_size, n_layers},
      "heads":   [[block, head], ...],
      "neurons": [[block, neuron], ...],
      "layers":  [block, ...] }
"""

from __future__ import annotations

import argparse
import json


def lowest_k(scores, frac):
    """Indices of the lowest `frac` fraction of a flat score list."""
    n = len(scores)
    k = int(round(n * frac))
    if k <= 0:
        return []
    order = sorted(range(n), key=lambda i: scores[i])
    return order[:k]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", required=True, help="scores JSON from pruning.py --out")
    ap.add_argument("--out", required=True, help="mask plan JSON to write")
    ap.add_argument("--head-frac", type=float, default=0.05)
    ap.add_argument("--neuron-frac", type=float, default=0.20)
    ap.add_argument("--layer-frac", type=float, default=0.0,
                    help="fraction of layers to drop (RISKY — keep small/0)")
    args = ap.parse_args()

    data = json.load(open(args.scores))
    meta = data["meta"]
    nh = meta["n_heads"]
    inter = meta["intermediate_size"]
    sc = data["scores"]

    plan = {"meta": meta, "heads": [], "neurons": [], "layers": []}

    for i in lowest_k(sc["heads"], args.head_frac):
        plan["heads"].append([i // nh, i % nh])
    for i in lowest_k(sc["neurons"], args.neuron_frac):
        plan["neurons"].append([i // inter, i % inter])
    plan["layers"] = lowest_k(sc["layers"], args.layer_frac)

    with open(args.out, "w") as f:
        json.dump(plan, f)

    print(f"plan: {len(plan['heads'])} heads, {len(plan['neurons'])} neurons, "
          f"{len(plan['layers'])} layers -> {args.out}")


# --- self-check -------------------------------------------------------------
def _demo():
    assert lowest_k([3.0, 1.0, 2.0, 0.0], 0.5) == [3, 1]  # two lowest, in order
    assert lowest_k([1, 2, 3], 0.0) == []
    assert lowest_k([5, 5, 5], 1.0) == [0, 1, 2]
    print("prune demo OK")


if __name__ == "__main__":
    import sys

    if "--demo" in sys.argv:
        _demo()
    else:
        main()
