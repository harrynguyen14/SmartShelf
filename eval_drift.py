"""Feature-drift evaluation: how much does masking change the encoder output?

Cách #1 (rẻ nhất, không cần label). Loads moonshotai/MoonViT-SO-400M once,
runs the SAME images twice — clean vs masked — and reports how far the output
features moved. MoonViT is an encoder (no class head), so "accuracy after
pruning" is measured indirectly as feature preservation:

    cos_sim      mean cosine similarity per token (1.0 = identical)
    rel_l2       ||f_masked - f_clean|| / ||f_clean||   (0.0 = identical)

Rule of thumb: cos_sim > 0.99 very safe; < 0.95 starting to break.

Masking (from mask.json, written by prune.py):
    heads   -> zero the head's slice of the input to `wo`
    neurons -> zero the neuron's column in the input to `mlp.fc1`
    layers  -> skip the block entirely (forward returns its input)

Reuses model/data loaders from pruning.py.

Usage
-----
    python eval_drift.py --mask mask.json --max-images 64 --max-side 1024
"""

from __future__ import annotations

import argparse
import json

import torch

from pruning import (
    DEFAULT_HF_DATASET,
    load_real_model,
    load_samples_hf,
    load_samples_glob,
)


class Masker:
    """Applies a mask plan via hooks; remove() restores the clean model."""

    def __init__(self, model, plan):
        self.model = model
        self.handles = []
        cfg = model.config
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads

        # group plan entries by block
        self.heads_by_block = {}
        for b, h in plan["heads"]:
            self.heads_by_block.setdefault(b, []).append(h)
        self.neurons_by_block = {}
        for b, n in plan["neurons"]:
            self.neurons_by_block.setdefault(b, []).append(n)
        self.dropped_layers = set(plan["layers"])

    def apply(self):
        for i, block in enumerate(self.model.encoder.blocks):
            if i in self.dropped_layers:
                # skip the whole block: forward returns its first input unchanged
                self.handles.append(
                    block.register_forward_hook(lambda _m, inp, _o: inp[0])
                )
                continue
            heads = self.heads_by_block.get(i)
            if heads:
                self.handles.append(
                    block.wo.register_forward_pre_hook(self._zero_heads(heads))
                )
            neurons = self.neurons_by_block.get(i)
            if neurons:
                self.handles.append(
                    block.mlp.fc1.register_forward_pre_hook(self._zero_neurons(neurons))
                )

    def _zero_heads(self, heads):
        hd = self.head_dim

        def hook(_m, inp):
            x = inp[0].clone()  # (L, hidden) = concat of heads
            for h in heads:
                x[:, h * hd:(h + 1) * hd] = 0
            return (x,)
        return hook

    def _zero_neurons(self, neurons):
        idx = torch.tensor(neurons)

        def hook(_m, inp):
            x = inp[0].clone()  # (L, intermediate)
            x[:, idx] = 0
            return (x,)
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


@torch.no_grad()
def encode(model, pv, ghw):
    """Return one flat (L_total, D) feature tensor for an image."""
    out = model(pv, ghw)
    return torch.cat([t.reshape(-1, t.shape[-1]) for t in out], dim=0)


@torch.no_grad()
def drift(model, samples, plan):
    masker = Masker(model, plan)
    cos_all, l2_all = [], []
    for i, (pv, ghw) in enumerate(samples):
        clean = encode(model, pv, ghw).float()
        masker.apply()
        masked = encode(model, pv, ghw).float()
        masker.remove()

        cos = torch.nn.functional.cosine_similarity(clean, masked, dim=-1).mean()
        rel = (masked - clean).norm() / clean.norm().clamp_min(1e-8)
        cos_all.append(cos.item())
        l2_all.append(rel.item())
        print(f"\rdrift {i + 1}/{len(samples)}", end="", flush=True)
    print()
    return sum(cos_all) / len(cos_all), sum(l2_all) / len(l2_all)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mask", required=True, help="mask plan JSON from prune.py")
    ap.add_argument("--dataset", default=DEFAULT_HF_DATASET)
    ap.add_argument("--split", default="train")
    ap.add_argument("--images", nargs="+")
    ap.add_argument("--max-images", type=int, default=64)
    ap.add_argument("--max-side", type=int, default=1024)
    args = ap.parse_args()

    plan = json.load(open(args.mask))
    print(f"masking {len(plan['heads'])} heads, {len(plan['neurons'])} neurons, "
          f"{len(plan['layers'])} layers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_real_model(device).eval()
    if args.images:
        samples = load_samples_glob(args.images, device, args.max_images, args.max_side)
    else:
        samples = load_samples_hf(args.dataset, args.split, device, args.max_images, args.max_side)

    cos, rel = drift(model, samples, plan)
    print(f"\nmean cos_sim = {cos:.4f}   (1.0 = identical)")
    print(f"mean rel_l2  = {rel:.4f}   (0.0 = identical)")
    verdict = "very safe" if cos > 0.99 else "ok" if cos > 0.95 else "BREAKING"
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
