"""Importance scoring for pruning the REAL moonshotai/MoonViT-SO-400M on SKU-110k.

Designed to run as a Kaggle cell / script with GPU + Internet. Loads the
actual checkpoint via transformers `trust_remote_code`, then scores every
prunable unit:

    Score = alpha * Activation + beta * Fisher + gamma * Diversity

computed independently for three granularities — attention heads, FFN (MLP)
neurons, whole encoder blocks — each z-score normalized WITHIN its own group
before the weighted sum. (DomainFrequency dropped: SKU-110k is a single
domain, so cross-domain frequency is undefined.)

Output: ranked tables (lowest score = best prune candidate). Nothing is
removed — scoring/ranking only.

Real module layout (from modeling_moonvit.py @ rev a889d399):
    patch_embed.proj            Conv2d 3 -> 1152, k=14 s=14
    encoder.blocks.{i}.wqkv     Linear 1152 -> 3456   (packed Q,K,V)
    encoder.blocks.{i}.wo       Linear 1152 -> 1152    (attn output proj)
    encoder.blocks.{i}.mlp.fc0  Linear 1152 -> 4304
    encoder.blocks.{i}.mlp.fc1  Linear 4304 -> 1152
    forward(pixel_values, grid_hws) -> list[Tensor] (one per image, merged)

Hooks:
    heads   -> pre-hook on `wo`      (input = concat of 16 heads x 72 dims)
    neurons -> pre-hook on `mlp.fc1` (input = FFN hidden, 4304, after act)
    layers  -> forward-hook on block (output residual stream)

Usage (Kaggle)
--------------
    !python pruning.py --images /kaggle/input/sku110k/images/*.jpg --max-images 64
    !python pruning.py --demo        # tiny self-check, random small model
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

REPO = "moonshotai/MoonViT-SO-400M"
PATCH = 14


# ---------------------------------------------------------------------------
# Image -> (pixel_values, grid_hws) the way MoonVisionPatchEmbed expects.
# proj is Conv2d(k=14,s=14); patch_embed does proj(x).view(L, -1), so x must be
# a stack of patches shaped (L, 3, 14, 14) and grid_hws = [[gh, gw], ...].
# ---------------------------------------------------------------------------


def image_to_patches(img: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
    """(3, H, W) float image -> ((gh*gw, 3, 14, 14) patches, (gh, gw)).

    Crops to the largest multiple of PATCH and of 2*PATCH (the 2x2 patch_merger
    at the end needs even grids), then unfolds into non-overlapping patches.
    """
    c, h, w = img.shape
    step = 2 * PATCH  # keep grid even for the merger
    gh = (h // step) * 2
    gw = (w // step) * 2
    if gh == 0 or gw == 0:
        raise ValueError(f"image too small for even grid: {h}x{w}")
    img = img[:, : gh * PATCH, : gw * PATCH]
    # (3, gh, 14, gw, 14) -> (gh, gw, 3, 14, 14) -> (gh*gw, 3, 14, 14)
    p = img.view(c, gh, PATCH, gw, PATCH).permute(1, 3, 0, 2, 4).contiguous()
    p = p.view(gh * gw, c, PATCH, PATCH)
    return p, (gh, gw)


# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------


@dataclass
class GroupStats:
    n_units: int
    act_sum: Tensor       # (n_units,)
    fisher_sum: Tensor    # (n_units,)
    dir_sum: Tensor       # (n_units, feat)
    count: int = 0

    @classmethod
    def zeros(cls, n_units: int, feat: int, device, dtype) -> "GroupStats":
        return cls(
            n_units=n_units,
            act_sum=torch.zeros(n_units, device=device, dtype=dtype),
            fisher_sum=torch.zeros(n_units, device=device, dtype=dtype),
            dir_sum=torch.zeros(n_units, feat, device=device, dtype=dtype),
        )


def _zscore(x: Tensor) -> Tensor:
    std = x.std()
    if std < 1e-8:
        return torch.zeros_like(x)
    return (x - x.mean()) / std


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class Scorer:
    """Collect Activation / Fisher / Diversity for heads, neurons, blocks."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        cfg = model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.hidden = cfg.hidden_size
        self.head_dim = self.hidden // self.n_heads
        self.inter = cfg.intermediate_size
        p0 = next(model.parameters())
        self.device, self.dtype = p0.device, torch.float32  # stats in fp32

        self.heads: Optional[GroupStats] = None
        self.neurons: Optional[GroupStats] = None
        self.layers: Optional[GroupStats] = None

        self._head_act: List[Tensor] = []
        self._neuron_act: List[Tensor] = []
        self._layer_act: List[Tensor] = []
        self._handles: List = []

    def attach(self) -> None:
        for block in self.model.encoder.blocks:
            self._handles.append(
                block.wo.register_forward_pre_hook(
                    lambda _m, inp: self._head_act.append(
                        self._keep(inp[0].view(inp[0].shape[0], self.n_heads, self.head_dim))
                    )
                )
            )
            self._handles.append(
                block.mlp.fc1.register_forward_pre_hook(
                    lambda _m, inp: self._neuron_act.append(self._keep(inp[0]))
                )
            )
            self._handles.append(
                block.register_forward_hook(
                    lambda _m, _i, out: self._layer_act.append(self._keep(out))
                )
            )

    @staticmethod
    def _keep(t: Tensor) -> Tensor:
        t.retain_grad()
        return t

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def step(self, pixel_values: Tensor, grid_hws: Tensor) -> None:
        self._head_act.clear()
        self._neuron_act.clear()
        self._layer_act.clear()

        self.model.zero_grad(set_to_none=True)
        out = self.model(pixel_values, grid_hws)
        # forward returns a list of per-image merged tensors; energy = sum of L2^2
        loss = sum(t.float().pow(2).sum() for t in out)
        loss.backward()

        self._accumulate(self._head_act, "heads")
        self._accumulate(self._neuron_act, "neurons")
        self._accumulate(self._layer_act, "layers")

    def _accumulate(self, captured: List[Tensor], group: str) -> None:
        acts, fishers, feats = [], [], []
        for t in captured:
            td = t.detach().float()
            grad = t.grad.float() if t.grad is not None else torch.zeros_like(td)
            if group == "heads":          # t: (L, n_heads, head_dim)
                a = td.norm(dim=2).mean(dim=0)              # (n_heads,)
                f = (td * grad).pow(2).sum(dim=(0, 2))      # (n_heads,)
                d = td.mean(dim=0)                          # (n_heads, head_dim)
            elif group == "neurons":      # t: (L, inter)
                a = td.abs().mean(dim=0)                    # (inter,)
                f = (td * grad).pow(2).sum(dim=0)           # (inter,)
                d = td.mean(dim=0, keepdim=True).T          # (inter, 1)
            else:                         # t: (L, hidden)
                a = td.norm(dim=1).mean().unsqueeze(0)              # (1,)
                f = (td * grad).pow(2).sum().unsqueeze(0)           # (1,)
                d = td.mean(dim=0, keepdim=True)                    # (1, hidden)
            acts.append(a)
            fishers.append(f)
            feats.append(d)

        act = torch.cat(acts)
        fisher = torch.cat(fishers)
        direction = F.normalize(torch.cat(feats, dim=0), dim=1)

        stats = getattr(self, group)
        if stats is None:
            stats = GroupStats.zeros(act.numel(), direction.shape[1], self.device, self.dtype)
            setattr(self, group, stats)
        stats.act_sum += act
        stats.fisher_sum += fisher
        stats.dir_sum += direction
        stats.count += 1

    def score(self, alpha: float, beta: float, gamma: float) -> Dict[str, Tensor]:
        results = {}
        for group in ("heads", "neurons", "layers"):
            stats: GroupStats = getattr(self, group)
            if stats is None or stats.count == 0:
                continue
            act = stats.act_sum / stats.count
            fisher = stats.fisher_sum / stats.count
            mean_dir = F.normalize(stats.dir_sum.mean(dim=0, keepdim=True), dim=1)
            unit_dir = F.normalize(stats.dir_sum, dim=1)
            diversity = 1.0 - (unit_dir * mean_dir).sum(dim=1)
            results[group] = (
                alpha * _zscore(act) + beta * _zscore(fisher) + gamma * _zscore(diversity)
            )
        return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_ranking(group: str, score: Tensor, n_heads: int, inter: int, top: int) -> None:
    order = score.argsort()  # ascending = prune first
    print(f"\n=== {group}: {score.numel()} units, lowest {top} (prune first) ===")
    for rank, idx in enumerate(order[:top].tolist()):
        if group == "heads":
            loc = f"block {idx // n_heads}, head {idx % n_heads}"
        elif group == "neurons":
            loc = f"block {idx // inter}, neuron {idx % inter}"
        else:
            loc = f"block {idx}"
        print(f"  #{rank:>3}  score={score[idx]:+.3f}  {loc}")


def run(samples, model, alpha, beta, gamma, top) -> Dict[str, Tensor]:
    scorer = Scorer(model)
    scorer.attach()
    try:
        for i, (pv, gh) in enumerate(samples):
            scorer.step(pv, gh)
            print(f"\rscored {i + 1}/{len(samples)} images", end="", flush=True)
    finally:
        scorer.detach()
    print()
    scores = scorer.score(alpha, beta, gamma)
    for group, s in scores.items():
        _print_ranking(group, s.cpu(), scorer.n_heads, scorer.inter, top)
    return scores


# ---------------------------------------------------------------------------
# Loading (real checkpoint)
# ---------------------------------------------------------------------------


def load_real_model(device):
    """Load moonshotai/MoonViT-SO-400M, working around the transformers-5 bug
    in the remote code (`all_tied_weights_keys` missing) by instantiating the
    class directly and loading the safetensors state dict ourselves."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"  # sdpa path runs in fp32 (no bf16 assert)
    Model = get_class_from_dynamic_module("modeling_moonvit.MoonVitPretrainedModel", REPO)
    model = Model(cfg).to(device).float()

    state = load_file(hf_hub_download(REPO, "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"warn: {len(missing)} missing, {len(unexpected)} unexpected keys")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return model


DEFAULT_HF_DATASET = "harryrobert/SKU-110k-reformat"


def _pil_to_sample(img, device):
    import torchvision.transforms.functional as TF

    patches, (gh, gw) = image_to_patches(TF.to_tensor(img.convert("RGB")))
    pv = patches.to(device).float()
    ghw = torch.tensor([[gh, gw]], device=device, dtype=torch.int32)
    return pv, ghw


def load_samples_hf(repo, split, device, max_images):
    """Load images from the HF Hub dataset (harryrobert/SKU-110k-reformat)."""
    from datasets import load_dataset

    ds = load_dataset(repo, split=split, streaming=True)
    samples = []
    for ex in ds:
        samples.append(_pil_to_sample(ex["image"], device))
        if len(samples) >= max_images:
            break
    if not samples:
        raise SystemExit(f"no images from {repo}:{split}")
    return samples


def load_samples_glob(patterns, device, max_images):
    from PIL import Image

    paths = sorted(p for pat in patterns for p in glob.glob(pat))[:max_images]
    if not paths:
        raise SystemExit(f"no images matched: {patterns}")
    return [_pil_to_sample(Image.open(p), device) for p in paths]


# ---------------------------------------------------------------------------
# Demo / self-check (small fake model with the SAME module names)
# ---------------------------------------------------------------------------


class _FakeCfg:
    num_hidden_layers = 3
    num_attention_heads = 4
    hidden_size = 64
    intermediate_size = 128


class _FakeBlock(nn.Module):
    def __init__(self, h, i, nh):
        super().__init__()
        self.norm0 = nn.LayerNorm(h)
        self.norm1 = nn.LayerNorm(h)
        self.wqkv = nn.Linear(h, h * 3)
        self.wo = nn.Linear(h, h)
        self.mlp = nn.Sequential()  # replaced below to get .fc0/.fc1
        self.mlp = _FakeMLP(h, i)

    def forward(self, x, *_a, **_k):
        x = x + self.wo(self.norm0(x))
        x = x + self.mlp(self.norm1(x))
        return x


class _FakeMLP(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.fc0 = nn.Linear(h, i)
        self.fc1 = nn.Linear(i, h)

    def forward(self, x):
        return self.fc1(F.gelu(self.fc0(x)))


class _FakeEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_FakeBlock(cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads)
             for _ in range(cfg.num_hidden_layers)]
        )


class _FakeModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.encoder = _FakeEncoder(cfg)

    def forward(self, pixel_values, grid_hws):
        x = pixel_values  # (L, hidden) already
        for b in self.encoder.blocks:
            x = b(x)
        return [x]  # list, like the real patch_merger output


def demo() -> None:
    torch.manual_seed(0)
    cfg = _FakeCfg()
    model = _FakeModel(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(True)

    samples = [
        (torch.randn(20, cfg.hidden_size), torch.tensor([[4, 5]], dtype=torch.int32))
        for _ in range(3)
    ]
    scores = run(samples, model, alpha=1.0, beta=1.0, gamma=1.0, top=5)

    assert set(scores) == {"heads", "neurons", "layers"}, scores.keys()
    assert scores["heads"].numel() == cfg.num_hidden_layers * cfg.num_attention_heads
    assert scores["neurons"].numel() == cfg.num_hidden_layers * cfg.intermediate_size
    assert scores["layers"].numel() == cfg.num_hidden_layers
    for g, s in scores.items():
        assert torch.isfinite(s).all(), f"{g} non-finite"
        assert abs(s.mean().item()) < 1e-3, f"{g} mean off: {s.mean()}"
    print("\ndemo OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=DEFAULT_HF_DATASET,
                    help="HF Hub dataset id (default: harryrobert/SKU-110k-reformat)")
    ap.add_argument("--split", default="train", help="dataset split")
    ap.add_argument("--images", nargs="+",
                    help="glob(s) for local images (overrides --dataset)")
    ap.add_argument("--max-images", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_real_model(device)
    if args.images:
        samples = load_samples_glob(args.images, device, args.max_images)
    else:
        samples = load_samples_hf(args.dataset, args.split, device, args.max_images)
    run(samples, model, args.alpha, args.beta, args.gamma, args.top)


if __name__ == "__main__":
    main()
