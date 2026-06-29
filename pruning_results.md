# MoonViT-SO-400M Pruning — Importance Scoring Results

**Model:** `moonshotai/MoonViT-SO-400M` (1152 hidden, 27 blocks, 16 heads, head_dim 72, FFN 4304)
**Dataset:** `harryrobert/SKU-110k-reformat` (split `train`)
**Score:** `α·Activation + β·Fisher + γ·Diversity` (z-scored per granularity group; δ·DomainFrequency bỏ — SKU-110k đơn domain)
**Loss proxy:** L2² của output (label-free)

---

## Run đáng tin (chốt)

```
--max-images 512 --max-side 1024 --checkpoint --alpha 0.5 --beta 2 --gamma 1
```
512 ảnh → ranking ổn định, trùng nhiều với run 64 ảnh ⇒ tín hiệu thật. Fisher-heavy (β=2) cho heads gần như giống α=β=γ=1 ⇒ ranking **không nhạy** với lựa chọn weight.

### Heads (432 units) — đáng tin nhất, prune trước
Lowest 20 (điểm thấp = cắt trước):

| # | score | vị trí |
|---|-------|--------|
| 0 | -4.600 | block 18, head 14 |
| 1 | -3.398 | block 23, head 7 |
| 2 | -3.209 | block 14, head 5 |
| 3 | -3.182 | block 9, head 8 |
| 4 | -3.143 | block 16, head 15 |
| 5 | -2.976 | block 17, head 2 |
| 6 | -2.949 | block 21, head 14 |
| 7 | -2.810 | block 5, head 1 |
| 8 | -2.792 | block 0, head 6 |
| 9 | -2.782 | block 9, head 2 |
| 10 | -2.679 | block 12, head 7 |
| 11 | -2.598 | block 18, head 10 |
| 12 | -2.593 | block 6, head 11 |
| 13 | -2.557 | block 18, head 2 |
| 14 | -2.518 | block 9, head 11 |
| 15 | -2.417 | block 14, head 14 |
| 16 | -2.416 | block 12, head 8 |
| 17 | -2.400 | block 11, head 11 |
| 18 | -2.270 | block 17, head 15 |
| 19 | -2.267 | block 16, head 2 |

`block 18 head 14` đứng đầu áp đảo ở cả 2 run (64 & 512 ảnh).

### Layers (27 units) — pattern ViT kinh điển
Block giữa **11–17** ít quan trọng nhất; block đầu (0–3) + cuối (24–26) KHÔNG nằm trong top-prune ⇒ hai đầu giữ thông tin, giữa dư thừa. Cắt nguyên layer là thô/rủi ro nhất — cắt ít.

| # | score | block |
|---|-------|-------|
| 0 | -1.366 | 14 |
| 1 | -1.363 | 13 |
| 2 | -1.361 | 15 |
| 3 | -1.343 | 12 |
| 4 | -1.331 | 16 |
| 5 | -1.299 | 11 |
| 6 | -1.281 | 17 |
| 7 | -1.261 | 10 |
| 8 | -1.204 | 9 |
| 9 | -1.154 | 18 |
| 10 | -1.123 | 8 |
| 11 | -1.045 | 19 |
| 12 | -0.965 | 7 |
| 13 | -0.945 | 20 |

### Neurons (116208 units) — tie -0.557, dồn block 0
20 unit thấp nhất cùng score `-0.557` = **neuron chết** (act≈0, Fisher≈0, div≈0 trên 512 ảnh). Đây là prune AN TOÀN nhất (cắt gần như không ảnh hưởng output). Tie = redundant, không phải bug. Dồn block 0 (layer đầu nhiều neuron chưa chuyên hóa cho domain kệ hàng).

---

## Lưu ý vận hành
- Score chỉ so sánh **trong cùng nhóm** (z-score riêng từng granularity). Không gộp/so head vs neuron vs layer.
- Giữ cố định `--max-side` khi so các run — đổi nó là đổi thang đo.
- OOM history: mask attention O(L²) materialize dense. Ảnh 3024px → ~46k patch → 8GB mask. Fix: resize `--max-side` (cap L) + `--checkpoint` (gradient checkpointing, ~5-10× ít VRAM) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. T4 16GB chịu được max-side ~1024-1536.
- Fisher dùng module backward hook (`register_full_backward_hook` → grad_input), KHÔNG retain_grad (checkpoint vứt mất).

## Chưa làm (next nếu cần)
- Export ranking ra CSV/JSON (--out flag) để khỏi scoring lại 512 ảnh.
- Cắt thật + lưu checkpoint pruned (chọn tỷ lệ cắt, mask/xóa unit điểm thấp).
