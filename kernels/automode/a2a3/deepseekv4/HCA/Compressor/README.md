# HCA/Compressor — KV gated-pooling compression

Auto-mode A3 split of the `Compressor.forward` path
([model.py:317](../../../../../../deepseek/model.py)).

## Reference (prefill path, `start_pos == 0`)

```python
# Compressor.__init__: model.py:284-306
# Compressor.forward:  model.py:317-378
ratio, overlap = self.compress_ratio, (compress_ratio == 4)
coff = 1 + overlap                 # 2 with overlap, 1 without

x  = x.float()
kv    = self.wkv  (x)              # [b, s, coff*d]     -- z_gemm   (cube)
score = self.wgate(x)              # [b, s, coff*d]     -- c_gemm   (cube)

kv    = kv.unflatten(1, (-1, ratio))      # [b, s/ratio, ratio, coff*d]
score = score.unflatten(1, (-1, ratio)) + self.ape   # + positional bias
if overlap:
    kv    = overlap_transform(kv,    0)         # [b, s/ratio, 2*ratio, d]
    score = overlap_transform(score, -inf)      # [b, s/ratio, 2*ratio, d]

kv = (kv * score.softmax(dim=2)).sum(dim=2)       # -- biased_softmax (vector)
                                                  # -- c_comp         (vector)

kv = self.norm(kv.to(dtype))                      # -- rms_norm       (vector)
apply_rotary_emb(kv[..., -rd:], freqs_cis)        # -- rope           (vector)
# (then upstream: rotate_activation + fp4/fp8 act_quant — see HCA/quant/)
```

## Leaves

| Leaf | HW | Op | Shapes (model defaults from ModelArgs) |
|------|----|----|----------------------------------------|
| [z_gemm/](z_gemm/) | cube | `kv = X @ Wkv^T` (FP32 in checkpoint, see model.py:298) | `X: [b*s, dim=4096]`, `Wkv: [coff*head_dim=1024 or 512, 4096]`, out FP32 |
| [c_gemm/](c_gemm/) | cube | `score = X @ Wgate^T` (FP32) | same shape as z_gemm |
| [biased_softmax/](biased_softmax/) | vector | `score = score + ape`, then `softmax(dim=2)` over `ratio` (or `2*ratio` with overlap) | `score: [b, s/ratio, ratio_eff, coff*d]` |
| [c_comp/](c_comp/) | vector | `kv_comp = sum(kv * softmax_score, dim=2)` — weighted-sum reduction across `ratio_eff` | `kv, score: [b, s/ratio, ratio_eff, d]`, out `[b, s/ratio, d]` |
| [rope/](rope/) | vector | `apply_rotary_emb` on last `rope_head_dim=64` dims of `kv_comp` | `kv_comp: [b, s/ratio, d]`, `freqs_cis: [s/ratio, rd/2]` complex |
| [rms_norm/](rms_norm/) | vector | `kv = kv * rsqrt(mean(kv^2) + eps)` along head_dim | `kv: [b, s/ratio, d=512]`, `eps=1e-6` |

## Family shape contract

Generated header: `build/generated_cases.h` (see [scripts/generate_cases.py](scripts/generate_cases.py)).
Case tuple: `B, S, DIM, HEAD_DIM, ROPE_DIM, COMPRESS_RATIO, OVERLAP`.

Defaults from [ModelArgs](../../../../../../deepseek/model.py):

- `dim=4096`, `head_dim=512`, `rope_head_dim=64`, `compress_ratio ∈ {4, 128}`
  (per layer; `(0,0,4,128,4,128,4,0)`).
- For prototype testing use a small case: `B=1, S=128, DIM=128, HEAD_DIM=64,
  ROPE_DIM=16, COMPRESS_RATIO=4, OVERLAP=1`.

## Auto-mode constraints (Compressor family)

- A3 only (`PTO_NPU_ARCH_A2A3`). Single AICORE per leaf.
- `wkv` / `wgate` weights are FP32 in the checkpoint (model.py:296-299) but
  loaded as BF16/FP8 on-chip; comment in each `_kernel.cpp` confirms the
  storage choice. `Assumption`: defaulting to BF16 for prototype.
- Compression runs in FP32 (model.py:323 `x = x.float()`). Numerical
  policy must match.
