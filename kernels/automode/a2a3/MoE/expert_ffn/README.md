# expert_ffn

Auto-mode A3 MoE kernel — Stage 3 of 3.

## What it does

Per-token element-wise expert FFN (gated activation):

```
output[t] = relu(x_packed[t] * w1[e]) * w2[e]
```

where `*` is element-wise (Hadamard product) and `e` is the expert assigned to token `t`. `w1[e]` and `w2[e]` are per-expert scale vectors of dimension `kD=64`.

- **x_packed** `(kT=256, kD=64)` float32 — tokens sorted by expert assignment
- **w1** `(kE=32, kD=64)` float32 — first gate weights
- **w2** `(kE=32, kD=64)` float32 — second gate weights
- **expert_count** `(kE=32,)` int32 — number of tokens per expert
- **expert_start** `(kE=32,)` int32 — packed-token start offset per expert
- **output** `(kT=256, kD=64)` float32

Tokens are pre-sorted by expert assignment (packed-token convention identical to `moe_segmented_gemm_relu`). The `expert_count` / `expert_start` arrays describe each expert's contiguous slice.

## Target platform

A3 (`PTO_NPU_ARCH_A2A3`). Vec path: `--cce-aicore-arch=dav-c220-vec`.

## Pipeline per token

```
TLOAD  w1[e]                    (reloaded each token; same as bMatTile in moe_segmented_gemm_relu)
TLOAD  w2[e]
TLOAD  x_packed[t]
TMUL   h    = x * w1            element-wise gating
TMAXS  relu = max(h, 0.0f)      ReLU via scalar max
TMUL   out  = relu * w2         output gate
TSTORE output[t]
```

## Auto-mode constraints

- Single AICORE; no `block_idx` work split.
- No manual sync.
- Tiles declared **inside** the inner token loop (topk lesson: per-iter liveness isolation for nested loops).
- `w1Global`/`w2Global` views declared in outer expert loop scope; TLOAD inside inner loop (confirmed-working pattern from moe_segmented_gemm_relu's bGlobal/bMatTile).
- TMAXS scalar arg: `static_cast<float>(0)` → maps to `vmaxs` hardware instruction.

## How to build and run

```bash
bash run.sh -r npu -v Ascend910B1
```

Expected output: `test data success`.

## How to compare against Python reference

The golden is `relu(x * w1[e]) * w2[e]` computed in float32. Tolerance 1e-4f.

## Known limitations

- Element-wise W1/W2 only — not a standard FFN with matrix projections. A matrix-projection variant (GEMM + type conversion for intermediate) requires FP32→FP16 conversion which is a separate kernel.
- Not connected to `moe_topk` output; standalone prototype. In a full pipeline the expert assignment would come from topk indices, and tokens would be reordered before being passed here.
- No claim of build/run success until user provides compiler output.
