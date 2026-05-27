# mla_basic

> **Status — design / skeleton.** Per [CLAUDE.md](../../../../CLAUDE.md), this
> is a first auto-mode A3 Multi-Head Latent Attention (DeepSeek V2/V3-style)
> kernel project. **Not** confirmed to compile or run on hardware yet — every
> nontrivial claim below is grounded in repo file paths, the docs in
> [docs_for_ai/](../../../../docs_for_ai/), or explicitly flagged as an
> `Assumption`. See *Open assumptions* at the bottom.

## Purpose

Implement Multi-Head Latent Attention (MLA), as introduced by DeepSeek V2 and
also used in DeepSeek V3, at the simplest possible auto-mode-A3 fidelity. MLA
compresses K and V into a low-rank latent (here `latent_dim = 64`, vs the
naive `num_heads * head_dim = 4096`), which dramatically shrinks the KV
cache. The "very basic" v1 here:

- single AICORE per kernel,
- no `block_idx` work split,
- no double / multi buffering,
- no `TPipe` / `TPUSH` / `TPOP`,
- no manual `set_flag` / `wait_flag` / `pipe_barrier` in kernel code,
- no `TASSIGN` aliasing or address pinning,
- prefill-only (KV cache for the current sequence; no append/decode),
- bidirectional attention (no causal mask).

## What the kernel does

Five logical stages, implemented as seven `__global__ AICORE` entries (cube vs
vec arch split necessitates two separate translation units). The host
launches all seven on the same ACL stream — stream-order is the cross-kernel
sync (same recipe as §A18
[moe_segmented_ffn_top1](../moe_segmented_ffn_top1/)):

| # | Stage | TU | Arch | Math | Shape |
|---|---|---|---|---|---|
| 1 | Query projection | cube | `dav-c220-cube` | `Q = X @ W_q` | `M=128 K=4096 N=4096` (split-K × 64, split-N × 64) |
| 2 | Latent KV compression | cube | `dav-c220-cube` | `C_kv = X @ W_dkv` | `M=128 K=4096 N=64` (split-K × 64) |
| 3 | Latent KV cache store | vec  | `dav-c220-vec`  | `C_cache = C_kv` (prefill-only) | one `[128, 64]` FP16 tile copy |
| 4 | KV reconstruction | cube | `dav-c220-cube` | `K = C_cache @ W_uk; V = C_cache @ W_uv` | each `M=128 K=64 N=4096` (split-N × 64) |
| 5a | Attention QK^T | cube | `dav-c220-cube` | `scores[h] = Q_h @ K_h^T` per head | `M=128 K=128 N=128` (split-K × 2, split-N × 2) × 32 heads |
| 5b | Attention softmax | vec  | `dav-c220-vec`  | `probs[h] = softmax(scale * scores[h])` per head | `[128, 128]` FP16 vec ops × 32 heads |
| 5c | Attention PV | cube | `dav-c220-cube` | `out_h = probs[h] @ V_h` per head | `M=128 K=128 N=128` × 32 heads |

The clear stage separation is the v1 design goal — performance comes later by
fusing across stages and adding the multi-core / pipelining infrastructure.

### Q · K^T without a physical transpose

Stage 5a needs `B = K_h^T` of shape `(head_dim, seq_len)`. Rather than emit a
separate transpose kernel, the cube `GlobalTensor` for B is built with
**swapped strides** (row-stride = 1, col-stride = `num_heads * head_dim`)
over `K`'s contiguous `[S, Nh, Hd]` storage. The cube `TLOAD` honors
arbitrary 5-D strides (
[tile_type_reference.md §1.3](../../../../docs_for_ai/tile_type_reference.md)
on `GlobalTensor::Stride`), so the matmul `C = A @ B` produces `Q_h @ K_h^T`
with no extra pass over GM.

### Dtype convention (FP16 inputs, FP32 accumulators, FP16 GM hand-off)

Inputs/weights are FP16. Every cube GEMM uses an FP32 accumulator
(`TileAcc<float, ...>` — proven combo in §A16 / §A17 / §A18) and writes FP16
to GM via the fused FP32→FP16 downcast in the FIX-pipe `TSTORE`. This last
step is a **soft Assumption** — §A18 proves the combo with
`ReluPreMode::NormalRelu`, and the underlying template ([
pto_instr.hpp:251-258
](../../../../include/pto/common/pto_instr.hpp#L251-L258)) supports
`ReluPreMode::NoRelu` as the other valid template arg, so combining
"downcast" with "no ReLU" is the natural composition. If this fails at
compile we will add a small vec downcast helper kernel.

The softmax in stage 5b uses FP16 vec math throughout (TROWMAX, TROWEXPAND,
TSUB, TEXP, TROWSUM, TDIV). This is a known precision divergence from a
pure-FP32 reference; the Python golden in `scripts/gen_data.py` mirrors the
FP16 step sequence to match the kernel byte-for-byte (modulo any vec-op
rounding-mode differences). The driver tolerates `0.5` absolute + `5%`
relative error on the final output to absorb the FP16-math fuzz.

## Case-driven test harness

The shape constants are NOT hardcoded — they are emitted into
`build/generated_cases.h` by `scripts/generate_cases.py` and pulled into
both kernel TUs (`mla_basic_cube_kernel.cpp`, `mla_basic_vec_kernel.cpp`) and
the host (`main.cpp`) via `#include "generated_cases.h"`. The binary
compiles against **one case at a time**; multi-case runs re-invoke
`generate_cases.py` + rebuild per case, orchestrated by `run.sh`.

### Case tuple format

```
S,H,Nh,Hd,L,qL,Rd
```

* `S`  — sequence length (`kSeqLen`); must be a multiple of `kTileM=128`, `kSoftmaxTileM=16`, and `kRopeTileM=kCacheTileM=64`
* `H`  — hidden dim (`kHidden`); must equal `Nh * Hd`
* `Nh` — number of heads
* `Hd` — per-head dim total = `kNopeDim + kRopeDim`
* `L`  — KV latent dim; current kernel assumes 64
* `qL` — Q latent dim; current kernel assumes 64
* `Rd` — rope per-head dim; current kernel assumes 64 (so `kNopeDim` is also 64)

### default_cases

When `bash run.sh -r ... -v ...` is invoked without `-c`/`-a`, two cases run:

```
128,4096,32,128,64,64,64    # canonical baseline
256,4096,32,128,64,64,64    # exercises multi-chunk vec path (S>128)
```

### Building & running

Requires the standard `ASCEND_HOME_PATH` env. Then:

```bash
bash run.sh -r npu -v Ascend910B1                                     # default cases
bash run.sh -r npu -v Ascend910B1 -c "128,4096,32,128,64,64,64"       # single case
bash run.sh -r npu -v Ascend910B1 -a "128,4096,32,128,64,64,64;256,4096,32,128,64,64,64"
bash run.sh -r sim -v Ascend910B4 -d -i                                # sim, debug, intermediate dumps
```

`run.sh` accepts: `-r/--run-mode`, `-v/--soc-version`, `-C/--compiler`,
`-n/--npu`, `-c/--case`, `-a/--cases`, `-i/--intermediate`, `-d/--debug`.

Per case, it does: `generate_cases.py` → `cmake` → `make` → `gen_data.py` →
`./mla_basic --case=<tuple>` → optional `compare_outputs.py`. Each case's
exit code feeds into the final summary; `run.sh` exits non-zero if any
case fails (per §7 of `docs_for_ai/kernel_test_guidance.md`).

## Layout

```text
mla_basic/
├── CMakeLists.txt              builds two SHARED libs (cube + vec) + executable
├── README.md                   this file
├── main.cpp                    host driver: read inputs, launch 13 kernels, per-stage validate
├── mla_basic_cube_kernel.cpp   Q compress/reconstruct, KV compress/reconstruct, attn QK/QK_rope/PV, Q/K_rope proj
├── mla_basic_vec_kernel.cpp    KV cache store, RoPE, attn softmax (nope+rope add inside)
├── run.sh                      multi-case driver (see above)
└── scripts/
    ├── generate_cases.py       emits build/generated_cases.h + .json
    ├── gen_data.py             produces ./input and ./output goldens for the active case
    └── compare_outputs.py      post-mortem per-stage comparison (optional)
```

## How to compare against the Python reference

`scripts/gen_data.py` writes both the inputs and a full set of stage-by-stage
goldens (`golden_q.bin`, `golden_c_kv.bin`, `golden_k.bin`, `golden_v.bin`,
`golden_scores.bin`, `golden_probs.bin`, `golden_out.bin`). `main.cpp`
validates only `output_out.bin` against `golden_out.bin`; for staged
debugging use a one-liner like:

```bash
python3 - <<'EOF'
import numpy as np
g = np.fromfile("output/golden_q.bin", dtype=np.float16)
d = np.fromfile("output/output_q.bin", dtype=np.float16)
err = np.abs(g - d).astype(np.float32)
print("max_abs_err:", err.max(), " | mean_abs_err:", err.mean())
EOF
```

## Known limitations (v1)

- **Single AICORE per kernel.** No `block_idx` work split. Production-quality
  MLA splits Q-rows across cube cores and per-head work across vec cores.
- **No KV cache append.** Stage 3 is a verbatim copy of the just-compressed
  latent. A decode-mode variant would index into `[S_max, latent_dim]` at
  `write_pos`.
- **No causal mask.** Bidirectional attention only. Adding a mask is a TCMP
  + TSEL or a TADDS-of-large-negative + softmax dance — see the manual FA
  references in `kernels/manual/common/flash_atten/` for the math.
- **No RoPE / no decoupled rotary.** The full DeepSeek MLA uses a small
  separate `head_dim_rope` per head; not implemented here.
- **No `W_o` output projection.** DeepSeek MLA includes a final
  `out = concat_heads @ W_o` step after attention. This v1 stops at the
  concatenated multi-head output `[S, num_heads * head_dim]` so the focus
  stays on the MLA-specific parts.
- **Untested compile / run.** No bisheng-CCE compiler access in this
  environment. See *Open assumptions*.

## Open assumptions (per CLAUDE.md vocabulary)

1. **FP32-Acc → FP16-GM TSTORE without ReLU.** Proven for the
   `ReluPreMode::NormalRelu` combo in §A18; assumed valid for
   `ReluPreMode::NoRelu`. **Risk**: low; **Likely first failure**: compile-time
   "no matching TSTORE_IMPL" error on the cube kernel.
2. **`GlobalTensor` with swapped strides as a K^T view in cube TLOAD.** No
   in-tree confirmed-built example, but [tile_type_reference.md §1.3](../../../../docs_for_ai/tile_type_reference.md)
   documents arbitrary 5-D strides and TLOAD's stride-honoring contract.
   **Risk**: medium; **Likely first failure**: incorrect scores values (off
   by transpose), not a compile error.
3. **Two cube `__global__` GEMMs sharing `aMatTile` across an inner-loop
   boundary** (stage 4 GEMM1 → GEMM2 sharing `aMatTile` / `aTile` for
   `C_cache`). §A18's split-kernel pattern avoided this on cube; consolidating
   here may surface a different auto-sync pattern. **Risk**: medium;
   **Likely first failure**: stale `aTile` content used in GEMM 2.
4. **Two SHARED libraries linked into the same executable.** Existing
   projects link one. **Risk**: low; **Likely first failure**: duplicate
   symbol / kernel-launch-table conflict at link time.
5. **`TROWMAX` / `TROWSUM` dst tile shape `[kSeqLen, 16]` with dynamic valid
   `(kSeqLen, 1)`.** Mirrors
   [trowsum_kernel.cpp](../../../../tests/npu/a2a3/src/st/testcase/trowsum/trowsum_kernel.cpp:30)
   but with FP16 instead of FP32. **Risk**: low; **Likely first failure**:
   compile-time assert on alignment.
6. **`TROWEXPAND` from `[kSeqLen, 16]` (valid 1 col) to `[kSeqLen, kSeqLen]`.**
   Tutorial uses `[M, 1]`; trowsum kernel uses `[M, 16, valid 1]`. The
   broadcast should read col 0 of src and write all cols of dst.
   **Risk**: medium; **Likely first failure**: incorrect probs values.
7. **FP16-throughout softmax precision.** May produce more error than the
   `0.5 abs + 5% rel` tolerance in `main.cpp` for some random seeds.
   **Risk**: medium; **Likely first failure**: `test data failed` with small
   element-wise mismatch even when the algorithm is correct.

If any of the above fail, log the symptom and the first meaningful compiler
error in [docs_for_ai/compile_error_logbook.md](../../../../docs_for_ai/compile_error_logbook.md)
per its §2 / §3 schema; cross-reference the assumption number above.

## Where this sits in the bigger picture

This v1 is intended as the smallest correct MLA. Follow-up milestones
already gated by other proven references:

- multi-AICORE work split using `block_idx` (after §A11 → §A18 lineage adds
  it for MoE),
- KV cache append (decode mode),
- causal mask,
- output projection `W_o`,
- FP32 softmax with bf16 storage,
- fused single-head attention via the FA macro family (currently manual-only
  per [auto_mode_bad_patterns.md §2.4 / §2.5](../../../../docs_for_ai/auto_mode_bad_patterns.md);
  re-evaluate once the auto-mode `TPipe` story is in tree),
- DeepSeek-V3 specific RoPE / decoupled rotary.
