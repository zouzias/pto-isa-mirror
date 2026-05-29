# deepseekv4

Auto-mode A3 kernel scaffolding for the **DeepSeek-V4-Pro** inference path
(flash, non-fused). Status: structure + per-leaf design notes only — **no
compile / run claim**. See [CLAUDE.md](../../../../CLAUDE.md) for the
hard-constraint policy.

## Reference

Source of truth for shapes and dtypes:

- TileLang CUDA kernels: [deepseek/kernel.py](../../../../deepseek/kernel.py)
- Model wiring + ModelArgs: [deepseek/model.py](../../../../deepseek/model.py)
- HuggingFace repo: `deepseek-ai/DeepSeek-V4-Pro/tree/main/inference`
  (kernel.py + model.py byte-sizes match the local copy — `Known`)

## Pipeline → kernel groups

DeepSeek-V4-Pro `Block.forward` (model.py:689) decomposes as:

```
hc_pre  →  attn_norm  →  Attention  →  hc_post
hc_pre  →  ffn_norm   →  MoE        →  hc_post
```

Where `Attention` (MLA + sparse-attn, model.py:485) internally does:

```
wq_a → q_norm → wq_b → rope(q[..., -rd:])
wkv  → kv_norm → rope(kv[..., -rd:]) → fp8/fp4 act_quant
Compressor(x, start_pos)  # produces compressed KV tail
Indexer(x, qr, ...)       # produces top-k compressed-cache indices
sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)
rope(o[..., -rd:], inverse=True)
wo_a (einsum) → wo_b
```

Groups in this scaffold (each leaf is its own kernel — vector OR cube, never
both):

| Group | Sub-group | Leaves | Role |
|-------|-----------|--------|------|
| [CSA/](CSA/) | — | qk_matmul, online_softmax, pv_matmul, output_rescale, gather_kv | The `sparse_attn` FlashAttention split into its cube + vector stages |
| [HCA/](HCA/) | [Compressor/](HCA/Compressor/) | z_gemm, c_gemm, biased_softmax, c_comp, rope, rms_norm | KV compressor pipeline (model.py:280) |
| [HCA/](HCA/) | [Indexer/](HCA/Indexer/) | wq_b_gemm, weights_proj_gemm, index_score | Top-k compressed-cache selector (model.py:381) |
| [HCA/](HCA/) | [quant/](HCA/quant/) | act_quant_fp8, act_quant_fp4 | Block-wise quantization (kernel.py:41, kernel.py:128) |
| [HC/](HC/) | — | hc_mix_gemm, hc_sinkhorn, hc_pre_reduce, hc_post_combine | Hyper-Connections residual mixing (model.py:648) |
| [MoE/](MoE/) | — | gate_logits, gate_softmax, gate_score_topk, gate_hash_routing, scatter, expert_ffn, shared_expert_ffn, gather | DeepSeek-V4 Gate/MoE (model.py:547, 610) — includes the hash-routing branch (model.py:559, 577) as its own leaf |

The **sparse_attn FlashAttention kernel** (kernel.py:277) is the heart of CSA;
it has been split into its hardware-distinct stages (cube vs vector) per the
"one-by-one not fused" directive.

## Cube vs vector per leaf

Authoritative classification (from each leaf's `CMakeLists.txt` —
`pto_example_cube_auto` vs `pto_example_vec_auto`). **9 cube, 19 vector,
28 total.**

### CSA — 2 cube / 3 vector

| Leaf | HW | Op |
|------|----|----|
| [CSA/qk_matmul/](CSA/qk_matmul/) | **cube** | `acc_s = Q @ K^T * scale` (BF16×BF16 → FP32) |
| [CSA/pv_matmul/](CSA/pv_matmul/) | **cube** | `acc_o += acc_s_cast @ V` (BF16×BF16 → FP32 RMW) |
| [CSA/gather_kv/](CSA/gather_kv/) | vector | indexed gather of KV rows by `topk_idx` |
| [CSA/online_softmax/](CSA/online_softmax/) | vector | FlashAttention running max / exp / rescale / sum |
| [CSA/output_rescale/](CSA/output_rescale/) | vector | `acc_o /= sum_exp` + `attn_sink` tail |

### HCA/Compressor — 2 cube / 4 vector

| Leaf | HW | Op |
|------|----|----|
| [HCA/Compressor/z_gemm/](HCA/Compressor/z_gemm/) | **cube** | `kv = X @ Wkv^T` |
| [HCA/Compressor/c_gemm/](HCA/Compressor/c_gemm/) | **cube** | `score = X @ Wgate^T` |
| [HCA/Compressor/biased_softmax/](HCA/Compressor/biased_softmax/) | vector | `score + ape` then softmax over ratio axis |
| [HCA/Compressor/c_comp/](HCA/Compressor/c_comp/) | vector | `kv_comp = sum(kv * softmax_score, dim=ratio)` |
| [HCA/Compressor/rope/](HCA/Compressor/rope/) | vector | `apply_rotary_emb` on last `rope_head_dim` |
| [HCA/Compressor/rms_norm/](HCA/Compressor/rms_norm/) | vector | `x * rsqrt(mean(x²) + eps) * weight` |

### HCA/Indexer — 2 cube / 1 vector

| Leaf | HW | Op |
|------|----|----|
| [HCA/Indexer/wq_b_gemm/](HCA/Indexer/wq_b_gemm/) | **cube** | `q = qr @ Wq_b^T` low-rank expansion |
| [HCA/Indexer/weights_proj_gemm/](HCA/Indexer/weights_proj_gemm/) | **cube** | `weights = (X @ W_wproj^T) * scale` |
| [HCA/Indexer/index_score/](HCA/Indexer/index_score/) | vector | `relu(einsum) * weights` reduced over heads |

### HCA/quant — 0 cube / 2 vector

| Leaf | HW | Op |
|------|----|----|
| [HCA/quant/act_quant_fp8/](HCA/quant/act_quant_fp8/) | vector | block-wise FP8 (E4M3) quant, BF16 inplace |
| [HCA/quant/act_quant_fp4/](HCA/quant/act_quant_fp4/) | vector | block-wise FP4 quant, BF16 inplace, E8M0 scale |

### HC — 1 cube / 3 vector

| Leaf | HW | Op |
|------|----|----|
| [HC/hc_mix_gemm/](HC/hc_mix_gemm/) | **cube** | `mixes = (x_flat @ hc_fn^T) * rsqrt` (FP32 GEMM) |
| [HC/hc_sinkhorn/](HC/hc_sinkhorn/) | vector | sigmoid + softmax + N Sinkhorn iters on 4×4 comb |
| [HC/hc_pre_reduce/](HC/hc_pre_reduce/) | vector | `y = sum(pre * x, dim=hc_axis)` |
| [HC/hc_post_combine/](HC/hc_post_combine/) | vector | `y = post*x + sum(comb * residual, dim=hc_axis)` |

### MoE — 3 cube / 5 vector

| Leaf | HW | Op |
|------|----|----|
| [MoE/gate_logits/](MoE/gate_logits/) | **cube** | `scores = X @ W^T` (FP32 — model.py:566) |
| [MoE/expert_ffn/](MoE/expert_ffn/) | **cube** | per-expert SwiGLU FFN (W1/W3/W2 GEMMs dominate) |
| [MoE/shared_expert_ffn/](MoE/shared_expert_ffn/) | **cube** | single-expert SwiGLU FFN over all tokens |
| [MoE/gate_softmax/](MoE/gate_softmax/) | vector | softmax / sigmoid / sqrt(softplus) activation |
| [MoE/gate_score_topk/](MoE/gate_score_topk/) | vector | `+ bias → topk → gather → normalize → scale` |
| [MoE/gate_hash_routing/](MoE/gate_hash_routing/) | vector | `tid2eid[input_ids]` lookup, no sort |
| [MoE/scatter/](MoE/scatter/) | vector | token → expert reorder |
| [MoE/gather/](MoE/gather/) | vector | expert output → token recombine (weighted scatter-add) |

### Cube/vector caveats

- **`MoE/expert_ffn` and `MoE/shared_expert_ffn`** are SwiGLU
  (`W1·x → silu·* → W2·(…)`). They contain vector ops (silu, elementwise
  mul, optional `swiglu_limit` clamp), but the kernel is built against the
  **cube AICORE pipe** (`--cce-aicore-arch=dav-c220-cube`) because the
  three GEMMs dominate. This mirrors the existing top-level
  [MoEv2/expert_ffn/](../MoEv2/expert_ffn/) pattern. If a pure split is
  preferred, refactor into `expert_w1w3_gemm` (cube), `expert_swiglu`
  (vector), `expert_w2_gemm` (cube).
- **`HCA/Indexer/index_score`** is scaffolded as vector, but the underlying
  einsum `bshd,btd→bsht` is cube-heavy. Its README documents this as an
  `Assumption` with a follow-up to split into `einsum_cube + reduce_vector`
  if profiling shows the inner product is the bottleneck.
- **`HC/hc_mix_gemm`** is built cube but has a `* rsqrt` epilogue that's
  vector — the scaffold takes `rsqrt` precomputed from the host to keep
  this leaf cube-only.

## MoE — hash vs score routing

DeepSeek-V4's `Gate` (model.py:547) supports **two routing modes**:

- **Hash routing** (model.py:559, 577) — first `n_hash_layers` layers use a
  fixed `tid2eid[input_ids]` table; no `topk` needed.
- **Score routing** (model.py:580) — remaining layers compute
  `scores + bias` then `topk`.

These are scaffolded as **separate leaves** (`gate_hash_routing/` vs
`gate_score_topk/`) because they have different I/O shapes and HW patterns
(hash = pure gather, score = topk + gather). The shared upstream is
`gate_logits/` (the FP32 weight × X linear) and `gate_softmax/` (one of
softmax / sigmoid / sqrt(softplus)).

## Out of scope for this scaffold

- `fp8_gemm` / `fp4_gemm` as standalone kernels — both are exercised inside
  `z_gemm`, `c_gemm`, `wq_b_gemm`, and `hc_mix_gemm` with their respective
  quant scale dtypes. A standalone copy would duplicate.
- `Attention` outer wiring (`wq_a`, `wo_a`, `wo_b`) — these are plain FP8
  linears already covered by the GEMM leaves in CSA/HCA; no separate scaffold.

## Memory-budget convention

Per [CLAUDE.md §Memory-budget-first kernel planning](../../../../CLAUDE.md):
every leaf README must state, before any kernel code:

- L1 / L0A / L0B / L0C / UB custom budgets
- Live tiles by memory level
- Smallest hardware operation (cube fractal or vector op shape)
- Loop tiling plan
- Test-shape plan (tiny / medium / model-inspired / tail)

## Test harness

Each leaf follows the contract in
[docs_for_ai/kernel_test_guidance.md](../../../../docs_for_ai/kernel_test_guidance.md):
`README.md`, `<name>_kernel.cpp`, `main.cpp`, `CMakeLists.txt`, `run.sh`,
`scripts/gen_data.py`. Each **family** (CSA, HCA/Compressor, HCA/Indexer,
HCA/quant, HC) owns a `scripts/generate_cases.py` and a
`build/generated_cases.{h,json}` manifest consumed by every leaf in that
family — same pattern as [MoE/scripts/generate_cases.py](../MoE/scripts/generate_cases.py).

## Status (auto-mode A3 only)

- **Branch:** `deepseekv4_structure`
- **Hardware target:** A3 (`PTO_NPU_ARCH_A2A3`). A5 mirror deferred.
- **What's done:** folder layout + per-leaf README + memory plan + signature
  + commented pseudocode + test harness skeleton.
- **What's NOT done:** real kernel bodies (only signatures + pseudocode),
  no build / run validation, no `run_all.sh` wiring.

`Assumption`: every leaf in this tree is currently **untested** and unbuilt.
Per CLAUDE.md, do not claim any of these compile or run until the user
provides compiler / runtime output.
