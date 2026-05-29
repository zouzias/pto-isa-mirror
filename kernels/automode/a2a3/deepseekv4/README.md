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
