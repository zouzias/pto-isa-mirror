---
name: card-table-pto-kernel-pattern-map
description: |
  触发：查找 PTO Kernel Pattern Map 对应的映射表、规格表或入口对照时。
  Trigger: Lookup the PTO Kernel Pattern Map mapping table, spec table, or entry cross-reference.
card_type: table
version: repo-current
chip: [950 A5, 910A3]
source: example
last_updated: 2026-04-20
---

# PTO Kernel Pattern Map

| Pattern | Representative kernels | Platforms | Typical PTO primitives | Recommended next stop |
|---|---|---|---|---|
| Standard gemm pipeline | `kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp`, `kernels/manual/a2a3/conv2d_forward/conv2d_forward_kernel.cpp` | A2/A3 | `TLOAD`, `TEXTRACT`, `TIMG2COL`, `TMATMUL`, `TSTORE` | `cards/patterns/gemm-pipeline.md` |
| TopK vector pipeline | `kernels/manual/a2a3/topk/topk_kernel.cpp` | A2/A3 | `TSORT32`, `TMRGSORT`, `TGATHER`, `TSTORE` | `cards/patterns/topk-vector-pipeline.md` |
| FlashAttention stages | `kernels/manual/common/flash_atten/fa_performance_kernel.cpp`, `pto_macro_matmul.hpp`, `pto_macro_fa_softmax.hpp` | common / A5 | `TMATMUL`, `TROWMAX`, `TROWEXPANDSUB`, `TEXP`, `TROWSUM` | `cards/patterns/flashattention-stages.md` |
| Compute / comm decouple | `kernels/manual/a5/allgather_gemm/allgather_gemm_compute_kernel.cpp`, `kernels/manual/a2a3/gemm_ar/comm_kernel.cpp` | A5 / A2A3 | load-store + queue/ready + comm primitives | `cards/patterns/compute-comm-decouple.md` |
| A5 MX / SIMT specialization | `kernels/manual/a5/matmul_mxfp4_performance/mxmatmul_performance_kernel.cpp`, `kernels/manual/a5/engram_simt/engram-simt_kernel.cpp` | A5 | `TMATMUL_MX`, sync/streaming primitives | `cards/patterns/a5-mx-simt-sync.md` |

## related
- `wiki/topics/pto-kernel-patterns.md`
- `cards/tables/pto-primitives-map.md`
