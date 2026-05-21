---
name: card-pattern-compute-comm-decouple
description: |
  触发：查找 Compute / comm decouple 相关 PTO 模式、代表 kernel、容量线索或常见误区时。
  Trigger: Route to the Compute / comm decouple PTO pattern, representative kernels, capacity cues, or common pitfalls.
card_type: pattern
version: repo-current
chip: [950 A5, 910A3]
source: example
last_updated: 2026-04-20
---

# Compute / comm decouple

## hit terms
- `compute comm decouple`
- `ready queue`
- `tile-ready polling`
- `allgather_gemm`
- `gemm_ar`
- `comm kernel`

## canonical intent
- 把 compute 与 comm 分文件、分阶段、流式协同的问题归一化到同一模式簇

## primary anchors
- `kernels/manual/a5/allgather_gemm/allgather_gemm_compute_kernel.cpp`
- `kernels/manual/a5/allgather_gemm/allgather_gemm_comm_kernel.cpp`
- `kernels/manual/a2a3/gemm_ar/gemm_compute_kernel.cpp`
- `kernels/manual/a2a3/gemm_ar/comm_kernel.cpp`

## common confusion
- 不要把它当纯通信原语问题；它强调 compute 与 comm 的协同。
- 不要把它当标准 gemm pipeline；这里多了 ready queue、streaming、分文件组织。
- 不要只看 tests；主事实锚点在 kernels/manual。

## fallback links
- comm 原语簇 → `cards/primitives/comm-primitives.md`
- kernel 模式总览 → `wiki/topics/pto-kernel-patterns.md`
