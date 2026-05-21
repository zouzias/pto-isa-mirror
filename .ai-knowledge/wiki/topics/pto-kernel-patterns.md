---
page_type: topic
title: pto-kernel-patterns
status: draft
version: [pto-isa repository current layout]
chip: [A2, A3, 910A3, 910B, A5, 950]
sources:
  - origin: code:kernels/manual/a2a3/topk/topk_kernel.cpp
    query: "pto topk vector pipeline"
    source_type: example
    reliability: high
    ingested_at: 2026-04-19
  - origin: code:kernels/manual/common/flash_atten/fa_performance_kernel.cpp
    query: "pto flashattention stages"
    source_type: example
    reliability: high
    ingested_at: 2026-04-19
  - origin: code:kernels/manual/a5/allgather_gemm/allgather_gemm_compute_kernel.cpp
    query: "pto compute comm decouple"
    source_type: example
    reliability: high
    ingested_at: 2026-04-19
last_updated: 2026-04-19
related:
  - pto-overview.md
  - ../../cards/tables/pto-kernel-pattern-map.md
---

# PTO Kernel Patterns

## canonical scope

本页是 PTO kernel 模式问题的 canonical router，用来把“某段 kernel 在用什么 PTO 模式”归一化到有限的模式簇上。本页不展开单个 kernel 教程，也不替代 card。

## hit terms

- `gemm pipeline`
- `topk`
- `flashattention`
- `compute comm decouple`
- `ready queue`
- `TMATMUL`
- `TSORT32`
- `TPUSH/TPOP`
- `A5 MX`
- `SIMT`

## normalized routing

- 如果问题是 `TLOAD/TEXTRACT/TMATMUL/TSTORE` 这类 cube 主流水 → `gemm-pipeline`
- 如果问题是 `TSORT32/TMRGSORT/TGATHER` 这类 vector-only 路线 → `topk-vector-pipeline`
- 如果问题是 cube/vector 多阶段协同 → `flashattention-stages`
- 如果问题是 compute 与 comm 分离、ready queue、streaming tile → `compute-comm-decouple`
- 如果问题是 A5 上的 MX / SIMT / sync 特化 → `a5-mx-simt-sync`

## primary anchors

- `kernels/manual/a2a3/topk/topk_kernel.cpp`
- `kernels/manual/common/flash_atten/fa_performance_kernel.cpp`
- `kernels/manual/a5/allgather_gemm/allgather_gemm_compute_kernel.cpp`

## common confusion

- 不要把本页当成 kernel 逐行讲解；逐行事实仍应回源码。
- 不要把原语问题直接路由到本页；原语簇问题应先看 `pto-primitives-map` 或 primitives cards。
- 不要把 testcase 问题直接路由到本页；示例入口问题应先看 `pto-test-entrypoints`。

## escalation path

- 需要模式到源码的快速落点 → `cards/tables/pto-kernel-pattern-map.md`
- 需要具体模式短答案单元 → `cards/patterns/*`
- 需要 testcase 闭环 → `wiki/topics/pto-test-entrypoints.md`

## current status

当前 PTO kernel 模式已经稳定收敛到五类：标准 gemm、TopK vector、FlashAttention stages、compute/comm decouple、A5 特化。后续新增 kernel 内容应优先命中这五类之一。

## Sources

- `kernels/manual/a2a3/topk/topk_kernel.cpp`
- `kernels/manual/common/flash_atten/fa_performance_kernel.cpp`
- `kernels/manual/a5/allgather_gemm/allgather_gemm_compute_kernel.cpp`

## Related

- `wiki/topics/pto-overview.md`
- `cards/tables/pto-kernel-pattern-map.md`
