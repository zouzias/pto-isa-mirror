---
name: card-pattern-gemm-pipeline
description: |
  触发：查找 GEMM pipeline 相关 PTO 模式、代表 kernel、容量线索或常见误区时。
  Trigger: Route to the GEMM pipeline PTO pattern, representative kernels, capacity cues, or common pitfalls.
card_type: pattern
version: repo-current
chip: [950 A5, 910A3]
source: example
last_updated: 2026-04-20
---

# GEMM pipeline

## hit terms
- `gemm pipeline`
- `TLOAD TEXTRACT TMATMUL TSTORE`
- `cube pipeline`
- `stepK`
- `ping-pong`

## canonical intent
- 把 cube/gemm 主流水归一化到标准 PTO 模式
- 识别 gemm/conv/部分 compute kernel 是否属于同一模式簇

## primary anchors
- `kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp`
- `kernels/manual/a2a3/conv2d_forward/conv2d_forward_kernel.cpp`
- `kernels/manual/a5/allgather_gemm/allgather_gemm_compute_kernel.cpp`

## capacity linkage
- 这类模式默认依赖 `Vec / UB`、`Mat / L1`、`Left / L0A`、`Right / L0B`、`Acc / L0C` 的容量预算。
- A2A3 与 A5 的关键差异不在 `L1/L0A/L0B`，而主要在 `UB` 与 `L0C`：A5 更宽。
- 若要判断某个 `stepK`、双缓冲或 accumulator 方案为什么成立，先回 `cards/tables/chip-memory-specs.md`。

## common confusion
- 不要把 gemm pipeline 和 vector-only 路线混用。
- 不要把 `stepK`/L1 staging 误当成单独模式；它们通常是 gemm pipeline 的内部增强。
- 不要把 compute/comm decouple 问题只看这张卡。

## fallback links
- kernel 模式总览 → `wiki/topics/pto-kernel-patterns.md`
- matmul 原语簇 → `cards/primitives/matmul-family.md`
- 容量规格 → `cards/tables/chip-memory-specs.md`
