---
name: card-pattern-a5-mx-simt-sync
description: |
  触发：查找 A5 MX / SIMT / sync 相关 PTO 模式、代表 kernel、容量线索或常见误区时。
  Trigger: Route to the A5 MX / SIMT / sync PTO pattern, representative kernels, capacity cues, or common pitfalls.
card_type: pattern
version: repo-current
chip: [950 A5]
source: example
last_updated: 2026-04-20
---

# A5 MX / SIMT / sync

## hit terms
- `A5 MX`
- `TMATMUL_MX`
- `SIMT`
- `engram_simt`
- `mxmatmul`
- `A5 sync`

## canonical intent
- 把 A5 平台上的 MX matmul、SIMT 对照、增强同步/流式策略归一化到同一模式簇

## primary anchors
- `kernels/manual/a5/matmul_mxfp4_performance/mxmatmul_performance_kernel.cpp`
- `kernels/manual/a5/matmul_mxfp8_performance/mxmatmul_performance_kernel.cpp`
- `kernels/manual/a5/engram_simt/engram-simt_kernel.cpp`
- `kernels/manual/a5/flash_atten/fa_performance_dn_kernel.cpp`

## capacity linkage
- 这类模式强依赖 A5 比 A2A3 更宽的 `UB` 与 `L0C`。
- `ScaleLeft / ScaleRight / FBuffer` 也是 A5 特化路径的重要预算来源。
- 若问题涉及 MX scale tile、accumulator 预算或 A5 才能放下的 buffering 方案，优先回 `cards/tables/chip-memory-specs.md` 和 `wiki/chips/950A5.md`。

## common confusion
- 不要把 `TMATMUL_MX` 当成通用 PTO matmul；它更偏 A5 特化。
- 不要把 SIMT 对照问题路由到普通 gemm pipeline。
- 不要把 A5 特化同步写法直接套回 A2A3。

## fallback links
- matmul 原语簇 → `cards/primitives/matmul-family.md`
- kernel 模式总览 → `wiki/topics/pto-kernel-patterns.md`
- A5 容量规格 → `wiki/chips/950A5.md`
