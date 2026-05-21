---
name: card-pattern-flashattention-stages
description: |
  触发：查找 FlashAttention stages 相关 PTO 模式、代表 kernel、容量线索或常见误区时。
  Trigger: Route to the FlashAttention stages PTO pattern, representative kernels, capacity cues, or common pitfalls.
card_type: pattern
version: repo-current
chip: [950 A5, 910A3]
source: example
last_updated: 2026-04-20
---

# FlashAttention stages

## hit terms
- `flashattention`
- `QK`
- `softmax`
- `PV`
- `GU`
- `pto_macro_matmul`
- `pto_macro_fa_softmax`

## canonical intent
- 把 FlashAttention 中的 PTO 多阶段协同归一化到一个模式簇
- 识别 cube/vector、fifo、running stats 相关问题

## primary anchors
- `kernels/manual/common/flash_atten/fa_performance_kernel.cpp`
- `kernels/manual/common/flash_atten/pto_macro_matmul.hpp`
- `kernels/manual/common/flash_atten/pto_macro_fa_softmax.hpp`
- `kernels/manual/common/flash_atten/pto_macro_fa_gu.hpp`
- `kernels/manual/a5/flash_atten/fa_performance_dn_kernel.cpp`

## common confusion
- 不要把 FlashAttention 拆成独立无关的 QK/softmax/PV 小问题；在 PTO 层它们是阶段链。
- 不要把它和普通 gemm pipeline 混成同一复杂度层。
- 不要先从 testcase 找 FlashAttention；主事实锚点在 kernels/manual。

## fallback links
- kernel 模式总览 → `wiki/topics/pto-kernel-patterns.md`
- matmul 原语簇 → `cards/primitives/matmul-family.md`
- vec 原语簇 → `cards/primitives/vec-elementwise-and-reduce.md`
