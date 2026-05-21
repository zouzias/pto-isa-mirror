---
name: card-pattern-topk-vector-pipeline
description: |
  触发：查找 TopK vector pipeline 相关 PTO 模式、代表 kernel、容量线索或常见误区时。
  Trigger: Route to the TopK vector pipeline PTO pattern, representative kernels, capacity cues, or common pitfalls.
card_type: pattern
version: repo-current
chip: [950 A5, 910A3]
source: example
last_updated: 2026-04-20
---

# TopK vector pipeline

## hit terms
- `topk`
- `TSORT32`
- `TMRGSORT`
- `TGATHER`
- `vector pipeline`

## canonical intent
- 把 TopK 类排序/merge/gather 路线归一化到 PTO vector-only 模式

## primary anchors
- `kernels/manual/a2a3/topk/topk_kernel.cpp`

## common confusion
- 不要把 TopK 路线混成一般 elementwise/reduce；它有明显的 sort/merge/gather 主线。
- 不要把 `TGATHER` 自动路由到通信原语页。

## fallback links
- vec 原语簇 → `cards/primitives/vec-elementwise-and-reduce.md`
- kernel 模式总览 → `wiki/topics/pto-kernel-patterns.md`
