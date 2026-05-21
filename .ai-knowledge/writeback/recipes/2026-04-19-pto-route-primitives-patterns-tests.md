---
type: recipe
date: 2026-04-19
env:
  cann: pto-isa repository current layout
  chip: A2 / A3 / 910A3 / 910B / A5 / 950
tags: [pto, knowledge-base, routing, patterns, tests]
reliability: verified
sources:
  - origin: code:include/pto/pto-inst.hpp
    query: "pto unified include entry"
    reliability: high
  - origin: code:kernels/manual/a2a3/topk/topk_kernel.cpp
    query: "pto topk vector pipeline"
    reliability: high
  - origin: code:tests/npu/a5/src/st/testcase/tadd/main.cpp
    query: "pto minimal testcase"
    reliability: high
---

# 先按 原语 → 模式 → 示例 → 验证 的顺序纳管 PTO 知识

## 适用场景

需要把 PTO 相关代码系统性纳入 `.ai-knowledge`，又不想把 `docs/isa`、kernel README、testcase 代码原样复制进知识库时。

## 做法

1. **先建体系页**：先写 PTO 总览、include 分层、kernel 模式页、tests 入口页，固定知识地图。
2. **再建映射表**：把原语族、模式、testcase 收成最小 map，而不是一开始就铺大量细页。
3. **模式优先于单原语**：优先沉淀 gemm pipeline、topk vector、flashattention、compute/comm decouple、A5 特化这类稳定模式。
4. **示例优先于目录遍历**：优先收高价值 testcase，如 `tadd`、`tpushpop_cv`、`tquant`、`ttrans_conv`、`tmatmul`、`tgather`。
5. **最后补原语簇卡**：按 load/store、manual-binding/sync、vec、matmul、comm 这样的原语簇补卡，而不是按单指令平铺。

## 边界

- 不复制 `docs/isa/*.md` 的单指令正文。
- 不复制 `kernels/manual/**/README.md` 的长段背景。
- 不把 testcase 代码大段粘进知识页，只保留路径、主题和阅读顺序。
- repo 当前事实最终仍以代码为准，知识库只做路由与理解骨架。
