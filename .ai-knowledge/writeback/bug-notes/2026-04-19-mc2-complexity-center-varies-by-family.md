---
type: bug-note
date: 2026-04-19
env:
  cann: ops-transformer mc2 mainline
  chip: 950 / 910A3 / 910B
  mode: deeper operator-layer mc2 readthrough
tags: [mc2, op_api, op_host, op_kernel, complexity-center]
reliability: verified
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/allto_all_matmul/README.md
    query: "mc2 alltoall matmul README"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/quant_reduce_scatter/op_api/aclnn_quant_reduce_scatter.cpp
    query: "mc2 quant reduce scatter aclnn"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/quant_reduce_scatter/op_kernel/quant_reduce_scatter.cpp
    query: "mc2 quant reduce scatter kernel"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/distribute_barrier/op_api/aclnn_distribute_barrier.cpp
    query: "mc2 distribute barrier aclnn"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/distribute_barrier/op_kernel/distribute_barrier.cpp
    query: "mc2 distribute barrier kernel"
    reliability: medium
---

# MC2 的复杂度不总在 kernel，很多时候在 README 和 op_api

## 问题

读 MC2 时容易默认“最复杂的地方一定在 op_kernel 模板里”。

## 根因

不同家族的复杂度中心并不一样：像 `QuantReduceScatter` 这类量化通信算子，很多关键复杂度在 README 约束和 op_api dtype/group 校验；像 `DistributeBarrier` 这类同步算子，kernel 很薄，真正重要的是通信域独占和使用方式；像 `AlltoAllMatmul` 这类融合算子，README 的通信顺序和布局变换决定了你后面怎么理解 op_host/op_kernel。

## 正确做法

每读一个 MC2 代表算子时，都先判断它的复杂度中心更偏 README、op_api、op_host 还是 op_kernel，再决定阅读重心。

## 反例

不分算子类型，一律把大部分时间花在 kernel 模板分支上，最后反而错过了真正限制使用的入口约束。
