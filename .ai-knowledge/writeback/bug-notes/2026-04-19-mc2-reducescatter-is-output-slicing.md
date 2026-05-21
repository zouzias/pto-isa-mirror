---
type: bug-note
date: 2026-04-19
env:
  source_repo: /home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2
  focus: MC2 collective semantics
  chip: [950, 910A3, 910B, A2]
tags: [mc2, reducescatter, collective, matmul, quant]
reliability: verified
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/matmul_reduce_scatter/README.md
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/matmul_reduce_scatter/op_api/aclnn_matmul_reduce_scatter.cpp
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/quant_reduce_scatter/README.md
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/quant_reduce_scatter/op_api/aclnn_quant_reduce_scatter.cpp
    reliability: high
---

# ReduceScatter 在 MC2 里更像“结果回切”，不是输入补全或布局重排

## 问题

在读 MC2 的 collective 家族时，很容易把 `AllGather`、`AlltoAll`、`ReduceScatter` 都看成只是通信原语不同。

## 根因

`MatmulReduceScatter` 的 README 明确把公式写成 `ReduceScatter(x1 @ x2 + bias)`，并说明“先计算后通信”。这和 `AllGatherMatmulV2` 的“先把左矩阵收齐”，以及 `AlltoAllMatmul` 的“先换路由再 permute/view”是不同的。

`QuantReduceScatter` 进一步证明了这点：它的核心是量化数据与 scale 经过通信后做 reduce，并且直接把通信域独占、`HCCL_BUFFSIZE` 和 world size 限制提升成接口前提。

## 正确做法

- 读 ReduceScatter 家族时，先问“完整结果在什么地方形成、在哪里被切回 rank 局部结果”。
- 对基础 `MatmulReduceScatter`，重点看输出 shape 缩回、平台分流，以及是否转到 V2 路径。
- 对 `QuantReduceScatter`，额外先看通信域约束和 `HCCL_BUFFSIZE` 预算，不要只盯 dtype/scale。

## 反例

- 把 ReduceScatter 也理解成“先补齐输入再算”。
- 用 AlltoAll 的布局重排心智模型去读 ReduceScatter。

## 参考来源

见 frontmatter `sources`。
