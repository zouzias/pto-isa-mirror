---
type: bug-note
date: 2026-04-19
env:
  cann: ops-transformer mc2 mainline
  chip: 950 / 910A3 / 910B
  mode: deeper op-layer mc2 readthrough batch 2
tags: [mc2, shape-check, grouped, composite-pipeline]
reliability: verified
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/all_gather_matmul/op_api/aclnn_all_gather_matmul.cpp
    query: "mc2 all gather matmul aclnn"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/allto_all_all_gather_batch_mat_mul/op_api/aclnn_all_to_all_all_gather_batch_matmul.cpp
    query: "mc2 alltoall allgather batch matmul aclnn"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/grouped_mat_mul_allto_allv/op_api/aclnn_grouped_mat_mul_allto_allv.cpp
    query: "mc2 grouped matmul alltoallv aclnn"
    reliability: medium
---

# 复合通算融合算子的第一复杂点常在 shape 关系和参数组合，而不是 kernel

## 问题

读复合通算融合算子时，很容易被 kernel 名字或模板参数吸引，误以为主要复杂度在 device 侧。

## 根因

像 `AlltoAllAllGatherBatchMatMul`、`GroupedMatMulAlltoAllv` 这类算子，op_api 层已经承担了大量结构性复杂度：shape 代数关系、`ep/tp` 维度约束、可选输出 shape 一致性、成组入参同时为空/同时存在的规则。这些如果没先理解，后面看 kernel 也会失真。

## 正确做法

遇到复合通信+计算算子时，先把 op_api 中的 shape/参数组合规则吃透，再去看 op_host 注册，最后才看 kernel 分派与实现细节。

## 反例

跳过 op_api 的 shape 校验逻辑，直接根据 README 公式或 kernel 名称猜输入输出关系。
