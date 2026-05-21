---
type: bug-note
date: 2026-04-19
env:
  cann: ops-transformer mc2 mainline
  chip: 950 / 910A3 / 910B
  mode: family-level mc2 readthrough
tags: [mc2, readthrough, family-map]
reliability: verified
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/matmul_all_reduce/README.md
    query: "mc2 matmul all reduce README"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine/README.md
    query: "mc2 moe distribute combine README"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/allto_all_matmul/README.md
    query: "mc2 alltoall matmul README"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/quant_all_reduce/README.md
    query: "mc2 quant all reduce README"
    reliability: medium
---

# MC2 不是单个算子，而是多家族并存的产品面

## 问题

如果把 `ops-transformer/mc2` 当成“一个技术点”，很容易在阅读时反复切换视角，难以形成稳定总结。

## 根因

主线目录本身就按多种算子家族组织：MoE distribute、Matmul+通信、Grouped/Quant/AlltoAllV、转换/屏障。不同家族的 README 关注点差异很大：有的强调配套路径，有的强调 dtype/量化矩阵，有的强调接口废弃与直调限制。

## 正确做法

先建立家族地图，再在每个家族中挑代表算子做 README → op_api → op_host → op_kernel 四层细读。

## 反例

把目录下所有算子等价看待，逐文件机械通读，最后只得到零散接口印象，没有形成家族级知识结构。
