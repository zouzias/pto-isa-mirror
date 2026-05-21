---
type: bug-note
date: 2026-04-19
env:
  cann: ops-transformer mc2 mainline
  chip: 950 / 910A3 / 910B
  mode: family-level mc2 readthrough batch 3
 tags: [mc2, README, constraints, communication-domains]
reliability: verified
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/all_gather_matmul/README.md
    query: "mc2 all gather matmul README"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/allto_all_all_gather_batch_mat_mul/README.md
    query: "mc2 alltoall allgather batch matmul README"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/grouped_mat_mul_allto_allv/README.md
    query: "mc2 grouped matmul alltoallv README"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/quant_reduce_scatter/README.md
    query: "mc2 quant reduce scatter README"
    reliability: medium
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/distribute_barrier/README.md
    query: "mc2 distribute barrier README"
    reliability: medium
---

# MC2 README 先看通信域独占与平台支持，再决定是否深入 kernel

## 问题

在 MC2 主线里，如果一开始只按算子名字理解，容易忽略一些决定性能和可用性的“硬前提”。

## 根因

多份 README 都把关键限制写在约束段，而不是写在 kernel 里：例如通信域独占、不能并发、平台仅支持 A3 或仅支持 950、HCCL_BUFFSIZE 约束、某些接口后续废弃等。这些信息比 kernel 细节更早决定“能不能用、该不该继续看”。

## 正确做法

读每个 MC2 家族时，先扫 README 的产品支持、约束说明、调用说明，再决定是否值得继续下沉到 op_api/op_host/op_kernel。

## 反例

还没确认算子是否支持当前平台、是否要求独占通信域，就先去分析 kernel 模板分支。
