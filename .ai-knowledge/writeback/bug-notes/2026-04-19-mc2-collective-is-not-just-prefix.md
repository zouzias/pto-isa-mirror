---
type: bug-note
date: 2026-04-19
env:
  source_repo: /home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2
  focus: MC2 collective semantics
  chip: [950, 910A3, 910B, A2]
tags: [mc2, allgather, alltoall, matmul, read-path]
reliability: verified
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/all_gather_matmul_v2/README.md
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/all_gather_matmul_v2/op_api/aclnn_all_gather_matmul_v2.cpp
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/allto_all_matmul/README.md
    reliability: high
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/allto_all_matmul/op_api/aclnn_allto_all_matmul.cpp
    reliability: high
---

# 不要把 AllGatherMatmulV2 和 AlltoAllMatmul 都读成“通信加 MatMul”

## 问题

初看 MC2 主线时，很容易把 `AllGatherMatmulV2` 和 `AlltoAllMatmul` 都归成同一种结构：前面做一次通信，后面接一次 MatMul。

## 根因

两者在 README 和接口层对“通信结果”的建模完全不同：
- `AllGatherMatmulV2` 把 `AllGather(x1)` 直接作为 `gatherOut` 暴露出来，说明通信后的完整左矩阵本身就是稳定中间结果。
- `AlltoAllMatmul` 则把 `AlltoAll -> permute -> view` 连成一个语义链，说明通信后的原始结果还不够，必须经过布局重建后才是 MatMul 真正消费的对象。

## 正确做法

- 读 `AllGatherMatmulV2` 时，重点看 gather 后 shape 扩张、`gather_out` 的语义、量化 scale 如何随 gather 对齐。
- 读 `AlltoAllMatmul` 时，重点看 `all2all_axes`、`world_size`、permute/view、以及通信后布局如何变成 MatMul 左矩阵。
- 把这两类算子的差异先理解成“通信结果是否天然可直接消费”，不要先陷入 kernel 实现细节。

## 反例

- 看到名字里都有 MatMul，就用同一阅读模板处理。
- 把 `alltoall_out` 误当成 `AllGatherMatmulV2` 里的 `gather_out` 同类中间结果。

## 参考来源

见 frontmatter `sources`。
