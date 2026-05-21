---
page_type: compare
title: mc2-collective-positioning
title_alias: [latest-allgather-matmul-vs-alltoall-matmul]
status: draft
version: [latest family entry in ops-transformer mc2 mainline]
chip: [950, 910A3, 910B, A2]
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/all_gather_matmul_v2/README.md
    query: "ops-transformer mc2 all gather matmul v2 README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/all_gather_matmul_v2/op_api/aclnn_all_gather_matmul_v2.cpp
    query: "ops-transformer mc2 all gather matmul v2 aclnn"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/all_gather_matmul_v2/op_host/all_gather_matmul_v2_def.cpp
    query: "ops-transformer mc2 all gather matmul v2 op host"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/allto_all_matmul/README.md
    query: "ops-transformer mc2 allto all matmul README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/allto_all_matmul/op_api/aclnn_allto_all_matmul.cpp
    query: "ops-transformer mc2 allto all matmul aclnn"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/allto_all_matmul/op_host/allto_all_matmul_def.cpp
    query: "ops-transformer mc2 allto all matmul op host"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
last_updated: 2026-04-19
related:
  - ../topics/mc2-overview.md
  - mc2-collective-triangle.md
  - mc2-vs-shmem.md
---

# Latest AllGatherMatmul vs AlltoAllMatmul

## canonical scope

本页是 AllGatherMatmulV2 与 AlltoAllMatmul 语义定位问题的 canonical router。它不负责给出 collective 三角全景，也不负责讲历史版本差异；它只回答“输入补全”和“路由+布局重建”这两种主语义该如何区分。

## hit terms

- `AllGatherMatmulV2`
- `AlltoAllMatmul`
- `gatherOut`
- `alltoall_out`
- `permute view`
- `all2all_axes`
- `world_size`

## normalized routing

- 问“先把输入补齐再做 MatMul” → 收敛到 `AllGatherMatmulV2`
- 问“通信后还要重排布局才能算” → 收敛到 `AlltoAllMatmul`
- 问“把 AllGather / AlltoAll / ReduceScatter 放到同一张图里比较” → 转 `mc2-collective-triangle`
- 问“旧版 AllGatherMatmul 怎么办” → 只补背景，不恢复旧入口

## primary anchors

- `ops-transformer/mc2/all_gather_matmul_v2/README.md`
- `ops-transformer/mc2/all_gather_matmul_v2/op_api/aclnn_all_gather_matmul_v2.cpp`
- `ops-transformer/mc2/allto_all_matmul/README.md`
- `ops-transformer/mc2/allto_all_matmul/op_api/aclnn_allto_all_matmul.cpp`

## canonical answer units

- `AllGatherMatmulV2` 的主语义是输入补全后再算
- `AlltoAllMatmul` 的主语义是换路由并重排布局后再算
- `gather_out` 是稳定中间结果，而 `alltoall_out` 更接近布局化后的中间结果
- 两者不是“collective 前缀不同”这么简单，而是通信结果在公式链中的位置不同

## common confusion

- 不要把 `AllGatherMatmulV2` 与 `AlltoAllMatmul` 都压成“通信 + MatMul”。
- 不要把 `alltoall_out` 当成原始通信返回值；它已经带了布局重建语义。
- 不要在本页展开 ReduceScatter；那应去 collective triangle 页。

## escalation path

- collective 三角全景 → `wiki/compare/mc2-collective-triangle.md`
- MC2 主题总览 → `wiki/topics/mc2-overview.md`
- repo 当前事实 → 回 README / op_api / op_host

## current status

当前知识层已把 AllGather 与 AlltoAll 的对比稳定压成“输入补全 vs 路由重建”这组 canonical answer unit。后续增补应继续维持这一压缩，而不是恢复按旧版本或按文件名分散叙事。

## Sources

- `ops-transformer/mc2/all_gather_matmul_v2/README.md`
- `ops-transformer/mc2/all_gather_matmul_v2/op_api/aclnn_all_gather_matmul_v2.cpp`
- `ops-transformer/mc2/all_gather_matmul_v2/op_host/all_gather_matmul_v2_def.cpp`
- `ops-transformer/mc2/allto_all_matmul/README.md`
- `ops-transformer/mc2/allto_all_matmul/op_api/aclnn_allto_all_matmul.cpp`
- `ops-transformer/mc2/allto_all_matmul/op_host/allto_all_matmul_def.cpp`
