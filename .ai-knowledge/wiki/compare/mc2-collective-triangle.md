---
page_type: compare
title: mc2-collective-triangle
status: draft
version: [latest family entry in ops-transformer mc2 mainline]
chip: [950, 910A3, 910B, A2]
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/all_gather_matmul_v2/README.md
    query: "ops-transformer mc2 all gather matmul v2 README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/allto_all_matmul/README.md
    query: "ops-transformer mc2 allto all matmul README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/matmul_reduce_scatter_v2/op_api/aclnn_matmul_reduce_scatter_v2.cpp
    query: "ops-transformer mc2 matmul reduce scatter v2 aclnn"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/matmul_reduce_scatter_v2/op_host/matmul_reduce_scatter_v2_def.cpp
    query: "ops-transformer mc2 matmul reduce scatter v2 op host"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/quant_reduce_scatter/README.md
    query: "ops-transformer mc2 quant reduce scatter README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/quant_reduce_scatter/op_api/aclnn_quant_reduce_scatter.cpp
    query: "ops-transformer mc2 quant reduce scatter aclnn"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
last_updated: 2026-04-19
related:
  - mc2-collective-positioning.md
  - ../topics/mc2-overview.md
---

# Latest AllGather / AlltoAll / ReduceScatter in MC2

## canonical scope

本页是 collective 三角定位问题的 canonical router。它把 AllGather、AlltoAll、ReduceScatter 三个最新版主入口放到同一张图里比较，但不展开某一边的内部实现细节。

## hit terms

- `AllGather`
- `AlltoAll`
- `ReduceScatter`
- `AllGatherMatmulV2`
- `AlltoAllMatmul`
- `MatmulReduceScatterV2`
- `quant_reduce_scatter`

## normalized routing

- 问“输入先补齐再算” → `AllGatherMatmulV2`
- 问“先换路由和布局再算” → `AlltoAllMatmul`
- 问“结果先算完再规约切回去” → `MatmulReduceScatterV2`
- 问“量化版 ReduceScatter 为什么限制这么强” → 本页直接回答 `quant_reduce_scatter` 的通信域治理约束
- 问“两条边的局部差异” → 转 collective-positioning 或对应家族页

## primary anchors

- `ops-transformer/mc2/all_gather_matmul_v2/README.md`
- `ops-transformer/mc2/allto_all_matmul/README.md`
- `ops-transformer/mc2/matmul_reduce_scatter_v2/op_api/aclnn_matmul_reduce_scatter_v2.cpp`
- `ops-transformer/mc2/quant_reduce_scatter/README.md`

## canonical answer units

- AllGather = 输入补全
- AlltoAll = 路由+布局重建
- ReduceScatter = 结果回切
- `quant_reduce_scatter` = 不只是量化参数变化，还带通信域治理约束

## common confusion

- 不要把三者都压成“collective + matmul”。
- 不要把 ReduceScatter 理解成输入补全；它更偏结果回切。
- 不要把 `quant_reduce_scatter` 当普通 ReduceScatter 小变体；它有更强的通信域前提。

## escalation path

- AllGather vs AlltoAll 二元语义定位 → `wiki/compare/mc2-collective-positioning.md`
- MC2 总主题入口 → `wiki/topics/mc2-overview.md`
- repo 当前实现 → 回 README / op_api / op_host

## current status

当前知识层已把 collective 家族稳定压成“三角页 + 二元定位页”的双页结构：三角页给三者全景，定位页给 AllGather/AlltoAll 的局部压缩。后续增补应继续维持这一边界。

## Sources

- `ops-transformer/mc2/all_gather_matmul_v2/README.md`
- `ops-transformer/mc2/allto_all_matmul/README.md`
- `ops-transformer/mc2/matmul_reduce_scatter_v2/op_api/aclnn_matmul_reduce_scatter_v2.cpp`
- `ops-transformer/mc2/matmul_reduce_scatter_v2/op_host/matmul_reduce_scatter_v2_def.cpp`
- `ops-transformer/mc2/quant_reduce_scatter/README.md`
- `ops-transformer/mc2/quant_reduce_scatter/op_api/aclnn_quant_reduce_scatter.cpp`

## Related

- `wiki/compare/mc2-collective-positioning.md`
- `wiki/topics/mc2-overview.md`
