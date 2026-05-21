---
page_type: topic
title: mc2-grouped-quant-alltoallv-latest
status: draft
version: [latest family entry in ops-transformer mc2 mainline]
chip: [950, 910A3, 910B, A2]
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/grouped_mat_mul_all_reduce/README.md
    query: "ops-transformer mc2 grouped matmul all reduce README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/grouped_mat_mul_allto_allv/README.md
    query: "ops-transformer mc2 grouped matmul alltoallv README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/allto_allv_grouped_mat_mul/README.md
    query: "ops-transformer mc2 allto allv grouped matmul README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/allto_allv_quant_grouped_mat_mul/README.md
    query: "ops-transformer mc2 allto allv quant grouped matmul README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/quant_grouped_mat_mul_allto_allv/README.md
    query: "ops-transformer mc2 quant grouped matmul alltoallv README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/quant_reduce_scatter/README.md
    query: "ops-transformer mc2 quant reduce scatter README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
last_updated: 2026-04-19
related:
  - mc2-overview.md
  - ../compare/mc2-collective-triangle.md
---

# MC2 Grouped / Quant / AlltoAllV Latest

## canonical scope

本页是 grouped / quant / alltoallv 家族的 canonical router。它不把一组长文件名当成平权入口，而是把查询归一化到三个主语义：grouped 主体、alltoallv 路由主体、quant 通信主体。

## hit terms

- `grouped_mat_mul_all_reduce`
- `grouped_mat_mul_allto_allv`
- `allto_allv_grouped_mat_mul`
- `allto_allv_quant_grouped_mat_mul`
- `quant_grouped_mat_mul_allto_allv`
- `quant_all_reduce`
- `quant_reduce_scatter`
- `grouped quant alltoallv`

## normalized routing

- 问 grouped 主体与通信融合 → 优先收敛到 `grouped_mat_mul_all_reduce` 或 grouped+alltoallv 主语义簇
- 问 alltoallv 路由如何与 grouped 交织 → 收敛到 grouped+alltoallv 主语义簇
- 问量化通信硬约束 → 优先收敛到 `quant_reduce_scatter`
- 问旧组合名字之间的关系 → 只补背景，不恢复并列入口

## primary anchors

- `ops-transformer/mc2/grouped_mat_mul_all_reduce/README.md`
- `ops-transformer/mc2/grouped_mat_mul_allto_allv/README.md`
- `ops-transformer/mc2/allto_allv_grouped_mat_mul/README.md`
- `ops-transformer/mc2/allto_allv_quant_grouped_mat_mul/README.md`
- `ops-transformer/mc2/quant_grouped_mat_mul_allto_allv/README.md`
- `ops-transformer/mc2/quant_reduce_scatter/README.md`

## canonical answer units

- **grouped + all-reduce 主入口**：`grouped_mat_mul_all_reduce`
- **grouped + alltoallv 主语义簇**：`grouped_mat_mul_allto_allv` / `allto_allv_grouped_mat_mul`
- **quant 通信主入口**：`quant_reduce_scatter`
- **命名规律**：grouped / alltoallv / quant 会继续做高组合度排列，但不应在知识层做平权一级入口

## common confusion

- 不要把不同命名组合都当成一级入口；它们更像同一高组合度家族的不同投影。
- 不要把 `quant_reduce_scatter` 当成普通 grouped 变体；它承载的是量化通信约束主线。
- 不要把 grouped 主体和 alltoallv 主体混成同一个问题。

## escalation path

- MC2 全景与其他家族位置 → `wiki/topics/mc2-overview.md`
- ReduceScatter 在 collective 家族中的位置 → `wiki/compare/mc2-collective-triangle.md`
- repo 当前事实 → 回 README / op_api / op_host

## current status

当前知识层已把 grouped / quant / alltoallv 家族稳定压回三个主语义簇。后续若继续增补，应优先拆 grouped+alltoallv 的 compare/topic，而不是重新并列文件名入口。

## Sources

- `ops-transformer/mc2/grouped_mat_mul_all_reduce/README.md`
- `ops-transformer/mc2/grouped_mat_mul_allto_allv/README.md`
- `ops-transformer/mc2/allto_allv_grouped_mat_mul/README.md`
- `ops-transformer/mc2/allto_allv_quant_grouped_mat_mul/README.md`
- `ops-transformer/mc2/quant_grouped_mat_mul_allto_allv/README.md`
- `ops-transformer/mc2/quant_reduce_scatter/README.md`

## Related

- `wiki/topics/mc2-overview.md`
- `wiki/compare/mc2-collective-triangle.md`
