---
page_type: compare
title: mc2-vs-shmem
status: draft
version: [generic, ops-transformer mc2 mainline, experimental mc2 shmem examples]
chip: [950, 910A3, 910B]
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch/README.md
    query: "ops-transformer mc2 mainline dispatch README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch/op_api/aclnn_moe_distribute_dispatch.cpp
    query: "ops-transformer mc2 mainline dispatch aclnn"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/experimental/mc2/moe_distribute_dispatch_shmem/README.md
    query: "ops-transformer experimental shmem README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
last_updated: 2026-04-19
related:
  - ../topics/mc2-overview.md
---

# MC2 vs SHMEM

## canonical scope

本页是“MC2 主线 vs experimental SHMEM 路线”问题的 canonical router。它不直接给性能输赢结论，而是固定比较维度，避免把主线 MC2 与实验性 SHMEM 混成同一类东西。

## hit terms

- `mc2 vs shmem`
- `experimental shmem`
- `mainline mc2`
- `productized layering`
- `shared memory path`
- `aclnn`

## normalized routing

- 问“MC2 主线和 SHMEM 的总体差异” → 本页直接回答
- 问“MC2 主线内部应该怎么分层看” → 转 `mc2-overview` / `mc2-layering`
- 问“某条 experimental shmem 代码细节” → 回 experimental 路线代码，不在本页展开

## primary anchors

- `ops-transformer/mc2/moe_distribute_dispatch/README.md`
- `ops-transformer/mc2/moe_distribute_dispatch/op_api/aclnn_moe_distribute_dispatch.cpp`
- `ops-transformer/experimental/mc2/moe_distribute_dispatch_shmem/README.md`

## canonical answer units

- **MC2 主线**：更产品化，具备 README / op_api / op_host / op_kernel / common 分层
- **SHMEM 路线**：更实验性，更直接暴露共享内存/上下文/接入细节
- **可控性差异**：MC2 把复杂度拆层，SHMEM 把更多底层细节直接暴露
- **维护差异**：MC2 更适合统一接入和长期维护，SHMEM 更适合局部实验和链路验证
- **调试差异**：MC2 更偏分层排障，SHMEM 更偏链路排障

## common confusion

- 不要把主线 MC2 和 experimental SHMEM 混成同一成熟度层级。
- 不要从单个 kernel 文件推出两条路线的总体优劣。
- 不要在本页下“拍板”性能结论；没有具体链路与样本时，本页只给比较框架。

## escalation path

- MC2 主线内部路由 → `wiki/topics/mc2-overview.md`
- repo 当前实现与实验代码 → 回 `ops-transformer/mc2` 或 `ops-transformer/experimental/mc2`

## current status

当前知识层已把 MC2 vs SHMEM 压成一组稳定比较维度：抽象层次、可控性、维护成本、调试方式、适用场景。后续若继续增补，应补证据样本，而不是先给绝对结论。

## Sources

- `ops-transformer/mc2/moe_distribute_dispatch/README.md`
- `ops-transformer/mc2/moe_distribute_dispatch/op_api/aclnn_moe_distribute_dispatch.cpp`
- `ops-transformer/experimental/mc2/moe_distribute_dispatch_shmem/README.md`
