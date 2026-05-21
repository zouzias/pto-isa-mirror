---
page_type: compare
title: mc2-dispatch-combine-platform-splits
status: draft
version: [ops-transformer mc2 mainline]
chip: [950, 910A3, 910B]
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v2.cpp
    query: "dispatch v2 aclnn entry"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/op_host/moe_distribute_dispatch_v2_def.cpp
    query: "dispatch v2 op host"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.cpp
    query: "dispatch v2 op kernel"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/op_api/aclnn_moe_distribute_combine_v2.cpp
    query: "combine v2 aclnn entry"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/op_host/moe_distribute_combine_v2_def.cpp
    query: "combine v2 op host"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2.cpp
    query: "combine v2 op kernel"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
last_updated: 2026-04-19
related:
  - mc2-dispatch-combine-closure.md
  - mc2-dispatch-combine-tiling-decisions.md
  - ../topics/mc2-moe-dispatch-combine-latest.md
---

# DispatchV2 / CombineV2 Platform Splits

## canonical scope

本页是 dispatch_v2 / combine_v2 平台分流问题的 canonical router。它不回答“默认看哪版”，也不回答“host/base/helper/tiling 决策链如何工作”；它只回答同样是 dispatch_v2 / combine_v2，A2、A3、950 的复杂度中心分别落在哪。

## hit terms

- `A2 A3 950`
- `platform split`
- `fullmesh`
- `hierarchy`
- `dataplane`
- `AICPU`
- `AIV`
- `A5`
- `arch35`

## normalized routing

- 问“A2 / A3 / 950 到底差在哪” → 本页直接回答
- 问“host/base/helper/tiling 三层怎么决策” → 转 `mc2-dispatch-combine-tiling-decisions`
- 问“为什么 dispatch/combine 必须一起看” → 转 `mc2-dispatch-combine-closure`
- 问“现在默认看哪版” → 转 `mc2-moe-dispatch-combine-latest`

## primary anchors

- `ops-transformer/mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v2.cpp`
- `ops-transformer/mc2/moe_distribute_dispatch_v2/op_host/moe_distribute_dispatch_v2_def.cpp`
- `ops-transformer/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.cpp`
- `ops-transformer/mc2/moe_distribute_combine_v2/op_api/aclnn_moe_distribute_combine_v2.cpp`
- `ops-transformer/mc2/moe_distribute_combine_v2/op_host/moe_distribute_combine_v2_def.cpp`
- `ops-transformer/mc2/moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2.cpp`

## canonical answer units

- **950**：A5 / arch35 专线，不应先套用 A3 心智模型
- **A3**：复杂度中心在 `FullMesh / Hierarchy / HasTp`
- **A2**：复杂度中心在 `DataplaneMode / LayeredMode / AICPU/AIV`
- **共同点**：真正的平台差异主要不在最外层 API，而在 host/kernel 下钻后的分流轴

## common confusion

- 不要把 A2、A3、950 理解成“同一套逻辑 + 少量 ifdef”。
- 不要先看 dtype 支持矩阵就下结论；真正的分流中心在拓扑/dataplane/专线模板。
- 不要把 950 误判成 A3 的小变体。

## escalation path

- host/base/helper/tiling 决策链 → `wiki/compare/mc2-dispatch-combine-tiling-decisions.md`
- 闭环协议语义 → `wiki/compare/mc2-dispatch-combine-closure.md`
- 默认入口与参数分组 → `wiki/topics/mc2-moe-dispatch-combine-latest.md`

## current status

当前知识层已把平台差异收敛为三条稳定分流轴：A2 看 dataplane，A3 看拓扑，950 看 A5 专线。后续增补应继续沿这三条轴，而不是把平台问题重新混成“大一统实现”。

## Sources

- `ops-transformer/mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v2.cpp`
- `ops-transformer/mc2/moe_distribute_dispatch_v2/op_host/moe_distribute_dispatch_v2_def.cpp`
- `ops-transformer/mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.cpp`
- `ops-transformer/mc2/moe_distribute_combine_v2/op_api/aclnn_moe_distribute_combine_v2.cpp`
- `ops-transformer/mc2/moe_distribute_combine_v2/op_host/moe_distribute_combine_v2_def.cpp`
- `ops-transformer/mc2/moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2.cpp`
