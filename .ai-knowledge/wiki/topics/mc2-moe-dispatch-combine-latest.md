---
page_type: topic
title: mc2-moe-dispatch-combine-latest
status: draft
version: [DispatchV4, CombineV4]
chip: [950PR, 950DT, 910A3, 910B]
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV4.md
    query: "dispatch v4 doc"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/docs/aclnnMoeDistributeCombineV4.md
    query: "combine v4 doc"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v4.cpp
    query: "dispatch v4 op api"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/op_api/aclnn_moe_distribute_combine_v4.cpp
    query: "combine v4 op api"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/op_api/moe_distribute_dispatch_v2_base.cpp
    query: "dispatch base"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/op_api/moe_distribute_combine_v2_base.cpp
    query: "combine base"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
last_updated: 2026-04-19
related:
  - ../compare/mc2-dispatch-combine-closure.md
  - ../compare/mc2-dispatch-combine-platform-splits.md
  - ../compare/mc2-dispatch-combine-tiling-decisions.md
---

# MC2 Moe Dispatch/Combine Latest

## canonical scope

本页是 Moe distribute dispatch/combine 家族的 canonical router。默认主入口固定为 `DispatchV4 + CombineV4`；V2/V3 只作为背景，不再作为平权入口。本页回答“现在默认看哪版、协议主线是什么、为什么接口最新版下面还能看到旧实现族谱”。

## hit terms

- `DispatchV4`
- `CombineV4`
- `aclnnMoeDistributeDispatchV4`
- `aclnnMoeDistributeCombineV4`
- `performanceInfoOptional`
- `assistInfoForCombine`
- `expandXOut`
- `V4 vs V2`
- `dispatch combine latest`

## normalized routing

- 问“最新版 dispatch/combine 看哪版” → 直接收敛到 `DispatchV4 + CombineV4`
- 问“dispatch/combine 为什么必须配套理解” → 转 `wiki/compare/mc2-dispatch-combine-closure.md`
- 问“不同平台为什么内部路径不同” → 转 `wiki/compare/mc2-dispatch-combine-platform-splits.md`
- 问“host 怎么决定实际分支” → 转 `wiki/compare/mc2-dispatch-combine-tiling-decisions.md`
- 问“为什么 V4 下面还能看到 V2” → 本页直接回答：接口演进叠加在旧实现族谱之上

## primary anchors

- `ops-transformer/mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV4.md`
- `ops-transformer/mc2/moe_distribute_combine_v2/docs/aclnnMoeDistributeCombineV4.md`
- `ops-transformer/mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v4.cpp`
- `ops-transformer/mc2/moe_distribute_combine_v2/op_api/aclnn_moe_distribute_combine_v4.cpp`

## canonical answer units

- **最新版入口**：`DispatchV4 + CombineV4`
- **协议主线**：dispatch 发散，combine 原路返还并聚合
- **显式新增点**：`performanceInfoOptional`
- **实现事实**：V4 接口层之下仍复用 V2/base/inner 路线
- **平台差异**：入口统一，不等于内部路径统一

## common confusion

- 不要把 `DispatchV4` 理解成“全新实现族谱”；它主要是接口层更新。
- 不要把 dispatch 和 combine 拆成两个独立入口；它们共享协议字段。
- 不要把 `performanceInfoOptional` 当调试附属信息；它是 V4 的正式接口能力。
- 不要把统一入口误解为 A2/A3/950 行为完全一致。

## escalation path

- 闭环协议细节 → `wiki/compare/mc2-dispatch-combine-closure.md`
- 平台分流中心 → `wiki/compare/mc2-dispatch-combine-platform-splits.md`
- host 决策链 → `wiki/compare/mc2-dispatch-combine-tiling-decisions.md`
- repo 当前实现事实 → 回 op_api / op_host / op_kernel

## current status

当前知识层已经把 Moe distribute 家族稳定收敛到 `DispatchV4 + CombineV4` 这一默认主入口。后续若继续增补，应优先补“高频问题的 canonical answer unit”，而不是恢复旧版本并列叙事。

## Sources

- `ops-transformer/mc2/moe_distribute_dispatch_v2/docs/aclnnMoeDistributeDispatchV4.md`
- `ops-transformer/mc2/moe_distribute_combine_v2/docs/aclnnMoeDistributeCombineV4.md`
- `ops-transformer/mc2/moe_distribute_dispatch_v2/op_api/aclnn_moe_distribute_dispatch_v4.cpp`
- `ops-transformer/mc2/moe_distribute_combine_v2/op_api/aclnn_moe_distribute_combine_v4.cpp`
- `ops-transformer/mc2/moe_distribute_dispatch_v2/op_api/moe_distribute_dispatch_v2_base.cpp`
- `ops-transformer/mc2/moe_distribute_combine_v2/op_api/moe_distribute_combine_v2_base.cpp`

## Related

- `wiki/compare/mc2-dispatch-combine-closure.md`
- `wiki/compare/mc2-dispatch-combine-platform-splits.md`
- `wiki/compare/mc2-dispatch-combine-tiling-decisions.md`
