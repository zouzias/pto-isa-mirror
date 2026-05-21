---
page_type: compare
title: mc2-dispatch-combine-closure
status: draft
version: [ops-transformer mc2 mainline]
chip: [950, 910A3, 910B, A2]
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch/README.md
    query: "ops-transformer mc2 moe distribute dispatch README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine/README.md
    query: "ops-transformer mc2 moe distribute combine README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine/op_api/aclnn_moe_distribute_combine.cpp
    query: "ops-transformer mc2 moe distribute combine aclnn"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine/op_host/moe_distribute_combine_def.cpp
    query: "ops-transformer mc2 moe distribute combine op host"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/README.md
    query: "ops-transformer mc2 moe distribute dispatch v2 README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/README.md
    query: "ops-transformer mc2 moe distribute combine v2 README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
last_updated: 2026-04-19
related:
  - ../topics/mc2-overview.md
  - ../topics/mc2-moe-dispatch-combine-latest.md
  - mc2-dispatch-combine-platform-splits.md
  - mc2-dispatch-combine-tiling-decisions.md
---

# Dispatch / Combine Closure in MC2

## canonical scope

本页是 dispatch/combine 闭环协议问题的 canonical router。它不回答“最新版入口是哪一版”，而是回答“为什么 dispatch 和 combine 必须被当成同一条闭环协议来读”。

## hit terms

- `dispatch combine closure`
- `expandIdx`
- `assistInfoForCombine`
- `epRecvCounts`
- `tpRecvCounts`
- `expandScales`
- `sharedExpertXOptional`
- `commAlg`

## normalized routing

- 问“最新版默认看哪版” → 转 `wiki/topics/mc2-moe-dispatch-combine-latest.md`
- 问“dispatch 和 combine 为什么不能拆开读” → 本页直接回答
- 问“哪些张量应被视为协议字段而不是业务结果” → 本页直接回答
- 问“平台内部怎么分流” → 转平台分流页，不在本页细展开

## primary anchors

- `ops-transformer/mc2/moe_distribute_dispatch/README.md`
- `ops-transformer/mc2/moe_distribute_combine/README.md`
- `ops-transformer/mc2/moe_distribute_combine/op_api/aclnn_moe_distribute_combine.cpp`
- `ops-transformer/mc2/moe_distribute_dispatch_v2/README.md`
- `ops-transformer/mc2/moe_distribute_combine_v2/README.md`

## canonical answer units

- `Dispatch` 和 `Combine` 是发散-返还闭环，不是前后处理的松散组合
- `expandIdx` / `assistInfoForCombine` / counts / scales 应优先当协议字段看
- V2 的核心意义在于把闭环所需同步信息显式化，而不只是接口变长
- shared expert 和 `commAlg` 说明闭环边界正在从隐式环境转成显式接口责任

## common confusion

- 不要把 `Combine` 当普通后处理；它是按 dispatch 路由账本原路返还。
- 不要把 `expandIdx`、`epRecvCounts` 这类张量当业务可解释结果。
- 不要把本页当最新版入口页；它讨论的是协议闭环，不是版本选择。

## escalation path

- 默认入口与参数分组 → `wiki/topics/mc2-moe-dispatch-combine-latest.md`
- 平台分流中心 → `wiki/compare/mc2-dispatch-combine-platform-splits.md`
- host 决策链 → `wiki/compare/mc2-dispatch-combine-tiling-decisions.md`

## current status

当前知识层已把 dispatch/combine 稳定压成“入口页 + 闭环页”的双页结构：入口页回答默认看哪版，闭环页回答为什么必须一起读。后续增补应继续维持这一边界。

## Sources

- `ops-transformer/mc2/moe_distribute_dispatch/README.md`
- `ops-transformer/mc2/moe_distribute_combine/README.md`
- `ops-transformer/mc2/moe_distribute_combine/op_api/aclnn_moe_distribute_combine.cpp`
- `ops-transformer/mc2/moe_distribute_combine/op_host/moe_distribute_combine_def.cpp`
- `ops-transformer/mc2/moe_distribute_dispatch_v2/README.md`
- `ops-transformer/mc2/moe_distribute_combine_v2/README.md`
