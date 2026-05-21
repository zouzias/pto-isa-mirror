---
page_type: compare
title: mc2-a5-design-to-implementation
status: draft
version: [ops-transformer mc2 mainline, arch35]
chip: [950PR, 950DT]
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/docs/MoeDistributeDispatch-Combine算子设计介绍.md
    query: "dispatch combine design intro"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/op_host/op_tiling/arch35/moe_distribute_dispatch_tiling_arch35.cpp
    query: "dispatch arch35 tiling"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/op_kernel/arch35/moe_distribute_dispatch_arch35.h
    query: "dispatch arch35 kernel"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/op_host/op_tiling/arch35/moe_distribute_combine_tiling_arch35.cpp
    query: "combine arch35 tiling"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_arch35.h
    query: "combine arch35 kernel"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
last_updated: 2026-04-19
related:
  - mc2-dispatch-combine-platform-splits.md
  - mc2-dispatch-combine-tiling-decisions.md
---

# A5 Dispatch/Combine: Design vs Implementation

## canonical scope

本页是 A5/arch35 设计文档到代码落点映射问题的 canonical router。它不讨论默认入口，也不讨论一般平台分流，而是回答“设计文档里的承诺在 arch35 实现里落到了哪里”。

## hit terms

- `A5 design`
- `arch35`
- `device-side autonomy`
- `double buffer`
- `window layout`
- `CCU`
- `AlltoAllvWrite`
- `perRankDataSize_`

## normalized routing

- 问“A5 设计文档里的术语在代码里怎么落地” → 本页直接回答
- 问“平台总体为什么分成 A2/A3/950 三条轴” → 转 `mc2-dispatch-combine-platform-splits`
- 问“host/base/helper/tiling 决策链怎么走” → 转 `mc2-dispatch-combine-tiling-decisions`

## primary anchors

- `ops-transformer/mc2/moe_distribute_dispatch_v2/docs/MoeDistributeDispatch-Combine算子设计介绍.md`
- `ops-transformer/mc2/moe_distribute_dispatch_v2/op_host/op_tiling/arch35/moe_distribute_dispatch_tiling_arch35.cpp`
- `ops-transformer/mc2/moe_distribute_dispatch_v2/op_kernel/arch35/moe_distribute_dispatch_arch35.h`
- `ops-transformer/mc2/moe_distribute_combine_v2/op_host/op_tiling/arch35/moe_distribute_combine_tiling_arch35.cpp`
- `ops-transformer/mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_arch35.h`

## canonical answer units

- **设备侧自治** → `CCU` server type、A5 专用 kernel 路线
- **双缓冲** → `BUFFER_NUM = 2`、status/count offset、窗口切换逻辑
- **批量通信与窗口布局** → `perRankDataSize_`、send/recv offset、`AlltoAllvWrite`
- **设计约束** → arch35 tiling 常量与硬边界
- **Combine 后处理** → `CalculateMoeResult` 等阶段函数

## common confusion

- 不要把设计文档里的术语当抽象口号；A5/arch35 实现里通常都有具体常量或函数落点。
- 不要把 950/A5 套回 A3 思路；本页只讨论 A5 专线。
- 不要把本页当平台总览；它是“设计承诺 → 代码落点”的映射页。

## escalation path

- 平台整体分流 → `wiki/compare/mc2-dispatch-combine-platform-splits.md`
- host 决策链 → `wiki/compare/mc2-dispatch-combine-tiling-decisions.md`
- 默认主入口 → `wiki/topics/mc2-moe-dispatch-combine-latest.md`

## current status

当前知识层已把 A5 dispatch/combine 压成“设计文档术语 → arch35 代码落点”的映射页。后续增补应继续沿这条映射链补量化路径，而不是回到泛化平台叙事。

## Sources

- `ops-transformer/mc2/moe_distribute_dispatch_v2/docs/MoeDistributeDispatch-Combine算子设计介绍.md`
- `ops-transformer/mc2/moe_distribute_dispatch_v2/op_host/op_tiling/arch35/moe_distribute_dispatch_tiling_arch35.cpp`
- `ops-transformer/mc2/moe_distribute_dispatch_v2/op_kernel/arch35/moe_distribute_dispatch_arch35.h`
- `ops-transformer/mc2/moe_distribute_combine_v2/op_host/op_tiling/arch35/moe_distribute_combine_tiling_arch35.cpp`
- `ops-transformer/mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_arch35.h`
