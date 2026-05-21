---
page_type: compare
title: mc2-dispatch-combine-tiling-decisions
status: draft
version: [ops-transformer mc2 mainline]
chip: [950, 910A3, 910B]
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/op_api/moe_distribute_dispatch_v2_base.cpp
    query: "dispatch v2 base"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_tiling_helper.cpp
    query: "dispatch v2 tiling helper"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp
    query: "dispatch v2 tiling"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/op_api/moe_distribute_combine_v2_base.cpp
    query: "combine v2 base"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_tiling_helper.cpp
    query: "combine v2 tiling helper"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_v2_tiling.cpp
    query: "combine v2 tiling"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
last_updated: 2026-04-19
related:
  - mc2-dispatch-combine-platform-splits.md
  - mc2-dispatch-combine-closure.md
---

# DispatchV2 / CombineV2 Tiling Decisions

## canonical scope

本页是 dispatch_v2 / combine_v2 host 决策链问题的 canonical router。它不讨论平台复杂度中心本身，而是回答 base / helper / tiling 三层如何共同决定最终策略路径。

## hit terms

- `base helper tiling`
- `workspace size base`
- `tiling helper`
- `fullmesh`
- `layered`
- `quantMode`
- `dynamicScales`
- `sharedExpertNum`

## normalized routing

- 问“平台差异的主分流轴是什么” → 转 `mc2-dispatch-combine-platform-splits`
- 问“合法输入最终为什么走到这条策略” → 本页直接回答
- 问“dispatch/combine 闭环协议是什么” → 转 `mc2-dispatch-combine-closure`

## primary anchors

- `ops-transformer/mc2/moe_distribute_dispatch_v2/op_api/moe_distribute_dispatch_v2_base.cpp`
- `ops-transformer/mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_tiling_helper.cpp`
- `ops-transformer/mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp`
- `ops-transformer/mc2/moe_distribute_combine_v2/op_api/moe_distribute_combine_v2_base.cpp`
- `ops-transformer/mc2/moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_tiling_helper.cpp`
- `ops-transformer/mc2/moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_v2_tiling.cpp`

## canonical answer units

- **base 层**：先做平台裁枝与 server type 归一化
- **helper 层**：把输入世界缩到合法 shape/dtype/format 子集
- **tiling 层**：在合法输入上决定 fullmesh/layered/quant/shared-expert 等真实策略
- 三层合起来才是 dispatch_v2 / combine_v2 的 host 侧决策链

## common confusion

- 不要一上来就盯 kernel；很多路径在 base/helper 已被裁掉。
- 不要把 helper 当成简单校验；它实际上在定义“允许进入策略空间的输入世界”。
- 不要把 tiling 看成纯性能参数；它也是功能分流中心。

## escalation path

- 平台复杂度中心 → `wiki/compare/mc2-dispatch-combine-platform-splits.md`
- 闭环协议语义 → `wiki/compare/mc2-dispatch-combine-closure.md`
- repo 当前实现 → 回对应 base/helper/tiling 代码

## current status

当前知识层已把 dispatch_v2 / combine_v2 的 host 决策链压成 base / helper / tiling 三层模型。后续增补应继续围绕“哪一层先裁掉了什么路径”展开，而不是直接跳进 kernel 模板。

## Sources

- `ops-transformer/mc2/moe_distribute_dispatch_v2/op_api/moe_distribute_dispatch_v2_base.cpp`
- `ops-transformer/mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_tiling_helper.cpp`
- `ops-transformer/mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp`
- `ops-transformer/mc2/moe_distribute_combine_v2/op_api/moe_distribute_combine_v2_base.cpp`
- `ops-transformer/mc2/moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_tiling_helper.cpp`
- `ops-transformer/mc2/moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_v2_tiling.cpp`
