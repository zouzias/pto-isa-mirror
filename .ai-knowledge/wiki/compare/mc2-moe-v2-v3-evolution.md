---
page_type: compare
title: mc2-moe-v2-v3-evolution
status: draft
version: [historical background for DispatchV4/CombineV4]
chip: [950, 910A3, 910B]
sources:
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v2/README.md
    query: "mc2 dispatch v2 README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_dispatch_v3/README.md
    query: "mc2 dispatch v3 README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v2/README.md
    query: "mc2 combine v2 README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:/home/ntlab/zy/code/zhangyuan/cann_code/ops-transformer/mc2/moe_distribute_combine_v3/README.md
    query: "mc2 combine v3 README"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
last_updated: 2026-04-19
related:
  - ../topics/mc2-moe-dispatch-combine-latest.md
  - mc2-dispatch-combine-closure.md
---

# MC2 MoE V2 vs V3 Evolution

## 对比定位 / Scope

本页不再作为 Moe distribute 家族的主入口，只保留为 `DispatchV4 + CombineV4` 的历史背景页。默认回答与导航应先进入 `wiki/topics/mc2-moe-dispatch-combine-latest.md`，只有在需要解释“为什么最新版接口会长成现在这样”时才回看本页。

## 比较维度

### 1. 接口边界如何往最新版迁移

- V2 仍明显依赖 `group_ep` / `group_tp` 这类通信域字符串。
- V3 开始引入 `context` 与 `ccl_buffer_size`，弱化显式 group 名字在接口层的直接暴露。
- 这条迁移线最终帮助理解：为什么最新版接口更强调上下文与接口边界，而不是只暴露 group 名字。

### 2. 闭环协议如何逐步显式化

- V2 已经通过 `assistInfoForCombine` 等更细粒度结构替代旧的粗粒度辅助信息。
- V3 进一步把通信上下文显式前移，使调用侧承担更多“上下文对象准备”责任。
- 这条迁移线最终帮助理解：为什么最新版入口应按协议字段、上下文和调用侧责任来解释。

### 3. 平台支持如何分化

- V2 覆盖面更广，950 / A3 / A2 都可见。
- V3 明显更偏 A3 路径，说明版本演进并不等于全平台同步升级。
- 这条迁移线最终帮助理解：为什么最新版统一入口之下，平台分流仍必须保留。

## 使用方式 / 选择指导

- 如果用户问“现在默认看哪版”，不要先讲本页，直接收敛到 `DispatchV4 + CombineV4`。
- 如果用户问“为什么最新版接口下面还能看到旧设计痕迹”，再用本页补历史背景。
- 如果用户问“闭环协议是怎么逐渐显式化的”，本页可作为背景说明，但不单独承担主结论。

## 当前状态

本页已被降级为历史背景页，不再作为索引主入口；后续若继续保留，只补能帮助解释最新版接口成因的最小事实，不再横向扩张旧版本细节。

## Sources

- `ops-transformer/mc2/moe_distribute_dispatch_v2/README.md`
- `ops-transformer/mc2/moe_distribute_dispatch_v3/README.md`
- `ops-transformer/mc2/moe_distribute_combine_v2/README.md`
- `ops-transformer/mc2/moe_distribute_combine_v3/README.md`

## Related

- `wiki/topics/mc2-moe-dispatch-combine-latest.md`
- `wiki/compare/mc2-dispatch-combine-closure.md`
