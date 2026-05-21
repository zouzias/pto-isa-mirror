---
page_type: topic
title: pto-isa-mapping
status: draft
version: [pto-isa repository current layout]
chip: [A2, A3, 910A3, 910B, A5, 950, CPU]
sources:
  - origin: code:include/README.md
    query: "PTO instruction implementation status"
    source_type: example
    reliability: high
    ingested_at: 2026-04-19
  - origin: code:docs/isa/manifest.yaml
    query: "PTO ISA manifest categories"
    source_type: example
    reliability: high
    ingested_at: 2026-04-19
  - origin: code:docs/PTOISA.md
    query: "PTO ISA generated index"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
last_updated: 2026-04-19
related:
  - pto-overview.md
  - ../../cards/tables/pto-primitives-map.md
---

# PTO ISA Mapping

## canonical scope

本页是 PTO 原语映射问题的 canonical router，用来把原语名、category、平台支持状态、kernel 模式和 testcase 入口归一化到同一条路由链上。本页不讲单指令细节定义，只讲“该去哪看”。

## hit terms

- `manifest.yaml`
- `PTOISA`
- `TLOAD`
- `TSTORE`
- `TASSIGN`
- `TSYNC`
- `TMATMUL`
- `TGATHER`
- `implementation status`
- `CPU A2 A3 A5`
- `category`

## normalized routing

- 如果问题是“这条原语属于什么类别” → 先看 `docs/isa/manifest.yaml`
- 如果问题是“这条原语在哪些后端可用” → 先看 `include/README.md`
- 如果问题是“单指令文档在哪” → 走 `docs/PTOISA.md` 和 `docs/isa/*.md`
- 如果问题是“这条原语在哪些 kernel/test 里最值得看” → 走 `cards/tables/pto-primitives-map.md`

## primary anchors

- `include/README.md`
- `docs/isa/manifest.yaml`
- `docs/PTOISA.md`
- `cards/tables/pto-primitives-map.md`

## common confusion

- 不要把 `manifest.yaml` 当成实现状态表；平台支持应以 `include/README.md` 为准。
- 不要把 `include/README.md` 当成原语语义定义；单指令语义仍应回 `docs/isa/*.md`。
- 不要在这页直接解释 kernel 写法；模式问题应转到 `pto-kernel-patterns`。

## escalation path

- 需要单指令定义 → `docs/isa/*.md`
- 需要原语簇级路由 → `cards/tables/pto-primitives-map.md`
- 需要模式级代码理解 → `wiki/topics/pto-kernel-patterns.md`
- 需要 testcase 闭环 → `wiki/topics/pto-test-entrypoints.md`

## current status

当前知识层已把 PTO 原语映射收敛到三个权威入口：实现状态矩阵、ISA manifest、原语映射表。后续若补新页，应优先挂到这三类入口之一。

## Sources

- `include/README.md`
- `docs/isa/manifest.yaml`
- `docs/PTOISA.md`

## Related

- `wiki/topics/pto-overview.md`
- `cards/tables/pto-primitives-map.md`
