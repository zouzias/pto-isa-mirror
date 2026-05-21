---
page_type: topic
title: pto-overview
status: stable
version: [pto-isa repository current layout]
chip: [A2, A3, 910A3, 910B, A5, 950]
sources:
  - origin: code:include/pto/pto-inst.hpp
    query: "pto unified include entry"
    source_type: example
    reliability: high
    ingested_at: 2026-04-19
  - origin: code:docs/PTOISA.md
    query: "PTO ISA generated index"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:kernels/README.md
    query: "kernels overview"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:tests/README_zh.md
    query: "tests overview"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:cards/tables/pto-primitives-map.md
    query: "pto primitive routing map"
    source_type: note
    reliability: medium
    ingested_at: 2026-04-20
last_updated: 2026-04-20
related:
  - ../concepts/pto-layering.md
  - pto-isa-mapping.md
  - pto-kernel-patterns.md
  - pto-test-entrypoints.md
  - ../../cards/tables/pto-primitives-map.md
  - ../../cards/tables/pto-kernel-pattern-map.md
  - ../../cards/tables/pto-test-entry-map.md
---

# PTO Overview

## canonical scope

本页是 PTO 主主题的 canonical router，用来把 PTO 相关问题归一化到四类子问题：
- include / API 分层
- ISA / category / implementation-status 映射
- kernels 中的 PTO 模式
- tests 中的 PTO testcase 入口

本页不回答单指令语义细节，不复制 `docs/isa`，也不展开具体 kernel 教程。

## 推荐阅读顺序

1. 先用本页判断你的问题属于分层、原语映射、kernel 模式还是 testcase 入口。
2. 若你还没建立 PTO 的仓内心智模型，先读 `wiki/concepts/pto-layering.md`。
3. 若你关心某条原语、某类原语或某个 category，转 `wiki/topics/pto-isa-mapping.md`。
4. 若你关心真实 kernel 怎样组合 PTO，转 `wiki/topics/pto-kernel-patterns.md`。
5. 若你要找能直接编译、运行或对照 golden 的入口，转 `wiki/topics/pto-test-entrypoints.md`。

## hit terms

- `pto`
- `pto-inst.hpp`
- `PTO ISA`
- `PTO primitive`
- `Tile`
- `GlobalTensor`
- `PTO kernel`
- `PTO testcase`
- `A5 PTO`
- `A2A3 PTO`

## normalized routing

- 如果问题是“PTO 统一入口 / include 怎么分层” → `wiki/concepts/pto-layering.md`
- 如果问题是“某条原语属于哪一类 / 哪些平台支持 / 对应文档在哪” → `wiki/topics/pto-isa-mapping.md`
- 如果问题是“真实 kernel 里 PTO 怎么组合” → `wiki/topics/pto-kernel-patterns.md`
- 如果问题是“tests 里哪个例子能直接看 / 能直接跑” → `wiki/topics/pto-test-entrypoints.md`
- 如果问题是“原语 / 模式 / testcase 的快速落点” → `cards/tables/pto-primitives-map.md`、`pto-kernel-pattern-map.md`、`pto-test-entry-map.md`

## 主题框架 / main subtopics

- **入口分层**：`include/pto/` 如何把统一 API 暴露出来，以及不同后端怎样分派。
- **原语清单**：manifest、category、实现状态、芯片形态与示例入口如何互相映射。
- **模式骨架**：kernel 中常见的 load/store、vec、matmul、comm、sync 组合方式。
- **测试闭环**：`tests/npu` 中哪些 testcase 适合用来建立最小运行闭环。

## primary anchors

- `include/pto/pto-inst.hpp`
- `docs/PTOISA.md`
- `kernels/README.md`
- `tests/README_zh.md`

## common confusion

- 不要把本页当成单指令说明书；单指令事实应回 `docs/isa/*.md`。
- 不要把本页当成 kernel 教程；kernel 组合问题应转到 `pto-kernel-patterns`。
- 不要把本页当成 testcase 列表；示例入口问题应转到 `pto-test-entrypoints` 或 test-entry map。
- 不要把 PTO topic 页当成仓内代码事实的替代品；具体实现位置、当前参数和调用链仍应以代码为准。

## 使用方式

当用户只说“看看 PTO 是怎么组织的”“某个 PTO 原语从哪看起”“哪个 testcase 最适合作入口”这类泛问题时，优先从本页收敛，再分流到四个稳定子主题。这样可以避免一开始就陷入单文件细节，丢掉 PTO 的整体分层关系。

## escalation path

- 需要 repo 当前事实 → 回代码
- 需要单指令文档 → 回 `docs/isa/manifest.yaml` 和 `docs/isa/*.md`
- 需要 testcase 运行闭环 → 回 `tests/npu/**/testcase/*`

## current status

当前 PTO 主线已经在知识库中形成四个稳定入口：分层、ISA 映射、kernel 模式、test 入口。后续新增 PTO 内容应优先挂到这四类入口之一，而不是新造平行主题页。

## Sources

- `include/pto/pto-inst.hpp`
- `docs/PTOISA.md`
- `kernels/README.md`
- `tests/README_zh.md`
- `cards/tables/pto-primitives-map.md`

## Related

- [pto-layering](../concepts/pto-layering.md)
- [pto-isa-mapping](./pto-isa-mapping.md)
- [pto-kernel-patterns](./pto-kernel-patterns.md)
- [pto-test-entrypoints](./pto-test-entrypoints.md)
- [pto-primitives-map](../../cards/tables/pto-primitives-map.md)
- [pto-kernel-pattern-map](../../cards/tables/pto-kernel-pattern-map.md)
- [pto-test-entry-map](../../cards/tables/pto-test-entry-map.md)
