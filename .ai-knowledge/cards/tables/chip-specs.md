---
name: card-table-chip-specs
description: |
  触发：需要快速查看芯片能力维度与后续待补项时。
  Trigger: Lookup chip capability dimensions and fields to be completed later.
card_type: tables
version: draft
chip: [950 A5, 910A3]
source: official
last_updated: 2026-04-19
---

# Chip Specs 占位表

## 使用说明

本表当前不追求填写具体数值，而是固定后续补数时必须关注的字段，避免把无关规格灌进知识库。

| Chip | UB | L1 | L2 | 关注点 | Notes |
| --- | --- | --- | --- | --- | --- |
| 910A3 | 待补 | 待补 | 待补 | tile 预算、buffer 规划、copy-compute 重叠 | 以官方文档复核后填写 |
| 950A5 | 待补 | 待补 | 待补 | tile 预算、buffer 规划、copy-compute 重叠 | 以官方文档复核后填写 |

## 填表规则

- 只填与 kernel 设计直接相关的规格字段。
- 只有在有官方来源时才补数值。
- 一旦补数值，需同步更新对应 wiki/chips 实体页。
