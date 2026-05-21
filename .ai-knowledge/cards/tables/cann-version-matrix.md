---
name: card-table-cann-version-matrix
description: |
  触发：需要比较不同 CANN 版本的能力差异与文档补齐计划时。
  Trigger: Compare CANN version capability differences and track completion status.
card_type: tables
version: draft
chip: [950 A5, 910A3]
source: official
last_updated: 2026-04-19
---

# CANN Version Matrix 占位表

## 使用说明

本表用于记录“哪些主题值得按版本分开维护”，不是把所有 API 都机械抄成大表。只有在确认某主题对版本敏感时，才值得新增一行。

| Topic | 8.x | 9.0 beta2 | 关注点 | Notes |
| --- | --- | --- | --- | --- |
| DataCopy 约束 | 待补 | 待补 | 对齐、参数边界、文档入口 | 需要官方来源支撑 |
| Cast 支持类型 | 待补 | 待补 | dtype 支持矩阵与精度行为 | 需要官方来源支撑 |
| Kernel/Tiling 文档覆盖 | 待补 | 待补 | 文档粒度与可查性 | 后续 ingest 时增补 |

## 填表规则

- 一行一主题，不一行一个零散 API 细节。
- 只有在存在版本差异时才写差异说明。
- 表格更新后，应同步更新对应 card/wiki 页的 `version` 描述。
