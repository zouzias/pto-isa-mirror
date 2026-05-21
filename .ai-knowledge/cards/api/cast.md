---
name: card-api-cast
description: |
  触发：查 Cast 的常见支持类型、精度注意项与排查入口。
  Trigger: Lookup Cast support, precision notes, and debugging entry points.
card_type: api
version: CANN 9.0 beta2
chip: [Atlas 900 A3, Atlas 800I A2]
source: official
last_updated: 2026-04-19
---

# Cast 速查卡

## 适用范围

本卡面向 CANN 9.0 beta2 下 Cast 相关的快速判断；精确支持矩阵、舍入细节与特化限制在提交实现前仍应由文档 RAG 复核。

## 速查要点

- 先确认源类型与目标类型是否合法，再讨论精度。
- Cast 异常要先区分为三类：类型不支持、量化/截断误差、同步/覆盖导致的伪精度问题。
- Cast 前后的 buffer 生命周期要清晰，避免读到被覆盖或尚未准备好的数据。
- 尾块路径若单独处理，必须确认它和主路径使用了相同的类型与同步约束。

## 首轮排查顺序

1. 源/目标 dtype 是否在当前版本与芯片上受支持。
2. 输入输出 buffer 是否别名重叠或生命周期冲突。
3. 问题是预期量化误差，还是明显异常值/脏数据。
4. 主路径与尾块路径的 Cast 行为是否一致。

## 常见陷阱

1. 误把同步或覆盖问题当作 Cast 精度问题。
2. 源/目标张量别名重叠，导致结果被覆盖。
3. 只验证主路径，忽略尾块路径导致精度异常只在边界出现。
4. 未先确认支持类型就直接使用 Cast。

## 联动入口

- 精度排查：`ascendc-precision-debug`
- API 用法：`ascendc-api-best-practices`

## 来源

- `mcp:user-local-rag-9.0_a5`，建议 query: `Cast support dtype CANN 9.0 beta2`
