---
page_type: entity
title: DataCopy
status: stable
version: [CANN 9.0 beta2]
chip: [Atlas 900 A3, Atlas 800I A2]
sources:
  - origin: mcp:user-local-rag-9.0_a5
    query: "DataCopy signature alignment CANN 9.0 beta2"
    source_type: official
    reliability: high
    ingested_at: 2026-04-19
last_updated: 2026-04-19
related:
  - ../concepts/pipeline-sync.md
  - ../../cards/api/datacopy.md
---

# DataCopy

## 定位

本页汇总 DataCopy 在日常 Ascend C 开发中最常被问到的维度：签名入口、约束边界、排障顺序以及与流水同步的关系。速查走卡片，需要完整上下文时回到本页。

## 使用心智模型

DataCopy 本质上是一次受空间、长度、对齐和生命周期共同约束的数据搬运。很多表面看起来像“拷贝错了”的问题，真实根因并不在指令本身，而在调用前后对 buffer 与流水状态的假设不成立。

## 关键约束与边界

- 先验证源和目的张量分别位于 API 预期的存储空间。
- 长度不能脱离 tile 与 UB 预算单独理解。
- 地址、长度和数据类型要同时满足约束；只看其中一个维度通常不够。
- 在 queue / 双缓冲场景中，DataCopy 是否正确还取决于 buffer 是否已经准备好、是否会被后续阶段覆盖。

## 排障顺序

1. 地址与长度约束。
2. 张量空间与 dtype 一致性。
3. host 侧 tiling 参数与 kernel 侧消费方式是否一致。
4. queue 生命周期、EnQue/DeQue/FreeTensor 是否成对。
5. 若问题只出现在边界块，单独检查尾块路径。

## 常见陷阱

- 把地址对齐问题误看成一般参数错误。
- tile 大小改了，但 `calCount` 和 buffer 预算没有同步更新。
- 复制方向、张量空间或 buffer 所有权理解错误。
- 真正的问题是流水同步或覆盖，而不是 DataCopy 本身。

## 与其他页面的关系

- 快速查入口：`cards/api/datacopy.md`
- 同步与生命周期背景：`wiki/concepts/pipeline-sync.md`
- 方法论层：`ascendc-api-best-practices`

## 版本差异

当前页已形成稳定入口；精确重载、参数矩阵和更细的版本差异仍可继续通过官方来源增补。

## Sources

- `mcp:user-local-rag-9.0_a5` / `DataCopy signature alignment CANN 9.0 beta2`

## Related

- [pipeline-sync](../concepts/pipeline-sync.md)
- [card-api-datacopy](../../cards/api/datacopy.md)
