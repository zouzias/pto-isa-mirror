---
name: card-api-datacopy
description: |
  触发：查 DataCopy 的签名、对齐要求、容量边界与常见坑。
  Trigger: Lookup DataCopy signature, alignment rules, capacity limits, and common pitfalls.
card_type: api
version: CANN 9.0 beta2
chip: [Atlas 900 A3, Atlas 800I A2]
source: official
last_updated: 2026-04-19
---

# DataCopy 速查卡

## 适用范围

本卡面向 CANN 9.0 beta2 下常见的 DataCopy 使用与排障场景；精确重载签名与特化参数在动手前仍应通过文档 RAG 复核。

## 速查要点

- 先检查源地址、目的地址与数据长度是否满足对齐约束。
- `calCount` 不能只按逻辑元素数理解，必须同时满足当前 tile 与 UB 容量预算。
- 使用前确认源/目的张量所在空间与 API 期望一致，不要把 GM/UB 角色混用。
- 如果问题出现在双缓冲或 queue 流水里，优先排查生命周期和覆盖顺序，不要只盯 DataCopy 本身。

## 首轮排查顺序

1. 地址是否对齐。
2. `calCount` 与 tile bytes 是否落在当前 buffer 预算内。
3. 张量空间、dtype、shape 是否和调用路径匹配。
4. host 侧 tiling 参数与 kernel 侧理解是否一致。
5. 若在流水中使用，检查 EnQue/DeQue/FreeTensor 是否成对。

## 常见陷阱

1. 地址未对齐，表现为运行时报参数/地址类错误。
2. `calCount` 超出当前 tile/UB 预算，表现为运行异常或结果错误。
3. 输入输出张量空间搞混，导致搬运方向和实际 buffer 不匹配。
4. 把同步或 buffer 覆盖问题误判为 DataCopy 本身故障。

## 联动入口

- 需要完整综合说明：`wiki/api/DataCopy.md`
- 需要流水与生命周期背景：`wiki/concepts/pipeline-sync.md`
- 需要 API 使用方法论：`ascendc-api-best-practices`

## 来源

- `mcp:user-local-rag-9.0_a5`，建议 query: `DataCopy signature alignment CANN 9.0 beta2`
- `wiki/api/DataCopy.md`
