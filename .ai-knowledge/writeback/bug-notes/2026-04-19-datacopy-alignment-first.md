---
type: bug-note
date: 2026-04-19
env:
  cann: 9.0 beta2
  chip: Atlas 900 A3
  mode: ascendc kernel
tags: [datacopy, alignment, parameters]
reliability: verified
sources:
  - origin: mcp:user-local-rag-9.0_a5
    query: "DataCopy signature alignment CANN 9.0 beta2"
    reliability: high
---

# DataCopy 问题先查对齐与 buffer 预算

## 问题

表面症状常表现为参数类报错、运行异常或数据不对，看起来像 DataCopy 本身失效。

## 根因

很多这类问题不是 DataCopy 逻辑复杂，而是调用前置条件没有成立：地址对齐、`calCount` 与 tile bytes 不匹配、buffer 空间角色理解错误。

## 正确做法

先检查地址对齐，再核算当前 tile 的 buffer 预算，然后确认源/目的张量空间与 dtype 一致，最后再看流水生命周期。

## 反例

一开始就把问题归因到 queue 或算子计算，跳过地址和长度约束检查。
