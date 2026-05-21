---
page_type: concept
title: pto-layering
status: draft
version: [pto-isa repository current layout]
chip: [A2, A3, 910A3, 910B, A5, 950]
sources:
  - origin: code:include/pto/pto-inst.hpp
    query: "pto unified include entry"
    source_type: example
    reliability: high
    ingested_at: 2026-04-19
  - origin: code:include/pto/common/pto_instr.hpp
    query: "pto common api"
    source_type: example
    reliability: high
    ingested_at: 2026-04-19
  - origin: code:include/pto/common/pto_instr_impl.hpp
    query: "pto impl dispatch"
    source_type: example
    reliability: high
    ingested_at: 2026-04-19
last_updated: 2026-04-19
related:
  - ../topics/pto-overview.md
---

# PTO Layering

## 定位 / Summary

本页固定 PTO include 体系的分层模型，回答“一个 PTO 原语从哪里进入、公共 API 在哪、后端实现如何分派”。这页不讲单条原语语义，而是讲 PTO 头文件体系应该怎么读。

## 核心心智模型

PTO include 层最自然的理解顺序是：

1. `include/pto/pto-inst.hpp`
2. `include/pto/common/pto_instr.hpp`
3. `include/pto/common/pto_instr_impl.hpp`
4. `include/pto/npu/*` / `include/pto/cpu/*` / `include/pto/costmodel/*` / `include/pto/comm/*`

也就是说：
- `pto-inst.hpp` 是统一入口
- `pto_instr.hpp` 是公共 API 面
- `pto_instr_impl.hpp` 是后端分派面
- 具体目录才是平台/后端实现面

## 关键机制或阶段关系

### 统一入口层

`pto-inst.hpp` 本身很薄，它的主要价值是把 PTO 使用者从平台差异里隔离出来：使用方通常先包含这个入口，再通过宏和模板分派到具体实现。

### 公共 API 层

`pto_instr.hpp` 是理解 PTO 原语“长什么样”的最佳入口。这里定义的是公共模板 API、参数形态、事件返回风格，以及很多原语在平台无关层面的调用方式。

### 公共类型与资源模型

除了 `pto_instr.hpp`，还应一起读：
- `pto_tile.hpp`
- `type.hpp`
- `memory.hpp`
- `event.hpp`
- `fifo.hpp`

因为 PTO 不只是 instruction name 列表，而是一套带 Tile、Layout、GlobalTensor、Event、FIFO 约束的模板式编程模型。

### 后端分派层

`pto_instr_impl.hpp` 负责把公共 API 分派到不同后端：
- `npu/a2a3`
- `npu/a5`
- `npu/kirin9030`
- `cpu`
- `costmodel`
- `comm`

因此读 PTO 时不要直接从某个 `npu/a5/T*.hpp` 开始，否则会跳过公共 API 面与分派逻辑。

## 常见失配模式 / failure modes

- 直接从某个后端头文件理解 PTO，误把平台特化写法当通用 API。
- 把 `docs/isa` 当成源码分层说明书，忽略 `pto-inst.hpp -> common -> impl -> backend` 这条真实 include 链。
- 只看 instruction 名，不看 `Tile / Layout / Event / FIFO` 等资源模型，导致对调用形态理解不完整。

## 与其他页面关系

- 想知道 PTO 整体知识如何组织，回到 `wiki/topics/pto-overview.md`。
- 想知道这些原语在 kernel 里如何组合，跳到 `wiki/topics/pto-kernel-patterns.md`。
- 想知道某个原语有没有测试样例，去看 `cards/tables/pto-test-entry-map.md`。

## Sources

- `include/pto/pto-inst.hpp`
- `include/pto/common/pto_instr.hpp`
- `include/pto/common/pto_instr_impl.hpp`

## Related

- `wiki/topics/pto-overview.md`
