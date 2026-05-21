---
page_type: concept
title: pipeline-sync
status: stable
version: [generic]
chip: [generic]
sources:
  - origin: writeback:writeback/pitfalls/2026-04-19-lifecycle-before-precision.md
    query: "lifecycle before precision"
    source_type: note
    reliability: medium
    ingested_at: 2026-04-19
  - origin: writeback:writeback/bug-notes/2026-04-19-datacopy-alignment-first.md
    query: "datacopy alignment and lifecycle first checks"
    source_type: note
    reliability: medium
    ingested_at: 2026-04-19
  - origin: code:kernels/manual
    query: "double buffer queue alloc enqueue dequeue free usage"
    source_type: example
    reliability: medium
    ingested_at: 2026-04-20
last_updated: 2026-04-20
related:
  - ../api/DataCopy.md
  - ../topics/mc2-overview.md
  - ../../cards/checklists/tiling-sanity.md
---

# Pipeline Sync

## 定位

本页统一描述 Queue、EnQue/DeQue、buffer 生命周期与流水阶段之间的同步语义，目的是把“数据错了”“结果漂了”“程序挂了”这几类表面症状放回同一个模型里理解。

## 核心心智模型

流水正确性的关键不是某一步单独正确，而是：谁拥有 buffer、什么时候可读、什么时候可写、什么时候必须释放。只要这四件事中有一件没说清，问题就会在精度、死锁、脏数据之间来回伪装。

## Queue 与生命周期

- AllocTensor 获得的是当前阶段可写的本地 buffer。
- EnQue 表示把该 buffer 的所有权交给下一阶段。
- DeQue 表示当前阶段开始消费这个 buffer。
- FreeTensor 表示当前阶段消费完成，buffer 可以重新进入循环。

## 阶段边界怎么判断

- 看一个 tile 是否已经完成生产，不能只看“数据写过了”，还要看是否已经通过 queue 或事件把可消费性显式交给下一阶段。
- 看一个 tile 是否还能被复用，不能只看“当前阶段用完了”，还要看下游是否已经真正消费完成。
- 双缓冲下最容易错的不是主循环，而是首轮 warmup、末轮 drain 和尾块路径；这些路径最容易漏掉一次 wait、free 或 flip。
- 如果 host 侧 tiling 把正常块与尾块分开，kernel 侧同步和 buffer 释放也必须对应分支收口，不能默认复用主路径节奏。

## 常见失配模式

- EnQue/DeQue 不成对，导致阶段间认知不一致。
- FreeTensor 过早，后续阶段读到被复用的数据。
- 只检查算子计算，不检查 buffer 所有权流转。
- 主路径正确但尾块路径少一步同步或释放。
- 只校验 DataCopy / Cast 的参数，却没有确认数据是否来自当前轮而非上一轮残留。

## 与精度和挂起的关系

- 精度异常不一定是算术误差，很多是旧数据、脏数据、覆盖数据。
- 卡死或超时也不一定是通信问题，可能是上游阶段没有按约定交付 buffer。
- DataCopy、Cast、双缓冲、通信融合都共享这套生命周期约束。

## 使用方式

遇到“结果不稳定、偶发错误、只在某些 tile 出问题、双缓冲一开就异常”时，先回到这个模型检查所有权与阶段边界，再决定是否继续查具体 API。

## 当前状态

当前页已足够作为稳定入口页使用：它固定了 queue 生命周期的统一术语、首轮/末轮/尾块的优先排查顺序，以及与精度/挂起问题的关系。后续如果需要补更细的 pipe/event 事实，应增加官方来源，但不影响本页作为概念路由页使用。

## Sources

- `writeback/pitfalls/2026-04-19-lifecycle-before-precision.md`
- `writeback/bug-notes/2026-04-19-datacopy-alignment-first.md`
- `code:kernels/manual`（双缓冲/queue 使用模式抽样）

## Related

- [DataCopy](../api/DataCopy.md)
- [mc2-overview](../topics/mc2-overview.md)
- [tiling-sanity](../../cards/checklists/tiling-sanity.md)
