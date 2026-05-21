---
type: decision-record
date: 2026-04-19
env:
  cann: generic
  chip: generic
tags: [knowledge-base, decision]
reliability: verified
sources:
  - origin: writeback:session
    query: "continue filling .ai-knowledge incrementally"
    reliability: medium
---

# 先补最小可用层，再补高精度内容

## 背景

`.ai-knowledge` 初始状态只有 router，其他层基本为空。如果直接追求高精度事实填充，会导致大量目录长期不可路由。

## 决策

先把 L2/L4/L5/L6 的最小可用骨架补齐，再逐步补实核心页与维护层。

## 为什么这样选

这样可以让知识库尽快具备路由、沉淀和扩展能力，后续增量内容也有明确落点。

## 未选择项

没有先做“只补一两页高精度卡片”的方案，因为那会让整体结构继续失衡。
