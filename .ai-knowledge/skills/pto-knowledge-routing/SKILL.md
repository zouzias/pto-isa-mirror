---
name: pto-knowledge-routing
description: |
  触发：需要把 PTO 问题在 wiki、cards、代码与文档之间做第一跳路由时。
  Trigger: Route PTO questions across wiki, cards, code, and docs before expanding the search.
status: stable
---

# PTO knowledge routing

## purpose
- 让 AI 在 PTO 相关问题里优先命中 `.ai-knowledge` 的正确层，而不是直接散读 docs/README/源码。

## routing order
1. **先判问题类型**
   - 原语/入口/分层问题 → `wiki/topics/pto-overview.md`、`wiki/concepts/pto-layering.md`
   - kernel 使用方式问题 → `wiki/topics/pto-kernel-patterns.md`、`cards/patterns/*`
   - testcase / 示例代码入口问题 → `wiki/topics/pto-test-entrypoints.md`、`cards/examples/*`
   - 原语到源码/样例的快速定位 → `cards/tables/pto-primitives-map.md`
2. **命中 map 或 topic 后，再决定是否回源代码**
   - repo 当前实现事实必须以代码为准
   - 原语定义和单指令事实优先回 `docs/isa/*.md` / `docs/isa/manifest.yaml`
3. **只有知识库未覆盖时，才扩大阅读范围**

## hard boundaries
- 不复制 `docs/isa` 的单指令说明到知识库。
- 不复制 `kernels/manual/**/README.md` 的长段背景。
- 不把 testcase 代码大段粘进知识页，只保留路径、主题和阅读顺序。
- 讨论 repo 当前行为时，最终以代码为准，不用 wiki 覆盖代码事实。

## preferred entry examples
- “PTO 统一入口在哪” → `wiki/concepts/pto-layering.md`
- “TopK 体现了什么 PTO 模式” → `cards/patterns/topk-vector-pipeline.md`
- “最简单的 PTO testcase 是哪个” → `cards/tables/pto-test-entry-map.md` + `cards/examples/test-tadd-a5-a2a3.md`
