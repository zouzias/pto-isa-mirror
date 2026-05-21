---
name: docs-rag-query
description: |
  触发：需要查询外部文档 RAG，并把返回结果压缩成可写入 card/wiki/writeback 的证据包时。
  Trigger: Query external documentation RAG and compress results into evidence packs for cards, wiki, or writeback.
status: stable
---

# Docs RAG Query

## 触发判断

当问题属于文档事实、API 签名、版本差异、芯片规格，并且需要调用 `user-local-rag-9.0_a5` 或 `user-cann-rag` 时读我。

## 核心方法

1. 先选路由：CANN 9.0 beta2 / A5 / A3 优先 `user-local-rag-9.0_a5`，跨产品线或非 9.0 用 `user-cann-rag`。
2. query 同时包含中文术语、英文术语、版本和主题。
3. 结果只保留 3–5 条摘要，不把原文整段塞进上下文。
4. 写回时保留 `origin`、`query`、`source_type`、`reliability`、`ingested_at`。

## 可执行清单

- 先判断是否真的是文档问题
- 组装 query
- 取 top-k
- 压缩成 <2K tokens 证据包
- 再决定写入 card、wiki 还是 writeback

## 反模式

- 同一问题轮流试所有 MCP。
- 把召回原文大段复制进知识层。
- 不记录 query，导致以后无法复跑。

## 相关 skill

- `meta-authoring-card`
- `meta-authoring-wiki-page`
