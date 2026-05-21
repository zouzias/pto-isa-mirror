---
name: pto-writeback-rules
description: |
  触发：需要把 PTO 相关探索结论沉淀回 .ai-knowledge，并判断该写到哪一层时。
  Trigger: Write PTO findings back into .ai-knowledge and choose the right destination layer.
status: stable
---

# PTO writeback rules

## purpose
- 规范如何把 PTO 相关探索结论从代码库沉淀进 `.ai-knowledge`。

## writeback admission
只有满足以下至少一条时，才新增 PTO 知识项：
- 能形成稳定阅读入口
- 能形成跨多个 kernel/test 的模式
- 能显著降低后续定位成本
- 能解释平台差异或同步/布局陷阱

## preferred targets
- 分层/入口/阅读顺序 → `wiki/`
- 原语簇、模式、示例入口 → `cards/`
- 抽取规则、可复用经验 → `writeback/recipes` 或 `writeback/bug-notes`
- AI 使用规则 → `skills/`

## anti-patterns
- 把 `docs/isa/*.md` 原文复制进 card/wiki
- 把 kernel README 大段复制进知识库
- 为每个单指令单独新建一页并重复说明书内容
- 为每个 testcase 单独建 wiki，而没有先做示例映射

## minimum evidence
每条 PTO 知识至少应带：
- 一个源码或文档入口路径
- 一个代表性 kernel 或 testcase 路径
- 若涉及平台差异，明确 SoC/架构范围

## sync requirements
任何新增 PTO 资产都必须同步：
- `wiki/index.md` 或相应 `_INDEX.md`
- `wiki/log.md`
- `meta/CHANGELOG.md`
