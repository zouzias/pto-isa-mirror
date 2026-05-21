---
name: meta-authoring-wiki-page
description: |
  触发：需要新建或改写 entity、concept、compare、topic 类型 wiki 页面时。
  Trigger: Create or revise entity, concept, compare, or topic wiki pages.
status: stable
---

# Meta Authoring Wiki Page

## 触发判断

当你要把零散资料综合成可长期复用的主题页、概念页、实体页或对比页时读我。

## 核心方法

1. 先决定 page_type：entity / concept / compare / topic。
2. frontmatter 必须有 `page_type`、`title`、`status`、`sources`、`last_updated`、`related`。
3. 正文先写“为什么要有这页”和“怎么用这页”，再写事实细节。
4. 知识不充分时允许 `draft`，但结构要稳定。

## 页面选择

- entity：某个 API / 芯片 / 组件的综合说明。
- concept：编程模型或原理的统一叙述。
- compare：两种或多种方案的比较框架。
- topic：大主题导航与总览。

## 可执行清单

- 选目录并命名
- 填 frontmatter
- 固定阅读/比较框架
- 交叉链接 related
- 更新 `wiki/index.md` 与 `wiki/log.md`

## 反模式

- 把 raw 文档整段粘进 wiki。
- 页面没有 Sources。
- 用 wiki 覆盖仓内当前事实。

## 相关 skill

- `docs-rag-query`
- `meta-authoring-writeback`
