---
name: meta-authoring-card
description: |
  触发：需要新建或规范化 API、错误码、checklist、table 类型卡片时。
  Trigger: Create or normalize API, error-code, checklist, or table cards.
status: stable
---

# Meta Authoring Card

## 触发判断

当知识点属于“读完就能直接答”的速查内容，而不是方法论或综述时读我。

## 核心方法

1. 先选 card_type：api / error-codes / checklists / tables。
2. 卡片优先短、准、可扫描；单卡不承载长篇背景说明。
3. 写“适用范围”“首轮排查/速查要点”“联动入口”。
4. 需要背景和原因时链到 wiki 或 skill，不在卡片里展开。

## 可执行清单

- 选目录命名
- 写 frontmatter
- 写 3–6 条高密度要点
- 补来源 query 或 related page
- 更新 `cards/_INDEX.md`

## 反模式

- 把卡片写成教程。
- 没有版本意识。
- 把未经验证的结论写成硬规则。

## 相关 skill

- `docs-rag-query`
- `meta-authoring-wiki-page`
