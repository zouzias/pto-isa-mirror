---
name: meta-authoring-skill
description: |
  触发：需要新建或规范化一个 skill，包括命名、frontmatter、正文结构与触发词写法时。
  Trigger: Create or normalize a skill, including naming, frontmatter, body structure, and trigger wording.
status: stable
---

# Meta Authoring Skill

## 触发判断

当你需要新增一个可路由的 skill，或者发现已有 skill 的命名、description、正文结构不稳定时读我。

## 核心方法

1. 先判断这个知识是不是方法论；如果是速查，应该做 card，不是 skill。
2. `name` 用 kebab-case，与目录名一致。
3. `description` 必须写清楚“什么情况下触发我”，中英双语优先。
4. 正文保持固定结构：触发判断、核心方法、可执行清单、反模式、相关 skill。
5. 单个 skill 解决一个问题族，不要把多个主题缠在一起。

## 可执行清单

- 先命名目录与 `name`
- 写 frontmatter
- 写 3–5 条核心方法
- 写一段可直接执行的步骤
- 写 2–4 条反模式
- 更新 `skills/_INDEX.md`

## 反模式

- 把 card、wiki、writeback 的内容硬塞进 skill。
- description 只写主题名，不写触发场景。
- skill 没有边界，导致任何问题都能“勉强命中”。

## 相关 skill

- `meta-authoring-card`
- `meta-authoring-wiki-page`
- `meta-authoring-writeback`
