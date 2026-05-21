---
name: patterns-harvest-from-codebase
description: |
  触发：需要从多个代码仓中提炼重复出现的 Ascend C 骨架，生成 patterns-* 草稿。
  Trigger: Harvest repeated Ascend C skeletons from codebases and generate draft patterns skills.
status: stable
---

# Patterns Harvest From Codebase

## 触发判断

当用户要“整理套路”“提炼模板”“看看代码库里有哪些稳定写法”时读我。

## 核心方法

1. 先 Glob 看目录，不先读内容。
2. 按主题分组：双缓冲、tiling、copy-compute、通信同步。
3. 每组只取 5–10 个代表文件做摘要卡。
4. 至少 3 个文件重复出现的骨架才可晋升为 pattern。
5. 新 pattern 默认 `draft`，等人工 review。

## 可执行清单

- 列候选目录与文件族
- 对每个样本提取 API、主循环、buffer、同步点
- 归并公共骨架
- 写 `patterns-*/SKILL.md` 草稿

## 反模式

- 从单一文件直接抽象出通用 pattern。
- 把具体 shape、dtype、项目常量写死进模板。
- 整段复制现有 kernel 到 skill。

## 相关 skill

- `code-search-tactics`
- `patterns-double-buffer`
