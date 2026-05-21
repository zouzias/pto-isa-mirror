---
name: code-search-tactics
description: |
  触发：面对代码问题时，需要决定何时用 Grep、何时用 Glob、何时用 Read，避免整库乱读。
  Trigger: Decide between Grep, Glob, and Read for code questions without reading whole repositories.
status: stable
---

# Code Search Tactics

## 触发判断

在问题属于“仓内事实”“代码在哪里实现”“某个符号在哪被调用”时读我；如果是文档事实问题，不读我。

## 核心方法

1. 精确符号、错误串、类名、函数名：先用 Grep。
2. 文件族、目录结构、命名模式：先用 Glob。
3. 已经知道文件且只需读局部：用 Read，优先带 offset/limit。
4. 先定位，再阅读；不要反过来。

## 可执行清单

- 调用链：`Grep pattern="FuncName\\("`
- 文件模式：`Glob pattern="**/*kernel*.cpp"`
- 大文件：先 Grep 行号，再 Read 局部
- 中小文件：已确认目标后可一次 Read

## 反模式

- 上来就 Read 整个目录里的多个文件。
- 用宽泛关键词 Grep 全仓后不收敛。
- 把 wiki 当当前代码事实来源。

## 相关 skill

- `patterns-harvest-from-codebase`
