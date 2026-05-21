---
name: meta-authoring-writeback
description: |
  触发：需要新建或规范化 bug-note、recipe、decision-record、pitfall 等 writeback 条目时。
  Trigger: Create or normalize bug-note, recipe, decision-record, and pitfall writeback entries.
status: stable
---

# Meta Authoring Writeback

## 触发判断

当你要把一次会话经验沉淀到 `writeback/`，或者要判断某个经验值不值得回写时读我。

## 核心方法

1. 先判断是否满足准入：非直觉、可复用、未来高概率再遇到、需要保留决策原因。
2. 按类型落目录：bug-notes、recipes、decision-records、pitfalls。
3. frontmatter 至少写日期、环境、可靠性、来源。
4. 正文只保留症状、根因、正确做法、边界，不抄整段原始日志。

## 模板选择

- `bug-note`：问题现象、根因、修复步骤。
- `recipe`：可复用做法与适用边界。
- `decision-record`：为什么这样选，而不是选了什么。
- `pitfall`：一个易错点及其反例。

## 可执行清单

- 选目录与文件名：`<date>-<short-name>.md`
- 补 `env`、`tags`、`reliability`、`sources`
- 写最小必要正文
- 更新 `writeback/_INDEX.md`
- 视价值决定是否晋升到 card/wiki/skill

## 反模式

- 记录一次性 typo 或临时分支状态。
- 把排查过程流水账全抄进去。
- 不写来源和可靠性，导致以后无法复核。

## 相关 skill

- `meta-authoring-wiki-page`
- `meta-authoring-card`
