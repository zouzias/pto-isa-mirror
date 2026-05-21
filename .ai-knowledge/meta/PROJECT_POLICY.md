# Project Policy

> `ANSWERING_POLICY.md` 的追加项：本项目特有的回答约束。若与 `ANSWERING_POLICY.md` 冲突，以本文件为准。

## 语言与沟通

- 默认使用中文回答。
- **知识库免责声明**：本项目内 `SKILL.md` 的 frontmatter 触发词（`description`）仅需中文即可，无需强制按标准文档要求提供中英双语。
- 代码注释、commit message、文件名、英文术语保留原文。

## 事实来源优先级（本项目落地版）

- 仓内事实（当前实现、当前参数、当前调用链、当前文件位置）优先查代码，不以 wiki 覆盖当前实现。
- 文档事实（API 签名、芯片规格、版本差异、错误码）优先查外部文档 RAG（`user-local-rag-9.0_a5` / `user-cann-rag`），不凭记忆补签名与参数。
- 本项目知识资产优先查 `.ai-knowledge/cards/`、`wiki/`、`writeback/`，命中即止。

## 项目主题范围

- 本项目主线研究对象：**PTO**（项目特有的 Tile 编程抽象）与 **MC2**（计算通信融合算子家族），默认基于 **CANN 9.0 beta2 / A5(950) / A3(910)** 理解。
- 对 `kernels/manual/`、`include/pto/`、`tests/npu/` 的问题，优先命中 `wiki/topics/pto-*` 与 `cards/patterns/`、`cards/examples/`。
- 对 `ops-transformer/mc2/` 的问题，优先命中 `wiki/topics/mc2-*` 与 `wiki/compare/mc2-*`。

## 工具使用约束

- 对 AscendC / PTO 源文件（含 `kernel_operator.h` 或位于 `kernels/` 下）**不跑 ReadLints**——IDE 无 CANN Toolkit include path，lint 全是假阳性（见 `.cursor/rules/no-lint-ascendc-projects.mdc`）。
- 编译 / 运行 / 环境探测类任务必须委派 shell 子任务，参考 `.cursor/rules/build-run-delegate-to-subtask.mdc` 与 `subtask-model-composer2.mdc`。
- 环境问题最多探测 1 次，失败即停止并汇报（`.cursor/rules/env-issue-stop-and-report.mdc`）。

## 回写边界

- PTO / MC2 探索结论按 `skills/pto-writeback-rules/SKILL.md` 与 `writeback/templates/` 沉淀。
- 仅限本项目代码仓现状的一次性事件，不升级为 wiki 综合页，只进 `writeback/bug-notes/`。
- 跨算子、跨家族、跨平台的稳定结论才进 `wiki/`。
