# Hard Rules

## no-full-doc-ingest

禁止将整篇文档、整页 PDF、整份 markdown 手册读入上下文。

理由：
- 文档类问题已经有外部 RAG，原文整读只会消耗 token。
- 当前知识库已经有可复用的 `cards/`、`wiki/`、`writeback/`，第一跳应该先命中这些层，而不是回到 raw 文档。

正确做法：
- 先查当前已落地的 `cards/wiki/writeback`。
- 确实缺证据时，再查 `user-local-rag-9.0_a5` 或 `user-cann-rag`。
- 只保留 3–5 条摘要证据进入上下文。

反例：
- 整篇读取 CANN API 手册。
- 把 MCP 召回的长段原文直接粘进 wiki/card/writeback。

## no-full-codebase-read

禁止整库读代码、整目录顺读、对大文件做无定位 Read。

理由：
- 仓内事实应靠 Grep / Glob / Read 定位式获取。
- 大范围 Read 会把代码检索退化成文本灌入。

正确做法：
- 先读 `skills/code-search-tactics/SKILL.md`。
- 先 Glob 缩文件族，再 Grep 定位符号/关键字，最后对已定位文件做 Read。

## docs-facts-must-come-from-rag

文档事实必须来自外部文档 RAG 或官方来源，不得凭记忆猜测。

正确做法：
- CANN 9.0 / A5 / A3 → `user-local-rag-9.0_a5`
- 非 9.0 或跨产品线 → `user-cann-rag`
- 需要沉淀查询结果时，先读 `skills/docs-rag-query/SKILL.md`

## repo-facts-must-come-from-code

仓内当前实现、调用链、参数、文件位置等事实必须以代码为准，wiki 只能作背景。

正确做法：
- 先代码搜索，再引用 `code:<path>#Lx-Ly`。

反例：
- 用 wiki 里的旧结论覆盖当前仓代码现状。

## no-code-rag

禁止自建 Code RAG、禁止为代码维护额外向量索引。

理由：
- 当前代码问题以精确符号和路径定位为主。
- Grep / Glob / Read 已足够构成最低可靠战术。

## always-route-to-existing-assets-first

面对知识问题时，优先复用当前已经存在的真实知识资产，不要重复造轮子。

当前应优先命中的资产包括：
- `cards/api/datacopy.md`
- `cards/api/cast.md`
- `cards/error-codes/aclnn-161xxx.md`
- `wiki/api/DataCopy.md`
- `wiki/concepts/pipeline-sync.md`
- `wiki/topics/mc2-overview.md`
- `wiki/compare/mc2-vs-shmem.md`
- `writeback/bug-notes/2026-04-19-datacopy-alignment-first.md`
- `writeback/pitfalls/2026-04-19-lifecycle-before-precision.md`

反例：
- 明明已有 DataCopy 卡片，却又重新写一份并行解释。
- 明明已有 pipeline-sync 概念页，却绕过它从零讲生命周期模型。

## l1-stays-thin

L1 只放路由表、硬约束、回答协议、wiki schema；不放方法论、不放长教程、不放一次性经验。

要求：
- `ROUTER.md` / `HARD_RULES.md` / `ANSWERING_POLICY.md` / `WIKI_SCHEMA.md` 合计保持精简。
- 方法论去 `skills/`，历史经验去 `writeback/`，综合页去 `wiki/`。

## l2-load-on-demand

L2 skill 永不整体加载，只读 `skills/_INDEX.md` 做决策，命中才读对应 `SKILL.md`。

补充约束：
- 维护知识库本身时，也按需加载 `docs-rag-query`、`meta-authoring-*`，不要把所有维护 skill 全读一遍。

## l4-is-for-fast-facts

L4 只放速查制品：API、错误码、checklist、表格。

要求：
- 单卡 < 2K tokens。
- 同一会话最多加载 2 张卡。
- 卡片不能写成长教程；需要背景时链接到 `wiki/` 或 `skills/`。

## l5-admission-only

L5 只回写高价值、可复用、已验证的新经验。

准入条件满足其一才可回写：
- 根因非直觉
- 以后高概率再遇到
- 官方文档未明确
- 需要保留决策原因
- 包含隐藏前提 / 易错顺序

不建议回写：
- 一次性 typo
- 当前分支临时状态
- 无复用价值的机械修改

## no-raw-copy-into-derived-layers

禁止把原始文档长段或原始 kernel 大段直接复制进 `skills/`、`cards/`、`wiki/`、`writeback/`。

正确做法：
- `skills/` 写方法与步骤
- `cards/` 写短事实与入口
- `wiki/` 写综合总结与阅读框架
- `writeback/` 写症状、根因、修复、边界

## version-and-source-awareness

涉及 API、参数、约束、芯片能力时必须显式写版本、芯片、来源。

反例：
- “DataCopy 就是 32 字节对齐。”

正确做法：
- “基于 CANN 9.0 beta2 / Atlas 900 A3 的官方文档摘要，DataCopy …”

## draft-before-generalization

从单次案例或少量样本抽出的 pattern、综述、规则，默认先标 `[DRAFT]` 或 `status: draft`。

理由：
- 避免把偶然写法或局部经验误当成稳定知识。
- 当前很多 wiki 页与 patterns 页就是先以 draft 框架存在，再逐步补实。

## keep-writeback-wiki-card-skill-boundaries

不要把不同层混写。

强制边界：
- 单次 bug / 决策 / 易错点 → `writeback/`
- 高频短事实 / 错误码 / checklist / table → `cards/`
- 综合理解 / 对比 / 主题导航 → `wiki/`
- 方法论 / 流程 / pattern / 维护规范 → `skills/`

反例：
- 把一次 bug 修复过程写进 wiki。
- 把错误码卡片写成方法论教程。

## changelog-and-log-required

凡是对知识库产生了新资产或规则变化，必须同步更新记录文件。

要求：
- 知识库层面的改动 → `meta/CHANGELOG.md`
- wiki 相关增补、结构调整、维护动作 → `wiki/log.md`
- 新建 wiki 页还要同步 `wiki/index.md`
- 新建 card / skill / writeback 条目还要同步各自 `_INDEX.md`

反例：
- 文件已经新增，但索引仍然指向“暂无条目”。
- 路由或规则变了，但 CHANGELOG 没有记录。

## eval-and-scripts-are-scaffolds-first

当前 `eval/` 与 `scripts/run-eval.sh` 处于最小可运行脚手架阶段，允许先以 inventory/样例驱动演进，但不允许把脚手架描述成完整评测系统。

正确做法：
- 明确标注当前是 scaffold。
- 逐步补 yaml 校验、frontmatter 检查、死链检查与回归逻辑。

反例：
- 把现有 `run-eval.sh` 说成已经完成全量回归。
