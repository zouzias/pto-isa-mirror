# Wiki Schema

## 页面类型

- `entity`：单一实体综合页，目录 `wiki/api/`、`wiki/chips/`、`wiki/ops/`
- `concept`：编程模型 / 原理页，目录 `wiki/concepts/`
- `compare`：方案 / 算法 / 实现对比页，目录 `wiki/compare/`
- `topic`：大主题综述页，目录 `wiki/topics/`

## 命名

- API / 芯片 / 算子页保留原名：`DataCopy.md`、`MatMulV3.md`、`910A3.md`
- 概念页使用小写连字符：`pipeline-sync.md`
- 对比页使用 `<a>-vs-<b>.md` 或 `<domain>-topologies.md`
- topic 页使用主题短名：`mc2-overview.md`

## frontmatter 最小字段集

每页必须包含：
- `page_type`
- `title`
- `status`
- `version`
- `chip`
- `sources`
- `last_updated`
- `related`

## status 约定

- `draft`：结构已确定，但事实仍在增补
- `stable`：结构与核心事实都较稳定
- `stale`：已知需要复核，命中时优先先查新证据

当前知识库刚搭建完成，现有大多数 wiki 页都允许先保持 `draft`，但不能是空壳页；至少要能回答“这页为什么存在、怎么用、下一步补什么”。

## 当前已存在页面模式

- `wiki/api/DataCopy.md`：entity 页，强调排障顺序、约束边界与 related card
- `wiki/chips/910A3.md`：entity 页，先固定看芯片时的阅读框架，再补规格值
- `wiki/concepts/pipeline-sync.md`：concept 页，先固定生命周期模型
- `wiki/compare/mc2-vs-shmem.md`：compare 页，先固定对比维度
- `wiki/topics/mc2-overview.md`：topic 页，先固定阅读顺序和主题导航

后续新页应尽量与这些已落地页面保持同一风格。

## entity 页必备段落

1. 定位 / Summary
2. 使用心智模型或阅读目标
3. 关键约束 / 关注维度
4. 排障顺序或常见陷阱
5. 与其他页面的关系
6. 版本差异 / 当前状态
7. Sources
8. Related

## concept 页必备段落

1. 定位 / Summary
2. 核心心智模型
3. 关键机制或阶段关系
4. 常见失配模式 / failure modes
5. 与其他页面关系
6. Sources
7. Related

## compare 页必备段落

1. 对比定位 / Scope
2. 比较维度
3. 各维度下的差异框架
4. 使用方式 / 选择指导
5. 当前状态或待补内容
6. Sources
7. Related

## topic 页必备段落

1. 定位 / Scope
2. 推荐阅读顺序
3. 主题框架 / main subtopics
4. 常见问题类型或关键决策轴
5. 使用方式
6. 当前状态
7. Sources
8. Related

## 写作规则

- wiki 不是 raw 文档收纳处，禁止粘贴大段原文。
- wiki 不是 card，不能只写 3 条 checklist 就结束。
- wiki 也不是 writeback，不能只记录一次具体事件。
- 允许先写“结构化草稿”，但必须让这页已经可导航、可引用、可继续增补。
- 如果某页引用了现有 `cards/` 或 `writeback/`，应在 `related` 或正文中明确连到具体文件。

## 三操作

- `Ingest`：发现新材料后，把已验证要点整合进对应页并追 `wiki/log.md`
- `Query`：先查 `wiki/index.md`，命中页再读正文；未覆盖再回落 L3
- `Lint`：检查 frontmatter、死链、孤立页、stale、跨页冲突，只记建议，不擅自大改结构

## 与当前知识库其他层的边界

- 原始文档原文不进 wiki
- 原始代码不进 wiki
- 方法论 / 工作流进 `skills/`
- 单次 bug / 决策 / 易错点优先进 `writeback/`
- 一句话速查、错误码、checklist、表格优先进 `cards/`
- 若某个 writeback 或 card 被反复引用，再考虑晋升为 wiki 页的一部分

## 更新要求

- 新建 wiki 页后同步更新 `wiki/index.md`
- 重要增补或新页创建后同步追加 `wiki/log.md`
- 若是知识库层面的规则变化，同时追加 `meta/CHANGELOG.md`
