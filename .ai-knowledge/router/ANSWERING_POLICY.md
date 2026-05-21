# Answering Policy

## 版本意识

- 每次涉及 API、参数、行为、芯片能力的回答，必须显式说明适用版本和芯片。
- 若用户未指定版本，默认写清“以下基于当前知识库基线：CANN 9.0 beta2；若与你的环境不一致，需要再确认”。
- 若用户未指定芯片或 SoC，默认对应关系写清：**CANN 9.0** 优先按 **A5 / 950** 形态理解，**CANN 8.5** 优先按 **A3 / 910** 形态理解；若用户明确指定芯片、SoC 或运行环境，以用户指定为准。
- 若问题明显版本敏感且现有证据不足，先确认再回答；不要把暂定版本说成默认事实。

## 来源标注

- 事实性断言后附来源类型与路径/查询，格式优先写成：`[source: official/example/note, <path-or-query>]`。
- `official`：外部文档 RAG 或官方材料，可作为事实依据。
- `example`：官方示例或代码样例，可作为写法参考，但不等于当前仓现状。
- `note` / `writeback`：经验来源，只能作为补充、提醒或待验证线索，不能单独升级成硬事实。
- 若引用本知识库已有页面，优先指向具体文件，如 `cards/api/datacopy.md`、`wiki/api/DataCopy.md`、`writeback/bug-notes/...`。

## 先判问题类型，再选层

- fact / 速查：优先 `cards/`，不够再看 `wiki/entity`，最后才回落文档 RAG。
- workflow / 方法论：优先 `skills/`，再看 `wiki/topic` / `wiki/concept`。
- compare / 综述：优先 `wiki/compare` / `wiki/topic`。
- incident / debug：优先 `writeback/bug-notes` / `writeback/pitfalls`，再看相关 skill。
- code / 仓内事实：优先代码搜索，不用 wiki 覆盖当前代码现实。

## wiki、card、writeback、skill 的使用边界

- `cards/` 回答“读完即可答”的短事实、错误码、checklist、表格。
- `wiki/` 回答“为什么、怎么理解、怎么比较”的综合问题。
- `writeback/` 复用历史问题、经验、决策和易错点。
- `skills/` 用于方法论、流程、套路和维护规范。
- 当同一问题横跨多层时，先用更轻的层收敛，再决定是否下钻。

## 证据不足先问

证据不足时，优先使用以下模式：
- 我没有找到官方证据，请先确认版本/芯片/环境。
- 现有证据只覆盖 `<范围>`，超出部分我不能下断言。
- 目前只有经验性来源支持该说法，请把它视为待验证线索。

## 推测与事实分离

- 事实：`根据 <source>，X。`
- 推测：`我推测 X，原因是 <reason>。`
- 不允许把推测伪装成事实，也不允许用“应该”“一般”掩盖证据空洞。

## 冲突显式化

- 多源冲突时必须摆出来，不擅自调和。
- 默认可靠性优先级：`official > example > note/writeback`。
- 若知识库现有页与新证据冲突，先说明冲突，再决定是否需要回写更新。

## 查询压缩协议

- 外部文档 RAG：最多保留 3–5 条摘要，每条控制在短摘要级别，不透传原文大段。
- 代码检索：最多保留 3–5 个代表文件的摘要或定位信息。
- 最终证据包总量控制在 < 2K tokens。
- 如果证据包已经足够回答，就停止继续检索，不为求全继续扩张。

## 当前知识库落地约束

- 当前 `cards/` 已有：DataCopy、Cast、ACLNN 161xxx、tiling-sanity、chip-specs、cann-version-matrix。
- 当前 `wiki/` 已有：DataCopy、910A3、pipeline-sync、mc2-vs-shmem、mc2-overview。
- 当前 `writeback/` 已有：DataCopy 对齐排查、结构优先搭建、最小可用层优先、生命周期先于精度判断。
- 回答时优先复用这些现有落点，而不是重复造轮子。

## 回答形状

推荐顺序：
1. 直接答案
2. 关键证据或入口页
3. 不确定性 / 冲突
4. 下一步建议

若用户明显是在做开发而不是做知识管理，优先给结论与下一步，不要先长篇解释知识库结构。

## 回写承诺

- 新 bug / 新坑 / 新决策 → 回写 `writeback/`
- 新综述 / 新对比 / 新主题 → 回写 `wiki/`
- 高频短事实 / checklist / 错误码 → 优先沉淀 `cards/`
- 通用流程与稳定套路 → 沉淀 `skills/`
- 若只是补了维护流程或模板，也要同步 `meta/CHANGELOG.md` 与 `wiki/log.md`
