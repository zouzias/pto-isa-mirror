2026-04-19 | expand ROUTER with PTO, MC2, and chip-memory assets | 将新增的 PTO / MC2 / 芯片容量资产系统接入 L1 路由表
2026-04-19 | wire ai-knowledge into project CLAUDE guidance | 将 .ai-knowledge 明确接入项目级 CLAUDE.md，作为内置知识路由层
2026-04-19 | link hardware memory constraints into PTO pattern and tiling pages | 把 TASSIGN 文档里的容量约束接到 PTO 设计、模式与 budget 判断页
2026-04-19 | add A3 and A5 TileType-visible memory specs from TASSIGN docs | 基于 TASSIGN 文档补充 A2A3/A5 的 UB/L1/L0/FBuffer 可见容量表
2026-04-19 | finish rewriting remaining MC2 compare pages into routing style | 将剩余 MC2 compare 页统一成 canonical routing 资产
2026-04-19 | start rewriting MC2 pages into routing style | 开始把 MC2 主 topic 与核心 compare/topic 页收紧为 canonical routing 资产
2026-04-19 | rewrite remaining PTO example cards into routing style | 将剩余 PTO examples 统一为 hit-terms / canonical-intent / anchors / confusion 结构
2026-04-19 | rewrite PTO primitive and pattern cards into routing style | 将 PTO primitives/patterns cards 收紧成更适合大模型命中的 canonical routing 单元
2026-04-19 | rewrite PTO pages into model-routing style | 将 PTO topic/example 页从教程式叙述收紧为 canonical routing 资产
2026-04-19 | add PTO ISA mapping and load-store-transform examples | 新增 PTO ISA 映射页，并补充 tload/tstore/ttrans 示例入口与映射表
2026-04-19 | expand PTO cards with primitive families and extra testcase entrypoints | 补充 PTO 原语簇卡、tgather/tload_mx_NZ 示例卡与 PTO 纳管 recipe
2026-04-19 | add PTO knowledge skeleton across primitives patterns and tests | 新增 PTO 主题页、映射表、模式卡、示例卡与知识路由 skill
2026-04-19 | add version-to-chip default mapping guidance | 补充 CANN 9.0→A5/950、CANN 8.5→A3/910 的默认对应规则
2026-04-19 | add latest grouped quant alltoallv topic entry | 新增 grouped / quant / alltoallv 家族的独立最新版主题入口并接入导航
2026-04-19 | add latest-version-first rule for grouped quant alltoallv families | 将 grouped / quant / alltoallv 家族也统一到最新版入口优先口径
2026-04-19 | remove remaining legacy version wording from mc2 overview | 清理 overview 中残留的基础版/v2/v3 叙事，统一到最新版主入口口径
2026-04-19 | bootstrap skills/cards/wiki indexes | 初始化 L2/L4/L5/L6 索引与首批占位内容
2026-04-19 | enrich core cards and wiki pages | 补实 DataCopy/Cast/ACLLN 161xxx 卡片与 DataCopy/pipeline-sync 页面
2026-04-19 | enrich mc2/chip/topic placeholders | 补实 mc2-overview、mc2-vs-shmem、910A3 与表格页结构
2026-04-19 | add maintenance layer skills and templates | 新增 docs-rag-query、meta-authoring-* 与 writeback 模板
2026-04-19 | add meta-authoring-skill and first writeback examples | 新增 skill 写作规范与首批 bug-note/recipe/decision-record 示例
2026-04-19 | add first pitfall and eval samples | 新增 pitfall 示例与首批 qa/code/debug 评测样例
2026-04-19 | add eval script scaffold and readmes | 新增 run-eval.sh 脚手架与 eval/scripts 说明
2026-04-19 | refine answering policy and wiki schema | 细化 ANSWERING_POLICY 与 WIKI_SCHEMA 以贴合当前已落地知识层
2026-04-19 | refine router to concrete file entry points | 将 ROUTER 从占位入口细化为当前已存在文件与路径的精确路由
2026-04-19 | refine hard rules to current knowledge layers | 细化 HARD_RULES 以贴合当前 cards/wiki/writeback/skills/eval 的真实状态
2026-04-19 | ingest ops-transformer mc2 mainline dispatch path | 读取主线 mc2 dispatch 的 README/op_api/op_host/op_kernel 并融入知识层
2026-04-19 | add mc2 family map and reading recipe | 基于 mc2 目录结构与代表 README 补充家族地图与阅读方法
2026-04-19 | synthesize mc2 versioning with flow-glue layer | 把 V2/V3 演进与 setup/teardown/route 配套层合并成更系统的认知
2026-04-19 | add mc2 wiki subpages for layering and version evolution | 新增 mc2-layering 与 mc2-moe-v2-v3-evolution 子页并接入索引
2026-04-19 | add mc2 thin-entry flow-op note and expand overview | 补充 MatmulReduceScatterV2、MatmulAllReduceAddRmsNorm、MoeUpdateExpert 读码结论
2026-04-19 | add collective-positioning compare page | 补充 AllGatherMatmulV2 与 AlltoAllMatmul 的通信语义对比
2026-04-19 | add collective-triangle compare page | 补充 AllGather / AlltoAll / ReduceScatter 三角定位
2026-04-19 | add dispatch-combine closure compare page | 补充 Dispatch/Combine 闭环协议视角
2026-04-19 | add dispatch-combine platform-splits compare page | 补充 dispatch_v2/combine_v2 在 A2/A3/950 的分流中心
2026-04-19 | add dispatch-combine tiling decisions page | 补充 base/helper/tiling 三层 host 决策链
2026-04-19 | add A5 design-to-implementation mapping | 补充 A5 设计文档与 arch35 实现的对应关系
2026-04-19 | collapse matmul and collective entries to latest-version-first navigation | 将 matmul / collective 家族默认入口统一到最新版，基础版只保留为背景
2026-04-19 | collapse mc2 entries to latest-version-first navigation | 将 MC2 主入口统一收敛到最新版，旧版本页降级为历史背景
2026-04-19 | refine latest v4 topic into answer-ready entry | 强化参数分组、必填/可选、平台差异与常见误读，提升直接答题能力
2026-04-19 | add latest v4 unified entry | 收敛主知识层到 DispatchV4/CombineV4 统一入口
2026-04-20 | wire 7 imported skills and align MCP names | 从 ~/.codex/skills 显式 cp 7 个 ascendc-*/comm-* skill 目录进项目、修正 local-rag-9.0_a5 与 cann-rag 为 user-* 前缀、新增 AGENTS.md、补 writeback/_INDEX.md 漏登 5 条 bug-notes、扩写 PROJECT_POLICY.md、补登 wiki/log.md
2026-04-20 | normalize knowledge metadata and wiki links | 补齐 PTO 路由 skill 与 cards 的 frontmatter，修正 wiki related 相对路径，统一 AGENTS/CLAUDE，并补登遗漏的 wiki 索引入口
