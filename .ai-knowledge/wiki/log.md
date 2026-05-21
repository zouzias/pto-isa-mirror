# Wiki Log

## [2026-04-20 11:30] restructure | normalize metadata, fix wiki related paths, sync project entry docs
- 补齐 2 个 PTO 路由 skill 的 YAML frontmatter：`pto-knowledge-routing`、`pto-writeback-rules`
- 补齐 25 个 PTO cards 的 YAML frontmatter，覆盖 examples / patterns / primitives / tables 缺口
- 统一修正 wiki frontmatter 中 `related` 的 repo-root 风格路径，改为相对路径，恢复后续 lint/导航可用性
- 修正 `wiki/api/DataCopy.md` 中 `status: stable` 与正文“当前页仍是 draft”的冲突表述
- 同步 `CLAUDE.md` 与 `AGENTS.md`，保持项目级入口内容一致
- 补登 `wiki/index.md` 中遗漏的 `mc2-moe-v2-v3-evolution` compare 页面入口

## [2026-04-20 09:00] restructure | wire missing skills, align MCP names, add AGENTS entry
- 新增占位 skill：`skills/ascendc-env-check/`、`ascendc-kernel-develop-workflow/`、`ascendc-tiling-design/`、`ascendc-api-best-practices/`、`ascendc-precision-debug/`、`comm-architecture-overview/`、`comm-debug-troubleshooting/`（从 `~/.codex/skills/` 显式 cp，按本项目独立演化）
- 同步更新 `skills/_INDEX.md`，把上述 7 个 skill 从"索引声明存在但目录缺失"修复为"索引与文件一致"
- 补登 `writeback/_INDEX.md` 漏登的 5 条 bug-notes：`mc2-v2-v3-are-interface-refactors`、`mc2-v2-and-postprocess-expand-whole-stack`、`mc2-has-flow-glue-operators`、`mc2-complexity-center-varies-by-family`、`mc2-shape-algebra-before-kernel`
- 新建项目根 `AGENTS.md`（内容与 `CLAUDE.md` 一致，作为 Cursor/Claude/Codex 共用文档入口）
- 全量修正 MCP server 名：`local-rag-9.0_a5` → `user-local-rag-9.0_a5`、裸 `cann-rag` → `user-cann-rag`，涉及 `router/ROUTER.md`、`router/HARD_RULES.md`、`skills/docs-rag-query/SKILL.md`、`wiki/api/DataCopy.md`、`cards/api/{datacopy,cast}.md`、`cards/error-codes/aclnn-161xxx.md`、`writeback/bug-notes/2026-04-19-datacopy-alignment-first.md` 共 13 处
- 扩写 `meta/PROJECT_POLICY.md`：补充 PTO/MC2 主题边界、工具使用约束（no-lint for AscendC）、回写边界
- 目的：把索引/路由与现场文件对齐，把 MCP 路由与真实 server 注册对齐，把跨平台入口补齐，消除"第一跳扑空"风险

## [2026-04-20 09:30] ingest | backfill 5 mc2 writeback log entries
- 本次同步补登 04-19 深入扫描期产出但未登 log 的 5 条 mc2 bug-notes（时间戳约 15:00–17:00 批次）
- 未修改文件内容，只做索引与日志补登

## [2026-04-19 00:00] ingest | bootstrap initial skills cards wiki skeleton
- 初始化 `skills/_INDEX.md`、`cards/_INDEX.md`、`wiki/index.md`、`writeback/_INDEX.md`
- 新建首批 draft 技能、卡片与 wiki 占位页
- 目的：先形成可路由的最小可用知识骨架，后续按查询与回写逐步增补

## [2026-04-19 00:30] ingest | enrich core cards and concept pages
- 补实 `cards/api/datacopy.md`、`cards/api/cast.md`、`cards/error-codes/aclnn-161xxx.md`
- 补实 `wiki/api/DataCopy.md` 与 `wiki/concepts/pipeline-sync.md`
- 目的：把首批占位页升级为可直接用于路由与答题的最小内容

## [2026-04-19 01:00] ingest | enrich mc2 topic compare and chip placeholders
- 补实 `wiki/topics/mc2-overview.md`、`wiki/compare/mc2-vs-shmem.md`、`wiki/chips/910A3.md`
- 补实 `cards/tables/chip-specs.md` 与 `cards/tables/cann-version-matrix.md`
- 目的：让主题页、对比页、芯片页和表格页具备稳定的阅读框架与后续补数规则

## [2026-04-19 01:30] ingest | add maintenance layer authoring guides
- 新增 `skills/docs-rag-query/`、`skills/meta-authoring-card/`、`skills/meta-authoring-wiki-page/`、`skills/meta-authoring-writeback/`
- 新增 `writeback/templates/` 下的 bug-note、recipe、decision-record、pitfall 模板
- 目的：把知识库的持续维护流程也沉淀成可路由的 skill 与模板

## [2026-04-19 02:00] ingest | add skill authoring guide and first writeback examples
- 新增 `skills/meta-authoring-skill/`
- 新增 `writeback/bug-notes/2026-04-19-datacopy-alignment-first.md`
- 新增 `writeback/recipes/2026-04-19-bootstrap-structure-first.md`
- 新增 `writeback/decision-records/2026-04-19-minimum-viable-layers-first.md`
- 目的：让维护链条具备从模板到真实条目的最小闭环

## [2026-04-19 02:30] ingest | add first pitfall and eval samples
- 新增 `writeback/pitfalls/2026-04-19-lifecycle-before-precision.md`
- 新增 `eval/qa-benchmark/datacopy-first-checks.yaml`
- 新增 `eval/coding-tasks/gm-ub-copy-compute-copyout.yaml`
- 新增 `eval/debugging-cases/double-buffer-lifecycle.yaml`
- 目的：让知识库从静态结构继续走向可评测的最小闭环

## [2026-04-19 03:00] ingest | add eval script scaffold and readmes
- 新增 `scripts/run-eval.sh`
- 新增 `eval/README.md` 与 `scripts/README.md`
- 目的：为后续 wiki 体检与评测回归提供可执行脚手架

## [2026-04-19 03:30] ingest | refine policy and schema to match current knowledge base
- 细化 `router/ANSWERING_POLICY.md`，补充当前已落地 cards/wiki/writeback 的复用优先级
- 细化 `router/WIKI_SCHEMA.md`，补充 draft 页写法、当前页面模式与更新要求
- 目的：让 L1 规则与实际 `.ai-knowledge` 落地状态保持一致

## [2026-04-19 04:00] ingest | refine router to concrete file entry points
- 细化 `router/ROUTER.md`，把占位入口改成当前已存在的 cards/wiki/writeback/skills 文件路径
- 增加当前已落地速查入口、wiki 入口、历史经验入口、维护知识库入口与评测入口
- 目的：让路由第一跳直接命中真实文件，而不是泛化占位路径

## [2026-04-19 04:30] ingest | refine hard rules to current knowledge layers
- 细化 `router/HARD_RULES.md`，补充当前真实资产优先复用、边界分层、索引/日志同步与 eval 脚手架约束
- 目的：让硬约束与当前 cards/wiki/writeback/skills/eval 的真实状态保持一致

## [2026-04-19 05:00] ingest | ops-transformer mc2 mainline dispatch readthrough
- 读取 `mc2/moe_distribute_dispatch/README.md`、`op_api/aclnn_moe_distribute_dispatch.cpp`、`op_host/moe_distribute_dispatch_def.cpp`、`op_kernel/moe_distribute_dispatch.cpp`
- 更新 `wiki/topics/mc2-overview.md` 与 `wiki/compare/mc2-vs-shmem.md`
- 新增 `writeback/bug-notes/2026-04-19-mc2-mainline-read-path.md`
- 目的：把主线 MC2 的分层结构（README → op_api → op_host → op_kernel）沉淀到知识层

## [2026-04-19 05:30] ingest | add mc2 family map and reading recipe
- 扫描 `ops-transformer/mc2` 主线目录，识别 MoE distribute、Matmul+通信、Grouped/Quant、转换/屏障等家族
- 更新 `wiki/topics/mc2-overview.md`
- 新增 `writeback/recipes/2026-04-19-read-mc2-by-operator-families.md`
- 目的：先建立 MC2 家族地图，再按家族深入单算子四层

## [2026-04-19 10:00] restructure | split stable mc2 conclusions into subpages
- 新增 `wiki/concepts/mc2-layering.md`
- 新增 `wiki/compare/mc2-moe-v2-v3-evolution.md`
- 更新 `wiki/index.md` 与 `wiki/topics/mc2-overview.md` 的阅读入口
- 目的：把稳定结论从 overview 中拆成可独立复用的 MC2 子页

## [2026-04-19 10:30] ingest | expand mc2 topic with v2 complexity and thin-entry flow ops
- 读取 `matmul_reduce_scatter_v2/README.md`、`op_api/aclnn_matmul_reduce_scatter_v2.cpp`、`op_host/matmul_reduce_scatter_v2_def.cpp`、`op_kernel/matmul_reduce_scatter_v2.cpp`
- 读取 `matmul_all_reduce_add_rms_norm/README.md`、`op_api/aclnn_matmul_all_reduce_add_rms_norm.cpp`、`op_host/matmul_all_reduce_add_rms_norm_def.cpp`
- 读取 `moe_update_expert/README.md`、`op_api/aclnn_moe_update_expert.cpp`、`op_host/moe_update_expert_def.cpp`
- 更新 `wiki/topics/mc2-overview.md`
- 新增 `writeback/bug-notes/2026-04-19-mc2-thin-entry-flow-ops.md`
- 目的：把 V2 类主复杂度算子与薄入口流程算子的阅读差异固化进知识层

## [2026-04-19 11:00] ingest | compare collective positioning in mc2 matmul family
- 读取 `all_gather_matmul_v2/README.md`、`op_api/aclnn_all_gather_matmul_v2.cpp`、`op_host/all_gather_matmul_v2_def.cpp`
- 读取 `allto_all_matmul/README.md`、`op_api/aclnn_allto_all_matmul.cpp`、`op_host/allto_all_matmul_def.cpp`
- 新增 `wiki/compare/mc2-collective-positioning.md`
- 新增 `writeback/bug-notes/2026-04-19-mc2-collective-is-not-just-prefix.md`
- 目的：把 AllGather 与 AlltoAll 在 Matmul 家族中的通信语义差异固定下来

## [2026-04-19 11:30] ingest | add reduce-scatter positioning into collective triangle
- 读取 `matmul_reduce_scatter/README.md`、`op_api/aclnn_matmul_reduce_scatter.cpp`、`op_host/matmul_reduce_scatter_def.cpp`
- 读取 `quant_reduce_scatter/README.md`、`op_api/aclnn_quant_reduce_scatter.cpp`、`op_host/quant_reduce_scatter_def.cpp`
- 新增 `wiki/compare/mc2-collective-triangle.md`
- 新增 `writeback/bug-notes/2026-04-19-mc2-reducescatter-is-output-slicing.md`
- 目的：把 AllGather / AlltoAll / ReduceScatter 在 Matmul 家族中的语义位置收成一张对照图

## [2026-04-19 12:00] ingest | add dispatch-combine closure view
- 读取 `moe_distribute_combine/README.md`、`op_api/aclnn_moe_distribute_combine.cpp`、`op_host/moe_distribute_combine_def.cpp`
- 读取 `moe_distribute_dispatch_v2/README.md`、`moe_distribute_combine_v2/README.md`
- 新增 `wiki/compare/mc2-dispatch-combine-closure.md`
- 新增 `writeback/bug-notes/2026-04-19-mc2-dispatch-combine-closure.md`
- 目的：把 dispatch/combine 从两个算子提升为一条发散-返还闭环协议

## [2026-04-19 12:30] ingest | add dispatch-combine platform split view
- 读取 `moe_distribute_dispatch_v2/op_api|op_host|op_kernel`
- 读取 `moe_distribute_combine_v2/op_api|op_host|op_kernel`
- 新增 `wiki/compare/mc2-dispatch-combine-platform-splits.md`
- 新增 `writeback/bug-notes/2026-04-19-mc2-platform-split-centers.md`
- 目的：固定 A2/A3/950 在 dispatch_v2/combine_v2 中的不同分流中心

## [2026-04-19 13:00] ingest | add dispatch-combine tiling decision chain
- 读取 `dispatch_v2_base.cpp`、`dispatch_tiling_helper.cpp`、`dispatch_v2_tiling.cpp`
- 读取 `combine_v2_base.cpp`、`combine_tiling_helper.cpp`、`combine_v2_tiling.cpp`
- 新增 `wiki/compare/mc2-dispatch-combine-tiling-decisions.md`
- 新增 `writeback/bug-notes/2026-04-19-mc2-base-helper-tiling-chain.md`
- 目的：把 host 决策链抽象成 base → helper → tiling 三层模型

## [2026-04-19 13:30] ingest | add A5 design-to-implementation mapping
- 读取 `MoeDistributeDispatch-Combine算子设计介绍.md`
- 读取 arch35 的 dispatch/combine tiling 与 kernel 特化文件
- 新增 `wiki/compare/mc2-a5-design-to-implementation.md`
- 新增 `writeback/bug-notes/2026-04-19-mc2-a5-design-mapping.md`
- 目的：把 A5 设计文档里的双缓冲/设备侧自治/批量通信映射到 arch35 实现

## [2026-04-19 22:00] ingest | link A3/A5 memory capacity constraints into PTO pattern and tiling pages
- 更新 `cards/checklists/tiling-sanity.md`
- 更新 `cards/patterns/gemm-pipeline.md`
- 更新 `cards/patterns/a5-mx-simt-sync.md`
- 更新 `cards/primitives/load-store.md`
- 更新 `cards/primitives/matmul-family.md`
- 目的：把 `TASSIGN_zh.md` 里的容量约束直接接到 PTO 的设计、模式与预算判断页面

## [2026-04-19 21:30] ingest | add A3 and A5 TileType-visible memory spec entries from TASSIGN docs
- 新增 `cards/tables/chip-memory-specs.md`
- 新增 `wiki/chips/950A5.md`
- 更新 `wiki/chips/910A3.md`
- 更新 `cards/_INDEX.md` 与 `wiki/index.md`
- 目的：把 `docs/isa/TASSIGN_zh.md` 里 A2A3/A5 的 TileType→内存空间→容量表直接纳入知识库

## [2026-04-19 21:00] restructure | finish rewriting remaining MC2 compare pages into routing style
- 更新 `wiki/compare/mc2-dispatch-combine-platform-splits.md`
- 更新 `wiki/compare/mc2-dispatch-combine-tiling-decisions.md`
- 更新 `wiki/compare/mc2-a5-design-to-implementation.md`
- 更新 `wiki/compare/mc2-vs-shmem.md`
- 目的：把剩余 MC2 compare 页统一到 canonical scope / hit terms / normalized routing / confusion 的大模型路由结构

## [2026-04-19 20:30] restructure | start rewriting MC2 topic and compare pages into routing style
- 更新 `wiki/topics/mc2-moe-dispatch-combine-latest.md`
- 更新 `wiki/topics/mc2-grouped-quant-alltoallv-latest.md`
- 更新 `wiki/compare/mc2-collective-positioning.md`
- 更新 `wiki/compare/mc2-collective-triangle.md`
- 目的：把 MC2 主入口页与核心 collective 对比页从说明性写法压成 canonical routing 资产

## [2026-04-19 20:00] restructure | rewrite remaining PTO example cards into routing style
- 更新 `cards/examples/test-tpushpop-cv.md`
- 更新 `cards/examples/test-tpushpop-vc-a5.md`
- 更新 `cards/examples/test-tquant-a5-a2a3.md`
- 更新 `cards/examples/test-ttrans-conv-a5.md`
- 更新 `cards/examples/test-tmatmul-a5-or-a2a3.md`
- 更新 `cards/examples/test-tgather-a5-comm.md`
- 更新 `cards/examples/test-tload-mx-nz-a5.md`
- 目的：让 PTO 全部 example cards 统一为 hit-terms / canonical-intent / anchors / confusion 的模型路由形状

## [2026-04-19 19:30] restructure | rewrite PTO primitive and pattern cards into routing style
- 更新 `cards/primitives/load-store.md`
- 更新 `cards/primitives/manual-binding-and-sync.md`
- 更新 `cards/primitives/vec-elementwise-and-reduce.md`
- 更新 `cards/primitives/matmul-family.md`
- 更新 `cards/primitives/comm-primitives.md`
- 更新 `cards/patterns/gemm-pipeline.md`
- 更新 `cards/patterns/topk-vector-pipeline.md`
- 更新 `cards/patterns/flashattention-stages.md`
- 更新 `cards/patterns/compute-comm-decouple.md`
- 更新 `cards/patterns/a5-mx-simt-sync.md`
- 目的：把 PTO cards 从说明性卡片收紧成更适合大模型命中的 canonical routing 单元

## [2026-04-19 19:00] restructure | rewrite PTO knowledge pages into model-routing style
- 更新 `wiki/topics/pto-overview.md`
- 更新 `wiki/topics/pto-isa-mapping.md`
- 更新 `wiki/topics/pto-kernel-patterns.md`
- 更新 `wiki/topics/pto-test-entrypoints.md`
- 更新 `cards/examples/test-tadd-a5-a2a3.md`
- 更新 `cards/examples/test-tload-gm2mat-a2a3.md`
- 更新 `cards/examples/test-tstore-mat2gm-a2a3.md`
- 更新 `cards/examples/test-ttrans.md`
- 目的：把 PTO 新页从“给人读的教程结构”收紧成更适合大模型命中的 canonical routing 资产

## [2026-04-19 18:30] ingest | add PTO ISA mapping page and more load-store-transform examples
- 新增 `wiki/topics/pto-isa-mapping.md`
- 新增 `cards/examples/test-tload-gm2mat-a2a3.md`
- 新增 `cards/examples/test-tstore-mat2gm-a2a3.md`
- 新增 `cards/examples/test-ttrans.md`
- 更新 `cards/tables/pto-primitives-map.md` 与 `cards/tables/pto-test-entry-map.md`
- 更新 `cards/_INDEX.md` 与 `wiki/index.md`
- 目的：把 PTO 的 manifest/category、实现状态矩阵和 load/store/transform 示例入口收得更系统

## [2026-04-19 18:00] ingest | expand PTO cards with primitive families and extra testcase entrypoints
- 新增 `cards/primitives/load-store.md`
- 新增 `cards/primitives/manual-binding-and-sync.md`
- 新增 `cards/primitives/vec-elementwise-and-reduce.md`
- 新增 `cards/primitives/matmul-family.md`
- 新增 `cards/primitives/comm-primitives.md`
- 新增 `cards/examples/test-tgather-a5-comm.md`
- 新增 `cards/examples/test-tload-mx-nz-a5.md`
- 新增 `writeback/recipes/2026-04-19-pto-route-primitives-patterns-tests.md`
- 更新 `cards/_INDEX.md` 与 `writeback/_INDEX.md`
- 目的：把 PTO 主线从总骨架继续补到原语簇和补充样例入口层

## [2026-04-19 17:30] ingest | add PTO knowledge skeleton across primitives patterns and tests
- 新增 `wiki/topics/pto-overview.md`
- 新增 `wiki/concepts/pto-layering.md`
- 新增 `wiki/topics/pto-kernel-patterns.md`
- 新增 `wiki/topics/pto-test-entrypoints.md`
- 新增 `cards/tables/pto-primitives-map.md`、`pto-kernel-pattern-map.md`、`pto-test-entry-map.md`
- 新增首批 `cards/patterns/*` 与 `cards/examples/*`
- 新增 `skills/pto-knowledge-routing/` 与 `skills/pto-writeback-rules/`
- 更新 `wiki/index.md`、`cards/_INDEX.md`、`skills/_INDEX.md`
- 目的：把 PTO 原语、kernel 模式与 testcase 入口正式纳入 `.ai-knowledge` 的统一知识骨架

## [2026-04-19 17:00] ingest | add latest grouped-quant-alltoallv topic entry
- 新增 `wiki/topics/mc2-grouped-quant-alltoallv-latest.md`
- 更新 `wiki/index.md`
- 更新 `wiki/topics/mc2-overview.md` 的推荐阅读顺序与 grouped/quant/alltoallv 入口描述
- 目的：把 grouped / quant / alltoallv 家族的 latest-version-first 规则落成独立主题入口

## [2026-04-19 16:30] restructure | add latest-version-first rule for grouped quant alltoallv families
- 更新 `wiki/topics/mc2-overview.md`
- 为 grouped / quant / alltoallv 家族补充“默认只保留最新版入口”的统一表述
- 目的：把 latest-version-first 规则从 moe / matmul / collective 扩展到剩余高组合度家族

## [2026-04-19 16:00] restructure | remove remaining legacy version wording from mc2 overview
- 更新 `wiki/topics/mc2-overview.md`
- 清理 matmul / collective / moe 残留的“基础版 vs v2/v3”叙事
- 统一改成“最新版主入口 + 历史背景”表述
- 目的：让 overview 与各子页的 latest-version-first 口径完全一致

## [2026-04-19 15:30] restructure | collapse matmul and collective families to latest-version-first entry
- 更新 `wiki/compare/mc2-collective-positioning.md`
- 更新 `wiki/compare/mc2-collective-triangle.md`
- 更新 `wiki/topics/mc2-overview.md` 中的 collective 导航表述
- 目的：让 matmul / collective 家族默认只保留最新版入口，基础版与旧路径只作为背景材料

## [2026-04-19 15:00] restructure | collapse mc2 operator knowledge to latest-version-first entry
- 更新 `wiki/topics/mc2-overview.md`
- 更新 `wiki/topics/mc2-moe-dispatch-combine-latest.md` 的上游导航定位
- 更新 `wiki/compare/mc2-dispatch-combine-closure.md`
- 降级 `wiki/compare/mc2-moe-v2-v3-evolution.md` 为历史背景页，并从 `wiki/index.md` 移除主入口
- 目的：让 MC2 家族默认只保留最新版入口，旧版本只作为背景材料存在

## [2026-04-19 14:30] ingest | refine latest v4 entry into answer-ready topic page
- 更新 `wiki/topics/mc2-moe-dispatch-combine-latest.md`
- 强化参数分组、必填/可选、平台差异、常见误读与使用方式
- 目的：把最新版入口页从“统一结论”进一步压成可直接答题的主题页

## [2026-04-19 14:00] ingest | add latest v4 unified dispatch-combine entry
- 读取 `aclnnMoeDistributeDispatchV4.md`、`aclnnMoeDistributeCombineV4.md`
- 读取 `aclnn_moe_distribute_dispatch_v4.cpp`、`aclnn_moe_distribute_combine_v4.cpp`
- 新增 `wiki/topics/mc2-moe-dispatch-combine-latest.md`
- 新增 `writeback/bug-notes/2026-04-19-mc2-latest-entry-is-v4.md`
- 目的：把主知识层默认入口统一收敛到 DispatchV4 / CombineV4
