# Ascend Task Router

本文件把常见任务映射到当前已落地的 L2 skill / L4 card / L6 wiki / L5 writeback / L3 MCP。AI 面对任务的第一步就读本文件。
句式：`- <触发词/场景> → skill: <name>` 或 `→ wiki: <path>` 或 `→ card: <path>` 或 `→ writeback: <path>` 或 `→ MCP: <server> query="<pattern>"`。

## 0. 查询优先级总则（当前落地版）

先判问题类型，再按固定优先级执行；命中即止，不为求全继续下钻。

- fact / 速查
  - 优先：`cards/`
  - 不够：`wiki/`
  - 仍不足：文档 RAG
- workflow / 方法论
  - 优先：`skills/`
  - 不够：`wiki/topics/` 或 `wiki/concepts/`
  - 仍不足：文档 RAG / 代码搜索
- compare / 综述
  - 优先：`wiki/compare/` 或 `wiki/topics/`
  - 不够：文档 RAG / 代码搜索
- incident / debug
  - 优先：`writeback/bug-notes/`、`writeback/pitfalls/`
  - 再看：相关 `skills/`、`cards/`、`wiki/concepts/`
- code / 仓内事实
  - 优先：代码搜索（Grep / Glob / Read）
  - 再看：`writeback/` / `skills/`
  - 最后才把 `wiki/` 当背景

补充约束：
- 知识问题默认优先复用当前已存在的 `cards/wiki/writeback` 文件。
- 仓内事实问题默认代码优先于 wiki。
- 同一任务最多加载 2 个 skill、1 页 wiki、2 次文档 RAG。
- 超过以上限制仍不收敛，先向用户确认边界。

## 1. 文档 RAG 路由

- CANN 9.0 beta2 / 950 A5 / 910 A3 / API 签名 / 参数 / Tiling / Kernel 细节 → MCP: `user-local-rag-9.0_a5` query="<中文术语> <English term> CANN 9.0 beta2 <topic>"
- 非 CANN 9.0 / 跨产品线 / FAQ / PyTorch / MindIE / MindStudio → MCP: `user-cann-rag` query="<中文术语> <English term> <topic>"
- 同一问题首次命中不足 → 允许中文术语与英文术语各重查 1 次；之后停止
- 外部 RAG 结果进入上下文前必须压缩为 3–5 条摘要，总量 < 2K tokens
- 需要把查询结果落到知识库时，优先读 skill: `docs-rag-query`

## 2. 代码检索路由

代码根：`/home/ntlab/zy/code/zhangyuan/cann_code/`

- 查精确符号 / 类名 / 函数名 / 错误串 → skill: `code-search-tactics`
- 查某类文件 / 仓库结构 / 路径模式 → skill: `code-search-tactics`
- 查“双缓冲怎么写”“queue pipeline 在哪实现”这类概念 → skill: `code-search-tactics`
- 研究某类 pattern 是否在代码库里稳定出现 → skill: `patterns-harvest-from-codebase`

已登记参考仓：
- `hccl/`
- `hccl_db/`
- `ops-nn/`
- `ops-transformer/`
- `shmem/`

## 3. 当前已落地的速查入口（优先级最高）

- DataCopy 签名 / 对齐 / buffer 预算 / 首轮排查 → card: `cards/api/datacopy.md`
- Cast 类型转换 / 精度误判 / 支持矩阵入口 → card: `cards/api/cast.md`
- ACLNN 161xxx 参数类错误 → card: `cards/error-codes/aclnn-161xxx.md`
- Tiling 自查 → card: `cards/checklists/tiling-sanity.md`
- 芯片规格字段入口（占位） → card: `cards/tables/chip-specs.md`
- 芯片可见容量速查（A2A3/A5） → card: `cards/tables/chip-memory-specs.md`
- CANN 版本差异字段入口 → card: `cards/tables/cann-version-matrix.md`
- PTO 原语总映射 → card: `cards/tables/pto-primitives-map.md`
- PTO kernel 模式映射 → card: `cards/tables/pto-kernel-pattern-map.md`
- PTO testcase 入口映射 → card: `cards/tables/pto-test-entry-map.md`

## 4. 当前已落地的 wiki 入口（优先复用）

- 不知道去哪 → wiki: `wiki/index.md`
- DataCopy 综合理解 / 约束边界 / 排障顺序 → wiki: `wiki/api/DataCopy.md`
- 910A3 可见容量与芯片入口 → wiki: `wiki/chips/910A3.md`
- 950A5 可见容量与芯片入口 → wiki: `wiki/chips/950A5.md`
- Queue / 生命周期 / 同步语义 → wiki: `wiki/concepts/pipeline-sync.md`
- PTO 总入口 / 原语-模式-示例总路由 → wiki: `wiki/topics/pto-overview.md`
- PTO include 分层 → wiki: `wiki/concepts/pto-layering.md`
- PTO ISA 映射 / category / 实现状态 → wiki: `wiki/topics/pto-isa-mapping.md`
- PTO kernel 模式 → wiki: `wiki/topics/pto-kernel-patterns.md`
- PTO testcase 路由 → wiki: `wiki/topics/pto-test-entrypoints.md`
- MC2 与手写 SHMEM 对比 → wiki: `wiki/compare/mc2-vs-shmem.md`
- MC2 全景入口 → wiki: `wiki/topics/mc2-overview.md`
- Moe distribute 默认入口 → wiki: `wiki/topics/mc2-moe-dispatch-combine-latest.md`
- grouped / quant / alltoallv 默认入口 → wiki: `wiki/topics/mc2-grouped-quant-alltoallv-latest.md`
- dispatch/combine 闭环协议 → wiki: `wiki/compare/mc2-dispatch-combine-closure.md`
- collective 二元定位 → wiki: `wiki/compare/mc2-collective-positioning.md`
- collective 三角定位 → wiki: `wiki/compare/mc2-collective-triangle.md`
- dispatch/combine 平台分流 → wiki: `wiki/compare/mc2-dispatch-combine-platform-splits.md`
- dispatch/combine host 决策链 → wiki: `wiki/compare/mc2-dispatch-combine-tiling-decisions.md`
- A5 设计到 arch35 实现映射 → wiki: `wiki/compare/mc2-a5-design-to-implementation.md`

## 5. 当前已落地的历史经验入口（incident / debug 优先）

- DataCopy 异常先查对齐与 buffer 预算 → writeback: `writeback/bug-notes/2026-04-19-datacopy-alignment-first.md`
- 生命周期问题常伪装成精度问题 → writeback: `writeback/pitfalls/2026-04-19-lifecycle-before-precision.md`
- 先写稳定结构，再逐步补事实细节 → writeback: `writeback/recipes/2026-04-19-bootstrap-structure-first.md`
- 先补最小可用层，再补高精度内容 → writeback: `writeback/decision-records/2026-04-19-minimum-viable-layers-first.md`
- PTO 纳管顺序：先原语→模式→示例→验证 → writeback: `writeback/recipes/2026-04-19-pto-route-primitives-patterns-tests.md`
- MC2 主线应先按家族理解 → writeback: `writeback/bug-notes/2026-04-19-mc2-is-multi-family-product-surface.md`
- MC2 闭环协议不要拆开读 → writeback: `writeback/bug-notes/2026-04-19-mc2-dispatch-combine-closure.md`
- MC2 平台分流中心 → writeback: `writeback/bug-notes/2026-04-19-mc2-platform-split-centers.md`
- MC2 host 决策链看 base→helper→tiling → writeback: `writeback/bug-notes/2026-04-19-mc2-base-helper-tiling-chain.md`

## 6. Kernel 开发与 Ascend C

- 环境检查 / CANN 版本 / NPU 设备 → skill: `ascendc-env-check`
- 新算子从零开发 → skill: `ascendc-kernel-develop-workflow`
- Tiling 设计 / 多核切分 / UB 预算 / buffer 规划 → skill: `ascendc-tiling-design`
- AscendC API 用法 / 调用前置条件 → skill: `ascendc-api-best-practices`
- 精度问题 / Cast 误差 / 尾块精度漂移 → skill: `ascendc-precision-debug`

## 7. PTO / 通信 / 融合

- PTO 知识路由与页面命中策略 → skill: `pto-knowledge-routing`
- PTO 探索结论如何回写 → skill: `pto-writeback-rules`
- HCCL / SHMEM / HCOMM / MC2 的能力边界与选型 → skill: `comm-architecture-overview`
- 通信死锁 / notify-wait 顺序 / rank 映射 / HCCL 报错 → skill: `comm-debug-troubleshooting`
- 计算通信融合背景 / 架构脉络 → wiki: `wiki/topics/mc2-overview.md`
- 要比较 MC2 与手写 SHMEM → wiki: `wiki/compare/mc2-vs-shmem.md`

## 8. Patterns 与模板

- 写双缓冲 / ping-pong pipeline → skill: `patterns-double-buffer`
- 做二维切分 / tile 网格 → skill: `patterns-tiling-2d-split`
- 组织 GM→UB→compute→GM 基本骨架 → skill: `patterns-gm-ub-copy-compute-copyout`
- 研究某类 pattern 是否已经在代码库中出现多次 → skill: `patterns-harvest-from-codebase`
- PTO 标准 cube/gemm 流水 → card: `cards/patterns/gemm-pipeline.md`
- PTO TopK vector 流水 → card: `cards/patterns/topk-vector-pipeline.md`
- PTO FlashAttention stages → card: `cards/patterns/flashattention-stages.md`
- PTO compute/comm decouple → card: `cards/patterns/compute-comm-decouple.md`
- PTO A5 MX / SIMT / sync → card: `cards/patterns/a5-mx-simt-sync.md`

## 9. PTO 示例代码入口

- 最小 PTO 读算写示例 → card: `cards/examples/test-tadd-a5-a2a3.md`
- FIFO / Cube->Vec → card: `cards/examples/test-tpushpop-cv.md`
- FIFO / Vec->Cube → card: `cards/examples/test-tpushpop-vc-a5.md`
- 量化 testcase → card: `cards/examples/test-tquant-a5-a2a3.md`
- 基础转置 testcase → card: `cards/examples/test-ttrans.md`
- 卷积布局变换 testcase → card: `cards/examples/test-ttrans-conv-a5.md`
- A5 MX/NZ 载入 → card: `cards/examples/test-tload-mx-nz-a5.md`
- A2A3 GM->Mat → card: `cards/examples/test-tload-gm2mat-a2a3.md`
- A2A3 Mat->GM → card: `cards/examples/test-tstore-mat2gm-a2a3.md`
- 矩阵主算子 testcase → card: `cards/examples/test-tmatmul-a5-or-a2a3.md`
- 通信原语 TGATHER testcase → card: `cards/examples/test-tgather-a5-comm.md`

## 10. 维护知识库本身

- 查文档 RAG 怎么查、怎么压缩结果 → skill: `docs-rag-query`
- 新建 card → skill: `meta-authoring-card`
- 新建 skill → skill: `meta-authoring-skill`
- 新建或改写 wiki 页 → skill: `meta-authoring-wiki-page`
- 新建 bug-note / recipe / decision-record / pitfall → skill: `meta-authoring-writeback`
- 需要模板 → writeback: `writeback/templates/`
- 需要评测样例与脚手架 → `eval/README.md`、`scripts/run-eval.sh`

## 11. 回写路由

- 单次 bug / 踩坑 / 修复经过 → `writeback/bug-notes/<date>-<short-name>.md`
- 可复用 recipe / 操作方案 → `writeback/recipes/<date>-<short-name>.md`
- 项目内决策原因 → `writeback/decision-records/<date>-<short-name>.md`
- 易错点 / 反模式 → `writeback/pitfalls/<date>-<short-name>.md`
- 新综述 / 新对比 / 新主题 → `wiki/` 对应目录，并追 `wiki/log.md`
- 高频短事实 / checklist / 错误码 → `cards/`
- 稳定方法论 / 维护流程 / pattern → `skills/`

## 12. 评测与脚本入口

- 看评测目录结构 → `eval/README.md`
- 跑最小评测脚手架 → `scripts/run-eval.sh`
- QA 样例 → `eval/qa-benchmark/datacopy-first-checks.yaml`
- Coding task 样例 → `eval/coding-tasks/gm-ub-copy-compute-copyout.yaml`
- Debug case 样例 → `eval/debugging-cases/double-buffer-lifecycle.yaml`

## 13. 术语表（防语义漂移）

- UB = Unified Buffer
- GM = Global Memory
- L1 / L0A / L0B / L0C = PTO TileType 可见片上存储层
- MTE = Memory Transfer Engine
- AICore = AI Core
- queue depth = Queue buffer 槽位数，不等同于 tile 数
- double buffer = ping-pong，两组 buffer 交替以重叠搬运与计算
- pattern = 从多个示例归并出的稳定骨架，不是单个工程的直接复制
