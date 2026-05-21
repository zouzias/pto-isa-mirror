# Skills Index

## kernel / workflow
- `ascendc-env-check`：检查 CANN 环境、驱动、设备、NPU 可见性。(imported)
- `ascendc-kernel-develop-workflow`：新 Ascend C 算子开发的阶段化工作流（7 阶段，不适用于 ops-* 算子仓）。(imported)
- `ascendc-tiling-design`：多核切分、UB 预算、buffer 规划（含 references/、scripts/、templates/）。(imported)
- `ascendc-api-best-practices`：Ascend C 基础 API 的正确用法与限制（含 references/）。(imported)
- `ascendc-precision-debug`：精度异常、Cast 误差、同步问题排查（含 references/）。(imported)

## communication / fusion
- `comm-architecture-overview`：HCCL、SHMEM、HCOMM、MC2 的能力边界与适用面。(imported)
- `comm-debug-troubleshooting`：通信死锁、notify/wait 顺序、rank 映射排查。(imported)

> `(imported)` 标记来自用户级 `~/.codex/skills/` 的显式 `cp`，按项目演化独立维护；如需同步上游，再次 `cp` 覆盖并 `git diff` 审阅。

## patterns
- `patterns-double-buffer`：双缓冲 / ping-pong pipeline 的稳定骨架。
- `patterns-tiling-2d-split`：二维切分与 tile 网格组织的骨架。
- `patterns-gm-ub-copy-compute-copyout`：GM→UB→计算→GM 的基本流水骨架。
- `patterns-harvest-from-codebase`：从代码仓批量提炼 pattern 草稿的流程。

## meta / tactics
- `code-search-tactics`：代码问题下 Grep、Glob、Read 的选择战术。
- `docs-rag-query`：外部文档 RAG 的 query 组装与证据压缩。
- `meta-authoring-card`：新建或规范化 card 的写法。
- `meta-authoring-skill`：新建或规范化 skill 的写法。
- `meta-authoring-wiki-page`：新建或改写 wiki 页面的方法。
- `meta-authoring-writeback`：沉淀 bug-note、recipe、decision-record、pitfall 的模板与准入规则。
- `pto-knowledge-routing`：PTO 问题下如何在 wiki/cards/code/docs 之间路由。
- `pto-writeback-rules`：PTO 探索结论如何回写进 `.ai-knowledge`。
