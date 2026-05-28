# moe_dispatch_combine_a8w8 Task Status Ledger

本文档只跟踪执行状态、owner、轻量 report 路径、Issue Log 和 Design Change Log。开发任务内容、依赖、文件范围
和验收标准以 `DESIGN.md` 第 13 章为准；不要在本文件重复维护任务说明。

当前目标：A3 A8W8/int8 主路径，PTO-only，不引入 Catlass/AscendC fallback。

## 执行状态台账

本节是当前 task state、owner、report、open issue、design change 和跨阶段 handoff
的单一台账。agent 领取、实现、交付、阻塞或发现设计问题时，必须更新本节，不能只写在聊天上下文里。

### 状态枚举

| State | 含义 | 谁可以写 |
| --- | --- | --- |
| `not_started` | 尚未领取 | coordinator |
| `claimed` | 已领取但未开始改代码 | task owner |
| `in_progress` | 正在实现或验证 | task owner |
| `review_ready` | 已交付，等待 review | task owner |
| `accepted` | 验收通过，可作为下游依赖 | reviewer/coordinator |
| `needs_fix` | review 未通过，需要返工 | reviewer/coordinator |
| `needs_user_decision` | 实现发现设计目标、精度、protocol、PTO-only 或验收口径需要用户确认 | task owner 或 coordinator |
| `blocked` | public primitive、硬件、构建环境或上游缺失导致无法继续 | task owner 或 reviewer |
| `superseded` | 被新的设计或任务替代 | coordinator |

### 阶段状态

| Stage | Scope | Required close condition | State | Report |
| --- | --- | --- | --- | --- |
| M0 | 工程初始化、脚本、layout、host smoke | M0.1-M0.6 全部 `accepted`，无 open `needs_user_decision` | in_progress | none |
| M1 | protocol mock 闭环 | M1.0-M1.11 全部 `accepted`，无 open `needs_user_decision` | not_started | none |
| M2 | A3 int8_int8 backend + MegaMoE-ready data path | M2.1-M2.8 全部 `accepted`，包含 M2.2a/M2.2b/M2.2c/M2.7a，dispatch route/pack/quant、Dispatch-GMM soft-sync ledger、Swiglu sync-group metadata、combine epilogue/return 和 tile-split return map 已合并到最终布局，int32 accumulator 精确对齐，epilogue/final output 按 tolerance 对齐，无 open `needs_user_decision` | not_started | none |
| M3 | overlap skeleton 与 timeline | M3.0-M3.9 全部 `accepted` 或明确 `blocked` 且不影响 M4 最小回归；只在 M2 固定的 edge/layout/signal 上启用或验证 async，无 open `needs_user_decision` | not_started | none |
| M4 | 最终 PTO 化收口 | M4.1-M4.2 全部 `accepted`，无 open `needs_user_decision` | not_started | none |

### Task 状态

| Task | State | Owner | Report | Last update |
| --- | --- | --- | --- | --- |
| M0.1 | accepted | design-agent/current-session | reports/M0.1.md | design bootstrap; implementation agents must not claim |
| M0.2 | review_ready | codex/current-session | reports/M0.2.md | DCL-58: single fused mixed-kernel target implemented and build-verified |
| M0.3 | review_ready | codex/current-session | reports/M0.3.md | DCL-58: explicit run script args, MPI rank default, and dry-run path implemented |
| M0.4 | review_ready | codex/current-session | reports/M0.4.md | DCL-58: workspace/window layout dump interfaces implemented |
| M0.5 | review_ready | codex/current-session | reports/M0.5.md | DCL-58: deterministic data and correctness-report skeleton implemented |
| M0.6 | review_ready | codex/current-session | reports/M0.6.md | DCL-58: host executable dry-run stdout proof implemented |
| M1.0 | not_started | unassigned | none | DCL-48: verify protocol invariants, do not redesign protocol |
| M1.1 | not_started | unassigned | none | initial |
| M1.2 | not_started | unassigned | none | initial |
| M1.3 | not_started | unassigned | none | initial |
| M1.4 | not_started | unassigned | none | initial |
| M1.5 | not_started | unassigned | none | initial |
| M1.6 | not_started | unassigned | none | initial |
| M1.6a | not_started | unassigned | none | initial |
| M1.7 | not_started | unassigned | none | initial |
| M1.8 | not_started | unassigned | none | initial |
| M1.9 | not_started | unassigned | none | initial |
| M1.10 | not_started | unassigned | none | initial |
| M1.11 | not_started | unassigned | none | DCL-48: depends on M1.0-M1.10 |
| M2.1 | not_started | unassigned | none | DCL-47: reserve control/sync/Sub-Tile/timeline fields |
| M2.2 | not_started | unassigned | none | initial |
| M2.2a | not_started | unassigned | none | initial |
| M2.2b | not_started | unassigned | none | initial |
| M2.2c | not_started | unassigned | none | initial |
| M2.3 | not_started | unassigned | none | initial |
| M2.4 | not_started | unassigned | none | initial |
| M2.5 | not_started | unassigned | none | DCL-47/48: fixes swigluSyncGroups/dequantSum metadata |
| M2.6 | not_started | unassigned | none | initial |
| M2.7 | not_started | unassigned | none | initial |
| M2.7a | not_started | unassigned | none | DCL-56: subTileReady typed view/layout is required, not optional |
| M2.8 | not_started | unassigned | none | DCL-54: includes swiglu_sync_groups acceptance field |
| M3.0 | not_started | unassigned | none | initial |
| M3.1 | not_started | unassigned | none | DCL-47: enables existing sync-group signals/counters |
| M3.2 | not_started | unassigned | none | initial |
| M3.3 | not_started | unassigned | none | initial |
| M3.4 | not_started | unassigned | none | initial |
| M3.5 | not_started | unassigned | none | initial |
| M3.6 | not_started | unassigned | none | initial |
| M3.7 | not_started | unassigned | none | DCL-47: opens M2.2c scoreboard ledger without redefining |
| M3.8 | not_started | unassigned | none | DCL-47/56: verifies M2.7a Sub-Tile plan; ready queue is optional only |
| M3.9 | not_started | unassigned | none | DCL-50: uses pre-reserved timeline fields only |
| M4.1 | not_started | unassigned | none | initial |
| M4.2 | not_started | unassigned | none | initial |

### Issue Log

| ID | Severity | Type | Affected tasks | State | Owner | Needs user decision | Question to user | User decision | Summary | Decision / next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| none | none | none | none | none | none | no | none | none | No open issue at initialization | none |

Severity:

- `P0`: 精度路线、protocol 不变量、PTO-only 约束或内存安全错误；下游必须停。
- `P1`: 当前 task 无法验收，或会影响后续阶段接口；相关下游停。
- `P2`: 局部实现缺陷，有 workaround，不影响下游领取。
- `P3`: 文档澄清、日志改善、非阻塞优化。

Type:

- `design-bug`: `DESIGN.md` 与代码事实、任务验收或实现路径冲突。
- `task-gap`: `DESIGN.md` 第 13 章的任务文件范围、依赖、验收证据或命令不够。
- `impl-bug`: 实现错误，设计仍成立。
- `primitive-gap`: PTO public primitive 缺失或能力不明。
- `env-gap`: 硬件、CANN、脚本或构建环境缺失。
- `test-gap`: reference、golden、dump 或验收命令不能证明任务完成。
- `user-decision`: 继续实现会改变最终目标、精度路线、MegaMoE protocol、PTO-only 约束或阶段验收口径，需要用户确认。

### Design Change Log

| ID | Source issue | Files changed | Affected tasks | Reviewer state | Summary |
| --- | --- | --- | --- | --- | --- |
| DCL-2026-05-28-01 | user-request-megamoe-alignment | `DESIGN.md`, `TASKS.md` | M0.1, M1.0, M3.7-M3.9, M4.2 | coordinator-authored | Clarified final target alignment with MegaMoE algorithm/overlap goals, added reference paths, removed misleading future-scope wording, and fixed DESIGN section numbering. |
| DCL-2026-05-28-02 | user-request-feedback-gate | `DESIGN.md`, `TASKS.md`, `reports/TEMPLATE.md` | all tasks | coordinator-authored | Added user confirmation gate for implementation/design divergence and required reports to record decision questions, options, and user decisions. |
| DCL-2026-05-28-03 | user-request-rule-review | `DESIGN.md`, `TASKS.md` | all stage gates | coordinator-authored | Tightened execution rules after design/task review: added `user-decision` to issue types, made open `needs_user_decision` block stage gates, and clarified that task-owner `done` maps to `review_ready`, not `accepted`. |
| DCL-2026-05-28-04 | user-clarified-m0-1-design-task | `TASKS.md`, `reports/M0.1.md` | M0.1, M0.2-M0.6 dependencies | coordinator-authored | Clarified that M0.1 is the design-agent bootstrap task from the current session, marked it accepted with a report, and kept implementation agents starting from M0.2. |
| DCL-2026-05-28-05 | user-request-correctness-e2e-perf | `DESIGN.md`, `TASKS.md`, `reports/TEMPLATE.md` | M0.3, M0.5, M0.6, M1.11, M2.8, M3.6, M3.9, M4.1 | coordinator-authored | Added explicit final-output correctness reports and E2E timing/performance reports, using `kernels/manual/a2a3/gemm_ar` host verification/perf pattern as reference. |
| DCL-2026-05-28-06 | user-request-human-goal-split | `TASKS.md` | M0.2-M4.2 | coordinator-authored | Earlier state, folded into `DESIGN.md` section 13 by DCL-2026-05-28-42: split the human-readable goal view into observable sub-goals without changing task ownership or scope. |
| DCL-2026-05-28-07 | user-request-console-reports | `DESIGN.md`, `TASKS.md`, `reports/TEMPLATE.md`, `reports/M0.1.md` | M0.3, M0.5, M0.6, M1.11, M2.8, M3.6, M3.9, M4.1 | coordinator-authored | Earlier state, superseded by DCL-2026-05-28-44: changed correctness/perf evidence from required report files to structured console reports. Task reports now keep only light summaries and must not copy stdout. |
| DCL-2026-05-28-08 | phase-one-implementation-review | `DESIGN.md`, `TASKS.md` | M0.2, M0.6, all dependency scans | coordinator-authored | Reviewed M0 implementation against existing PTO manual projects; clarified CMake/host executable handoff, placeholder headers, and forbidden AscendC build helpers. |
| DCL-2026-05-28-09 | user-clarified-no-catlass-reference | `DESIGN.md`, `TASKS.md` | all dependency scans, all source/build files | coordinator-authored | Strengthened PTO-only rule: project source, build scripts, and run scripts must not reference catlass in any letter case, including headers, libraries, symbols, build helpers, or hidden fallback paths. |
| DCL-2026-05-28-10 | phase-two-task-review | `DESIGN.md`, `TASKS.md` | M2.1-M2.8 | coordinator-authored | Earlier M2 review added a separate GMM1-input reorder/routing-quant bridge plus scale/SwiGLU checks; the separate reorder main path is superseded by DCL-2026-05-28-12. |
| DCL-2026-05-28-11 | m1-m2-acceptance-review | `DESIGN.md`, `TASKS.md` | M1.6a-M1.11, M2.8 | coordinator-authored | Re-reviewed M1/M2 acceptance granularity: moved cumsum/preSum construction before GatherDispatch, clarified exact-vs-tolerance checks, and kept E2E timing as recorded evidence rather than a performance gate. |
| DCL-2026-05-28-12 | user-feedback-megamoe-gap | `DESIGN.md`, `TASKS.md`, `reports/TEMPLATE.md` | M2.1-M2.8, M3.0-M3.6 | coordinator-authored | Moved MegaMoE data-path merge points into M2: dispatch route/pack/quant writes peer-visible int8 payload, dispatch gather writes direct GMM1 input, and GMM2 epilogue writes remote return payload directly. M3 now focuses on overlap scheduling instead of data-layout rewrite. |
| DCL-2026-05-28-13 | user-feedback-softsync-tilesplit-stage2 | `DESIGN.md`, `TASKS.md`, `reports/TEMPLATE.md` | M2.2c, M2.7a, M2.8, M3.7, M3.8 | coordinator-authored | Moved Dispatch-GMM soft-sync ledger and GMM-Combine tile-split return mapping into M2 as semantic/layout requirements; M3.7/M3.8 now only prove async execution and primitive capability on top of those structures. |
| DCL-2026-05-28-14 | user-feedback-swiglu-sync-granularity | `DESIGN.md`, `TASKS.md`, `reports/TEMPLATE.md` | M3.1, M3.3, M3.4, M3.9 | coordinator-authored | Added the SwiGLU/quant coarse-to-fine sync plan: `swigluSyncGroups` is dumped in layout, activation consumes grouped `dequantSum` row ranges, GMM2 waits group-ready ranges instead of full activation, and console timeline plus light report summary must expose activation group intervals. |
| DCL-2026-05-28-15 | design-task-final-review | `DESIGN.md`, `TASKS.md`, `reports/TEMPLATE.md` | M1.0-M4.2 | coordinator-authored | Re-reviewed markdown responsibilities, M1-M4 stage goals, and task acceptance evidence for handoff to implementation agents; removed reference-function wording from the main path and clarified report template boundaries. |
| DCL-2026-05-28-16 | user-restated-no-ascendc-catlass-interface | `DESIGN.md`, `TASKS.md` | all tasks | coordinator-authored | Re-emphasized PTO-only as a non-negotiable interface constraint: project source cannot include, call, wrap, forward, link, or hide Catlass/AscendC APIs; missing PTO capability must become a PTO primitive task or blocked issue. |
| DCL-2026-05-28-17 | user-request-performance-feature-coverage | `TASKS.md` | M2.2a-M4.2 | coordinator-authored | Earlier state, folded into `DESIGN.md` sections 0, 5, 6, 11, and 13 by DCL-2026-05-28-42: added MegaMoE performance-feature coverage for core fusion, dispatch, combine, metadata, overlap, soft sync, Sub-Tile return, timeline, risk reporting, and SwiGLU grouping. |
| DCL-2026-05-28-18 | user-request-dependency-sync-first | `DESIGN.md`, `TASKS.md` | M1.0-M3.9 | coordinator-authored | Required the final dependency graph and synchronization contract to be defined before implementation: M1/M2 may use overlap-off/BSP wait or ledger-only, but must use the final producer/consumer edges, signals, counters, row ranges, and layout so M3 opens overlap without second refactor. |
| DCL-2026-05-28-19 | user-request-final-code-first | `DESIGN.md`, `TASKS.md` | M1.0-M3.9 | coordinator-authored | Strengthened the implementation route: accepted code must start from the final stage graph and final layout; mock may only replace GMM numeric output or missing primitive behavior, while multi-launch/temporary host barriers are debug-only and cannot be accepted main path. |
| DCL-2026-05-28-20 | user-request-remove-stale-design-wording | `DESIGN.md`, `TASKS.md` | M1.0-M3.9 | coordinator-authored | Removed stale wording that could imply a later performance-version rewrite. DESIGN now states that M1/M2 already use the final fused stage graph and that M3 only verifies or enables async execution on existing producer/consumer edges. |
| DCL-2026-05-28-21 | user-question-status-vs-alignment | `DESIGN.md`, `TASKS.md` | M0.1, M1.0-M3.9 | coordinator-authored | Merged DESIGN status and final-alignment conclusions into one section so implementation agents have a single opening source for scope, baseline, MegaMoE alignment, and async-overlap boundaries. |
| DCL-2026-05-28-22 | user-review-project-position-and-basis | `DESIGN.md`, `TASKS.md` | M0.2-M2.8 | coordinator-authored | Tightened DESIGN project positioning and design-basis sections: clarified this is a PTO manual kernel rather than an external-wrapper project, listed main-path coverage, and fixed A8W8 scale wording to use the uint64 deq-scale view with host-equivalent decode. |
| DCL-2026-05-28-23 | user-review-goals-and-non-goals | `DESIGN.md`, `TASKS.md` | M1.0-M4.2 | coordinator-authored | Rewrote DESIGN goals as current A3 A8W8/int8 acceptance targets, renamed the precision table to current implementation requirements, and tightened non-goals/PTO-only forbidden dependencies to match TASKS scan rules. |
| DCL-2026-05-28-24 | user-question-precision-table-duplication | `DESIGN.md`, `TASKS.md` | M2.1-M2.8 | coordinator-authored | Removed duplication between DESIGN 3.1 and 5: 3.1 is now the dtype/stage acceptance source, while section 5 defines numeric computation order, scale decode, quant/requant, restore, and tolerance evidence. |
| DCL-2026-05-28-25 | user-question-need-for-5-1 | `DESIGN.md`, `TASKS.md` | M2.1-M2.8 | coordinator-authored | Removed the extra DESIGN 5.1 subsection heading and renamed section 5 to numeric contract and acceptance semantics; kept the content because it is the canonical source for scale decode, quant/requant, restore order, and tolerance evidence. |
| DCL-2026-05-28-26 | user-review-overall-architecture | `DESIGN.md`, `TASKS.md` | M1.0-M3.9 | coordinator-authored | Reworked DESIGN overall architecture around host runtime, single fused MPMD kernel, ProtocolAndLayout, AIV/AIC stages, and current-only A3 int8 numeric stage set; clarified that PTO primitives are direct stage calls and mock may only replace GMM numeric output. |
| DCL-2026-05-28-27 | user-review-kernel-stage-sections | `DESIGN.md`, `TASKS.md` | M1.0-M3.9 | coordinator-authored | Tightened DESIGN section 7: removed loose first-version wording, added WaitCounts and BuildCumsumAndPreSumBeforeRank to the stage graph, clarified GMM1 metadata/dispatch gates, changed M3 language to verify/enable async rather than rewrite, and aligned 7.2/7.3 wording with current task gates. |
| DCL-2026-05-28-28 | user-review-dataflow-overlap-view | `DESIGN.md`, `TASKS.md` | M2.2c, M2.7a, M3.0-M3.9 | coordinator-authored | Added a separate MegaMoE overlap execution view to DESIGN section 8 so the linear dataflow is not mistaken for a full-stage serial schedule; tied expert-group, sync-group, scoreboard, Sub-Tile return, restore, and timeline evidence to existing M2/M3 tasks. |
| DCL-2026-05-28-29 | user-review-workspace-window-layout | `DESIGN.md`, `TASKS.md` | M0.4, M2.1-M2.8, M3.7-M3.9 | coordinator-authored | Promoted DESIGN section 9 from suggested layouts to current layout contracts, added dtype/row-byte headers, scale typed views, scoreboard timeout counters, Sub-Tile ready fields, timeline storage, and explicit debug-mirror exclusions from the accepted main path. |
| DCL-2026-05-28-30 | user-review-pto-primitive-mapping | `DESIGN.md`, `TASKS.md` | M1.2-M3.8 | coordinator-authored | Reworked DESIGN section 10 into a stage-level PTO primitive mapping, removed loose first-stage/article wording, and clarified that layout helpers may compute typed views but must not hide copy/wait/notify/matmul/quant operations. |
| DCL-2026-05-28-31 | user-review-precisionbackend-duplication | `DESIGN.md`, `TASKS.md` | M2.1-M2.8, M3.1-M3.5 | coordinator-authored | Narrowed DESIGN section 11 from precision semantics to A3 int8 stage-set boundaries, leaving dtype facts in section 3.1 and numeric/tolerance semantics in section 5; removed repeated scale/bias/TQUANT strategy details and kept only function-boundary acceptance rules. |
| DCL-2026-05-28-32 | user-question-merge-sync-overlap-with-dataflow | `DESIGN.md`, `TASKS.md` | M2.2c, M2.7a, M3.0-M3.9 | coordinator-authored | Kept DESIGN section 8 as the dataflow/runtime overlap view and narrowed section 12 to sync contracts, signal ownership, publish ordering, blocked handling, and evidence gates; removed duplicated linear schedule text from section 12. |
| DCL-2026-05-28-33 | user-question-tiling-host-sections | `DESIGN.md`, `TASKS.md` | M0.2-M0.6, M1.0-M4.2 | coordinator-authored | Merged standalone DESIGN tiling/shape and host/engineering-organization chapters into section 6 as architecture contracts; removed redundant sections 13/14 and renumbered verification/task/risk sections. |
| DCL-2026-05-28-34 | user-review-verification-task-chapters | `DESIGN.md`, `TASKS.md` | M0.1-M4.2 | coordinator-authored | Earlier state, superseded by DCL-2026-05-28-42: kept DESIGN verification as the project-level correctness/perf/timeline contract, while detailed agent steps later moved back into DESIGN section 13 and TASKS became status-only. |
| DCL-2026-05-28-35 | wording-avoid-staged-rewrite | `DESIGN.md`, `TASKS.md` | M2.1-M3.9 | coordinator-authored | Reworded overlap and current-M2 task phrasing so implementation agents read M3 as verifying/enabling existing producer/consumer edges, not as a later performance rewrite or second implementation path. |
| DCL-2026-05-28-36 | user-rename-non-goals-to-constraints | `DESIGN.md`, `TASKS.md` | M0.1-M4.2 | coordinator-authored | Renamed DESIGN section 4 from non-goals to constraints and changed its intro to describe implementation boundaries and hard gates rather than optional out-of-scope items. |
| DCL-2026-05-28-37 | refresh-status-boundary-megamoe-keypoints | `DESIGN.md`, `TASKS.md` | M1.0-M4.2 | coordinator-authored | Expanded DESIGN section 0 with MegaMoE alignment checkpoints: core fusion, dispatch remote read, combine remote write, control metadata, AIC/AIV MPMD overlap, SwiGLU grouping, scoreboard soft sync, Sub-Tile return, counter/timeline evidence, and performance risks. |
| DCL-2026-05-28-38 | user-review-project-position-section | `DESIGN.md`, `TASKS.md` | M0.1-M4.2 | coordinator-authored | Replaced DESIGN section 1 project-position prose with a short design-entry map; moved its facts to existing single sources in sections 0, 4, and 6 to avoid duplicate project scope, dependency, and directory descriptions. |
| DCL-2026-05-28-39 | merge-goals-into-section-zero | `DESIGN.md`, `TASKS.md` | M0.1-M4.2 | coordinator-authored | Renamed DESIGN section 0 to goals/boundaries/alignment, merged the former goals overview into section 0, deleted the design-entry chapter, promoted precision requirements to their own section, and renumbered later DESIGN sections. |
| DCL-2026-05-28-40 | merge-precision-and-numeric-contract | `DESIGN.md`, `TASKS.md` | M2.1-M2.8 | coordinator-authored | Merged DESIGN precision dtype/stage requirements and numeric acceptance semantics into one section so A3 A8W8 dtype, compute order, checksum, scale decode, tolerance, and final restore rules have one source. |
| DCL-2026-05-28-41 | split-swiglu-sync-and-move-overlap-summary | `DESIGN.md`, `TASKS.md` | M3.1-M3.9 | coordinator-authored | Moved the sync/overlap overview into DESIGN section 6 dataflow, kept section 10 focused on signal ownership and publish ordering, and promoted SwiGLU/quant coarse-fine sync grouping to its own top-level section. |
| DCL-2026-05-28-42 | simplify-task-status-ledger | `DESIGN.md`, `TASKS.md` | M0.1-M4.2 | coordinator-authored | Moved detailed M0-M4 task descriptions and acceptance criteria into DESIGN section 13, and reduced TASKS.md to status tracking, Issue Log, DCL, and handoff rules. |
| DCL-2026-05-28-43 | align-doc-source-rules-after-ledger-split | `DESIGN.md`, `TASKS.md` | M0.1-M4.2 | coordinator-authored | Tightened markdown responsibility rules after TASKS slimming: DESIGN section 13 owns task details/acceptance, TASKS owns only execution status, issues, DCL, and handoff state. |
| DCL-2026-05-28-44 | user-request-lightweight-reports | `DESIGN.md`, `TASKS.md`, `reports/TEMPLATE.md`, `reports/M0.1.md` | all tasks | coordinator-authored | Narrowed reports to lightweight acceptance summaries only. Reports must not store compile logs, stdout raw text, counter raw text, timeline raw text, or long command logs; console output remains runtime evidence and reports only record pass/fail, key metric summaries, blocked reasons, and handoff notes. |
| DCL-2026-05-28-45 | user-request-m0-gemm-ar-scaffold | `DESIGN.md`, `TASKS.md` | M0.1-M0.6 | coordinator-authored | Reworked M0 so development starts from cut-down `gemm_ar` CMake/run.sh/host scaffolding plus MoE manual examples, while M0.1 markdown/directory bootstrap is already accepted and must not be claimed by implementation agents. |
| DCL-2026-05-28-46 | user-request-design-stage-task-navigation | `DESIGN.md`, `TASKS.md` | M0.1-M4.2 | coordinator-authored | Added stage and task navigation tables to DESIGN section 13 only, keeping TASKS as a status ledger; clarified that M3 opens runtime overlap on M2-fixed edges and must not require row/order/layout/stage-graph rewrites. |
| DCL-2026-05-28-47 | task-detail-design-alignment-review | `DESIGN.md`, `TASKS.md` | M0.3-M0.4, M2.1, M2.5, M3.1, M3.7-M3.8 | coordinator-authored | Aligned task details with sections 4-12: added missing GMM tiling args, removed M0.3's premature layout dependency, moved Swiglu sync-group metadata/layout responsibility to M2, and narrowed M3 tasks to enabling existing signals/scoreboard/Sub-Tile fields instead of redefining layout. |
| DCL-2026-05-28-48 | task-chain-protocol-review | `DESIGN.md`, `TASKS.md` | M0.3-M0.4, M1.0, M1.11, M2.5 | coordinator-authored | Tightened task navigation and dependency wording: M1.0 now verifies existing protocol design and host reference instead of asking agents to redesign it, M1.11 depends on M1.0-M1.10, and navigation text reflects GMM tiling/control fields plus M2.5 sync-group responsibility. |
| DCL-2026-05-28-49 | dependency-scan-scope-clarification | `DESIGN.md`, `TASKS.md` | all tasks | coordinator-authored | Clarified that dependency scans apply to project source, CMake, and scripts only, excluding DESIGN/TASKS/reports because markdown intentionally records forbidden dependency names and decisions. |
| DCL-2026-05-28-50 | timeline-layout-boundary | `DESIGN.md`, `TASKS.md` | M3.9 | coordinator-authored | Clarified that M3.9 timestamp/timeline work may only use or size pre-reserved timeline fields and must not change payload, row-order, scoreboard, or Sub-Tile segment offset semantics. |
| DCL-2026-05-28-51 | task-ledger-alignment-review | `TASKS.md` | M0.3-M0.4, M1.0, M1.11, M2.1, M2.5, M3.1, M3.7-M3.9 | coordinator-authored | Updated TASKS status ledger to reflect recent DESIGN task-boundary changes in Last update fields and tightened M2/M3 stage close conditions without moving detailed task requirements back into TASKS. |
| DCL-2026-05-28-52 | risk-section-review | `DESIGN.md`, `TASKS.md` | M0.2-M4.2 | coordinator-authored | Expanded DESIGN section 14 risks to cover M0 scaffold copying, PTO-only leakage, dependency-scan scope, M1 mock/protocol drift, M2 metadata/merge prerequisites, M3 open-only boundaries, timeline isolation, and lightweight report discipline. |
| DCL-2026-05-28-53 | design-0-12-megamoe-review | `DESIGN.md`, `TASKS.md` | M2.5, M2.7a, M3.8, M3.9 | coordinator-authored | Reviewed DESIGN sections 0-12 against the MegaMoE target: changed M2 prerequisites from four to five items by adding Swiglu sync-group metadata, narrowed PTO stride wording to an M3.8 capability gate, and kept timeline files as debug-only while stdout remains acceptance evidence. |
| DCL-2026-05-28-54 | full-design-task-readthrough | `DESIGN.md`, `TASKS.md` | M1.0, M2.5, M2.8, M3.1-M3.4 | coordinator-authored | Full readthrough aligned DESIGN/TASKS: M2 stage navigation now includes Swiglu sync-group metadata, M1.0 is phrased as verify-and-freeze, CorrectnessReport/M2.8 include `swiglu_sync_groups`, and topK=1 reference wording points to the host A8W8 MoE reference. |
| DCL-2026-05-28-55 | user-question-m2-m3-no-refactor-swiglu | `DESIGN.md`, `TASKS.md` | M2.5, M3.1-M3.4 | coordinator-authored | Strengthened DESIGN section 0 so the opening MegaMoE alignment explicitly covers Swiglu/quant coarse-fine sync, limited sync-event motivation, M2 row-range freeze, and M3 signal enablement without activation-layout refactor. |
| DCL-2026-05-28-56 | user-readyqueue-as-backup | `DESIGN.md`, `TASKS.md` | M2.7a, M3.8 | coordinator-authored | Completed the MegaMoE design confirmation pass, tightened GMM2 ready projection to expert/tile/sub-tile return segments, made `subTileReady` a required M2/M3 field, and downgraded ready queue to an optional local AIC/AIV handoff candidate rather than a required path. |
| DCL-2026-05-28-57 | user-request-m0-design-fix | `DESIGN.md`, `TASKS.md` | M0.2-M0.6, dependency scans | coordinator-authored | Fixed M0 design review gaps: M0.2 now requires a single fused mixed-kernel arch instead of vec-only, M0.3 depends on M0.2 and derives rank from MPI by default, M0.4/M0.5 expose dump/report interfaces while M0.6 owns executable stdout proof, and dependency scan wording excludes pure CANN SDK search paths from AscendC interface violations. |
| DCL-2026-05-28-58 | m0-implementation-handoff | `TASKS.md`, `reports/M0.2.md`, `reports/M0.3.md`, `reports/M0.4.md`, `reports/M0.5.md`, `reports/M0.6.md` | M0.2-M0.6 | task-owner-authored | Recorded M0 implementation handoff: M0.2-M0.6 are delivered as `review_ready` with lightweight reports, while M0 stage remains `in_progress` until reviewer acceptance. |

### Handoff Rules

1. 任务领取前先读 `DESIGN.md` 第 13 章对应小节，再把对应 task state 从 `not_started` 改为 `claimed`，填写 owner。
2. 开始改代码前，把 state 改为 `in_progress`，确认依赖 task 都是 `accepted` 或明确 `blocked` 且当前任务允许继续。
3. 交付时创建轻量 `reports/Mx.y.md`，把 state 改为 `review_ready`，并在 `Task 状态` 表填写 report 路径；report
   不记录编译日志、stdout 原文、counter 原文或 timeline 原文。
4. reviewer 验收通过后把 state 改为 `accepted`；未通过改为 `needs_fix` 并在 `Issue Log` 增加问题。
5. 任何 `P0/P1` issue 未关闭前，受影响的下游 task 不得从 `not_started` 进入 `claimed`。
6. 任何触发用户确认门禁的 issue 未获得用户明确决策前，受影响 task 必须保持 `needs_user_decision` 或 `blocked`，不得交付为 `accepted`。
7. 修改 `DESIGN.md` 或 `TASKS.md` 时，必须新增一条 `Design Change Log`，并列出受影响 task。
8. task owner 不能 self-accept 自己的交付。
