# moe_dispatch_combine_a8w8 Task Status Ledger

本文档只保留当前可领取任务、轻量 Issue Log、Design Change Log 和 handoff rules。历史 M0/M1/M2/M3/M3N
任务清单已按用户要求清空；历史验收事实以 `reports/`、git history 和当前代码为准。

当前目标：A3 A8W8/int8 主路径，PTO-only，不引入 Catlass/AscendC fallback。

## 执行状态台账

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
| `archived` | 历史任务已从 active 列表清空，仅保留报告/git 作为事实来源 | coordinator |

### 阶段状态

| Stage | Scope | Required close condition | State | Report |
| --- | --- | --- | --- | --- |
| Baseline | 历史 M0-M3/M3N 已清空出 active 任务列表，仅作为当前代码和报告事实基线 | 不再领取历史任务；后续只按 M3O/M4 编号推进 | archived | reports/M3.md |
| M3O | 多核切分、report 真实性和 overlap 优化收口 | M3O.1-M3O.7 全部 `accepted`；每个任务验收都覆盖统一 small/large case | in_progress | reports/M3O.1.md, reports/M3O.2.md, reports/M3O.3.md, reports/M3O.4.md, reports/M3O.5.md |
| M4 | 最终回归命令、文档和状态收口 | M4.1 `accepted`，无 open P0/P1 或 `needs_user_decision` | not_started | none |

### Task 状态

| Task | State | Owner | Report | Last update |
| --- | --- | --- | --- | --- |
| M3O.1 | review_ready | codex | reports/M3O.1.md | DCL-2026-06-01-129: stale report claims renamed to requested/claim fields; active worker fields verified on required small/large cases |
| M3O.2 | review_ready | codex | reports/M3O.2.md | DCL-2026-06-01-130: route shard rescan removed; worker/expert count-prefix cursor path verified on required small/large cases |
| M3O.3 | review_ready | codex | reports/M3O.3.md | DCL-2026-06-01-131: fused GMM1/GMM2 tile tasks now use active multi-AIC round-robin scheduling; required small/large cases pass |
| M3O.4 | blocked | codex | reports/M3O.4.md | DCL-2026-06-01-135: documented current activation shard diagnosis and ruled-out failure points in `DESIGN.md`; next probe is forced single logical activation worker |
| M3O.5 | review_ready | codex | reports/M3O.5.md | DCL-2026-06-01-133: combine now uses dense combine-local AIV worker policy; required small/large cases pass with `combine_active_aiv_workers=8` and non-worker0 segment distribution |
| M3O.6 | not_started | unassigned | none | DCL-2026-06-01-128: 重新打开 sync-group / combine overlap；small/large 必跑 |
| M3O.7 | not_started | unassigned | none | DCL-2026-06-01-128: 回归矩阵、性能拆解与交付口径；small/large 必跑 |
| M4.1 | not_started | unassigned | none | DCL-2026-06-01-128: 文档、状态和最终回归命令收口 |

### Issue Log

| ID | Severity | Type | Affected tasks | State | Owner | Needs user decision | Question to user | User decision | Summary | Decision / next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M3O4-ACTIVATION-SHARD-HANG | P1 | impl-bug | M3O.4, M3O.6 | open | codex | no | none | none | Activation stage-local AIV sharding hangs or is unsafe after entering the real row loop / post-compute sync path. A non-overlap `ActivationToGmm2` completion-edge experiment and PTO helper activation experiment both timed out, so correctness is recovered only by keeping coarse scalar activation with `activation_active_aiv_workers=1`. | Keep M3O.4 blocked for activation worker acceptance; M3O.5 may proceed because combine worker policy is independent. Do not enable M3O.6 GMM1->activation or activation->GMM2 overlap until this is resolved or redesigned. |

Severity:

- `P0`: 精度路线、protocol 不变量、PTO-only 约束或内存安全错误；下游必须停。
- `P1`: 当前 task 无法验收，或会影响后续阶段接口；相关下游停。
- `P2`: 局部实现缺陷，有 workaround，不影响下游领取。
- `P3`: 文档澄清、日志改善、非阻塞优化。

Type:

- `design-bug`: `DESIGN.md` 与代码事实、任务验收或实现路径冲突。
- `task-gap`: `DESIGN.md` 第 14 章的任务文件范围、依赖、验收证据或命令不够。
- `impl-bug`: 实现错误，设计仍成立。
- `primitive-gap`: PTO public primitive 缺失或能力不明。
- `env-gap`: 硬件、CANN、脚本或构建环境缺失。
- `test-gap`: reference、golden、dump 或验收命令不能证明任务完成。
- `user-decision`: 继续实现会改变最终目标、精度路线、MegaMoE protocol、PTO-only 约束或阶段验收口径，需要用户确认。

### Design Change Log

| ID | Source issue | Files changed | Affected tasks | Reviewer state | Summary |
| --- | --- | --- | --- | --- | --- |
| DCL-2026-06-01-128 | user-request-clear-task-list-renumber-m3o | `DESIGN.md`, `TASKS.md`, `ffn_original_code_split_report.md` | M3O.1-M3O.7, M4.1 | coordinator-authored | Cleared historical DESIGN/TASKS task lists, removed old M0-M3N active task entries, and renumbered the current optimization plan as M3O.1-M3O.7 plus M4.1. M3R/M3S and M3N10 scoreboard remain discarded; every M3O task must validate the user-provided small and large cases. |
| DCL-2026-06-01-129 | m3o1-report-truth-cleanup | `host/main.cpp`, `TASKS.md`, `reports/M3O.1.md` | M3O.1-M3O.3 | task-owner-authored | M3O.1 delivery records launch/requested facts separately from active worker evidence. Current active fused GMM remains `gmm*_active_aic_blocks=1`, which M3O.3 must replace with real multi-AIC evidence. |
| DCL-2026-06-01-130 | m3o2-dispatch-prefix-development | `TASKS.md`, `reports/M3O.2.md`, `host/main.cpp`, `kernel/moe_dispatch_combine_a8w8_kernel.cpp`, `kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp` | M3O.2, M3O.3 | task-owner-authored | Delivered M3O.2 as `review_ready`: dispatch workers now produce worker/expert counts, merge derives worker/expert prefixes, and route pack uses `expertBase + workerExpertPrefix + localOrdinal` instead of rescanning prior shard routes. Required small/large A3 runs on devices 4-5 pass with `final_output.err_count=0`; source scan confirms removed rescan helpers. A transient stage-11 hang was traced to a redundant parallel prefix rewrite after the serial merge had already populated the prefix table; that dead path was removed. M3O.3 still owns real multi-AIC GMM scheduling. |
| DCL-2026-06-01-131 | m3o3-gmm-active-multi-aic | `TASKS.md`, `reports/M3O.3.md`, `host/main.cpp`, `include/moe_dispatch_combine_a8w8_layout.hpp`, `kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp` | M3O.3, M3O.4, M3O.6 | task-owner-authored | Delivered M3O.3 as `review_ready`: fused GMM1/GMM2 tile tasks now run with block-stride AIC scheduling instead of block0-only execution, scalar tile helper no longer filters block0, and host report reads device-written GMM active block/task evidence from expanded peer debug counters. Required small/large A3 runs on devices 4-5 pass with `final_output.err_count=0`; large reports `gmm1_active_aic_blocks=24` and `gmm2_active_aic_blocks=24` with first/last task and per-block task distribution. Fine-grained GMM/activation/combine overlap remains deferred to M3O.6. |
| DCL-2026-06-01-132 | M3O4-ACTIVATION-SHARD-HANG | `TASKS.md`, `reports/M3O.4.md`, `host/main.cpp`, `include/moe_dispatch_combine_a8w8_types.hpp`, `kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp` | M3O.4, M3O.5, M3O.6 | task-owner-authored | Recorded M3O.4 as blocked rather than `review_ready`: restore evidence now reports 8 active AIV token shards with per-worker token/route/skipped counts and required small/large correctness passes, but activation stage-local AIV sharding remains unsafe after real row-loop/post-compute sync probes. Current valid path is coarse scalar activation with `activation_active_aiv_workers=1`; M3O.5 may proceed, while M3O.6 must not reopen activation overlap until this issue is resolved. |
| DCL-2026-06-01-133 | m3o5-combine-dense-worker-policy | `TASKS.md`, `reports/M3O.5.md`, `include/moe_dispatch_combine_a8w8_types.hpp`, `kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`, `kernel/moe_dispatch_combine_a8w8_kernel.cpp` | M3O.5, M3O.6 | task-owner-authored | Delivered M3O.5 as `review_ready`: combine assignment is decoupled from dispatch assignment by using dense combine-local AIV workers, and combine debug probes no longer pollute worker segment evidence slots. Required small/large A3 runs on devices 4-7 pass with `final_output.err_count=0`; both cases report `combine_active_aiv_workers=8`, non-worker0 segment distribution, and `combine_mode=subtile_stride`. `TPUT_ASYNC` stride remains recorded as a primitive gap; fine-grained overlap remains M3O.6 scope and activation overlap remains blocked by M3O4-ACTIVATION-SHARD-HANG. |
| DCL-2026-06-01-134 | M3O4-ACTIVATION-SHARD-HANG | `TASKS.md`, `reports/M3O.4.md` | M3O.4, M3O.6 | task-owner-authored | Added follow-up diagnosis for the activation shard blocker: the red acceptance signal is reproducible, adding a non-overlap `ActivationToGmm2` completion edge still times out, scalar probes 45/47 time out while 41/42/44/46 pass, and the existing PTO vector activation helper path also times out at debug-stop 43/full small. Source was restored to the coarse activation correctness path and rebuilt; small sanity passes with `final_output.err_count=0` and `activation_active_aiv_workers=1`. |
| DCL-2026-06-01-135 | M3O4-ACTIVATION-SHARD-HANG | `DESIGN.md`, `TASKS.md` | M3O.4, M3O.6 | task-owner-authored | Recorded the current M3O.4 diagnosis in `DESIGN.md`: debug-gated AIV activation shard is now reachable, probes isolate the hang to scalar sigmoid/reduction/quant or PTO requant boundaries, and already ruled out fallback, restore/combine policy, basic GMM1 output reads, routingScale/gate/up reads, and the activation-to-GMM2 completion edge. Next diagnostic split is forced single logical activation worker. |

### Handoff Rules

1. 任务领取前先读 `DESIGN.md` 第 14 章对应小节，再把对应 task state 从 `not_started` 改为 `claimed`，填写 owner。
2. 开始改代码前，把 state 改为 `in_progress`，确认依赖 task 都是 `accepted` 或明确 `blocked` 且当前任务允许继续。
3. 交付时创建轻量 `reports/Mx.y.md`，把 state 改为 `review_ready`，并在 `Task 状态` 表填写 report 路径；report
   不记录编译日志、stdout 原文、counter 原文或 timeline 原文。
4. reviewer 验收通过后把 state 改为 `accepted`；未通过改为 `needs_fix` 并在 `Issue Log` 增加问题。
5. 任何 `P0/P1` issue 未关闭前，受影响的下游 task 不得从 `not_started` 进入 `claimed`。
6. 任何触发用户确认门禁的 issue 未获得用户明确决策前，受影响 task 必须保持 `needs_user_decision` 或 `blocked`，不得交付为 `accepted`。
7. 修改 `DESIGN.md` 或 `TASKS.md` 时，必须新增一条 `Design Change Log`，并列出受影响 task。
8. task owner 不能 self-accept 自己的交付。
