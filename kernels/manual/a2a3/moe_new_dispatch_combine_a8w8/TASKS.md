# moe_new_dispatch_combine_a8w8 Task Status Ledger

本文档只保留当前可领取任务、轻量 Issue Log、Design Change Log 和 handoff rules。历史 M0/M1/M2/M3/M3N
任务清单已按用户要求清空；历史验收事实以 `reports/`、git history 和当前代码为准。

当前目标：A3 A8W8/int8 主路径，PTO-only，不引入 Catlass/AscendC fallback。设计事实源为
`design.md`；本文只跟踪任务状态、issue 和设计变更。

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


### Active Tasks: initquant 前重排 PTO 重写

| ID | State | Owner | 依赖 | initquant 阶段 | 交付物 | 验收/Report | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T0 | `accepted` | Codex | 无 | 全局前置 | API 审计脚本或命令 | `check_pto_native_boundaries.sh` PASS | 固定 PTO-only 边界 |
| T1 | `accepted` | Codex | T0 | 全局前置 | `blockTokenPerExpert`、`blockPrefixPerExpert` 扩成 `[maxWorkers, globalExpertNum]` | small/large workspace + stop17 无覆盖，均 pass | host/device layout 同步 |
| T2 | `accepted` | Codex | T1 | 全阶段验收前置 | initquant 最终验收摘要 | stop17 字段完整 | 常规日志只保留最终汇总和 mismatch |
| T3 | `accepted` | Codex | T1 | 全阶段验收前置 | host golden 稳定分桶 | small/large `expandedRowIdx mismatches=0` | golden 按 worker token range 拼接 |
| T4 | `accepted` | Codex | T1 | Stage A/B/C 共用 | AIV worker 调度 helper | small activeWorkers=1，large activeWorkers=4 | token range 切分 |
| T5 | `accepted` | Codex | T2,T4 | Stage A: route count | PTO count 写 `blockTokenPerExpert[worker, expert]` | stop17 `init_quant_route_count_mismatches=0` | 不使用共享 atomic cursor |
| T6 | `accepted` | Codex | T5 | Stage B: expert prefix | main AIV 归并 count，生成 expert base 和 per-worker base | stop17 prefix/cumsum/preSum/expertTokenNums mismatch=0 | 写 `tokenOwnerRankOffsets`/`blockPrefixPerExpert` |
| T7 | `accepted` | Codex | T5,T6 | Stage A/B/C 共用同步 | AIV-only PTO phase sync helper | small/large stop17 无卡顿，均 `pass=true` | count/prefix/scatter barrier 已覆盖 |
| T8 | `accepted` | Codex | T6,T7 | Stage C-1: scatter metadata | `expandedRowIdx`、`packedRowToRouteIndex`、invalid/clipped 哨兵 | stop17 small 32、large 8194 elements mismatch=0 | packed row 稳定分桶 |
| T9 | `accepted` | Codex | T8 | Stage C-2: quant bring-up | PTO route quant 输出 payload/scale | stop17 small/large payload 和 scale mismatch=0 | scalar fallback 关闭 |
| T10 | `accepted` | Codex | T9 | Stage C-2: production quant | mixed AIV 固定 UB PTO-lowered Vec dynamic quant | `route_quant_path=pto_vec`，fallback reason `none`，stop17 pass | 避开 public tile wrapper 卡顿 |
| T11 | `accepted` | Codex | T10 | Stage C-3: padding | payload row tail padding | small/large dispatchPayloadInt8 mismatch=0 | 覆盖 rowBytes 存储路径 |
| T12 | `accepted` | Codex | T8,T10 | Stage C-4: capacity | capacity clip 集成 | stop17 small/large `pass=true` | `packedRow >= maxOutputSize` 哨兵随 expandedRow golden 验证 |
| T13 | `accepted` | Codex | T12 | initquant 后置衔接 | 多 rank count matrix 集成 | stop17 small/large tokenPerExpertMatrix mismatch=0 | 接后续 count sync/dispatch metadata |
| T14 | `accepted` | Codex | T13 | initquant 后置衔接 | 后续 prefix/GMM1 ready 回归 | small/large stop17 均 `pass=true` | prefix、GMM1 input、ready flag |
| T15 | `accepted` | Codex | T10,T13 | e2e timing hook | 前重排入口到最后全量同步后 e2e cycles/us | stop17 输出 `init_quant_e2e_us`；small/large acceptance pass | 只证明可观测性和基本多核，不等价商用性能闭环 |
| T16 | `accepted` | Codex | T15 | 全局异常路径 | PTO fallback 决策和 main-AIV fallback | 正常路径 reason=`none`；scratch 超限 case stop107 path=`pto_main_aiv` reason=`global_expert_scratch_limit` 且 pass | fallback 仍走 PTO host/device 路径，不退 AscendC/Catlass |

### Active Tasks: production performance hardening

| ID | State | Owner | 依赖 | 阶段 | 交付物 | 验收/Report | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T17 | `not_started` | - | T15 | baseline 对标 | 同 shape/seed 下原 FFN initquant 与 PTO 前重排 e2e 对比脚本 | small/large 记录原 FFN 与 PTO `init_quant_e2e_us`，给出 gap | 没有 baseline 前不能宣称商用性能达标 |
| T18 | `not_started` | - | T17 | sync 优化 | 将前重排 hard sync 收敛为设计中的必要 AIV-only phase sync | stop17 pass，sync 点数量和位置可审计 | 不新增中间验收日志 |
| T19 | `not_started` | - | T17 | quant pipeline | row quant 做 UB 双缓冲/ping-pong，重叠 GM load、Vec compute、GM store | large `init_quant_e2e_us` 相比 T15 基线下降或瓶颈解释清楚 | 当前是单 tile 两遍 row 扫描，不是完整流水 |
| T20 | `not_started` | - | T17 | GM 访问优化 | count/scatter 元数据批量化，减少 per-route scalar GM load/store 和 cache invalidation | large 计数/scatter 阶段不成为主瓶颈 | 保持 PTO-only |
| T21 | `not_started` | - | T17 | worker 调度 | 根据 M/topK/K/专家分布调优 worker 数和分片，避免固定 token shard 负载倾斜 | skew route case activeWorkers 有效，stop17 pass | 当前 large case 证明 4 worker 可用，不证明负载均衡最优 |
| T22 | `not_started` | - | T17 | swizzle/L1/L0 边界确认 | 明确原 FFN 中 swizzle/L1/L0 属于 GMM/Catlass 还是 initquant，并给 PTO 替代方案 | design.md 更新边界，若属于前重排则拆实现任务 | 不能把 GMM 的 L1/L0/swizzle 误计入前重排完成项 |

### Issue Log

| ID | Severity | State | Affected Tasks | Description | Decision/Next Step |
| --- | --- | --- | --- | --- | --- |
| - | - | - | - | 暂无 active issue | - |

### Design Change Log

| Date | Change | Affected Tasks |
| --- | --- | --- |
| 2026-06-01 | 新增 T0-T16 active task 状态跟踪表；任务状态统一维护在 `TASKS.md`，`design.md` 只保留设计、任务定义和阶段对应关系。 | T0-T16 |
| 2026-06-01 | 收紧 initquant 验收日志策略：常规输出只保留汇总和 first mismatch；临时 debug probe/debug stop 用完删除；验收 case 固定为 small/large。 | T0,T2,T15 |
| 2026-06-02 | route quant 改为 mixed AIV 固定 UB PTO-lowered Vec 路径；删除临时 10231/102311 等细分探针；small/large stop17 验收通过。 | T9-T14 |
| 2026-06-02 | 完成 T15/T16：stop17 输出前重排 e2e 时间；中间 timeline 验收日志从常规路径删除；scratch 超限走 `pto_main_aiv` fallback；`dispatchOffset` layout 收敛为 `[expertPerRank]`。T15 只代表可观测性和基本多核，不代表商用性能闭环。 | T15,T16 |
| 2026-06-02 | 新增 T17-T22 生产性能 hardening 任务：baseline 对标、sync 收敛、quant ping-pong、GM 访问优化、worker 调度、swizzle/L1/L0 边界确认。 | T17-T22 |

### Handoff Rules

1. 任务领取前先读 `design.md` 对应小节，再把对应 task state 从 `not_started` 改为 `claimed`，填写 owner。
2. 开始改代码前，把 state 改为 `in_progress`，确认依赖 task 都是 `accepted` 或明确 `blocked` 且当前任务允许继续。
3. 交付时创建轻量 `reports/Tx.md`，把 state 改为 `review_ready`，并在 active task 表填写 report 路径；report
   不记录编译日志、stdout 原文、counter 原文或 timeline 原文。
4. reviewer 验收通过后把 state 改为 `accepted`；未通过改为 `needs_fix` 并在 `Issue Log` 增加问题。
5. 任何 `P0/P1` issue 未关闭前，受影响的下游 task 不得从 `not_started` 进入 `claimed`。
6. 任何触发用户确认门禁的 issue 未获得用户明确决策前，受影响 task 必须保持 `needs_user_decision` 或 `blocked`，不得交付为 `accepted`。
7. 修改 `design.md` 或 `TASKS.md` 时，必须新增一条 `Design Change Log`，并列出受影响 task。
