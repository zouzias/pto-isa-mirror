# moe_new_dispatch_combine_a8w8 Task Status Ledger

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


### Handoff Rules

1. 任务领取前先读 `DESIGN.md` 第 对应小节，再把对应 task state 从 `not_started` 改为 `claimed`，填写 owner。
2. 开始改代码前，把 state 改为 `in_progress`，确认依赖 task 都是 `accepted` 或明确 `blocked` 且当前任务允许继续。
3. 交付时创建轻量 `reports/Mx.y.md`，把 state 改为 `review_ready`，并在 `Task 状态` 表填写 report 路径；report
   不记录编译日志、stdout 原文、counter 原文或 timeline 原文。
4. reviewer 验收通过后把 state 改为 `accepted`；未通过改为 `needs_fix` 并在 `Issue Log` 增加问题。
5. 任何 `P0/P1` issue 未关闭前，受影响的下游 task 不得从 `not_started` 进入 `claimed`。
6. 任何触发用户确认门禁的 issue 未获得用户明确决策前，受影响 task 必须保持 `needs_user_decision` 或 `blocked`，不得交付为 `accepted`。
7. 修改 `DESIGN.md` 或 `TASKS.md` 时，必须新增一条 `Design Change Log`，并列出受影响 task。
8. task owner 不能 self-accept 自己的交付。
