# moe_new_dispatch_combine_a8w8 Dispatch Task Status Ledger

本文档只跟踪 `moe_new_dispatch_combine_a8w8` 的 dispatch 阶段开发任务。设计事实源为
`design_dispatch.md`；本文只记录开发顺序、依赖、验收条件和同步问题处理要求。

当前目标：前重排/initquant 已完成后，按 FFN `dispatch_ffn_combine` 的业务语义，用 PTO 实现 dispatch。
dispatch 的业务就是本 rank 的 AIV worker 遍历本地 expert，从各 token owner rank 的 packed buffer 拉
payload/scale rows，直接写成本 rank 的 GMM1 输入。

## 状态枚举

| State | 含义 | 谁可以写 |
| --- | --- | --- |
| `not_started` | 尚未领取 | coordinator |
| `claimed` | 已领取但未开始改代码 | task owner |
| `in_progress` | 正在实现或验证 | task owner |
| `review_ready` | 已交付，等待 review | task owner |
| `accepted` | 验收通过，可作为下游依赖 | reviewer/coordinator |
| `needs_fix` | review 未通过，需要返工 | reviewer/coordinator |
| `blocked` | public primitive、硬件、构建环境或上游缺失导致无法继续 | task owner 或 reviewer |
| `needs_user_decision` | 继续实现会改变 FFN 语义、PTO-only 约束或验收口径，需要用户确认 | task owner 或 coordinator |
| `superseded` | 被新的设计或任务替代 | coordinator |

## Active Tasks: dispatch PTO 重写

| ID | State | Owner | 依赖 | 阶段 | 交付物 | 验收/Report | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D0 | `accepted` | Codex | 前重排 T15/T16 已完成 | 设计冻结 | `design_dispatch.md` 明确 FFN 业务语义、D0-D4 阶段、e2e 口径、同步定位标准 | 文档说明 `preSumBeforeRank` 是源端 packed row prefix；目的端位置由 `dispatchOffset + cumsumMM prefix` 表达 | 不接受把同步问题改串行定位 |
| D1 | `accepted` | Codex | D0 | Count publish/wait | 接通 `PublishCounts` / `WaitCounts`，让每个 expert owner rank 可见所有 token owner 的 count row | small/large 下 `tokenPerExpertMatrix[*][*][*]` 与 CPU reference 一致；无 count wait 卡住 | notify 必须在远端 count row 写完之后 |
| D2 | `accepted` | Codex | D1 | Metadata | 构造 `preSumBeforeRank/cumsumMM/dispatchOffset/expertTokenNums` | small/large 下 metadata mismatch 为 0；capacity clip 后 `expertTokenNums`、GMM1 行数一致 | `preSumBeforeRank` 按完整 global expert raw count prefix；`cumsumMM` 按目的端有效 rows |
| D3 | `accepted` | Codex | D2 | Payload gather | 用 PTO row-block `TGET` 拉 remote `dispatchPayload` 到 `gmm1InputInt8`；local rows 走 PTO tile copy | local/remote payload mismatch 为 0；rows=0、tail row-block、capacity clip 覆盖 | 不用 scalar remote byte copy 作为验收路径 |
| D4 | `accepted` | Codex | D2 | Scale gather | 用 PTO row-block `TGET` 拉 remote `dispatchScale` 到 `routingPerTokenScale`；local scale 走 PTO tile copy | local/remote scale mismatch 为 0；scale 行号与 `gmm1InputInt8` 行号一致 | scale 的 `dstStart` 必须和 payload 一致 |
| D5 | `accepted` | Codex | D3,D4 | Expert ready | 每个 local expert 的全部 owner rows 写完后置 `dispatchGroupReady[localExpert]` | `dispatchGroupReady` ready count 等于本 rank local expert 数；ready mismatch 为 0 | 若后续 rowBlock 并行，需要 per-expert 完成计数或 AIV-only 收口 |
| D6 | `accepted` | Codex | D5 | Dispatch e2e timing | 记录 D0 `PublishCounts` 开始到 D4 `dispatchGroupReady` 完成的 dispatch e2e 时间 | 输出 start/end/cycles/us；每 rank 一个固定 owner core 记录；当前验收版允许 D4 后临时全同步并在报告中标明 | 下一阶段 GMM1 overlap 时删除临时全同步 |
| D7 | `accepted` | Codex | D1-D6 | Correctness acceptance | 跑 dispatch-only small/large 验收 | metadata、payload、scale、ready mismatch 全为 0；常规日志只保留 summary/first mismatch/e2e | 不加 token/row 级大 dump |
| D8 | `accepted` | Codex | D7 | Code review | 检视字段语义、TGET 路径、ready 顺序、e2e 口径和无 debug 膨胀 | 无 remote scalar copy 调用；无源端/目的端 prefix 混用；无 ready 早置 | 见 `reports/D_dispatch.md` |
| D9 | `accepted` | Codex | D8 | Report/commit | 生成轻量交付报告并只提交 dispatch 相关文件 | report 记录命令、case、结果和未覆盖风险；git stage 不包含无关脏改 | report 已生成；本阶段提交只包含 dispatch 相关文件 |
| D10 | `not_started` | - | D7 | Hot expert rowBlock 优化 | 把 `(localExpert, tokenOwnerRank)` 继续切成 rowBlock task，提高 hot expert 并行度 | 不改变 row order；skew case correctness pass；ready 仍晚于全部 rowBlock 完成 | 性能优化项，不阻塞 dispatch correctness |

第一轮开发顺序：D1 -> D2 -> D3/D4 -> D5 -> D6 -> D7 -> D8 -> D9。D10 是后续优化，不阻塞第一版
dispatch correctness。

## 验收矩阵

| Case | 命令口径 | 必须验证 |
| --- | --- | --- |
| small | `ffn-v3-small`, `M=16`, `K=128`, `N=128`, `topK=2`, `expertPerPe=2`, `maxOutputSize=32`, `dispatch-only=1` | count、metadata、payload、scale、ready、e2e |
| large | `ffn-v3-4097`, `M=4097`, `K=128`, `N=128`, `topK=2`, `expertPerPe=2`, `maxOutputSize=8194`, `dispatch-only=1` | count、metadata、payload、scale、ready、capacity clip、e2e |
| follow-up sweep | `M/topK/K/expertPerPe/distribution` 参数族扩展 | 非 small/large 特化、hot expert rowBlock、tail row-block |

常规验收日志只保留：

- shape/rank summary。
- metadata pass/fail。
- payload/scale pass/fail 和 first mismatch。
- ready/e2e summary。

## 同步问题处理要求

dispatch 只允许围绕必要同步点定位问题，不允许为了定位卡住把并发改串行。

必要同步点：

1. count row 可见：远端 count row 写完后 notify；wait 完成后才能读 count matrix。
2. metadata 可见：`BuildDispatchMetadata` 完成后，gather workers 才能读 prefix/offset。
3. expert rows 可见：当前 expert 所有 payload/scale rows 搬完后，才能置 ready。

卡住时只按下面五项分析：

- 参与者：producer/consumer 的 rank、worker 集合是否一致。
- 地址：signal 地址、count matrix row 地址、peer window rank 偏移是否一致。
- 值协议：ready 初值、递增值、等待值是否一致，是否复用了旧值。
- 顺序：远端写后 notify、wait 后读、rows 写完后 ready。
- 可见性：`TPUT/TGET` 完成、必要 event/barrier、GM cache invalidate 是否在正确位置。

临时日志只允许同步点前后的摘要计数、signal 值或 timeout summary；不用 token/row 级 dump。

## Issue Log

| ID | Severity | State | Affected Tasks | Description | Decision/Next Step |
| --- | --- | --- | --- | --- | --- |
| DISPATCH-SYNC-RULE | P1 | closed | D1-D7 | 定位 dispatch 卡住时不得通过串行化 AIV worker 或关闭并发来试结果 | 本轮按地址协议定位并修复 host/device peer timeline offset；未做串行降级实验 |
| DISPATCH-SCALAR-REMOTE-COPY | P1 | closed | D3,D4,D8 | remote payload/scale 不接受 scalar byte copy 作为验收路径 | 已使用 PTO row-block `TGET`；D8 源码扫描确认无旧 scalar remote copy 调用 |

## Handoff Rules

1. 任务领取前先读 `design_dispatch.md` 对应阶段，再把对应 task state 从 `not_started` 改为 `claimed`。
2. 开始改代码前，把 state 改为 `in_progress`，确认依赖 task 已 `accepted` 或当前任务允许并行推进。
3. 交付时创建轻量 report，把 state 改为 `review_ready`；report 不记录完整 stdout、counter 原文或 token/row dump。
4. reviewer 验收通过后把 state 改为 `accepted`；未通过改为 `needs_fix` 并在 Issue Log 增加问题。
5. 任何 `P0/P1` issue 未关闭前，受影响下游 task 不得从 `not_started` 进入 `claimed`。
6. 任何会改变 FFN 语义、PTO-only 约束或 e2e 计时口径的实现，必须进入 `needs_user_decision`。
