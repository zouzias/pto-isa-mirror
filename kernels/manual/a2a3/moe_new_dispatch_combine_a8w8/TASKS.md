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
| T1 | `accepted` | Codex | T0 | 全局前置 | 正式 `initQuantWorkerTokenPerExpert/initQuantWorkerPrefixPerExpert` worker scratch；`blockTokenPerExpert/blockPrefixPerExpert` 保留 expert 汇总 | small/large workspace + stop17 无覆盖，均 pass | host/AIV/AIC layout 同步 |
| T2 | `accepted` | Codex | T1 | 全阶段验收前置 | initquant 最终验收摘要 | stop17 字段完整 | 常规日志只保留最终汇总和 mismatch |
| T3 | `accepted` | Codex | T1 | 全阶段验收前置 | host golden 稳定分桶 | small/large `expandedRowIdx mismatches=0` | golden 按 worker token range 拼接 |
| T4 | `accepted` | Codex | T1 | Stage A/B/C 共用 | AIV worker 调度 helper | small activeWorkers=1，large activeWorkers=4 | token range 切分 |
| T5 | `accepted` | Codex | T2,T4 | Stage A: route count | PTO count 写 `initQuantWorkerTokenPerExpert[worker, expert]`，main AIV 汇总到 `blockTokenPerExpert[expert]` | stop17 `init_quant_route_count_mismatches=0` | 不使用共享 atomic cursor |
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
| T16 | `accepted` | Codex | T15 | 全局异常路径 | PTO fallback 决策和 main-AIV fallback | 正常路径 reason=`none`；无效 expert/workspace 条件走 `pto_main_aiv` 且 pass | fallback 仍走 PTO host/device 路径，不退 AscendC/Catlass |

### Active Tasks: production performance hardening

| ID | State | Owner | 依赖 | 阶段 | 交付物 | 验收/Report | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T17 | `accepted` | Codex | T15 | baseline 对标 | `scripts/run_initquant_baseline_compare.sh` | `reports/T17.md`；small/large 记录 PTO `init_quant_e2e_us`；原 FFN 命令缺失时 `original_status=missing_command` | 脚本和报告可作为后续依赖；没有 baseline 前不能宣称商用性能达标 |
| T18 | `accepted` | Codex | T17 | sync 优化 | 仅将 count/prefix/scatter 三个 initquant phase barrier 收敛为 AIV-only；后置/final 继续 hard/mix 配对 | `reports/T18.md`；small/large stop17 acceptance PASS | 未新增中间日志；stop13/18-21 调试 stop 保持原 mixed 轮次 |
| T19 | `accepted` | Codex | T17 | UB-fits fast path | full-load/UB-fits 路径按 UB resident count/prefix/scatter 和 token-centric quant 融合，减少 GM 中间结果和同步 | `reports/T19.md`；small anchor、UB-fits `topK=4` expansion、large regression stop17 PASS | 对标原 FFN `tilingKey=21000`；保守阈值由 `R/K/globalExpertNum/UB fit` 决定，不按 case name 或固定 topK |
| T20 | `accepted` | Codex | T17 | large-token quant pipeline | `R=M*topK` 较大路径 row quant 做任意 topK token-centric 复用和 UB 双缓冲/ping-pong，重叠 GM load、Vec compute、GM store | `reports/T20.md`；large anchor、`M=8192/16384` token scale、`topK=4/8` sweep stop17 PASS | 主路径任意 topK token-centric；UB packed row cache + ping/pong quant buffer；完整 large-K 性能验收留给 T24 |
| T21 | `accepted` | Codex | T17 | worker/cache 调度 | 根据 M/topK/K/专家分布调优 worker 数、UB resident metadata 和批量 GM 写回，避免固定 token shard 负载倾斜 | `reports/T21.md`；large、`M=8192`、`topK=8`、skew topK4、more experts stop17 PASS | selector 按 `M/topK/K/globalExpertNum/logicalAIV` 选 worker；route-pack expert base/localOrdinal 进 UB cache；T23 继续处理 metadata GM 批量化 |
| T22 | `accepted` | Codex | T17 | swizzle/L1/L0 边界确认 | 明确原 FFN 中 swizzle/L1/L0/preload 属于 GMM/Catlass 阶段，并给后续 GMM PTO `TMATMUL` tiling、L1/L0 cache、swizzle 替代任务 | `reports/T22.md`；design.md 边界清晰，GMM 阶段单独拆任务验收 | 不能把 GMM 的 L1/L0/swizzle 误计入前重排完成项；GMM 设计继承 `M/topK/K/currentM` 参数族 |
| T23 | `accepted` | Codex | T17,T21 | metadata GM 批量化 | scatter 同步生成当前 token packed-row UB cache，quant 直接吃 cache；`packedRowToRouteIndex` 按 expert/worker 连续段 flush，删除 worker 级全 `R` flush | `reports/T23.md`；small/large、`M=8192/16384` token scale、topK sweep、skew 和 more experts stop17 PASS；常规日志无中间 dump | 解决 tok/topK 更多时 metadata GM store/flush 放大；capacity clip 的稀疏 expanded-row 更新仍保留最终全量 flush |
| T24 | `accepted` | Codex | T20,T23 | 大 K 行/列 tiling | quant 支持大 K column chunk、row tile/ping-pong，不能因整行放不下 UB 退成单核或 per-route 重复量化 | `reports/T24.md`；`K=1024/7168,topK=4` pass，small/large、`M=16384`、`topK=8` 回归 pass，route_quant_path=`pto_vec` | 对标 FFN gather dynamic quant 的 row/col tiling；新增 metadata store fence 只处理写后可见性，不做 shape 特化 |
| T25 | `not_started` | - | T22 | GMM PTO 性能设计 | 后续 GMM1/GMM2 用 PTO `TMATMUL`、L1/L0 ping-pong、preload、swizzle 替代 Catlass/Catcoc 的设计任务 | 新 GMM design/task 拆分完成；不把 initquant pass 当整体商用性能完成；覆盖 `M=8192/16384`、`topK>2`、large K 和 skew | PTO 重写整体性能闭环任务，不按 small/large anchor 估算 GMM tile |
| T26 | `not_started` | - | T19,T20,T21,T23,T24 | 参数化非特化验收矩阵 | stop17 脚本支持 token/topK/K/distribution sweep，并做 device 非特化代码审计 | `M={16,512,2048,4097,8192,16384}`、`topK={1,2,4,8}`、`K={128,1024,7168}`、skew、capacity clip 均 pass；device 主路径无 small/large/topK==2 写死 | 作为 T19-T24 的共享验收门禁，不新增中间日志；性能结论按参数族输出 |

### Issue Log

| ID | Severity | State | Affected Tasks | Description | Decision/Next Step |
| --- | --- | --- | --- | --- | --- |
| IQ-SYNC-001 | P1 | closed | T18 | AIV-only 批量替换会破坏 stop17 AIC/AIV `SYNCALL<Mix>` 配对 | T18 已按配对表只替换 count/prefix/scatter 三个内部 phase；后置/final 继续 hard/mix，small/large stop17 pass |

### Design Change Log

| Date | Change | Affected Tasks |
| --- | --- | --- |
| 2026-06-01 | 新增 T0-T16 active task 状态跟踪表；任务状态统一维护在 `TASKS.md`，`design.md` 只保留设计、任务定义和阶段对应关系。 | T0-T16 |
| 2026-06-01 | 收紧 initquant 验收日志策略：常规输出只保留汇总和 first mismatch；临时 debug probe/debug stop 用完删除；验收 case 固定为 small/large。 | T0,T2,T15 |
| 2026-06-02 | route quant 改为 mixed AIV 固定 UB PTO-lowered Vec 路径；删除临时 10231/102311 等细分探针；small/large stop17 验收通过。 | T9-T14 |
| 2026-06-02 | 完成 T15/T16：stop17 输出前重排 e2e 时间；中间 timeline 验收日志从常规路径删除；正式 per-worker workspace 替代 debug scratch；`dispatchOffset` layout 收敛为 `[expertPerRank]`。T15 只代表可观测性和基本多核，不代表商用性能闭环。 | T15,T16 |
| 2026-06-02 | 新增 T17-T22 生产性能 hardening 任务：baseline 对标、sync 收敛、quant ping-pong、GM 访问优化、worker 调度、swizzle/L1/L0 边界确认。 | T17-T22 |
| 2026-06-02 | `design.md` 补充 small/full-load fast path 和 large-token 优化路径：明确原 FFN small 走 `tilingKey=21000`，large 走 `tilingKey=11010`；T19/T20 调整为 full-load fast path 与 large quant pipeline 两条性能任务。 | T17-T22 |
| 2026-06-02 | `design.md` 新增商用性能对标门禁：前重排必须考虑 UB resident、multi-AIV、ping-pong、token-centric quant、metadata 批量化和同步收敛；GMM 阶段单独对标 L1/L0/swizzle/preload。 | T17-T22 |
| 2026-06-02 | 补充非特化要求：worker scratch 使用正式 `initQuantWorker*` workspace，token-centric quant 必须支持任意 `topK`；新增 T23-T25 覆盖 metadata 批量化、大 K 行/列 tiling 和后续 GMM PTO 性能设计。 | T17-T25 |
| 2026-06-02 | 新增 T17 baseline 脚本，默认提取 PTO small/large stop17 e2e、worker、path/reason；原 FFN command 未配置时显式输出 `missing_command`，避免伪造 baseline。 | T17 |
| 2026-06-02 | 将 T17 标为 accepted 作为后续 hardening 依赖；补充 T19-T24 非特化验收矩阵，more tokens/topK expansion/skew/more experts/large K 作为参数族覆盖，small/large 只作为 anchor。 | T17-T24 |
| 2026-06-02 | T18 AIV-only 同步替换尝试在 small stop17 卡住，失败实现已撤回；记录为同步配对问题，后续需先建立 AIC/AIV mixed sync 配对表再改代码。 | T18 |
| 2026-06-02 | 增加 token/topK/K 参数化 sweep 和 device 非特化代码审计门禁；新增 T26 作为 T19-T24 的共享验收矩阵，避免只针对 small/large anchor 优化。 | T19-T26 |
| 2026-06-02 | T18 accepted：仅 count/prefix/scatter 三个 initquant phase 改 AIV-only；AIC stop17 等待轮次同步扣减，后置/final mixed 配对保留，small/large stop17 acceptance PASS。 | T18 |
| 2026-06-02 | 收紧 T19-T26 more-token 门禁：`M>4097`、`topK>2`、large K、skew 和 capacity clip 均作为生产性能 hardening 的参数族，不允许 worker 数、full-load 选择或 quant 逻辑围绕 small/large anchor 特化。 | T19-T26 |
| 2026-06-02 | T19 accepted：新增 PTO full-load/UB-fits fast path，按 `R/K/globalExpertNum/capacity` 保守选择，单 AIV 融合 count/prefix/scatter 和 token-centric quant，去掉内部三次 initquant phase sync；small、`topK=4` UB-fits expansion 和 large regression stop17 PASS。 | T19 |
| 2026-06-02 | 强化 T20-T25 非 anchor 交付门禁：每个性能任务报告都必须覆盖至少一个超出 large anchor 的 token/topK/K 参数族，T26 只作为共享脚本化矩阵，不能把 more-token 泛化延后补测。 | T20-T26 |
| 2026-06-02 | T20 accepted：large-token quant 主路径去掉单 route 量化分支，任意 `topK` 先 UB cache packed rows，再单次 token quant 写多个 packed rows；row quant 增加 ping/pong UB buffer 和独立量化 scratch，新增 store 同步使用 PTO `PtoSetWaitFlag`。large、`M=8192/16384`、`topK=4/8` stop17 PASS。 | T20,T24 |
| 2026-06-02 | T21 accepted：worker selector 改为按 `M/topK/K/globalExpertNum/logicalAIV/lane slot` 参数化；fused 与直调 route-pack scatter 均使用 UB expert-base/localOrdinal cache。large、`M=8192`、`topK=8`、skew topK4、more experts stop17 PASS。 | T21,T23 |
| 2026-06-02 | T22 accepted：明确 L1/L0/swizzle/preload 属于 GMM1/GMM2 Catlass 阶段，不能计入 initquant 完成项；T25 GMM PTO 设计继承 `M/topK/K/currentM` 参数族和 skew/capacity clip 分布。 | T22,T25,T26 |
| 2026-06-02 | T23 accepted：route scatter 生成当前 token packed-row UB cache，token-centric quant 不再立即回读 `expandedRowIdx`；`packedRowToRouteIndex` flush 收敛为 full-load expert 段或 multi-worker expert/worker 段。small/large、`M=8192/16384`、`topK=8`、skew topK4、more experts stop17 PASS。 | T23,T24,T26 |
| 2026-06-02 | T24 accepted：large-K column chunk 路径在 `K=1024/7168,topK=4` 下保持 PTO Vec，无 scalar fallback；补充 route-pack metadata store fence 解决长列切分后 `expandedRowIdx/packedRowToRouteIndex` 可见性风险，同时回归 small/large、`M=16384` 和 `topK=8`。 | T24,T26 |

### Handoff Rules

1. 任务领取前先读 `design.md` 对应小节，再把对应 task state 从 `not_started` 改为 `claimed`，填写 owner。
2. 开始改代码前，把 state 改为 `in_progress`，确认依赖 task 都是 `accepted` 或明确 `blocked` 且当前任务允许继续。
3. 交付时创建轻量 `reports/Tx.md`，把 state 改为 `review_ready`，并在 active task 表填写 report 路径；report
   不记录编译日志、stdout 原文、counter 原文或 timeline 原文。
4. reviewer 验收通过后把 state 改为 `accepted`；未通过改为 `needs_fix` 并在 `Issue Log` 增加问题。
5. 任何 `P0/P1` issue 未关闭前，受影响的下游 task 不得从 `not_started` 进入 `claimed`。
6. 任何触发用户确认门禁的 issue 未获得用户明确决策前，受影响 task 必须保持 `needs_user_decision` 或 `blocked`，不得交付为 `accepted`。
7. 修改 `design.md` 或 `TASKS.md` 时，必须新增一条 `Design Change Log`，并列出受影响 task。
