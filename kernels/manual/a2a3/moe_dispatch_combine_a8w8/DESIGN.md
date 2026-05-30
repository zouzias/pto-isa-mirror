# moe_dispatch_combine_a8w8 详细设计

## 文档分工与单一事实源

本项目保留 3 类核心 markdown。它们不是平级重复文档，而是不同层级的单一事实源：

| 文件 | 单一职责 | 何时修改 | 不应包含 |
| --- | --- | --- | --- |
| `DESIGN.md` | 架构事实、精度路线、protocol 不变量、PTO-only 硬约束、当前项目边界、agent 任务拆分、依赖、文件范围、验收证据、阶段门禁 | 架构事实、精度口径、protocol 不变量、项目边界、任务粒度、依赖、验收方式或阶段门禁变化时 | 当前 task owner、执行进度、一次性日志 |
| `TASKS.md` | 执行状态台账：stage/task state、owner、report、Issue Log、Design Change Log、handoff rules | task 状态、owner、report、issue、design change 或 handoff 状态变化时 | 架构长解释、任务实现细节、验收标准、一次性命令长日志 |
| `reports/Mx.y.md` | 单个 task 的轻量交付结论、验收摘要、风险和下游提示 | 每个 task 交付时新增或更新一份 | 全局规则、替代 `DESIGN.md` / `TASKS.md` 的新要求、编译日志、stdout 原文、counter/timeline 原文 |

当前不创建 `README.md`。项目入口直接使用 `DESIGN.md` 和 `TASKS.md`：前者读取设计边界和第 14 章任务细节，
后者领取任务、更新状态和记录 issue/DCL。后续如果确实要补入口文档，它只能写最小导航和运行命令，不能成为新的
单一事实源。

冲突优先级：

1. 架构、精度、protocol、PTO-only 约束冲突时，以 `DESIGN.md` 为准。
2. 任务边界、依赖、文件范围、验收证据和阶段门禁，以 `DESIGN.md` 第 14 章为准。
3. 当前状态、owner、open issue、blocked 原因和 design change 是否接受，以 `TASKS.md` 为准。
4. 单个 task 的实际交付结论、验收摘要、blocked 原因和下游提示，以对应 `reports/Mx.y.md` 为准。
5. 聊天上下文和临时笔记不能覆盖上述核心 markdown。

发现文档冲突或设计缺口时，agent 不能私自选择一个版本继续实现，必须按以下流程处理：

1. 在 `TASKS.md` 的 `Issue Log` 新增 `design-bug`、`task-gap`、`primitive-gap`、`test-gap` 或
   `user-decision`。
2. 暂停受影响 task 的功能实现；若是 P0/P1，受影响下游不得进入 `claimed`。
3. 同步提交 `DESIGN.md` 或 `TASKS.md` 的修正草案，并在 `TASKS.md` 的 `Design Change Log` 记录受影响 task。
4. 如果偏差会改变最终目标、精度路线、MegaMoE protocol、PTO-only 约束或阶段验收口径，必须进入
   `TASKS.md` 的用户确认门禁，找用户明确决策。
5. 只有 reviewer/coordinator 接受修正，且需要用户确认的问题已获得明确决策后，相关 task 才能继续；
   task owner 不能 self-accept 自己的交付。

## 0. 目标、边界与对齐结论

本文档是 `kernels/manual/a2a3/moe_dispatch_combine_a8w8` 的当前详细设计。目标是把 MegaMoE 的
dispatch、FFN compute、combine 闭环按 A3 A8W8/int8 主路径和 PTO 风格重新组织。当前任务池只覆盖
A3 A8W8/int8 路线，其他精度路线和其他芯片形态不在当前范围内。

MegaMoE 对齐结论：

- 核心融合：当前最终目标是把
  `AlltoAll Dispatch -> GMM1 -> SwiGLU/Quant -> GMM2 -> AlltoAll Combine -> Restore`
  融成一个 single fused kernel 闭环。它不是简单串接 Dispatch/Combine 通信和 FFN，而是重构通信方向、数据布局和
  stage 依赖，让 GMM 能在对应 row range ready 后尽早启动。
- Dispatch 侧：采用前同步后的远端读。token owner 先把 route、dynamic quant、pack 后的 int8 payload 和
  per-token scale 写到 peer-visible window；expert owner 再用 `TGET` 拉取，直接形成 expert-major contiguous
  `gmm1InputInt8/gmA`。主路径不允许先 dispatch 到临时 buffer、再做二次 GMM1-input reorder。
- Dispatch 输入合并：`MoeInitRoutingQuant` 对应的重排、动态量化、pack 语义在本项目中由
  `RoutePackQuantLocal` 前移并合并，M2 起就写 peer-visible int8 dispatch payload 和 routing per-token scale。
- Combine 侧：采用计算后的远端写。expert owner 在 GMM2 epilogue 后直接把 output dtype return payload 写回
  token owner 的 `offsetD/returnPayload`；`RunGmm2EpilogueAndReturn` 必须融合 GMM2 epilogue 和 `TPUT` return，
  不能先全量写 `gmm2Out` 再单独 copy combine。
- Restore 侧：final restore 由 token owner 按 `expandedRowIdx + probs` 做 topK weighted restore。`probs`
  使用 float 语义，最终输出为当前固定的 FP16/BF16 等 output dtype，不能把 final output 改成 int8。
- 控制面：`tokenPerExpertMatrix`、`cumsumMM`、`preSumBeforeRank` 是核心不变量。它们同时决定 GMM group 的
  M 维范围、dispatch remote read offset、combine remote write offset 和 restore row/order。没有这三者的统一
  layout，后续 overlap 很容易出现 row、order、offset 对不上的问题。
- AIC/AIV MPMD：single fused kernel 内 AIV 负责 route、gather、activation/quant、combine return 和 restore；
  AIC 负责 GMM1/GMM2。PTO primitive 必须在对应 stage 主流程直接调用，不能通过 helper 隐藏 `TGET/TPUT`、
  `TWAIT/TNOTIFY/TTEST`、`TMATMUL` 或 quant/vector 操作。
- 计算通信重叠：Dispatch/GMM1 以 expert group ready 为最小消费边；GMM1/Activation 和 Activation/GMM2 按
  `IsSyncTask/dequantSum` 与 `swigluSyncGroups` 切 row range；GMM2/Combine 以 group、tile 或 sub-tile ready
  后尽早 return 为目标。M1/M2 可以用 `overlap_mode=off` 做 correctness，但必须使用同一套最终 stage graph、
  signal/counter、row range 和 layout。
- AIC/AIV 精细同步备选：本地 Cube/Vector handoff 可以评估 `gemm_ar` 的 ready-queue 模式，但当前不作为
  M2/M3 的必选验收路径。主路径仍先使用已定义的 sync group、`gmm2GroupReady/subTileReady` 和 counter；
  只有普通 ready signal 无法表达足够精确的 1:1 tile 依赖，或 AIV 轮询开销成为阻塞点时，才把 ready queue
  作为 design issue / primitive-gap 后的备选方案。该队列只承载 tile id / dependency id，不能封装 PTO copy、
  matmul、quant 或 return 指令；payload 仍由对应 stage 直接调用 PTO primitive。
- SwiGLU/quant 粗细粒度同步：SwiGLU 是必做计算，优化目标不是减少计算量，而是在有限同步事件资源下避免
  Swiglu/quant 推迟 GMM2/Combine 关键路径。GMM/AlltoAll 仍按专家、group、tile 或 sub-tile 细粒度展开；
  Swiglu/quant 使用 `swigluSyncGroups/dequantSum` 做前粗后细分组，前段较大 group 批量处理 AIV 工作并节省同步事件，
  后段逐步变小以减少尾部 activation/quant 对 Combine 的阻塞。M2 必须固定 group plan 和连续 row range，
  M3 只能在这个既有分组上启用 `gmm1SyncGroupReady/activationSyncGroupReady`，不能重做 activation row layout。
- 软同步：Dispatch-GMM 的多 producer / 多 consumer 不靠所有 worker 轮询所有 producer，也不靠全核 BSP 拉齐。
  M2 固定 `scoreboardTaskMap/producerStatus/scoreboardMinStatus` 语义账本、producer publish 时机和
  consumer dependency domain；M3.7 在同一账本上验证或启用 async scoreboard，让 worker 只看本 expert/tile
  依赖域的聚合状态并记录 wait/timeout counter。`scoreboardMinStatus` 不是全局最小 task id，不能让无关 expert
  或 tile 被慢 producer 拉齐。
- Sub-Tile 通信：GMM-Combine overlap 需要把一个 GMM2 tile 可能跨多个 token owner 的 return 拆成
  `OwnerSegment`，用 Sub-Tile、stride 或多段 `TPUT` 做非连续远端写。已从 PTO primitive 实现确认（见 §11、M3N.11）：
  **同步 `TPUT`（`TPut.hpp`，src/dst 各自 5 维独立 stride + `AtomicAdd`）可表达 strided/多段 sub-tile 远端写**，
  因此 sub-tile/stride return 的**功能形态可 enabled 交付**，不会整条 blocked；唯一 blocked 的是**异步 SDMA + stride**
  （`TPUT_ASYNC` 在 A3 硬性要求 flat-contiguous-1D，见 `TPutAsyncCommonDetail.hpp`），即"sub-tile return 与 GMM2 计算异步重叠"
  是真实 primitive-gap，按 blocked 记录、locator `TPUT_ASYNC requires flat-contiguous-1d`，不引入 AscendC `DataCopy` fallback。
- 观测与验证：必须有 counter 和 timeline。counter 覆盖 producer、consumer、timeout、scoreboard、subTile；
  timeline 覆盖 dispatch、GMM1、ActivationQuant、GMM2、RunGmm2EpilogueAndReturn、RestoreOutput，用来证明
  overlap 形态、等待空泡和 blocked 原因。不能只看 E2E 数字。
- 性能风险：small shape 可能因为前同步和首个 expert token transfer 无法隐藏而更慢；compute/comm 并行可能抢
  MTE/带宽并拉长 GMM 本身耗时。因此验收必须同时看 correctness、timeline、等待空泡、stage 膨胀和 E2E。
- 性能门槛口径：M3/M3N 的首要目标是**对标 `ffn.md` 把五层切分与并发"形态"做出来**（功能正确 + group/tile/sub-tile
  ownership 切分到位 + counter/timeline 证据完整），**不把"overlap on 的 E2E 必须快于 off"作为 accept 硬门**。
  E2E delta 是必须如实记录可对照的量（可正/负/平），性能调优（K width、preload/swizzle、worker 数、sub-tile 粒度）
  是形态完成后的后续迭代，不阻塞 M3N/M4。若要把某具体 shape 的性能为正设成门槛，需走 issue/DCL 单独追加。
- 当前边界：correctness baseline 固定为 A3 A8W8/int8 主路径；M3.0-M3.6 只能证明 fused overlap skeleton；
  M3.7-M3.9 的 scoreboard async、Sub-Tile/stride combine 和 timeline 通过或明确 blocked 后，才可以声称接近
  文章级 async overlap。外部实现只作为行为、协议、精度和性能对照，不能把 Catlass/AscendC 或外部融合算子依赖搬进
  本 PTO 项目源码。

当前必须落地的验收目标：

1. 语义闭环：在同一 fused stage graph 中覆盖
   `dispatch -> GMM1 -> SwiGLU/quant -> GMM2 -> combine -> restore`，并保持 final output 与 A8W8/int8
   host reference 对齐。
2. 数据面 PTO 化：payload view、Tile staging、本地 `TLOAD/TSTORE`、跨 rank `TGET/TPUT`、readiness
   `TNOTIFY/TWAIT/TTEST` 必须在对应 stage/backend 主流程中直接出现，不能藏到通用 helper 或外部 fallback。
3. 控制面协议固定：`tokenPerExpertMatrix`、`cumsumMM`、`preSumBeforeRank`、`expandedRowIdx`、ready/soft flag、
   cursor 和 offset 留在 GM control protocol 中，不强行 Tile 化。
4. A3 int8 GMM：GMM1/GMM2 使用 A3 PTO `TMATMUL` 的 `(int32_t, int8_t, int8_t)` 组合，输出 int32
   accumulator，再显式执行 dequant、cast、SwiGLU、requant 和 weighted restore。
5. MegaMoE 通信方向：Dispatch 必须是前同步后的远端读并直接形成 GMM1 input；Combine 必须是 GMM2 epilogue
   后的远端写并直接形成 return payload，主路径不能有独立二次 reorder/copy 阶段。
6. 最终依赖图前置：从 M1/M2 起固定最终 layout、producer/consumer 边、signal/counter 名称和 row range 粒度。
   `overlap_mode=off` 只是一种执行模式；M3 只能验证或启用既有边的 async 调度，不能重写 protocol、offset、
   layout 或同步关系。
7. 可观测性：correctness、accumulator/checksum/scale dump、counter、E2E timing 和 timeline 必须能在控制台
   证明目标是否达成；task report 只记录结论摘要，不粘贴 stdout 或原始日志。性能数字不能替代 correctness 和
   overlap 形态证据。

### 0.1 参考路径索引

后续 agent 优先从下表取证，不要重新从大目录里盲读。

| 路径 | 用途 | 重点查看 |
| --- | --- | --- |
| `/mnt/data/ntlab/zy/megamoe/昇腾高效支持MegaMoE融合算子，实现MoE计算通信深度融合.html` | MegaMoE 算法目标、通信策略、overlap 目标和性能分析 | 融合目标、Dispatch 前同步远端读、Combine 远端写、scoreboard soft sync、Sub-Tile/stride combine、timeline 打点 |
| `/mnt/data/ntlab/zy/megamoe/昇腾高效支持MegaMoE融合算子，实现MoE计算通信深度融合_files/` | 文章配图资产 | 只在需要核对图示阶段编号或 timeline 形态时查看；不要把网页脚本当技术来源 |
| `kernels/manual/a2a3/gemm_ar/main.cpp` | host 端 correctness/performance 记录参考 | `VerifyOutput`、`PerfStats`、warmup/measure、compute-only/sequential/pipelined timing、rank0 perf report |
| `kernels/manual/a2a3/gemm_ar/run.sh` | build/run 脚本参考 | HCCL window sizing、MPI discovery、显式 run 参数、rank 启动 |
| `/mnt/data/ntlab/zy/code/zhangyuan/vllm-ascend-zy/csrc/mc2/dispatch_ffn_combine` | mc2的 megamoe实现案例 |
| `kernels/manual/a2a3/moe_dispatch_combine_a8w8/ffn.md` | FFN 融合算子切分与并发精读 | 五层切分、group/tile 正交、init_routing 多阶段、多 AIV dispatch、GMM tile scheduler、SwiGLU 分组、Combine V1/V2 与 Sub-Tile 同步取舍 |

## 1. 设计依据

本章只列会直接约束实现的事实依据，不放任务拆解、验收步骤或阶段状态。目标结论见第 0 章；具体
stage、layout、同步和验收规则在后续章节展开。

PTO 与工程依据：

- `docs/isa/TMATMUL.md`：A2/A3 `TMATMUL` 的实现检查列出 `(int32_t, int8_t, int8_t)`、
  `(float, half, half)`、`(float, float, float)`、`(float, bfloat16_t, bfloat16_t)`；因此当前 A3 backend
  以 `int8 x int8 -> int32` 为唯一 GMM 数值路线。
- `.ai-knowledge/writeback/decision-records/2026-05-26-pto-data-plane-vs-control-plane.md`：payload tensor
  用 PTO Tile / GlobalTensor / primitive，scalar metadata 用 raw GM control protocol；因此 count、prefix、offset、
  ready flag 等控制面不强行 Tile 化。
- `.ai-knowledge/writeback/pitfalls/2026-05-26-pto-control-metadata-over-tiling.md`：不要为了 PTO 风格把
  count、flag、offset、prefix 等标量 metadata 强行 Tile 化；实现时只把 payload 数据面写成 PTO primitive
  的一等公民调用。

A8W8/int8 主路径事实依据：

- 外部输入 `a/x` 为 FP16 或 BF16；当前任务池必须显式固定一种输入/输出 dtype，并在脚本和 report 中标注。
- `w1/w2` 为 INT8，GMM 使用 `int8 x int8 -> int32` accumulator。
- `scale1/scale2` 是 A8W8 GMM epilogue 的 per-channel dequant scale 输入；kernel 侧按 `uint64_t` deq-scale view
  取数，host reference 必须使用同一 decode/等价浮点语义，不能把它们当 bias 或普通额外输入。
- routing quant 生成 int8 dispatch payload 和 float per-token scale；GMM1 epilogue 执行 per-token scale、SwiGLU、
  dynamic requant，并生成 GMM2 per-token scale。
- GMM2 epilogue 对 int32 accumulator 做 dequant/cast，并把 FP16/BF16 payload 写回 token owner；final restore 使用
  float `probs` 加权 topK 输出。

MegaMoE 文章事实依据：

- 融合目标是把 `AlltoAll Dispatch + GMM1 + SwiGLU + GMM2 + AlltoAll Combine + Restore` 组织成单算子闭环，
  让 GMM 可以在对应 row range ready 后启动，而不是等全量通信完成。
- Dispatch 采用前同步后的远端读，Combine 采用计算后的远端写；中间依赖应由卡内 C/V 生产消费、soft status
  或 PTO readiness 表达，不以 host barrier 作为主路径同步。
- 融合数据布局的核心是让 expert owner 得到 contiguous expert rows；因此 routing、重排、动态量化和 pack
  必须前移到 dispatch payload 生成阶段。
- count/control matrix 语义是 `rankNum * rankNum * expertPerRank`；`tokenPerExpertMatrix`、`cumsumMM`、
  `preSumBeforeRank` 是 GMM M 维范围、dispatch offset、combine offset 和 restore order 的共同依据。
- Dispatch-GMM 的多 producer / 多 consumer 依赖需要 soft sync / scoreboard 降低轮询复杂度；GMM-Combine 需要
  Sub-Tile/stride return 将一个 GMM2 tile 内的多 owner segment 拆成可验证的远端写。
- small shape 下 fused path 可能因为前同步和首个 expert token transfer 难以隐藏而退化；compute/comm 并行还可能引入
  带宽/MTE 抢占。验收必须同时看 correctness、counter、timeline、等待空泡和 stage 膨胀，不能只看单点 E2E。

## 2. 全路径精度、数值契约与验收语义

本章是 A3 A8W8/int8 主路径 dtype、数值计算顺序和 tolerance 验收的单一事实源。实现和 host reference
必须按本章检查中间结果；task report 只记录 dtype/tolerance/关键结论摘要。

当前必须逐段固定 A8W8/int8 主路径 dtype：

| Stage | 主路径事实 | 当前实现/验收要求 |
| --- | --- | --- |
| 外部输入 `a/x` | FP16 或 BF16 | host 数据生成支持 FP16/BF16；当前实现可先固定一个 dtype，但脚本和 task report 摘要必须显式标注 |
| `w1/w2` | INT8，当前实现固定一种显式布局 | 只生成 int8 weight；其他 layout fail-fast，不做静默转换 |
| `scale1/scale2` | `DT_INT64` 输入，kernel 侧按 `uint64_t` deq-scale view 使用 | reference 和 kernel 使用同一 decode/等价浮点语义；dump bit pattern、checksum 和 float 解释；不把它们当 bias |
| routing quant payload | 生成 int8 dispatch payload 和 float per-token scale，并把结果放进 peer-visible window | M2 起 `RoutePackQuantLocal` 把 route/pack/quant 合并，直接写 peer-visible int8 dispatch payload 和 routing per-token scale |
| Dispatch gather -> GMM1 input | 前同步后从各 token owner peer window 远端读，写入本 rank contiguous dispatch/GMM1-input 语义 buffer | M1 必须用真实 `TGET` 写 `workspace.dispatchedA` mock payload target；M2 只把同一 stage 的 target 切到 `gmm1InputInt8/gmA`，不经过二次 reorder workspace |
| GMM1 | int8 A x int8 W1，输出 int32 accumulator | PTO `TMATMUL` 先得到 int32，再显式做等价 per-channel dequant/cast 到 C workspace |
| GMM1 epilogue | C workspace 乘 per-token scale，SwiGLU，dynamic quant 到 int8 D1，生成 float per-token scale2 | `RunActivationAndQuant` 输出 int8 GMM2 input 和 float per-token scale2 |
| GMM2 | int8 D1 x int8 W2，输出 int32 accumulator | PTO `TMATMUL` 先得到 int32，再显式做等价 per-channel dequant/cast |
| GMM2 epilogue + combine return | C2 workspace 乘 per-token scale2，cast 到 FP16/BF16 后按 `preSumBeforeRank` 写回 token owner `offsetD` | M1 必须用真实 `TPUT` 写回 mock expert output；M2 只把 producer 换成 GMM2 epilogue/cast payload，`gmm2Out` 只能是 debug mirror |
| restore/unpermute | token owner 从 `offsetD` return payload 读取；`probs` 为 float，加权 topK restore | final restore 使用 float probs，输出 FP16/BF16；不输出 int8 |
| bias | int8 主路径接口没有 bias | 禁止把 bias 纳入 correctness |

数值计算契约：

1. `RoutePackQuantLocal` 对输入 `x/a` 做 routing、local pack 和 dynamic quant，生成 int8 dispatch payload
   以及 float routing per-token scale。rounding、saturation、zero point 策略必须在控制台结构化输出中可见；
   M2.2a report 只记录策略摘要和是否对齐；
   deterministic smoke case 下 payload 应能 bitwise 对齐 host reference。
2. `GMM1` 使用 PTO `TMATMUL` / `TMATMUL_ACC` 得到 int32 accumulator。small shape 的 int32 accumulator
   checksum 必须与 CPU int32 reference 完全一致，padding/tail 不能污染有效行。
3. `GMM1` epilogue 按 `scale1` 做 per-channel dequant。`scale1` 是 `DT_INT64` 输入、kernel `uint64_t`
   deq-scale view；reference 必须打印 bit pattern、checksum 和等价浮点解释。当前主路径不加 bias。
4. `ActivationQuant` 的数值顺序是：GMM1 dequant output 乘 routing per-token scale，执行 SwiGLU，再 dynamic/per-token
   quant 到 int8，生成 `gmm2InputInt8` 和 float `gmm2PerTokenScale`。Swiglu sync group 只改变调度粒度，
   不能改变单 row 的数值顺序。
5. `GMM2` 使用 `gmm2InputInt8 x weight2Int8 -> int32 accumulator`。accumulator 的 checksum 与 CPU reference
   对齐后，epilogue 按 `scale2` dequant，再乘 `gmm2PerTokenScale`，cast 到 `dtype_out` return payload。
6. `RunGmm2EpilogueAndReturn` 把 GMM2 epilogue 和 remote return 合并；主路径不能先写全量 `gmm2Out` 再 copy combine。
   debug mirror 允许存在，但 correctness 以 token owner 的 return payload 和 final restore 为准。
7. `RestoreOutput` 使用 float `probs` 按 `expandedRowIdx` 对 topK return slot 加权还原。accumulation/cast 顺序必须和
   host reference 一致；最终输出 dtype 只能是 M2.1 固定的 FP16/BF16 等 `dtype_out`，不能输出 int8。被 drop 的 slot
   （`expandedRowIdx >= num_out_tokens`）按第 6.4 节跳过且不重新归一 probs。
8. 验收分层：int8 payload 和 int32 accumulator 优先 bitwise/checksum 对齐；dequant、SwiGLU、cast 和 final output
   使用 M2.1/M2.8 固定的 dtype tolerance。最大误差、首个错误位置和相关 scale dump 由控制台结构化输出承载；
   task report 只写 pass/fail、tolerance 和关键误差摘要。

M2.8 当前集成回归的固定口径：

- 当前 A3 int8 主路径固定 `dtype_in=fp16`、`dtype_out=fp16`、`w1/w2=int8`、`scale1/scale2` 为
  `uint64_lower32_float_bits` dequant scale view，默认 acceptance tolerance 为 `atol=1e-2, rtol=1e-2`。
- 完整链路必须按
  `RoutePackQuantLocal -> GatherDispatchToGmm1Input -> GMM1 -> ActivationQuant -> GMM2 -> RunGmm2EpilogueAndReturn -> RestoreOutput`
  执行；`RoutePackQuantLocal` 直接写 peer-visible int8 payload 和 routing scale，
  `GatherDispatchToGmm1Input` 直接形成 `gmm1InputInt8`，不走 `workspace.dispatchedA` 二次 reorder。
- GMM1/GMM2 的 int32 accumulator checksum 必须精确对齐 host reference；GMM1 scale/dequant、
  SwiGLU/requant、GMM2 scale/dequant+cast、return payload 和 final output 使用上述 dtype tolerance。
- Combine 合并点固定在 `RunGmm2EpilogueAndReturn`：同一 stage 内完成 `scale2 * gmm2PerTokenScale`、FP16 cast
  和 token-owner `offsetD/returnPayload` 写回；`gmm2Out` 仅为该 stage 内的 debug/staging mirror。
- M2 final correctness report 必须同时出现 `dispatch_merge=true`、`combine_merge=true`、
  `soft_sync_ledger=true`、`swiglu_sync_groups=true`、`tile_split_return_map=true` 和 `restore_from_offsetD=true`。
  M2 使用 `overlap_mode=off`，E2E perf report 仍按固定 `warmup_iters=3`、`measure_iters=5` 输出
  `samples/avg/min/max/stddev`；M3 才验证或启用 async overlap。

## 3. 约束

本章定义当前任务池的实现边界和硬约束。实现如果需要越过这些边界，必须按 `TASKS.md` 的
`user-decision` 门禁找用户确认，不能在 task 内自行改目标或引入旁路实现。

- 不做 W4A8、FP4/FP8、A5 backend、其他量化精度、bias 路线或 int8 final output。
- 不把独立 Dispatch/Combine 通信算子、外部融合算子或 sibling 分支源码搬进本项目；本项目只实现 PTO manual
  fused 主路径。
- 不用 host barrier、多 launch 或临时 host copy/reorder buffer 替代 device-side remote readiness。它们只能作为
  本地 debug/repro 辅助，不能作为 accepted 主路径。
- 不把 `tokenPerExpert`、`expandedRowIdx`、`softFlagBase`、cursor、offset、prefix 等 metadata 当成普通 tensor
  payload 规划独立 L1/L0 双缓冲。
- 不要求当前版本覆盖所有 shape、所有 topK、所有 expert count、所有 weight layout 或 small-shape 性能优化。
  small shape 只作为 correctness smoke 和风险记录，不作为独立性能目标。
- 容量受限 drop 是 in-scope，且必须对齐参考实现 `dispatch_ffn_combine` 主 kernel 的做法：`maxOutputSize` 是本 rank
  所有 local expert 合计的输出 row 预算，drop 采用位置序前缀和截断，restore 对 dropped slot 跳过且不重新归一。
  具体语义、各 stage 截断公式、`expandedRowIdx` sentinel 和 golden 规则见第 6.4 节，**不允许**用 per-expert
  capacity-padded 布局替换现有 compact `cumsumMM` 布局。
- 不为了补 PTO primitive 缺口引入 Catlass/AscendC fallback。缺能力时只能补 PTO public primitive、标记 blocked
  或触发用户确认门禁。

### 3.1 PTO 化依赖硬约束

本项目最终验收口径是 PTO 化工程，不接受 Catlass/AscendC 作为项目源码依赖，也不允许直接使用它们的任何接口。
这里的“依赖/接口”同时包括 header、library、symbol、source API、build helper、fallback path 和临时调试路径。
允许 PTO 框架自己的 backend 在 `include/pto/**` 内部使用平台实现细节；但 `moe_dispatch_combine_a8w8` 项目源码
不能直接 include、调用、封装或转发这些实现层 API。

项目源码禁止出现：

```text
kernel_operator.h
matmul_intf.h
AscendC::
catlass / Catlass / CATLASS 任意大小写
LocalTensor
GlobalTensor from AscendC namespace
TQue / TBuf / TPipe
DataCopy / DataCopyPad
BlockMmad / MmadAtlas*
ascendc_library / ascendc.cmake / ascendc_kernel_cmake
```

项目源码允许的 device 入口依赖：

```cpp
#include <pto/pto-inst.hpp>
#include <pto/comm/pto_comm_inst.hpp>
#include <pto/common/constants.hpp>
```

如后续需要某个 PTO public primitive 尚未提供的能力，处理方式只能是：

1. 在 PTO 框架内补 public primitive；
2. 或把对应阶段标记为 blocked；
3. 不能在本项目下绕过 PTO 直接接 Catlass/AscendC，也不能用项目内 helper 把 Catlass/AscendC 接口藏起来。

依赖扫描口径：

- 对源码扫描时，禁止项目源码 include、调用、包装、转发或 fallback 到上述 AscendC/Catlass 接口。
- 对 CMake 和脚本扫描时，禁止 `ascendc_library`、`ascendc.cmake`、`ascendc_kernel_cmake`、Catlass target、
  外部融合算子 target 和隐藏 fallback 路径。
- CMake 中仅作为 CANN 编译器/SDK 搜索路径存在的系统 include 目录，不单独视为 AscendC 接口依赖；但这些路径
  不能伴随项目源码中的 AscendC API include/call/wrap，也不能引入 AscendC build helper。

## 4. 总体架构

本项目的架构分为 host runtime、single fused device kernel、protocol/layout contract、A3 int8 stage set 和观测输出。
这些是文件/职责边界，不是用来隐藏 PTO primitive 的通用 helper 层。

```text
host/example
  |
  +-- RuntimeSubstrate
  |     - ACL/HCCL/MPI init
  |     - HCCL window bootstrap
  |     - ordinary GM workspace allocation
  |     - tiling/shape validation
  |     - correctness / perf / timeline stdout report
  |
  +-- kernel/MegaMoeKernel
        - single fused kernel
        - MPMD AIV/AIC role split
        - final stage graph exists from M1/M2
        |
        +-- ProtocolAndLayout
        |     - WorkspaceLayout typed GM views
        |     - tokenPerExpertMatrix / cumsumMM / preSumBeforeRank
        |     - expandedRowIdx / probs / owner segment metadata
        |     - ready signal / soft-sync ledger / counter / timestamp views
        |
        +-- AIVStages
        |     - route / pack / dynamic quant
        |     - peer count publication
        |     - remote dispatch gather direct to GMM1 input
        |     - GMM2 epilogue fused with remote combine return
        |     - final weighted restore
        |
        +-- AICStages
              - GMM1 PTO TMATMUL int8 x int8 -> int32
              - GMM2 PTO TMATMUL int8 x int8 -> int32
```

关键边界：

- `RuntimeSubstrate` 只存在于 host 侧，负责 HCCL window、rank、stream、launch、shape config 和报告输出；
  device stage 不查询 runtime，也不能依赖 host barrier 作为主路径同步。
- `ProtocolAndLayout` 是所有 stage 的共享契约，集中定义 GM typed view、row order、owner segment、ready signal、
  soft-sync ledger、counter 和 timestamp；后续任务不能绕过它私自解释 offset。
- `AIVStages` 负责 routing、payload 搬运、quant/activation、combine return 和 restore；跨 rank payload 通信必须在
  对应 stage 主流程中直接调用 `TGET/TPUT/TNOTIFY/TWAIT/TTEST`。
- `AICStages` 只负责 GMM1/GMM2 的 PTO int8 matmul；`TMATMUL/TMATMUL_ACC` 必须在 GMM stage 主流程中直接调用。
- `A3Int8Backend` 是当前唯一 numeric stage set。若后续需要其他精度或芯片 backend，必须重新进入设计/用户确认门禁，
  不能在当前工程里预埋未验收的 runtime backend 切换。
- M1 的 mock 只允许替代 GMM 数值输出；routing、prefix、dispatch gather、combine return、restore、layout 和同步对象
  必须已经按最终架构落地。

### 4.1 Host/device 职责与工程目录

当前工程目录契约：

```text
moe_dispatch_combine_a8w8/
  DESIGN.md
  CMakeLists.txt
  host/
    main.cpp
    args.hpp
    reference.hpp
    comm_mpi.hpp
    hccl_context.hpp
    workspace_layout.hpp
    hccl_window.hpp
  include/
    moe_dispatch_combine_a8w8_runtime_types.hpp
    moe_dispatch_combine_a8w8_m1_layout.hpp
    moe_dispatch_combine_a8w8_types.hpp
    moe_dispatch_combine_a8w8_layout.hpp
  kernel/
    moe_dispatch_combine_a8w8_kernel.cpp
    kernel_launchers.hpp
    protocol_core.hpp
    a3_int8_backend.hpp
    control_metadata.hpp
  scripts/
    run_a3.sh
    gen_data.py
```

host 侧职责：

- 解析显式 shape 参数，不隐藏 case。
- 初始化 ACL/HCCL/MPI，绑定 rank/device。
- 分配 ordinary GM workspace 和 HCCL peer window。
- 初始化输入、expertId、probs、weights、scale1/scale2、per-token scale workspace 和 golden reference；当前 int8
  主路径不生成 bias。`xActiveMask[M]`（bool）为可选输入：传入时按其标记把 inactive token drop，不传时全部 active；
  golden 必须按同一 mask 语义生成。
- launch single fused kernel，回收输出，打印 correctness/perf/timeline stdout 报告。

device 侧职责：

- 不查询 runtime，不做 host resource bootstrap。
- 只接收已切好的 local workspace、peer window、HCCL context、tiling data。
- 所有 stage 通过 `StageContext` 访问 layout、shape 和 rank 参数；stage 内 PTO primitive 仍然直接调用。

### 4.2 Shape 与 tiling 契约

当前 launch-time shape / tiling 参数：

| 参数 | 含义 |
| --- | --- |
| `rankNum` | EP / rank 总数 |
| `rankId` | 当前 rank |
| `expertPerRank` | 每 rank local expert 数 |
| `topK` | 每 token expert 数 |
| `M` | local token 数 |
| `hiddenSize` | input/output hidden |
| `intermediateSize` | FFN intermediate |
| `maxTokensPerExpert` | 每 local expert 容量上限 |
| `payloadTileCols` | dispatch/combine hidden chunk |
| `gmmBlockM/N/K` | backend GMM tile |

shape / tiling 约束：

- `hiddenSize`、`intermediateSize` 必须满足 A3 PTO int8 matmul K/N alignment。
- topK 先固定为主路径常用值；扩展 topK 前先验证 `expandedRowIdx` 和 combine weighted sum。
- `maxOutputSize` 是本 rank 所有 local expert 合计的输出 row 预算（workspace/window 按它预分配），不是 per-expert
  capacity。token 总量超 `maxOutputSize` 时按第 6.4 节的位置序前缀和截断 drop，不 fail-fast、不静默改写其他 stage 的
  offset 语义。当前 smoke/balanced shape 给足预算时不触发 drop，但截断公式、sentinel 和 no-renorm restore 必须就位
  并由 golden 覆盖。
- `rankNum * expertPerRank` 必须覆盖 expert id 范围；非法 expert id 进入 debug counter 并触发 host 检查失败。
- 所有 peer-visible buffer 按 window layout 计算上限，host launch 前检查 `peerWindowOffset + bytes <= winSize`。

## 5. Kernel 与 stage 划分

当前主路径使用一个 single fused device kernel。kernel 内按 MPMD 方式区分 AIV/AIC 角色，但 stage 名称、layout、
signal 和 counter 从 M1/M2 起就是最终 fused 结构；`overlap_mode=off` 只是同一结构的执行模式。

| Stage | 主要 core | 职责 | PTO 化边界 |
| --- | --- | --- | --- |
| `RouteLocalTokens` | AIV | 读取 `expertId/probs`（存在 `xActiveMask` 时先把 inactive token 的 expertId 改写为越界 invalid），统计 local token per expert，生成 per-block count | metadata raw GM，批量 compare/pack 直接用 PTO Vec primitive |
| `RoutePackQuantLocal` | AIV | 把 routing、local pack、dynamic quant 合并，直接写本 rank peer-visible int8 dispatch payload 和 per-token scale | payload 用 `GlobalTensor + Tile + TLOAD/TSTORE/TQUANT`，不生成主路径二次 reorder workspace |
| `PublishCounts` | AIV | 发布 count matrix row，形成 dispatch 前同步 | `TPUT` 或 remote GM view + `TNOTIFY` |
| `WaitCounts` | AIV | 等待所有 token owner count row 可见 | metadata raw GM + `TWAIT/TTEST` |
| `BuildCumsumAndPreSumBeforeRank` | AIV | 构造 `cumsumMM/preSumBeforeRank/expertTokenNums` | 输出必须能解释 dispatch/gmm/combine offset |
| `GatherDispatchToGmm1Input` | AIV | 前同步后从 token owner rank 远端读属于本 rank local experts 的 int8 rows，直接落入 `gmm1InputInt8/gmA` | `TWAIT/TTEST + TGET`，不经过额外 GMM1-input reorder stage |
| `Gmm1` | AIC | local expert-major grouped matmul | A3 int8 `TMATMUL` |
| `ActivationQuant` | AIV | SwiGLU、per-token scale、int8 requant | PTO Vec + int8 requant |
| `Gmm2` | AIC | second grouped matmul | A3 int8 `TMATMUL` |
| `RunGmm2EpilogueAndReturn` | AIV | GMM2 epilogue 后立即按 `preSumBeforeRank` 写回 token owner rank `offsetD`；同一 stage 保留 Sub-Tile/stride ready 对象 | `TCVT/TMUL + TPUT + TNOTIFY` 在同一 stage 主流程中直接出现 |
| `RestoreOutput` | AIV | token owner rank 等待全部 return，按 `expandedRowIdx + probs` 加权还原 | PTO Vec tile 做 weighted sum |

依赖和同步关系必须从首次实现就按最终 fused 目标建模，不能先写一套临时串行依赖，等 M3 再二次重构。M1/M2 可以用
`overlap_mode=off`、BSP wait 或 `*_mode=ledger_only` 做 correctness，但必须在同一套最终 stage 图、ABI、
workspace layout、signal/counter 名称、producer/consumer 关系和 row range 粒度上执行。多 launch 只能作为本地
debug/repro 辅助，不能作为任何 M1/M2/M3 task 的 accepted 主路径。

最终依赖图：

| Edge | Producer | Consumer | 数据依赖 | 同步对象 | 首次落地 |
| --- | --- | --- | --- | --- | --- |
| count publish -> dispatch gather | `PublishCounts` | `GatherDispatchToGmm1Input` | `tokenPerExpertMatrix` 完整可见 | `countReadySignal[tokenOwnerRank]` | M1.6 |
| prefix -> dispatch/gmm/combine | `BuildCumsumAndPreSumBeforeRank` | `GatherDispatchToGmm1Input` / `Gmm1` / `RunGmm2EpilogueAndReturn` | `cumsumMM`、`preSumBeforeRank`、`expertTokenNums` | metadata completion counter | M1.6a |
| dispatch gather -> GMM1 | `GatherDispatchToGmm1Input` | `Gmm1` | contiguous `gmm1InputInt8/gmA` row range + routing scale | `dispatchGroupReady[expert]` and M2 `scoreboardTaskMap` | M2.2b/M2.2c |
| GMM1 -> activation | `Gmm1` | `ActivationQuant` | int32 accumulator dequant row range | `swigluSyncGroups/dequantSum` + `gmm1SyncGroupReady[syncIdx]` | M2.1/M2.5 metadata, M3.1 signal |
| activation -> GMM2 | `ActivationQuant` | `Gmm2` | `gmm2InputInt8` + per-token scale2 row range | `swigluSyncGroups/dequantSum` + `activationSyncGroupReady[syncIdx]` | M2.1/M2.5 metadata, M3.1 signal |
| GMM2 -> fused return | `Gmm2` | `RunGmm2EpilogueAndReturn` | GMM2 accumulator tile/sub-tile | `gmm2GroupReady[expert]` / `subTileReady[aicTile]` | M2.7/M2.7a metadata, M3 signal/async |
| fused return -> restore | `RunGmm2EpilogueAndReturn` | `RestoreOutput` | token owner `offsetD` return payload | `combineDoneSignal[expertOwnerRank]` | M1.9/M2.7 |

这张表是实现依赖关系的单一目标。后续任务只能把某条边从 `overlap_mode=off` / BSP wait / ledger-only 切到
async 执行，不能改 producer/consumer、offset 语义或同步粒度；如果必须改，按 `TASKS.md` 的 `design-bug` /
`user-decision` 流程处理。

代码合入标准：

- 非 mock stage 一开始就按最终函数边界、layout、signal/counter 和 row range 实现。
- mock 只允许替代尚未实现的 GMM 数值结果，不能替代 routing、dispatch gather、prefix、combine return 或 restore
  的协议关系。
- 如果某条边缺 PTO public sync primitive，可以保留最终 signal/counter 和 producer/consumer call site，以
  `primitive-gap` 标记 blocked 或 ledger-only；不能改写成另一套临时 host barrier / multi-launch 主路径。
- M3 的工作是验证或启用 async 调度并补验证据，不应重写 M1/M2 已合入的数据布局、依赖图或同步对象。

### 5.1 主路径必须守住的关键点

MegaMoE 主路径的核心不是单纯串联两个 GMM，而是一个 mixed AIC/AIV kernel：

- MegaMoE 的核心改动是通信算法重构，不是复用后同步远端写路径。PTO 版
  `RoutePackQuantLocal` 必须先把本 rank 的重排/量化结果放在 peer-visible window；目标 rank 再通过
  `GatherDispatchToGmm1Input` `TGET` 远端读形成 local expert-major 连续矩阵。
- AIC 分支只做 `GMM1(params); GMM2(params)`。
- AIV 分支负责 routing、跨 rank count publish/wait、remote payload 拉取、
  activation preprocess、GMM1 epilogue/SwiGLU/quant、GMM2 epilogue/combine 和 final unpermute。
- `GMM1` 的第一道门是 `BuildCumsumAndPreSumBeforeRank` 完成 `cumsumMM/preSumBeforeRank`，第二道门是当前 expert group
  的 dispatch rows 已被 `GatherDispatchToGmm1Input` 写入 `gmm1InputInt8/gmA`。
- `GMM1` 不是任意 source segment ready 就无条件开始；主路径先按 expert group 顺序消费，并在每个 group 预处理完成后
  放行对应 GMM group。
- `GMM1 -> ActivationQuant -> GMM2` 的同步边界不是任意 tile queue，而是 baseline `IsSyncTask(groupIdx, expertPerRank)`
  形成的 sync group，并用 `dequantSum` 描述每个 sync group 的 row range。
- Combine return path 依赖 `tokenPerExpert` 和 `preSumBeforeRank`，并把 GMM2 epilogue 与写回 token owner rank 的
  `offsetD` return payload 结合；PTO M2 不能先做全量 `gmm2Out` 再独立 copy combine。
- final restore 不是简单 copy：token owner rank 等待跨 rank combine 完成后，按 `expandedRowIdx + probs` 对 topK 输出加权
  unpermute 到最终 `output`。

因此实现阶段允许 M1/M2 在最终 stage 图内用 `overlap_mode=off` 或 BSP wait 做 correctness，但主路径不能退化为
多 launch 串行链。M3 通过时必须证明 single-kernel MPMD 或等价 device-side producer-consumer
调度有效；否则只能称为 correctness 版本，不能称为 fused overlap 版本。

#### 5.1.1 A3 fused 的 PTO mixed ELF 模型

当前 A3 上 fused 能力的判断口径如下：

- A3 fused mixed AIC/AIV 本身可行。参考 MC2 `dispatch_ffn_combine` 的 fused 形态依赖 mixed task metadata
  (`KERNEL_TYPE_MIX_AIC_1_2`) 和 op-host tiling/registration；它不是把所有 Vec/Cube primitive 硬塞进一个
  普通 `--cce-aicore-arch=dav-c220` target。
- 本项目保持 PTO-only，不引入 `kernel_operator.h`、`AscendC::`、Catlass 或 custom-op host 注册接口。PTO 侧等价
  抓手是 `include/pto/common/kernel_meta.hpp` 的 `.ascend.meta` metadata 宏，以及
  `tests/npu/a2a3/src/st/testcase/syncall` 展示的 mixed ELF/registration 构建方式。
- 直接设置 `FUSED_KERNEL_ARCH=dav-c220` 编译现有 Vec/comm kernel 失败，只能说明“单 raw mixed target”这条路
  不成立；不能作为“A3 不能 fused”或“M2.8 不需要 single fused MPMD”的证据。
- M2.8/M3.0 的正确实现路径是：AIC side 以 `dav-c220-cube` 编译，AIV side 以 `dav-c220-vec` 编译，通过
  PTO `.ascend.meta` 和 registration ELF/handle launch 形成一个 single fused MPMD launch。

M2.8 后续开发必须先做最小 mixed launch spike，再迁移主链：

1. M2.8a 最小 PTO mixed launch：AIC/AIV 两侧共用同一 ABI，只写入各自 heartbeat/counter，host 通过
   registration ELF 单次 launch 验证 AIC 和 AIV 都执行。
2. M2.8b fused stage graph skeleton：保留现有 M2 workspace/layout/scoreboard 字段，构造单 launch 的
   AIC/AIV stage orchestrator；先使用 `overlap_mode=off` 的 device-side wait，替代 host stage 间
   `aclrtSynchronizeStream` / `MpiBarrier` 语义。
3. M2.8c 全链迁移：AIV side 接入 dispatch、epilogue、activation/requant、return、restore，AIC side 接入
   GMM1/GMM2；host 只发起 single fused launch，multi-launch 仅保留为 `multi_launch_debug` 回归路径。

M2.8 系列的 stage 边界同步与 M3 的 overlap 同步是两套粒度，必须区分，不能混为一谈：

- M2.8a/b/c 用卡内 `SYNCALL<SyncCoreType::Mix>` 全核栅栏即可做 overlap-off correctness，stage 之间硬同步是允许的。
- 真正的 `GMM1↔SwiGLU↔GMM2` per-sync-group overlap 由第 10 章命名的 `pto::Event<...>::Init<CrossCoreId>() /
  Wait<CrossCoreId>()`（`ffts_cross_core_sync`/`wait_flag_dev`）或 `TSYNC_CVID` 在 M3 打开，等价于 `ffn.md` 的
  `CrossCoreSetFlag/WaitFlag<0x2>(flagId)`，**不是 primitive-gap**。
- 因此 M3 把"每个 stage 一个 `SYNCALL<Mix>`"细化为"按 sync group/tile 放行的 per-`CrossCoreId` 握手"是设计内的
  正常演进，不需要重写 protocol、layout 或 row order。

### 5.2 主路径策略到 PTO stage 对照

MegaMoE 主路径到本项目 stage/backend 的对照：

| 主路径对象 | PTO 项目对应 | 必须保留的不变量 |
| --- | --- | --- |
| routing + dispatch payload | `RoutingMetadata` + `RoutePackQuantLocal` | expert routing、expanded row、peer-visible int8 payload、per-token scale 语义一致；M2 起合并 routing/pack/quant |
| count publish/wait | `PublishCounts` + `WaitCounts` | 每个 token owner rank 的 count row 对所有 expert owner rank 可见 |
| prefix/cumsum metadata | `BuildCumsumAndPreSumBeforeRank` | `tokenPerExpertMatrix -> cumsumMM -> preSumBeforeRank` 三者同源 |
| dispatch gather rows | `GatherDispatchToGmm1Input` | 前同步后从 token owner rank remote read，直接形成 contiguous `gmA/gmm1InputInt8` |
| GMM1 group loop | `A3Int8Backend::RunGmm1` | `currentM` 来自 `cumsumMM[rankNum - 1][localExpert]`，按 expert group 消费 |
| GMM1 activation/quant | `RunActivationAndQuant` | scale1 dequant、per-token scale、SwiGLU、int8 requant 边界一致 |
| GMM2 group loop | `A3Int8Backend::RunGmm2` | GMM2 input 只消费 activation ready 的 sync group |
| GMM2 epilogue + return | `RunGmm2EpilogueAndReturn` | scale2 dequant、per-token scale2、output dtype cast 后按 `tokenPerExpert + preSumBeforeRank` 直接写回 token owner `offsetD` |
| final restore | `RestoreOutput` | 按 `expandedRowIdx + probs` 对 topK expert output 做 weighted restore |

MegaMoE 策略到本项目 stage 的对照：

| 策略 | PTO 项目对应 | 阶段验收 |
| --- | --- | --- |
| routing/quant 输出放到 peer-visible window | `RoutePackQuantLocal` | M2.2a white-box dump |
| Dispatch 前同步 + 远端读，形成 contiguous expert rows | `PublishCounts` / `BuildCumsumAndPreSumBeforeRank` / `GatherDispatchToGmm1Input` | M1.6-M2.2b counter、prefix 和 payload dump |
| Dispatch-GMM 多生产者/多消费者 soft sync | M2 固定 `scoreboardTaskMap/producerStatus/scoreboardMinStatus`、producer publish 时机和 consumer dependency domain，M3 验证或启用 async scoreboard 调度 | M2.2c white-box dump + M3.7 counter-log |
| AIC/AIV 同 kernel 并行推进 | single-kernel MPMD stage functions | M3.0-M3.6 black-box-run + counter-log |
| `IsSyncTask/dequantSum` 粗粒度同步 | `gmm1SyncGroupReady` / `activationSyncGroupReady` | M2.5 固定 group row range，M3.1-M3.4 启用 signal/counter |
| Swiglu/quant 前粗后细分组 | `swigluSyncGroups` + `dequantSum` | M2.5 group plan dump，M3.3-M3.4 counter-log |
| GMM-Combine Sub-Tile/stride return | M2 固定 tile owner mapping/source row/destination `offsetD` segment plan，M3 验证 stride 或多段 `TPUT` async 执行能力 | M2.7a white-box dump + M3.8 enabled 或 blocked 证据 |
| kernel 内 timestamp 转 timeline | `debugCounters` / `timeline` export | M3.9 timeline |

### 5.2.1 FFN 五层切分与并发契约

`ffn.md` 对本项目的直接约束是：FFN 融合不是把 dispatch、GMM、SwiGLU、combine 串到一个文件里，而是把
空间切分和时间依赖拆开表达。M2/M3 后续实现必须遵守下面的五层契约：

| 层级 | 本项目语义 | M2 必须固定 | M3 才打开或验证 |
| --- | --- | --- | --- |
| rank / expert ownership | `tokenOwnerRank` 与 `expertOwnerRank/localExpert` 决定 dispatch 读源、GMM group、combine 目的和 restore order | `tokenPerExpertMatrix/cumsumMM/preSumBeforeRank/expandedRowIdx` 同源，host/device dump 可追溯 | 不重排 owner 语义，只在已固定 owner segment 上打开 async |
| core worker | AIV 负责 route/count/gather/activation/return/restore，AIC 负责 GMM1/GMM2 | M2.8c 如实打印每个 AIV stage worker facts；worker=1 不阻塞 M2 | M3.2/M3.3/M3.5/M3.8 把对应 payload stage 提升为多 AIV worker，并用 counter/timeline 证明 |
| expert group / sync group | group 是 ready/sync 边界，不是“某个核负责某个 expert”的核分配单位 | `dispatchGroupReady`、`swigluSyncGroups/dequantSum`、`gmm2GroupReady` 的 row range 和 producer/consumer 边固定 | M3 只改变这些边的执行时序和 worker 分摊，不重新定义 group row order |
| tile / sub-tile | AIC 负载按 GMM tile 分配；combine 可把 GMM2 tile 投影成 owner segment 或 sub-tile | GMM tile task、`ReturnSegmentPlan/OwnerSegment/subTileReady` schema 固定，segment correctness 已验证 | M3.5 做 continuous owner-segment 多 AIV return；M3.8 再验证 Sub-Tile/stride async |
| L1/L0 / pipe | GMM 内部按 cache-level tile 和 ping-pong state 组织，epilogue/dispatch/return 内部用 Vec/comm pipe 排空 | M2.GMM 记录 L1/L0 tile shape、stage 数、tail、multi-AIC evidence；Vec epilogue/requant 有 PTO primitive 证据 | M3 可补 swizzle、L2 hint、preload/pipe overlap、SyncAll 空泡分析，但不能把 M2 GMM 降回 fixed smoke |

由此得到三个硬性口径：

- **group/tile 正交**：所有 AIC 都可以遍历所有 expert group；真正分给 AIC 的是 GMM tile。`dispatchGroupReady` 或
  `gmm1SyncGroupReady` 放行的是 group/sync group 的数据依赖，不表示该 group 被某个固定 core 独占。
- **并发证据分层**：`mixed_aiv_blocks>0` 只能证明 fused launch 中 AIV side 存在；只有 per-stage worker
  processed/skipped counters、owned row/segment/tile range 和 timeline 才能证明 payload-level AIV 数据并行。
- **访存局部性与同步微优化分层**：`ffn.md` §3.2/§5.2/§10 的 swizzle 遍历、L2 cache hint（冷门 expert 旁路 L2 防污染）、
  preload drain（`SynchronizeBlock` 排空欠账）、`icache_preload` 和 count 同步的 DataAsFlag（数据兼到达信号，省一次
  独立信号往返）属于访存局部性 / 同步微优化层，不影响 correctness、offset 语义和 ready 边。它们是 M3 可选项
  （DataAsFlag 在 PTO 侧的等价表达是 payload `TPUT` 落地后用 `TTEST` 轮询 count 区）；M2 不要求实现，实现时也不得
  改变已固定的 row order、offset 或 ready 语义。

#### 5.2.1.1 参考实现的具体切分粒度（基线参考，非强制对齐）

下表把参考工程 `dispatch_ffn_combine` 的具体切分数值抽出来，和本项目当前值并排，作为 M2.GMM / M3 调优的基线参考。
本项目 canonical 值以 `include/moe_dispatch_combine_a8w8_types.hpp` 和 M2.3a 已验收实现为准；参考列只用于解释差异和提供
perf 调优方向，**不作为强制对齐目标**。参考来源：`op_kernel/dispatch_ffn_combine.h`、`op_kernel/utils/const_args.hpp`、
`op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp`。

| 切分维度 | 参考实现值 | 本项目当前值 | 说明 / 差异 |
| --- | --- | --- | --- |
| GMM tile (M×N) | `128 × 256` | `kGmmBaseM=128 × kGmmBaseN=256` | 一致；都是 AIC 的工作分配单位 |
| L1 tile K（每次进 L1 的 K 宽） | `512`（`L1TileShape=GemmShape<128,256,512>`） | `kGmmBaseK*kGmmStepK = 64*4 = 256` | **差异**：参考 L1-K=512，本项目=256（一半）。两者都按 4 个 K-slice 组织 L1 panel，但绝对宽度不同，参考更利于摊薄 weight 读 |
| L0 tile K（每次进 L0 的 K 宽） | `128`（`L0TileShape=GemmShape<128,256,128>`） | `kGmmBaseK = 64` | **差异**：参考 L0-K=128，本项目=64（一半） |
| L1 ping-pong stages | `l1Stages=2` | `kGmmL1Stages=2` | 一致（双缓冲） |
| L0A / L0B stages | `l0AStages=2 / l0BStages=2` | `kGmmL0AStages=2 / kGmmL0BStages=2` | 一致 |
| L0C stages | `l0CStages=1` | `kGmmL0CStages=1` | 一致（accumulator 单缓冲） |
| preload 异步级数 | `preloadStages=1` | 当前无显式 preload-async（M3 可评估） | 参考用 `MmadAtlasA2PreloadAsyncFixpipe`；本项目 preload/pipe overlap 列为 M3 可选 |
| tile 遍历 swizzle | `GemmIdentityBlockSwizzle<9, 1>` | 暂无（行优先调度） | 参考用 swizzle 提升 weight L2 命中；本项目 swizzle 列为 M3 访存局部性可选项 |
| int8 L1 占用（A/B，双缓冲） | A `128×512=64KB`、B `512×256=128KB`、合计双缓冲约 `388KB`（< 512KB） | A `128×256=32KB`、B `256×256=64KB`（更宽松） | 参考贴近 L1 上限以最大化 K 摊薄；本项目留更多余量 |
| SwiGLU epilogue 分组粒度 | `epilogueGranularity = expertPerRank - 3`（`≤4` 时 `-1`），即两段 | `swigluSyncGroups` 幂指数 `{8,4,2,1,1}`（前粗后细） | 本项目用升级版多段分组，机制兼容、更细 |
| 跨核 flag 复用上限 | `CROSS_CORE_FLAG_MAX_SET_COUNT = 15` | FFTS 物理 0-15，用户区 0-10（见 §10） | 见 §10 计数信号量 + 折叠 |
| init_routing 量化列 loop 上限 | `MAX_COLS_ONE_LOOP_QUANT = 8192` | 由 `payloadTileCols` 控制 | 本项目无独立 init_routing 子系统（见 §6 无 sort 说明） |
| 多核归并排序路数 | `MAX_MRGSORT_LIST = 4`（VBS/VMS/SortOut） | 不适用 | 本项目不实现 multi-core sort |
| AIC:AIV mixed launch 比例 | AIV = 2 × AIC subblock（1:2）；`blockDim = CalcTschBlockDim(aivNum, aicNum, aivNum)` | `kAicBlocks=24`，AIV=48（`subblockdim=2`），即 1:2 | **固定 launch/硬件事实**：A3 每个 cube 核配 2 个 vector subblock，由 mixed ELF meta 决定，不是自由可调比例 |
| 逐阶段核分配（如 epilogue/dispatch 用几核） | `epilogueCoreNum`、`aivNumInitRouting=2*BLOCK_NUM` 等可调参数 | M2 多为 worker=1，按 stage 上报 | **不在设计固定具体数值**：属 M3 调优量，由 runtime logical core count 推导，M2.8c 只如实上报 worker facts |

关于核分工的两层区分（重要）：

- **AIC:AIV 总比例（1:2）是固定 launch 事实**，和 tile 尺寸一样属于切分粒度，已列入上表；`kAicBlocks/kAivRatio` 是它的工程入口。
- **各 stage 的 AIV worker 数不是设计常量**，而是 §5.2.1 core-worker 层和 M3.2/M3.3/M3.5 负责的调优量：M2 可以是 worker=1，
  只需如实打印 `dispatch_aiv_workers/activation_aiv_workers/combine_return_aiv_workers/restore_aiv_workers` 等字段；M3 按
  runtime logical core count 提升并用 counter/timeline 证明。设计不钉死这些数字，避免与"M2 worker=1 不阻塞、M3 调优"自相矛盾。

由此给 M3 的两个可量化调优方向（仅参考，不阻塞 M2）：

- **L1/L0 K 宽**：本项目当前 L1-K=256 / L0-K=64 是参考的一半。M3 性能阶段可评估提到 L1-K=512 / L0-K=128 是否在 PTO
  `TMATMUL` + L1 预算下可行，以更好摊薄 weight 读；若 PTO tile 类型或 L1 余量不支持，记 `primitive-gap` 或保留当前值。
- **swizzle / preload-async**：参考用 `GemmIdentityBlockSwizzle<9,1>` + `preloadStages=1` 提升 weight L2 命中和搬运掩盖；
  本项目把它们列为 M3 访存局部性可选项，启用时不得改变 GMM tile task 的 row/order 语义。

> **实现 tip**：写代码时若某处切分（tile 边界、K-loop 步进、group/sync-group row range、owner segment 拆分、core 分工）
> 一时想不清楚，可对照参考实现 `/mnt/data/ntlab/zy/code/zhangyuan/vllm-ascend-zy/csrc/mc2/dispatch_ffn_combine`
> （`op_kernel/dispatch_ffn_combine.h`、`block_mmad_preload_async_fixpipe_quant.hpp`、`dispatch_ffn_combine_kernel.hpp`）
> 看它的切分**思路与数值**。这只是**只读参考**：禁止拷贝/include/链接 AscendC/Catlass 代码；本项目仍用 PTO primitive 重写，
> 且 row order/offset/owner segment 语义以本设计为准，不能因为参考不同就改契约。

### 5.3 M2 必须前置的 MegaMoE 合并点

如果 M2 只做“协议 mock + 普通 int8 backend”，M3 再补 MegaMoE 的数据流合并，会把 dispatch row layout、
GMM1 input layout、GMM2 output layout 和 combine return offset 全部重写一遍，overlap 阶段会留下过多 gap。
因此 M2 的定位不是纯 backend correctness，而是 MegaMoE-ready correctness。

M2 必须先固定六个点：

1. Dispatch 输入侧合并：`RoutePackQuantLocal` 把 local routing、expanded row index、dynamic quant、local pack 合并，
   直接把 int8 hidden row 和 routing per-token scale 写入本 rank peer-visible window。`GatherDispatchToGmm1Input`
   前同步后用 `TGET` 从各 token owner 读取，并直接落入本 rank `gmm1InputInt8/gmA` 的 expert-major contiguous
   row range。
2. Combine 输出侧合并：`RunGmm2EpilogueAndReturn` 把 GMM2 int32 accumulator 的 scale2 dequant、per-token scale2、
   output dtype cast 和 `TPUT` remote return 放在同一个 stage 主流程。写回目的地由
   `tokenPerExpertMatrix + preSumBeforeRank` 推导，直接落到 token owner `offsetD` return payload。
3. Dispatch-GMM soft-sync 语义账本：M2 不要求真正异步 scoreboard 提速，但必须定义 task id 到 token owner
   segment、expert group 或 row/tile range 的映射，定义 producer status publish 时机、consumer dependency domain、
   `scoreboardMinStatus`、worker wait counter 和 timeout counter 的 layout/dump。M2 可在 BSP 或 soft-sync-off
   模式执行，但 `GatherDispatchToGmm1Input -> GMM1` 的依赖必须能被这个账本解释，不能到 M3 才重新发明 task 粒度。
4. Swiglu/quant sync-group 元数据：M2 必须生成 `swigluSyncGroups/dequantSum`，并能把每个 group 映射到
   `cumsumMM[rankNum - 1][localExpert]` 推导出的连续 row range。M2 可按 `overlap_mode=off` 顺序执行
   activation/quant，但 row range 和 group plan 必须已经是 M3.3/M3.4 要启用的结构。
5. GMM-Combine Tile 切分 return 映射：M2 不要求完成文章级非连续异步通信，但必须定义 GMM2 tile/sub-tile 到
   token owner rank 的分段映射、source row range、destination `offsetD` segment view 和 owner segment
   counter。M2 的 fused combine return 至少要按这个映射验证 payload correctness；如果某个 strided/multi-segment
   `TPUT` 形态需要 M3 才验证，M2 report 要把 primitive-gap gate 写清。
6. FFN 切分证据：M2.8c 必须在结构化输出中记录第 5.2.1 节的五层切分事实，包括
   `ffn_partition_model=rank_core_group_tile_l1l0`、`group_is_sync_boundary=true`、
   `gmm_tile_is_aic_work_unit=true`、各 AIV stage worker 数和 `aiv_data_parallel_deferred_to_m3`。这些字段用于
   防止把 single fused launch 误读为全阶段 payload 多核，也给 M3 timeline 提供基线。

主路径只按 MegaMoE 数据流拆子目标，不按参考源文件数量拆任务：

| 子目标 | 列入理由 | 不列入的相邻内容 |
| --- | --- | --- |
| `RoutingMetadata` | expert routing 的 row order、`expandedRowIdx`、expert count/cumsum 是 dispatch、GMM 和 restore 的共同 offset contract；没有它无法判断后续 payload 是否写到正确 token/expert 位置 | 不单独做 one-core/multi-core sort 形态；这些是实现方式，验收只看 metadata contract |
| `RoutePackQuantLocal` | 输入侧必须在一次主路径中完成 route、pack、dynamic quant，并直接落入 peer-visible dispatch payload；这是 Dispatch-GMM overlap 的数据入口 | 不做 fixed scale/offset quant、不做 non-quant gather、不做独立 expanded FP payload 再二次转换 |
| `GatherDispatchToGmm1Input` | expert owner 前同步后远端读，直接形成 local expert-major contiguous GMM1 input；这是 GMM1 能按 expert row range 消费的前提 | 不做先远端写到临时 dispatch buffer、再二次 reorder 的主路径 |
| `RunActivationAndQuant` | GMM1 后的 scale dequant、SwiGLU、dynamic requant 和 per-token scale2 是 A8W8 两段 GMM 之间的精度边界 | 不把 activation/requant 延后成全量 barrier；debug mirror 不能成为主路径 |
| `RunGmm2EpilogueAndReturn` | GMM2 epilogue 后按 owner segment 直接 remote return，是 GMM-Combine overlap 的核心边界 | 不做先写全量 `gmm2Out`、再单独 combine copy 的主路径 |
| `RestoreOutput` | token owner 最终按 `expandedRowIdx + probs` 做 weighted restore，验证 return payload 是否能被最终输出消费 | 不把 final output 改成 int8，不引入 bias 或额外精度路线 |

small shape 只作为 correctness smoke case，不作为独立性能子目标。理由是 small shape 下启动、前同步和首个 expert
transfer 开销占比过高，融合收益不稳定；当前项目只要求它走同一主路径语义并产出正确 `expandedRowIdx`、count/cumsum、
int8 payload 和 per-token scale，不要求实现额外 full-load fast path。

PTO 处理一个 GMM2 tile 跨多个 token owner rank 的规则：

1. GMM2 output 仍保持 localExpert-major，并尽量让同一 localExpert 内部按 tokenOwnerRank segment 连续排列。
   一个 GMM2 tile 的 row interval 可能跨多个 tokenOwnerRank segment，但不能丢失
   `tokenPerExpertMatrix + preSumBeforeRank` 推导出的 owner 边界。
2. `RunGmm2EpilogueAndReturn` 先构造 `ReturnSegmentPlan`：用 tile/sub-tile row range 去 intersect 每个
   tokenOwnerRank 的 row range，得到若干 `OwnerSegment{tokenOwnerRank, srcRowBegin, rowCount, dstOffsetD}`。
   相邻且目标 rank、source rows、destination rows 都连续的 segment 必须 coalesce，避免退化成逐 row 小包。
3. 每个 owner segment 直接形成 PTO `GlobalTensor Shape/Stride` view：local source view 指向 GMM2 epilogue/cast
   后的当前 segment，remote destination view 指向 token owner rank 的 `offsetD` segment，然后在
   `RunGmm2EpilogueAndReturn` 主流程里直接调用 `TPUT`。如果 PTO A3 支持远端 strided view，就用
   `GlobalTensor Shape/Stride + TPUT`；如果不能把多个离散目的地表达成一次 strided view，就拆成多条 `TPUT`，
   不能引入 AscendC scatter fallback。
4. segment 发完后按 owner segment 计数或 `TNOTIFY` 发布 ready。M2 可顺序执行这些 `TPUT` 验证 correctness；
   M3.8 再验证 `TPUT_ASYNC`、event wait/test 或等价 PTO async 路径能否让 sub-tile return 和 GMM2 overlap。
5. 如果某个 shape 下 segment 数过多，优先调整 return sub-tile 粒度或 GMM2 M tile 边界去贴近 owner segment；
   不能为了性能把多个 owner 混写到同一远端连续地址，也不能先写全量 `gmm2Out` 再做单独 combine copy。

M2 仍可以按 `overlap_mode=off` 或 BSP wait 做 correctness，但它的 ABI、workspace、peer window 布局、
soft-sync ledger 和 tile-split return map 必须已经是 M3 overlap 要用的布局。M3 只允许改调度方式和 ready
granularity，不能再新增
“先 expanded FP payload、后二次 GMM1-input reorder”或“先 gmm2Out、后独立 combine copy”的主路径，也不能重做
Dispatch-GMM task id 或 GMM-Combine owner segment 语义。

### 5.4 rank 术语与控制矩阵维度

后续实现禁止用裸 `srcRank/dstRank` 描述 MoE 控制矩阵维度，因为 dispatch 和 combine 的源/目的方向相反。
统一使用以下术语：

```text
tokenOwnerRank:
  原始 input token 所在 rank，也是 final output 的 owner。
  Dispatch 阶段它是数据生产者；Combine 阶段它是 return payload/offsetD 的接收者。

expertOwnerRank:
  global expert 所在 rank。
  Dispatch 阶段它是远端读的发起者/消费者；Combine 阶段它是 GMM 输出的生产者。

localExpert:
  expertOwnerRank 内部的 expert index。
```

控制矩阵统一维度：

```text
tokenPerExpertMatrix[tokenOwnerRank][expertOwnerRank][localExpert]
  tokenOwnerRank 有多少 token 要发给 expertOwnerRank.localExpert。
  语义维度是 rankNum * rankNum * expertPerRank；A3 PTO count-row TPUT 的物理 layout 按
  AlignUp(rankNum * expertPerRank, 16) 个 int32 作为 tokenOwner row stride，padding 元素必须保持 0。

cumsumMM[tokenOwnerRankPrefix][localExpert]
  对当前 expertOwnerRank，沿 tokenOwnerRank 维度做 cumulative sum。
  cumsumMM[rankNum - 1][localExpert] 是该 localExpert 的总 M。

preSumBeforeRank[tokenOwnerRank][localExpert]
  对当前 expertOwnerRank，某 tokenOwnerRank 的 expanded/return buffer 中，
  当前 global expert 之前的 token prefix。
  Dispatch remote read 用它定位 tokenOwnerRank peer window 中当前 expert 的起始 row；
  Combine remote write 用它定位 tokenOwnerRank offsetD 中当前 expert 的起始 row。
```

这三个对象必须共同实现，不能只实现其中一个。尤其 `preSumBeforeRank` 不是 source-rank cumsum，也不是 local
expert 内部 row offset；它是按 token owner 的全局 expert 顺序 prefix。

## 6. 数据流

端到端逻辑：

```text
input x[M, hidden]
expertId[M, topK]
probs[M, topK]
xActiveMask[M]            (optional, bool; absent => 全部 active)
  |
  v
RouteLocalTokens
  -> 若 xActiveMask 存在：inactive token 的 expertId 改写为越界 invalid expert，使其在路由阶段被 drop
  -> tokenPerExpertMatrix[tokenOwnerRank, expertOwnerRank, localExpert]
  -> expandedRowIdx[M * topK]
  -> dispatchOffset[M * topK]
  |
  v
RoutePackQuantLocal
  -> local peerWindow.dispatchPayloadByGlobalExpert[globalExpert][row][int8 hidden]
  -> local peerWindow.dispatchScaleByGlobalExpert[globalExpert][row][float scale]
  -> peer-visible tokenPerExpertMatrix[tokenOwnerRank, expertOwnerRank, localExpert]
  |
  v
PublishCounts + WaitCounts + BuildCumsumAndPreSumBeforeRank
  -> workspace.cumsumMM[tokenOwnerRank cumulative][localExpert]
  -> workspace.preSumBeforeRank[tokenOwnerRank][localExpert]
  |
  v
GatherDispatchToGmm1Input
  -> TGET from tokenOwnerRank peerWindow.dispatchPayloadByGlobalExpert[globalExpert]
  -> workspace.gmm1InputInt8/gmA[localExpert][contiguous row][hidden]
  -> workspace.routingPerTokenScale[localExpert][contiguous row]
  |
  v
GMM1
  -> workspace.gmm1AccInt32[localExpert][row][intermediate]
  |
  v
GMM1 epilogue + SwiGLU + Quant
  -> workspace.gmm2InputInt8[localExpert][row][intermediate / 2]
  -> workspace.gmm2PerTokenScale[localExpert][row]
  |
  v
GMM2
  -> workspace.gmm2AccInt32[localExpert][row][hidden]
  |
  v
RunGmm2EpilogueAndReturn
  -> scale2 dequant + per-token scale2 + output dtype cast
  -> tokenOwnerRank peerWindow.returnPayload/offsetD[row][hidden]
  |
  v
RestoreOutput
  -> output c[M, hidden]
```

这张图只描述数据依赖顺序，不表示主路径必须按全量 stage 串行执行。MegaMoE 的实现目标是在同一套数据流和
offset 语义下，让已经 ready 的 expert group、sync group 或 return segment 尽早被下游 stage 消费。

关于 dispatch 排序：本项目**不实现** `ffn.md` init_routing 的 multi-core merge sort（VBS/VMS/SortOut）。token 的
expert-major 有序性由 `RoutePackQuantLocal` 按 `globalExpert -> row` offset 直接 pack，再由
`GatherDispatchToGmm1Input` 按 `cumsumMM/preSumBeforeRank` 远端读 gather 完成——这等价于 `ffn.md`"把通信后重排
折叠进 gather 地址映射"的结论，offset-table 已经承担了排序职责，不需要独立 sort 子系统。因此 M3.2 的 dispatch
多 AIV 只覆盖 route/count/pack/gather 的 worker 分摊，**不包含**排序；任何"补 sort 子系统"的要求都属于
误读，需先走 design-bug / 用户确认门禁。

### 6.1 MegaMoE overlap 执行视图

M1/M2 可以用 `overlap_mode=off` 做 correctness，但数据流必须已经按下面的 producer/consumer 边建模；M3 只验证或
启用这些边的 async 执行能力。

本视图按 `ffn.md` 的分层口径阅读：AIV/AIC role split 是执行角色，expert group 是 ready 边界，GMM tile 是
AIC 工作分配单位，Sub-Tile/OwnerSegment 是 combine consumer 边界。不能把任意一个层级直接替代另一个层级。

```text
time --->

AIV route/dispatch:
  RoutePackQuantLocal
    -> PublishCounts / WaitCounts / BuildCumsumAndPreSumBeforeRank
    -> GatherDispatchToGmm1Input(expertGroup 0) -> dispatchGroupReady[0]
    -> GatherDispatchToGmm1Input(expertGroup 1) -> dispatchGroupReady[1]
    -> ...

AIC GMM1:
  wait dispatchGroupReady[expert 0] -> GMM1(expert 0)
  wait dispatchGroupReady[expert 1] -> GMM1(expert 1)
  ...
  when all experts covered by swigluSyncGroups[sync 0] finish GMM1
    -> gmm1SyncGroupReady[sync 0]

AIV activation:
  wait gmm1SyncGroupReady[sync 0]
    -> ActivationQuant(row range dequantSum[sync 0]..dequantSum[sync 1])
    -> activationSyncGroupReady[sync 0]
  wait gmm1SyncGroupReady[sync 1] -> ...

AIC GMM2:
  wait activationSyncGroupReady[sync 0]
    -> GMM2(row range dequantSum[sync 0]..dequantSum[sync 1])
    -> publish gmm2GroupReady[expert] for each completed expert interval
    -> publish subTileReady[aicTile] for each completed tile/sub-tile when Sub-Tile mode is enabled
  wait activationSyncGroupReady[sync 1] -> ...

AIV combine/restore:
  wait gmm2GroupReady or subTileReady
    -> RunGmm2EpilogueAndReturn(owner segments) -> combineDoneSignal[expertOwnerRank]
    -> RestoreOutput waits required topK return slots, then writes final output chunk
```

并发视图的约束：

- Dispatch/GMM1 overlap 的最小消费粒度是 expert group ready；`GatherDispatchToGmm1Input` 写完某 group 的
  contiguous rows 后，`GMM1` 可以消费该 group，不等所有 expert 全量 dispatch 完成。
- GMM1/Activation/GMM2 overlap 的最小同步语义由 `IsSyncTask/dequantSum` 和 `swigluSyncGroups` 给出；
  `ActivationQuant` 和 `GMM2` 只消费对应 row range，不等全量 GMM1 或全量 activation。
- GMM2/Combine overlap 的语义由 `ReturnSegmentPlan`、`gmm2GroupReady` 和 `subTileReady` 给出；
  `RunGmm2EpilogueAndReturn` 对 ready 的 owner segment 直接 `TPUT`，不等整个 expert 或全量 `gmm2Out`。
- `RestoreOutput` 可以分块消费已完成的 return segment，但某个 output token 的最终写出不能越过缺失的 topK return slot。
- M2 必须已经 dump `scoreboardTaskMap/producerStatus/scoreboardMinStatus`、dependency domain、
  `swigluSyncGroups/dequantSum` 和
  `subTileReturnPlan/subTileOwnerSegments`；M3.7/M3.8 只验证 scoreboard 和 Sub-Tile/stride remote write 能否异步执行，
  不能重新定义 row order、offset 或 segment。
- timeline 里至少应能区分 `dispatch`、`GMM1`、`ActivationQuant`、`GMM2`、`RunGmm2EpilogueAndReturn`、
  `RestoreOutput` 的区间；若某个 overlap 点 blocked，task report 只写 blocked 的 primitive 或调度原因摘要。

同步 overlap 调度粒度：

| Edge | baseline ready 粒度 | 高性能方向 | consumer 规则 |
| --- | --- | --- | --- |
| Dispatch -> GMM1 | local expert / expert group；该 expert 的所有 token-owner rows 收齐并写入 contiguous `gmm1InputInt8/gmA` | 可选细化只能使用 M2.2c 已预留的 source segment 或 tile row range task，不允许 M3 重新定义 row order | `GMM1` 可以启动该 expert，不等其他 expert 的 dispatch |
| GMM1 -> Swiglu/quant | `swigluSyncGroups` / `dequantSum` 描述的连续 row range；一个 sync group 可以覆盖多个 expert，例如 `{8,4,2,1,1}` | 组内 row/tile 可由多个 AIV 分摊，但同步事件仍按 sync group 发布 | `ActivationQuant` 等当前 sync group 覆盖的 expert rows 都完成 GMM1，不要求单 expert 逐个同步，也不等全量 GMM1 |
| Swiglu/quant -> GMM2 | 同一个 `swigluSyncGroups` / `dequantSum` row range | GMM2 内部可按 GMM block 调度，但不能改变 sync group row range 语义 | `GMM2` 消费已完成 activation/quant 的 row range，不等全量 activation |
| GMM2 -> Combine | baseline 是 expert group；该 expert 的 GMM2 output ready 后可做连续段 return | M3.8 目标是 tile/sub-tile；一个 GMM2 tile 可拆成多个 `OwnerSegment` 远端写；ready queue 仅作为 AIC/AIV 1:1 或静态 N:1 handoff 的备选 | `RunGmm2EpilogueAndReturn` baseline 不等所有 expert；Sub-Tile 模式不等整个 expert/GMM2，只等对应 AIC tile 或 owner segment |

因此不能把主路径理解为“Dispatch -> GMM1 -> Swiglu 都按 expert 粒度，只有 Combine 是 tile 粒度”。本项目的同步
口径是：Dispatch/GMM1 先以 expert group 为 baseline，Swiglu/quant 以多 expert sync group 做粗细结合，
GMM2 跟随 Swiglu sync group row range 或 GMM block 调度，Combine baseline 为 expert group、高性能目标为
tile/sub-tile owner segment。

运行时 ready 对象之间存在一次投影关系：`activationSyncGroupReady[syncIdx]` 放行的是一段连续 activation row
range，GMM2 可以按这段 row range 启动；但 Combine 需要的不是 sync group 本身，而是 GMM2 完成后落到
expert interval、tile 或 sub-tile 的输出 ready。实现必须用 `cumsumMM/expertTokenNums` 把 GMM2 的完成 row range
投影回 `gmm2GroupReady[expert]`，再用 `ReturnSegmentPlan/OwnerSegment` 投影到 token owner 的 `offsetD`
segment。这样 Swiglu 可以按 `{8,4,2,1,1}` 这类多 expert 分组节省同步事件，而 Combine 仍能按 expert、tile 或
sub-tile 尽早远端写回，不被 Swiglu group 边界强行拉齐。

Ready queue 暂不进入当前主路径，只作为本地 AIC/AIV 精细依赖的备选形态。若后续普通 signal/counter 无法证明
tile/sub-tile 级 overlap，或者每个 AIV 轮询过多 AIC flag 成为瓶颈，可以按 issue/DCL 流程评估如下队列模型：

```text
AIC producer p:
  compute/store tile payload
  pipe_barrier + DDR visibility
  readyQueue[p].data[tail] = tileOrDependencyId
  readyQueue[p].count = tail + 1

AIV consumer c:
  statically owns queues {c, c + aivConsumerNum, ...}
  TTEST(readyQueue[p].count >= head + 1)
  acquire fence on first hit in a drain pass
  read tileOrDependencyId
  consume exactly the payload described by that id
  if no progress, TWAIT on the next non-exhausted queue
```

这个模型可候选用于 `GMM2 -> RunGmm2EpilogueAndReturn` 的 tile/sub-tile handoff，也可候选用于
`GMM1 -> ActivationQuant` 在 sync group 内的 tile 级生产消费；但它不能替代 `swigluSyncGroups/dequantSum` 的
粗细同步语义，也不能替代跨 rank 的 Dispatch-GMM scoreboard。`swigluSyncGroups` 决定哪些 row range 可以作为一组
进入 Activation/GMM2；ready queue 如果启用，也只是在这个合法 row range 内，把 AIC/AIV 的实际 tile 交接做得更精确。
若 `aicProducerNum == aivConsumerNum`，queue ownership 可以退化为 1:1；若 AIV 更少，一个 AIV 静态轮询多条队列。
当前任务不要求实现该队列；若启用，禁止使用抢占式全局队列或跨 consumer 原子仲裁作为首版设计，因为它会引入新的
同步不确定性。

### 6.2 同步/overlap 总体契约

本节定义数据流里的同步对象、发布顺序、粒度、blocked 处理和验收证据。MegaMoE 同步约束先于具体实现拆分：

1. Dispatch-GMM 必须基于前同步后的远端读，让 GMM 看到连续 expert rows；不能退回后同步远端写路径。
2. GMM-Combine 必须基于计算后的远端写；文章级 overlap 要求 Sub-Tile/stride 切分，让每个 AIV 只通信对应 AIC
   已完成的那部分数据。
3. 中间阶段避免多次跨卡同步。跨 rank 同步只应出现在 dispatch communication 开始前和 combine communication
   结束后；其他依赖通过卡内 C/V signal、soft status 或 PTO readiness 表达。
4. Dispatch-GMM 的多生产者/多消费者等待不能靠全核 BSP 拉齐。M2 必须先落 producer status、task id 映射、
   producer publish 时机、consumer dependency domain 和 `scoreboardMinStatus` 账本；M3.7 在同一账本上验证或启用
   async scoreboard：producer 在 segment payload/scale 可见后写 GM status，AIV Ctrl 按 expert 或 GMM tile 依赖域
   计算聚合状态，worker AIV/AIC 只轮询自己要消费的聚合状态。
5. GMM-Combine 的 return path 在 M2 必须先按 tile/sub-tile 规划 owner segment、source row range 和
   destination `offsetD` view；M3.8 再验证这些 segment 能否用 strided `TPUT` 或多段 `TPUT` 异步执行。
6. `TGET/TPUT` 可作为本项目跨 rank payload 的 public PTO 表达；普通连续段必须在 M1/M2 使用 direct PTO
   调用落地。Sub-Tile 非连续 combine 是否能用 `GlobalTensor Shape/Stride + TPUT` 或多段 direct `TPUT`
   完整表达原文章的 stride remote copy，需要作为 M3.8 独立验收项；若必须用 AscendC `DataCopy` 才能做到，
   则 PTO 版该点 blocked。

同步验收分层：

- M1/M2 的 correctness 可以在最终 stage graph 内用 `overlap_mode=off`、BSP wait 或 ledger-only 执行，但 signal、
  counter、row range 和 producer/consumer 边必须与 `6.1` 的运行视图一致。
- M3.0-M3.6 验证 fused overlap skeleton：`dispatchGroupReady`、`gmm1SyncGroupReady`、
  `activationSyncGroupReady`、`gmm2GroupReady` 能驱动 single-kernel MPMD 或等价 device-side producer-consumer 调度。
- M3.7/M3.8 在 M2 已验收的 `scoreboardTaskMap/producerStatus/scoreboardMinStatus` dependency domain 和
  `subTileReturnPlan/subTileOwnerSegments/subTileReady` 上验证 async scoreboard 与 Sub-Tile/stride return 能力；
  不能重新定义 row order、offset、task id 或 owner segment。
- M3.9 用 timeline 证明 `dispatch`、`GMM1`、`ActivationQuant`、`GMM2`、`RunGmm2EpilogueAndReturn`、
  `RestoreOutput` 的区间和等待空泡；如果某个 overlap 点 blocked，task report 只记录 primitive-gap 或调度原因摘要。

### 6.3 Dispatch/Combine 维度约定

dispatch/combine 维度约定：

- `globalExpert = expertId[token, k]`
- `expertOwnerRank = globalExpert / expertPerRank`
- `localExpert = globalExpert % expertPerRank`
- token owner 的 peer-visible dispatch payload 以 `globalExpert -> row` 或等价可计算 offset 为主序；expert owner
  远端读后直接写入 `gmm1InputInt8/gmA`，local workspace 以 `localExpert -> tokenOwnerRank -> row` 为主序，
  便于 local expert grouped GMM 连续消费。
- `tokenPerExpert` 的语义必须覆盖 `(tokenOwnerRank, expertOwnerRank, localExpert)`。
- `cumsumMM[tokenOwnerRankPrefix, localExpert]` 是对 token owner rank 维度做 cumulative sum；`cumsumMM[EP-1, localExpert]`
  是本 local expert 的总 row 数，也是 GMM group 的 `currentM` 来源。
- `preSumBeforeRank[tokenOwnerRank, localExpert]` 是当前 expert owner 在该 token owner 的 expanded/return buffer 中的
  row offset；`GatherDispatchToGmm1Input`、`RunGmm2EpilogueAndReturn` 和 final restore 都依赖它。
- combine return 以 `ownerToken -> topK slot -> hidden chunk` 还原，便于按 `probs` 做加权。

### 6.4 容量受限 drop 语义（对齐参考实现）

drop 模型直接对齐参考 `dispatch_ffn_combine` 主 kernel（`op_kernel/dispatch_ffn_combine_kernel.hpp` 的 GMM1/GMM2/
dispatch gather/combine return 截断，以及 `op_kernel/unpermute/moe_token_unpermute.h` 的 restore）。要点如下，M2/M3
实现和 host golden 必须一致：

1. **`maxOutputSize` 是本 rank 合计预算**：它是本 rank 所有 local expert 输出 row 的总上限（= workspace/window 预分配
   行数），不是 per-expert capacity。本项目继续用 compact `cumsumMM` 布局，**不**切换到 per-expert capacity-padded 布局。
2. **位置序前缀和截断**（与参考一致）：按 local expert 顺序累计 `preCurrentmSum`，对每个 expert 的 `currentM`：
   - `preCurrentmSum >= maxOutputSize` → `currentM = 0`（整段丢弃）；
   - `preCurrentmSum + currentM > maxOutputSize` → `currentM = maxOutputSize - preCurrentmSum`（边界 expert 部分丢弃）；
   - 否则全保留；随后 `preCurrentmSum += currentM`。被丢的是前缀顺序靠后的 token，不按 router score 选择。
3. **全 stage 一致截断**：同一 `maxOutputSize` 公式必须统一应用在所有消费 `cumsumMM` 的 stage——`GatherDispatchToGmm1Input`
   不远端读超额行（`rowStart >= maxOutputSize` 跳过，`rowStart + rows > maxOutputSize` 截断）、`Gmm1/Gmm2` 不计算被丢行、
   `swigluSyncGroups/dequantSum` 同步截断、`RunGmm2EpilogueAndReturn` 不回写被丢行。任何 stage 漏截断都会让 row range 对不齐。
4. **drop 标记走 `expandedRowIdx` sentinel**：被丢的 `(token, slot)` 在 `expandedRowIdx` 写一个 `>= num_out_tokens`
   的 sentinel（`num_out_tokens` = 有效 return row 上界，即 `maxOutputSize`）。不新增独立 drop mask buffer。
5. **restore 不重新归一**：`RestoreOutput` 对每个 output token 的 topK slot，若 `expandedRowIdx < num_out_tokens` 则按
   原始 `probs[slot]` 加权累加，否则跳过该 slot（贡献 0）。**不**把剩余 slot 的 `probs` 重新归一到和为 1；首个 slot 若被丢，
   累加器初始化为 0。这与参考 `moe_token_unpermute` 的 `cal_token_idx < num_out_tokens` 跳过语义逐字一致。
6. **golden 同源**：host reference 必须用同一套截断 + sentinel + no-renorm 规则生成 expected output，否则 drop case
   无法验收。
7. **验收**：至少补一个 over-capacity skewed case（某 expert token 数 > 其在预算内可占额度），证明 drop 行被一致截断、
   sentinel 正确、final output 与 golden 在 tolerance 内一致，且不死锁、不越界。balanced/smoke case 仍走相同代码路径但不触发截断。
8. **`xActiveMask` 可选输入（drop 前端）**：`xActiveMask[M]`（bool）是可选输入，语义上只是 drop 的前端触发器，复用上面
   同一条 drop 路径，不新增独立分支：
   - `RouteLocalTokens` 在统计 count 前，对每个 inactive token（`xActiveMask[token] == false`）把其全部 topK
     `expertId` 改写为越界 invalid expert（如 `rankNum * expertPerRank`），使这些 `(token, slot)` 不进入任何 expert
     的 count/cumsum，自然在路由阶段被 drop，其 `expandedRowIdx` 写 sentinel。
   - mask 不传（host 传空指针）时等价于全部 active，行为与无 mask 完全一致。
   - host golden 必须用同一 mask 语义：inactive token 不参与 routing/GMM，对应 slot 在 restore 跳过且不重新归一。
   - 它与第 2 项的容量截断是两个独立的 drop 来源（mask 在路由前、容量在预算边界），但都收敛到同一 sentinel +
     no-renorm restore 口径。验收可在 over-capacity case 之外，另补一个含 inactive token 的 mask case。

## 7. Workspace 与 window layout

所有 rank 的 HCCL window layout 必须一致。只要字段会被 peer rank 访问，就必须放在 peer-visible window 内；只被本 rank
访问的中间 buffer 放普通 GM workspace。

当前 peer-visible window layout 契约：

```text
PeerWindow
  header
    magic/version
    rank/ep/topK/expertPerRank
    dtype_in/dtype_out
    dispatchPayloadRowBytes
    returnPayloadRowBytes
  routingMetadata
    int32 tokenPerExpert[rankNum][AlignUp(rankNum * expertPerRank, 16)]
      valid view remains tokenPerExpert[tokenOwnerRank][expertOwnerRank][localExpert]
    cache-line aligned countReadySignal[rankNum]
  dispatchPayload
    int8 hidden payload produced by this token owner rank and read remotely by expert owner ranks
    rowBytes = explicit int8 hidden row stride
  dispatchScale
    float routingPerTokenScale rows aligned with dispatchPayload rows
  returnPayload
    output dtype payload returned to token owner rank; this region is offsetD
  combineDoneSignal
    cache-line aligned signal[rankNum]
  returnSegmentCounters
    owner segment counters for Sub-Tile or multi-segment combine return
  debugCounters
    per-stage counters for producer/consumer/timeout/scoreboard/subTile evidence
  timeline
    optional timestamp ring or fixed-size per-stage timestamp buffer
```

当前 local workspace layout 契约：

```text
LocalWorkspace
  routing
    tokenPerExpertMatrix
    blockTokenPerExpert
    blockPrefixPerExpert
    expandedRowIdx
    dispatchOffset
    cumsumMM
    preSumBeforeRank
    expertTokenNums
    tokenOwnerRankOffsets
  dispatch
    dispatchedA                // M1 mock/debug mirror only after M2; M2 main path writes gmm1InputInt8 directly
    dispatchedScale            // M1 mock/debug mirror only after M2
  a3Int8
    gmm1InputInt8
    gmm1WeightInt8
    scale1Uint64
    gmm1AccInt32
    gmm1Out
    swigluOut
    gmm2InputInt8
    gmm2PerTokenScale
    gmm2WeightInt8
    scale2Uint64
    gmm2AccInt32
    gmm2Out                    // optional debug mirror; M2 main combine path writes peerWindow.returnPayload/offsetD
  control
    softFlagBase
    readyCounters
    dispatchGroupReady
    gmm1SyncGroupReady
    activationSyncGroupReady
    gmm2GroupReady
    stageStatus
    swigluSyncGroups
    dequantSum
    scoreboardTaskMap          // M2 定义语义账本；M3.7 切换 async scoreboard 调度
    producerStatus
    scoreboardMinStatus
    workerWaitCounters
    scoreboardTimeoutCounters
    subTileReturnPlan          // M2 定义 tile/sub-tile 到 owner segment 的映射；M3.8 验证执行能力
    subTileOwnerSegments
    subTileReady
    timelineScratch
```

offset 规则：

1. 每个 field 由 `WorkspaceLayout` / `PeerWindowLayout` 统一计算 offset。
2. device stage 通过 typed view 获取 `GlobalTensor` 或 raw metadata pointer；PTO primitive 在 stage 主流程直接调用。
3. remote pointer 计算只能做 `windowsIn[peerRank] + localOffset` 映射，不封装 copy、wait、notify。
4. signal 与高频 counter 独占 cache line，避免 payload 写入和 control polling 互相污染。
5. `tokenPerExpertMatrix`、`cumsumMM`、`preSumBeforeRank` 的维度和 layout 必须在 host reference 与 device view 中
   完全一致；这三者是 combine offset 和 GMM group row range 的共同契约。
6. layout 里必须显式记录 `dispatchPayloadRowBytes`，不能让不同 stage 临时 reinterpret row stride。
7. `dispatchedA/dispatchedScale/gmm2Out` 是 mock/debug mirror。M2 之后主路径分别写 `gmm1InputInt8` 和
   `peerWindow.returnPayload/offsetD`；task report 摘要必须标注 debug mirror 不参与 accepted 主路径。
8. `scoreboardTaskMap/producerStatus/scoreboardMinStatus`、scoreboard dependency domain 和
   `subTileReturnPlan/subTileOwnerSegments/subTileReady`
   是 M2/M3 overlap 共享账本，不能在 M3 重新定义 offset、row range 或 owner segment 语义。
9. `scale1Uint64/scale2Uint64/routingPerTokenScale/gmm2PerTokenScale` 必须有 typed view 或 layout dump；控制台输出
   承载 bit pattern、checksum 或 tolerance 证据，M2 report 只写结论摘要。

## 8. PTO primitive 映射

本章只定义本项目 stage 主流程允许直接调用的 PTO primitive。`WorkspaceLayout` / `PeerWindowLayout` 可以计算
typed view 和 offset，但不能封装 copy、wait、notify、matmul 或 quant 操作。

| Stage / 对象 | PTO 表达 | 直接调用位置与约束 |
| --- | --- | --- |
| `RouteLocalTokens` | `GlobalTensor + Tile<TileType::Vec> + TLOAD/TSTORE` plus Vec compare/pack primitives | 读取 `expertId/probs`、写 `expandedRowIdx/dispatchOffset`；metadata 计数仍是 raw GM typed view |
| `RoutePackQuantLocal` | `TLOAD/TSTORE`, `TABS/TROWMAX/TMUL`, `TQUANT<INT8_SYM/INT8_ASYM>` | 直接写 peer-visible int8 dispatch payload 和 routing per-token scale；不能先写 expanded FP payload 再二次转换 |
| `PublishCounts` / `WaitCounts` | `TSTORE/TPUT` for count row, `TNOTIFY/TWAIT/TTEST` for readiness | count row publish/wait 不用 host barrier 替代；如果 PTO ordering 不足，标记 primitive-gap |
| `BuildCumsumAndPreSumBeforeRank` | raw GM typed view + local scalar/vector loops | 控制面 prefix/cursor/offset 不强制 Tile 化；输出必须和 host reference 同 layout |
| `GatherDispatchToGmm1Input` | `TWAIT/TTEST + TGET + TSTORE` | 从 token owner peer window 远端读，直接写 `gmm1InputInt8/gmA` 和 routing scale；不能经过二次 reorder workspace |
| `Gmm1` / `Gmm2` | `TMATMUL` / `TMATMUL_ACC` | A3 `int8 x int8 -> int32`；`TMATMUL` 必须直接出现在 GMM stage 主流程 |
| `ActivationQuant` | `TLOAD/TCVT/TMUL/TADD` etc., `TABS/TROWMAX/TMUL`, `TQUANT<INT8_*>`, `TSTORE` | 直接完成 scale1 dequant、routing scale、SwiGLU、requant、`gmm2PerTokenScale` 写回；当前主路径无 bias |
| `RunGmm2EpilogueAndReturn` | `TLOAD/TCVT/TMUL/TSTORE + TPUT + TNOTIFY` | GMM2 epilogue 与 remote return 同 stage；`TPUT` 直接写 token owner `offsetD`，不能先全量写 `gmm2Out` 再 copy |
| Sub-Tile/strided return（功能） | 同步 `GlobalTensor Shape/Stride + TPUT` or multiple direct `TPUT` calls | M2 固定 segment plan；同步 `TPUT` 已确认支持 strided/多段非连续远端写（`TPut.hpp` 5 维独立 stride + `AtomicAdd`），M3N.11 功能形态可 enabled accepted |
| Sub-Tile return 异步重叠（性能） | `TPUT_ASYNC` (SDMA) | A3 `TPUT_ASYNC` 只接受 flat-contiguous-1D（`TPutAsyncCommonDetail.hpp` 硬性 assert），无法表达 strided/多段，故"sub-tile return 与 GMM2 异步 overlap"按 primitive-gap blocked，不引入 AscendC scatter fallback |
| AIC↔AIV C/V handoff | `pto::Event<SrcOp, DstOp>::Init<CrossCoreId>() / Wait<CrossCoreId>()`（`ffts_cross_core_sync`/`wait_flag_dev`）或 `TSYNC_CVID`；卡内全核栅栏用 `SYNCALL<SyncCoreType::Mix>` | per-group/per-tile readiness 用 per-`CrossCoreId` 握手；`SYNCALL<Mix>` 仅 M2.8c overlap-off stage 边界。粒度层级与约束见第 10 章 |
| `RestoreOutput` | `TLOAD/TMUL/TADD/TSTORE` | 按 `expandedRowIdx + probs` 做 weighted restore；等待缺失 topK return slot，不能输出 int8 |
| scoreboard/timeline/counter | raw GM typed view + explicit publish order | `scoreboardTaskMap/producerStatus/scoreboardMinStatus/subTileReady/timeline` 是控制面账本；scoreboard 必须记录 producer publish 时机和 consumer dependency domain，不通过 helper 隐藏 readiness 或 remote copy |

当前 M2 review 口径：

- `RoutePackQuantLocal` 的主 payload quant 已要求并允许用 PTO Vec `TCVT/TABS/TROWMAX/TQUANT/TSTORE`
  表达；若当前代码已经直接使用这些 primitive，不能再把 M2.2a 归为“标量 route/quant”缺口。
- 仍未完成的 PTO Vec data-plane 缺口集中在 GMM1 epilogue、SwiGLU 后 requant、GMM2 epilogue/cast：
  scalar GM loop 可以作为 correctness preflight，但不能关闭 M2 data-plane PTO 化验收。
- `RunGmm2EpilogueAndReturn` 可以在同 stage 写 `gmm2Out` debug mirror 或 per-segment staging；验收关注点是
  return payload 的 source 不能是全量 `gmm2Out` 二次 combine copy，结构化输出必须能证明 `TPUT` 消费的是
  per-segment epilogue payload 或等价 direct tile。
- `swigluSyncGroups/dequantSum/SwigluGroup` 和 GMM2 task preview 属于 M2.5 metadata handoff。若代码已经生成并
  host 比对这些字段，不能继续把 ISSUE-06 当作未落地；剩余 activation 数值 PTO 化缺口归入上面的 Vec issue。

PTO 化判断：

```text
payload tensor 或批量向量计算
  -> PTO Tile / GlobalTensor / primitive

跨 rank payload
  -> TGET / TPUT

跨 rank readiness
  -> TNOTIFY / TWAIT / TTEST

标量调度状态、offset、cursor、flag
  -> raw GM metadata + 明确发布顺序
```

## 9. A3 int8 stage set 边界

`2` 是 dtype、数值计算顺序和 tolerance 的单一事实源；本章只定义当前
A3 int8 numeric stage set 的文件/函数边界。这里的 `PrecisionBackend` 不是 runtime 多态框架，也不是 helper
封装层；它只是让 protocol core 用固定 stage 名称调度 GMM/activation/return 相关代码。

PTO 指令必须直接写在对应 stage 主流程里：`RunGmm1/RunGmm2` 内直接调用 `TMATMUL/TMATMUL_ACC`，
`RunActivationAndQuant` 内直接调用 `TQUANT` 或等价 PTO Vec primitive 序列，`RunGmm2EpilogueAndReturn`
内直接调用 `TPUT`。不能再引入隐藏 `TLOAD/TSTORE/TGET/TPUT/TMATMUL/TQUANT` 的二次封装。

M2 active runtime 迁移后的文件口径：

- 当前 active host 入口是 `host/main.cpp`。
- 当前 active AIV/comm/activation/return kernel 入口是 `kernel/moe_dispatch_combine_a8w8_kernel.cpp`。
- 当前 active AIC GMM kernel 入口是 `kernel/moe_dispatch_combine_a8w8_gmm_kernel.cpp`。
- `kernel/a3_int8_backend.hpp`、`kernel/protocol_core.hpp` 和 `kernel/control_metadata.hpp` 仍是 stage 边界、
  typed view 和未来拆分目标的设计文件，但 review 当前实现时必须以 active runtime 文件为准。

当前 stage 接口：

```cpp
struct PrecisionBackend {
    void RunGmm1(StageContext& ctx, ExpertSegment segment);
    void RunActivationAndQuant(StageContext& ctx, ExpertSegment segment);
    void RunGmm2(StageContext& ctx, ExpertSegment segment);
    void RunGmm2EpilogueAndReturn(StageContext& ctx, ExpertSegment segment, OwnerSegment owner);
};
```

接口边界：

- `RoutePackQuantLocal` 和 `GatherDispatchToGmm1Input` 属于 protocol/data-path stage，不在 `PrecisionBackend` 中再做
  二次 GMM1-input 转换。M2 起 GMM1 输入已经是最终 `gmm1InputInt8/gmA` layout。
- `RunGmm1` 只对本 rank local experts 的 contiguous segment 工作。
- `RunActivationAndQuant` 消费 GMM1 accumulator row range，生成 `gmm2InputInt8/gmm2PerTokenScale`。
- `RunGmm2` 只消费 activation-ready row range，输出 GMM2 accumulator。
- `RunGmm2EpilogueAndReturn` 把 GMM2 epilogue 和 remote return 写回合并，按 owner segment 直接 `TPUT` 到
  `offsetD`。
- `RestoreOutput` 不属于 `PrecisionBackend`；它是 token owner 侧的 protocol/final-output stage，按
  `expandedRowIdx + probs` 消费 return payload。

### 9.1 A3Int8Backend

当前唯一 numeric stage set：

```cpp
struct A3Int8Backend : PrecisionBackend {
    void RunGmm1(StageContext& ctx, ExpertSegment segment);
    void RunActivationAndQuant(StageContext& ctx, ExpertSegment segment);
    void RunGmm2(StageContext& ctx, ExpertSegment segment);
    void RunGmm2EpilogueAndReturn(StageContext& ctx, ExpertSegment segment, OwnerSegment owner);
};
```

边界验收：

- stage 函数内能看到直接 PTO primitive 调用位置；不能只有通用 wrapper 调用。
- `RunGmm1/RunGmm2` 的 segment 输入来自 `cumsumMM/expertTokenNums`，不能自行重排 GMM input。
- `RunActivationAndQuant` 的 ready 粒度使用 `dequantSum/swigluSyncGroups`，不能退回全量 activation barrier。
- `RunGmm2EpilogueAndReturn` 使用 `ReturnSegmentPlan/OwnerSegment`，不能先全量写 `gmm2Out` 再单独 combine copy。
- `A3Int8Backend` 是当前唯一 numeric stage set；新增其他 backend 或 runtime backend 切换必须走设计变更和用户确认门禁。

## 10. 同步对象与发布顺序

本章只维护 signal ownership、同步粒度和 payload/signal 发布顺序。同步与 overlap 的总体约束见第 6.2 节；
SwiGLU/quant 的粗细粒度分组策略见第 11 章。

同步对象：

| Signal | Producer | Consumer | 粒度 |
| --- | --- | --- | --- |
| `countReadySignal[tokenOwnerRank]` | `PublishCounts` | `GatherDispatchToGmm1Input` | token owner row |
| `dispatchGroupReady[expert]` | `GatherDispatchToGmm1Input` | `GMM1` | original expert group |
| `gmm1SyncGroupReady[syncIdx]` | `GMM1` | `ActivationQuant` | `swigluSyncGroups` / `dequantSum` row range |
| `activationSyncGroupReady[syncIdx]` | `ActivationQuant` | `GMM2` | `swigluSyncGroups` / `dequantSum` row range |
| `gmm2GroupReady[expert]` | `GMM2` | `RunGmm2EpilogueAndReturn` | original expert group |
| `scoreboardMinStatus[dependencyDomain]` | AIV Ctrl | worker AIV/AIC | Dispatch-GMM soft sync；按 local expert 或 GMM tile 聚合，不是全局 task min |
| `subTileReady[aicTile]` | `GMM2` | `RunGmm2EpilogueAndReturn` | GMM-Combine Sub-Tile |
| `combineDoneSignal[expertOwnerRank]` | `RunGmm2EpilogueAndReturn` | token owner `RestoreOutput` | expert owner return |

发布顺序必须是：

```text
payload write complete
local pipe / DDR visibility satisfied
signal write or TNOTIFY
consumer TWAIT/TTEST success
payload read
```

如果 PTO comm primitive 已经内建对应 ordering，stage 只直接调用 primitive；如果需要额外 fence，必须先进入
PTO public primitive 或明确标记该 overlap 点 blocked，不能在本项目里绕开 PTO 自建通信封装。

本项目不直接使用 `AscendC::CrossCoreWaitFlag` / `CrossCoreSetFlag`，但 AIC↔AIV 的 C/V handoff 不是
primitive-gap。PTO 已提供等价的跨核同步能力，必须按下列层级使用，不能用一个全核栅栏替代所有粒度：

| 同步范围 | PTO 原语 | 对应 `ffn.md` 机制 | 用在哪 |
| --- | --- | --- | --- |
| 卡内 AIC+AIV 全核栅栏 | `SYNCALL<SyncCoreType::Mix>` | `SyncAll<true>` 全核 + 阶段收口 | 仅 M2.8c overlap-off 的 stage 边界；M3 关键路径不得每个 stage 都用 |
| AIC↔AIV per-group/per-tile 握手 | `pto::Event<SrcOp, DstOp>::Init<CrossCoreId>() / Wait<CrossCoreId>()`（封装 `ffts_cross_core_sync` / `wait_flag_dev`），或 `TSYNC_CVID` | `CrossCoreSetFlag/WaitFlag<0x2>(flagId)` | M3.0/M3.3/M3.4 的 `GMM1↔Activation↔GMM2` sync-group 放行 |
| 跨 rank readiness | `TNOTIFY / TWAIT / TTEST` | DataAsFlag / `gm_signal_wait_until_ne` | dispatch 前 count 同步、combine 后 restore |

约束：

- FFTS cross-core event id 是 4-bit 字段，物理上只有 16 个 flag（0-15），是全平台共享的硬件上限，PTO 同步也受此限制。
  PTO `SyncAll` 已占用 `SYNC_AIC_FLAG=11`、`SYNC_AIV_FLAG=12`、`SYNC_AIC_AIV_FLAG=13`、`SYNC_AIV_ONLY_ALL=14`
  （见 `include/pto/common/type.hpp`），`TSYNC_CVID` 的 CV-comm 控制额外占用 12-15。因此当 fused kernel 同时使用
  `SYNCALL` 时，用户 per-group/per-tile 握手的 `CrossCoreId` **安全区是 0-10**。
- 物理 flag 数有限不限制 sync group 数：FFTS flag 是计数信号量（set N 次、wait 按序消费 N 次）。当逻辑 sync group 数
  超过可用物理 flag 数时，按计数信号量 + 索引折叠复用同一物理 flag，例如 `flagId = syncGroupIdx % availableFlags`
  或参考实现的 `flagId = syncGroupIdx / CROSS_CORE_FLAG_MAX_SET_COUNT(=15)` 滚动（每 15 次 set 换下一个物理 flag，
  防计数器溢出）。复用映射规则必须在 report 记录，且不能让无关 group 因共享 flag 被错误放行。
- `SYNCALL<SyncCoreType::Mix>` 是卡内 AIC+AIV 栅栏，既不是跨卡同步，也不是 per-group 放行。M2.8c 允许用它表达
  stage 边界做 overlap-off correctness；但 M3.0 起，`GMM1 -> ActivationQuant -> GMM2` 等 overlap 边必须改用
  per-`CrossCoreId` 的 `pto::Event` 握手或 `TSYNC_CVID`，否则属于"串行未 overlap"，不能声称 M3 overlap 完成。
- 只有当某个 overlap 点确实超出上述 PTO 能力时，才标记 primitive-gap blocked，不能把 AscendC flag API 藏进本项目代码。

## 11. Swiglu/quant 粗细粒度同步策略

Swiglu 是必做计算，优化点不是减少计算量，而是避免 AIV 的 Swiglu/quant 工作推迟 GMM2 和 Combine 的关键路径。
在量化路径中，Swiglu 前后还有 per-token scale、dequant、requant、cast 等 AIV 工作；如果每个 expert 都独立同步，
会消耗大量同步事件并打碎 AIV 调度；如果一次等待太多 expert，又会让前面已经 ready 的 row range 无法及时进入 GMM2。

本项目采用三层粒度：

1. GMM/AlltoAll 仍按细粒度展开，因为这是融合收益来源：Dispatch-GMM 至少按 expert group ready，GMM-Combine
   在 M3.8 进一步按 tile/sub-tile ready。
2. Swiglu/quant 按粗细结合的 sync group 展开：前段分组较大，用 AIC 正在跑后续 GMM 的窗口批量处理 AIV
   工作并节省同步事件；后段分组逐步变小，避免尾部 Swiglu/quant 阻塞已经 ready 的 GMM2/Combine。
3. sync group 内部仍按 activation tile/sub-tile 给 AIV 分工；分工只影响本地执行，不改变 group ready 事件的
   语义。也就是说，group 是同步边界，tile/sub-tile 是 AIV 工作切分边界。

M2 必须生成以下 metadata，M3 只能启用信号或调度，不能重新定义这些字段：

```text
SwigluGroup {
  syncIdx
  expertBegin        // inclusive localExpert
  expertEnd          // exclusive localExpert
  rowBegin           // flattened expert-major row begin
  rowEnd             // flattened expert-major row end
  tileBegin          // activation tile id begin, optional but reserved
  tileEnd            // activation tile id end, optional but reserved
}

swigluSyncGroups[syncIdx] = expertEnd - expertBegin
dequantSum[syncIdx] = rowBegin
dequantSum[syncIdx + 1] = rowEnd
```

`rowBegin/rowEnd` 的计算规则：

```text
expertRowStart[0] = 0
expertRowStart[e + 1] = expertRowStart[e] + cumsumMM[rankNum - 1][e]

rowBegin = expertRowStart[expertBegin]
rowEnd   = expertRowStart[expertEnd]
```

因此每个 sync group 必须覆盖一段 contiguous flattened expert-major rows。zero-token expert 仍可以出现在 group 内，
但不会增加 row count；如果一个 group 的 `rowBegin == rowEnd`，该 group 必须显式标记 empty，并且 producer/consumer
counter 不得等待一个永远不会产生 payload 的事件。

当前 host 生成 group size 使用幂指数递减策略。规则是：从 remaining experts 中取不超过当前一半的 2 的幂，
直到尾部退化到 1；同时保证 group size 之和等于 `expertPerRank`。以 `expertPerRank=16` 为例：

```text
swigluSyncGroups = {8, 4, 2, 1, 1}

group0: expert 0..7   -> one gmm1SyncGroupReady / activationSyncGroupReady event
group1: expert 8..11  -> one gmm1SyncGroupReady / activationSyncGroupReady event
group2: expert 12..13 -> one gmm1SyncGroupReady / activationSyncGroupReady event
group3: expert 14     -> one gmm1SyncGroupReady / activationSyncGroupReady event
group4: expert 15     -> one gmm1SyncGroupReady / activationSyncGroupReady event
```

对非 2 的幂或较小 expert 数，必须使用同一规则生成可解释分组；例如 smoke `expertPerRank=2` 退化为 `{1,1}`。
如果实现选择不同但等价的递减策略，必须先记录 design issue 并说明为什么不影响 sync 事件数量、row range 和
GMM2/Combine 尾部阻塞。

这不是分配 8 个 AIV，而是把 8 个 expert 的 Swiglu/quant row range 合成一个同步组。具体执行时，组内 row/tile
由 AIV grid-stride 分摊：

```text
for tileId in [tileBegin + aivId, tileEnd) step aivCount:
  rows = intersect(tileId, rowBegin..rowEnd)
  load GMM1 dequant input + routing scale
  apply routing scale, SwiGLU, dynamic quant
  store gmm2InputInt8 + gmm2PerTokenScale
```

activation tile size 必须是 launch-time 或 layout 中可见的参数，例如 `activationTileRows` 和
`activationTileCols/intermediateChunk`。M2 可以顺序执行所有 tile，但必须输出 `tileBegin/tileEnd` 或等价统计；
M3.3 才启用 group 内 AIV 并行和 `gmm1SyncGroupReady` 消费。

与 GMM2 的关系：

- `activationSyncGroupReady[syncIdx]` 表示 `dequantSum[syncIdx]..dequantSum[syncIdx + 1]` 覆盖的所有 activation
  tile 已经写入 `gmm2InputInt8/gmm2PerTokenScale`。
- GMM2 scheduler 必须把这个 row range intersect 到 local expert/tile work item。不能为了方便让 GMM2 等全量
  `gmm2InputInt8` 完成后再生成所有 GMM2 tile。
- GMM2 完成后要再投影为 `gmm2GroupReady[expert]` 和 `subTileReady[aicTile]`；不能把
  `activationSyncGroupReady` 直接当成 combine-ready。

控制台结构化输出至少包含：

```text
swiglu_group_count=<...>
swiglu_group_sizes=[...]
swiglu_group_row_ranges=[{syncIdx,rowBegin,rowEnd,empty}, ...]
swiglu_group_tile_ranges=[{syncIdx,tileBegin,tileEnd}, ...]
activation_tile_rows=<...>
activation_aiv_workers=<...>
activation_empty_groups=<count>
```

验收口径：

- `swigluSyncGroups` 的 group size 之和等于 `expertPerRank`，每个 expert 只属于一个 group。
- `dequantSum` 必须能从 `cumsumMM[EP-1][localExpert]` 推导出每个 group 的连续 row range。
- `dequantSum[0] == 0`，`dequantSum` 单调不降，最后一个元素等于本 rank 所有 local expert 的总 row 数。
- 每个 non-empty group 的 activation tile range 覆盖 `rowBegin..rowEnd` 且无重复；empty group 必须 skip，不产生
  永久等待。
- `GMM1 -> ActivationQuant` 只等待当前 Swiglu group 覆盖的 expert rows ready，不等所有 GMM1。
- `ActivationQuant -> GMM2` 按 Swiglu group 发布 ready，GMM2 可消费该 group 的 row range，不等全量 activation。
- GMM2 消费的是 group row range 对应的 expert/tile work item；Combine 消费的是 GMM2 完成后的 expert/tile/sub-tile
  ready，不能混用这两个 ready 语义。
- timeline/counter 要能区分 `gmm1SyncGroupReady`、`activationSyncGroupReady`、`gmm2GroupReady`；如果后段
  Swiglu/quant 推迟了 Combine，task report 只写 overlap gap 摘要，而不是放宽 correctness。

## 12. 验证与观测策略

本章只定义项目级验证口径。每个 agent step 的具体命令、文件范围和逐条验收标准，以第 14 章为单一事实源；
状态流转、owner、report、Issue Log 和 DCL 以 `TASKS.md` 为准。

正确性验证矩阵：

| Case | 目标 | 预期 |
| --- | --- | --- |
| single rank, topK=1 | 验证 routing/GMM/restore 基本闭环 | output 等于 host A8W8 MoE reference |
| single rank, topK=2 | 验证 expanded row 和 probs weighted sum | output tolerance 内一致 |
| multi rank, balanced experts | 验证 count publish、dispatch gather、combine return | 每 rank output 一致 |
| multi rank, skewed experts | 验证 capacity、prefix、empty expert segment | 不越界，空段不触发错误 wait |
| odd token tail | 验证 tile valid region 和 GM offset | tail 行正确 |
| int8 GMM1/GMM2 | 验证 A3 PTO `int8 x int8 -> int32` backend | 与 int8 reference tolerance 内一致 |
| int8 activation requant | 验证 SwiGLU 后 GMM2 输入 | scale、saturation、zero point 策略一致 |
| hidden/intermediate alignment edge | 验证 int8 tile tail | padding 不污染输出 |
| zero-token expert | 验证 signal 与 segment skip | 不死锁 |

性能/overlap 验证：

- M1/M2 可以在最终 stage 图中用 `overlap_mode=off` 先做 correctness；M3 只能在同一 stage graph、layout、
  signal/counter 和 row range 上验证或启用 async producer/consumer 边，不能重写主路径。
- 每个 signal 增加 debug counter：producer count、consumer wait count、timeout count。
- kernel 内部为 route/dispatch/GMM1/SwiGLU/GMM2/combine/restore 设置 timestamp；host 默认转出可读
  `[Timeline]` stdout。额外文件导出只能作为本地 debug 辅助，不能作为 task acceptance 的唯一证据，也不能写入
  `reports/Mx.y.md`。
- 性能验收只要求证明 overlap 形态、等待点和膨胀原因；small shape 可能因为前同步而退化，Prefill 大 shape 可能因为
  MTE/带宽抢占导致 GMM 变慢，不能把这些直接判成 correctness 失败。
- A3 硬件运行前用 `npu-smi info` 确认卡空闲；本机 A3 可运行 A3 用例。

### 12.1 Correctness report

结果精度正确性必须独立成结构化 stdout 报告，不能只打印“PASS”。host 端参考
`kernels/manual/a2a3/gemm_ar/main.cpp` 的 `VerifyOutput()` 写法，但本项目按 MoE/int8 主路径记录以下字段：

```text
[CorrectnessReport]
  case_name
  seed
  rankNum/rankId/expertPerRank/topK/M/hiddenSize/intermediateSize/maxTokensPerExpert
  dtype_in/dtype_out
  reference_source = a8w8-int8-host-reference
  final_output:
    max_abs_diff
    max_rel_diff
    err_count
    err_threshold
    tolerance_atol
    tolerance_rtol
    checksum_actual
    checksum_expected
    pass
  intermediate:
    tokenPerExpertMatrix_checksum
    cumsumMM_checksum
    preSumBeforeRank_checksum
    dispatch_merge
    soft_sync_ledger
    swiglu_sync_groups
    combine_merge
    tile_split_return_map
    gmm1_accumulator_checksum
    gmm2_accumulator_checksum
    scale_dequant_checksum
  failure_locator:
    first_bad_token
    first_bad_hidden
    first_bad_rank
    first_bad_stage
```

M1 mock 阶段可以没有 GMM accumulator，但必须有 protocol metadata checksum。M2 之后必须记录 GMM1/GMM2
int32 accumulator checksum、scale/dequant checksum、dispatch/combine merge evidence 和 final output checksum。benchmark 结果只有在同一轮
correctness report `pass=true` 时才可采信。

### 12.2 E2E performance report

E2E 时间校验也必须独立成结构化 stdout 报告，host 端参考 `gemm_ar` 的 `PerfStats`、warmup/measure 和
rank0 汇总模式。
本项目固定 `warmup_iters=3`、`measure_iters=5`，不暴露成 run script 参数。

1. 每个性能用例先执行 3 次 warmup，warmup 不进入统计。
2. 每次测量前 reset local workspace、peer window signal/counter，并做 rank barrier。
3. E2E wall time 从 host 发起本轮第一段 kernel/launch 前开始，到本轮最后一个相关 stream 同步完成后结束。
4. M1/M2 使用最终 stage 图的 `overlap_mode=off` 记录 `overlap_off_e2e_us`；M3 之后同时记录
   `overlap_off_e2e_us` 和 `overlap_on_e2e_us`。
5. rank0 汇总每个 metric 的 `avg/min/max/stddev`；单次异常值不删除，但 task report 只写汇总结论，不记录每轮
   raw samples。
6. M3/M4 才允许计算 overlap 指标；M1/M2 只记录 E2E，不作为性能门槛。

性能报告字段：

```text
[PerfReport]
  case_name
  warmup_iters = 3
  measure_iters = 5
  rankNum/rankId/shape
  correctness_pass
  e2e_us:
    samples
    avg
    min
    max
    stddev
  stage_us:
    route
    count_sync
    dispatch_gather
    gmm1
    activation_quant
    gmm2
    fused_combine_return
    restore
  overlap:
    overlap_off_avg_us
    overlap_on_avg_us
    time_saved_us
    speedup
    overlap_efficiency
  counters:
    producer_count
    consumer_count
    timeout_count
```

正确性和性能报告默认打印到控制台。`reports/Mx.y.md` 只写 pass/fail、关键 metric 摘要、观察到的 overlap
形态或 blocked 原因，不复制 `[CorrectnessReport]` / `[PerfReport]` stdout 原文，也不要求生成报告文件。

E2E 性能通过标准不是固定 speedup 数字。M3/M4 只要求：

- correctness 不退化；
- E2E report 和 timeline 对同一 case/seed/shape 可关联；
- overlap on/off 至少能解释等待空泡、time saved 或 blocked 原因；
- small shape 退化时必须记录原因，不能把退化直接当 correctness 失败。

## 13. FFN 精读驱动的 M2/M3 PTO MegaMoE 实施导引

本章把 `ffn.md` 的精读结论和前 0-12 章已有设计约束收束成后续 M2.8c、M3 的实施导引。它不是新的
protocol，也不替代第 14 章逐任务验收标准；它的作用是防止开发时只盯某个 kernel 文件或某个 task，忘掉
MegaMoE FFN 的切分层级、同步边界和观测证据。

阅读顺序建议：

1. 先读第 0 章目标和硬边界，确认当前只做 A3 A8W8/int8 PTO-only 主路径。
2. 再读第 5-6 章 stage graph、五层切分和 overlap 执行视图，确认 producer/consumer 边不允许 M3 重写。
3. 再读本章，把 `ffn.md` 的算法事实映射成 M2.8c/M3 的开发路线。
4. 最后回第 14 章领取具体 task，按 `TASKS.md` 更新状态和 DCL。

### 13.1 `ffn.md` 精读后必须固定的十个事实

`ffn.md` 不是外部代码照搬清单，而是本项目判断“是否接近 PTO MegaMoE 目标形态”的语义依据。下面事实必须在
M2/M3 的设计、代码和验收输出里保持可追溯：

| `ffn.md` 事实 | 本项目绑定 | 不能误读成 |
| --- | --- | --- |
| AIC/AIV 角色分离：AIC 做 GMM1/GMM2，AIV 做 route、count、dispatch、SwiGLU、combine、restore | single fused MPMD launch 中保留 AIC/AIV 分支；M2.8c 先迁移全数据路径，M3 再打开更多 overlap | 把多 launch host 编排称为 fused，或把 AIV heartbeat 当成 payload 并行 |
| 五层切分是 rank/core/group/tile/L1-L0 | `ffn_partition_model=rank_core_group_tile_l1l0` 必须出现在 M2.8c/M3 结构化输出 | 只按 expert 切核，或只按 tile 解释所有同步 |
| group 是 ready/sync 边界，不是 core ownership | `dispatchGroupReady`、`gmm1SyncGroupReady`、`activationSyncGroupReady`、`gmm2GroupReady` 表达依赖边 | “AIC0 负责 expert0” 这类固定 expert-to-core 分配 |
| GMM tile 是 AIC 工作单元 | GMM task scheduler、multi-AIC evidence、L1/L0 tile policy 证明 tile ownership | 用单 AIC block 或 smoke shape 证明 GMM 已达目标 |
| init_routing 在 `ffn.md` 是 route/sort/count/srcToDst/gather+quant 子系统；本项目用 offset-table pack+gather 等价替代 sort | M3.2 的 dispatch 多 AIV 覆盖 route/count/pack/gather worker 分摊证据，不实现 multi-core merge sort | 把 PTO 简化误读成"必须补 sort 子系统"，或只把 `TGET` 循环并行化就声称 dispatch 多核 |
| count 同步是 dispatch 的前置点对点协议 | `tokenPerExpertMatrix`、count ready、prefix/cumsum 必须先对齐，再 remote gather | 把 count/prefix 做成 host barrier 或临时 host copy |
| SwiGLU 是 AIV 工作，按 sync group 粗细结合 | M2.5 固定 `swigluSyncGroups/dequantSum`，M3.3/M3.4 只打开 worker 分摊和 group overlap | 到 M3 重排 activation row layout |
| epilogue pipe 有 prefill/drain 生命周期 | M3.3/M3.5/M3.9 要记录 SetFlag/Finalize 或等价 PTO pipe lifecycle evidence | 只看 final output pass，不证明 pipe 没有悬空/脏 flag |
| Combine V1/V2 是同步成本取舍 | M3N.8 做 V1 continuous owner segment 轮动 + M3N.3 切分；M3N.11 用同步 `TPUT` 做 V2 Sub-Tile/stride（异步 SDMA stride 为 primitive-gap） | 把 Sub-Tile 理解成“DMA 越大越好”或以为同步 sub-tile return 做不了 |
| final restore 是 topK reduce，不能被通信寻址折叠 | `RestoreOutput` 继续按 `expandedRowIdx + probs` 加权输出 | combine 写回后直接把 return payload 当 final output |

### 13.2 目标形态和 M2/M3 边界

最终目标形态是一个 PTO-only single fused MPMD kernel：AIV 分支推进 route/count/dispatch、activation/requant、
combine return 和 restore，AIC 分支推进 GMM1/GMM2。host 只负责资源初始化、单次 fused launch、debug/preflight
模式选择和结构化报告，不再用 stage 间 `aclrtSynchronizeStream` / `MpiBarrier` 构成 accepted 主路径。

M2.8c 的边界是“全数据路径迁移到 single fused launch，overlap-off 可以接受”。它必须证明 active M2 full chain 已经
在 single fused MPMD graph 中跑通，multi-launch 只保留为 `multi_launch_debug` preflight。M2.8c 不要求
dispatch、activation、combine return 的 payload-level AIV 多 worker 分摊，但必须如实打印 worker facts，不能把
`mixed_aiv_blocks>0`、AIV heartbeat 或 stage 参与数量当成“payload 已多核”的证据。

M3 的边界是“只在 M2 固定的 edge/layout/signal 上打开并发”。M3 可以改变 worker partition、ready 触发时序、
poll/wait 策略和 timeline 打点；不能改变 `tokenPerExpertMatrix/cumsumMM/preSumBeforeRank/expandedRowIdx`、
`swigluSyncGroups/dequantSum`、`ReturnSegmentPlan/OwnerSegment`、GMM tile policy 或 final restore 语义。

### 13.3 M2.8c 必须交付的 fused data-path 形态

M2.8c 关闭 ISSUE-2026-05-29-07 的条件不是“mixed ELF 能 launch”，M2.8a/M2.8b 已经证明这个前提；M2.8c 要把当前
active M2 full chain 从 host multi-launch 迁入 single fused MPMD stage graph。最小 accepted 形态如下：

| 维度 | M2.8c 必须做到 | M2.8c 不要求 |
| --- | --- | --- |
| launch 形态 | `stage_graph_mode=single_fused_mpmd`，host 单次 fused handle launch，AIC/AIV side 都执行 active data path | overlap on 性能收益 |
| 数据路径 | `RoutePackQuantLocal -> GatherDispatchToGmm1Input -> GMM1 -> ActivationQuant -> GMM2 -> RunGmm2EpilogueAndReturn -> RestoreOutput` 全链在 fused graph 内 | M3 级 async scoreboard 或 Sub-Tile async |
| debug 路径 | multi-launch 只能作为 `multi_launch_debug=true` preflight，报告里标清不参与 accepted main path | 删除已有 multi-launch 调试能力 |
| 同步模式 | 可以用卡内 hard `SYNCALL<SyncCoreType::Mix>`（AIC+AIV 全核栅栏，非跨卡）或 overlap-off device wait 表达 stage 边；跨卡同步仍只在 dispatch 前 count 同步和 combine 后 restore 两点 | host stage 间 barrier；M3 才把中间卡内栅栏细化为第 10 章的 per-`CrossCoreId` CV 握手 |
| GMM | 沿用 M2 accepted 的 `gemm_ar` cache-level policy、multi-AIC task partition 和 accumulator exactness | 重做 tile size 或降回 small smoke |
| AIV worker facts | 对 route/count/gather/activation/return/restore 打印 worker count、processed/skipped rows 或 segments | payload-level 多 AIV 分摊 |
| FFN partition facts | 打印 rank/core/group/tile/L1-L0 五层事实、group/tile 正交事实和 AIV 并行延后事实 | 用单字段笼统说 “megamoe=true” |

M2.8c 的结构化输出至少包含：

```text
stage_graph_mode=single_fused_mpmd
single_fused_payload_migrated=true
multi_launch_debug_only=true
overlap_mode=off
ffn_partition_model=rank_core_group_tile_l1l0
group_is_sync_boundary=true
gmm_tile_is_aic_work_unit=true
aiv_data_parallel_deferred_to_m3=true
dispatch_aiv_workers=<count>
activation_aiv_workers=<count>
combine_return_aiv_workers=<count>
restore_aiv_workers=<count>
dispatch_payload_parallel=false|true
activation_payload_parallel=false|true
combine_payload_parallel=false|true
mixed_aiv_blocks=<count>
mixed_aic_blocks=<count>
```

其中 `*_payload_parallel=false` 在 M2.8c 可以 accepted，但必须和 `aiv_data_parallel_deferred_to_m3=true` 同时出现。
如果某段已经天然多 AIV，也必须用 processed rows/segments counter 证明，不能只靠 block 数推断。

### 13.4 M3 打开并发的推荐顺序

M3 的顺序必须从“先保证 fused graph 和信号可观测”到“逐段打开 payload worker 分摊”，最后用 timeline 证明等待空泡。
推荐顺序如下，除非发现 primitive-gap 或 env-gap，否则不要跳过前置证据直接优化局部循环。

| Task | 目标 | 关键证据 |
| --- | --- | --- |
| M3.0 | 在 M2.8c fused data path 上打开 M3 runtime 开关，确保所有 stage 在同一 launch 内可按信号推进 | `stage_graph_mode=single_fused_mpmd`、stage enter/exit counter、no host stage barrier |
| M3.1 | 启用 `dispatchGroupReady/gmm1SyncGroupReady/activationSyncGroupReady/gmm2GroupReady` 计数和 wait dump | 每类 signal 的 producer/consumer count、timeout count、empty group skip |
| M3.2 | 做 dispatch 多 AIV（route/count/pack/gather worker 分摊，不含 sort 子系统） | route rows、count rows、pack rows、gather source-rank rows 的 per-worker counter |
| M3.3 | 做 Activation/SwiGLU/requant 多 AIV tile partition，并补 epilogue pipe prefill/drain evidence | `swiglu_group_tile_ranges`、activation tile processed/skipped、pipe prefill/drain/finalize counter |
| M3.4 | 让 GMM2 消费 `activationSyncGroupReady` row range，不等全量 activation | sync-group row range 到 GMM2 tile task 的投影、GMM2 wait histogram |
| M3.5 | 做 Combine V1-style continuous owner segment 多 AIV return，并证明 group/expert-level overlap | owner segment per-worker counter、continuous segment coalesce、SyncAll/CV wait 空泡 |
| M3.6 | 补 timeout dump 和 overlap 回归，确保 blocked 原因能定位到 signal、worker 或 primitive | per-signal timeout、last producer/consumer、rank/expert/tile locator |
| M3.7 | 在 M2.2c scoreboard ledger 上打开 async scoreboard，不重定义 task id 或 dependency domain | dependency-domain aggregation、worker poll/wait count、scoreboardMinStatus by domain |
| M3.8（→M3N.11） | 在 M2.7a segment schema 上用**同步 `TPUT`** 做 Combine V2 Sub-Tile/stride return（功能可 enabled，`TPut.hpp` 5 维 stride 已确认）；仅"异步 SDMA stride overlap"是 primitive-gap blocked | `subtile_rows`、strided/multi-segment 同步 `TPUT` evidence、subtile SyncAll reduction；异步 overlap 标 `primitive_gap=TPUT_ASYNC_flat_contiguous_1d` |
| M3.9 | 输出 timeline，区分 launch-level participation 和 payload-level worker evidence | stage intervals、group/tile/sub-tile ownership、SyncAll/CV wait bubble、overlap on/off 对照 |

M3.2、M3.3、M3.5 是 payload-level AIV 数据并行的主体；M3.7/M3.8 是更细同步和更细 return 的验证项；
M3.9 是验收证据汇总项。M3.0/M3.1 只打开 graph/signal，不等于性能形态完成。

### 13.5 Combine V1/V2 与 Sub-Tile 的本项目落点

本项目已经在 M2 固定 `ReturnSegmentPlan/OwnerSegment/subTileReady` schema，因此 M3 不再重新设计 combine offset。
后续只在同一 schema 上区分两类执行形态：

| 形态 | 适用目标 | 本项目任务 | 验收重点 |
| --- | --- | --- | --- |
| V1 continuous owner segment | 大 shape / prefill 方向，尽量合并连续目的 rank、source rows、destination rows | M3.5 | 多 AIV 分摊连续 owner segment，减少单 worker return 尾巴，记录 SyncAll/CV wait 空泡 |
| V2 Sub-Tile/stride | 小 shape / decode 方向，AIV consumer 与对应 AIC tile 对齐，降低全核同步空泡 | M3.8（→M3N.11） | 同步 strided/multi-segment `TPUT` 已确认可在 A3 表达，功能可 enabled；仅"异步 SDMA stride overlap"是 primitive-gap blocked |

Sub-Tile 的价值是减少同步等待，不是追求更大的 DMA。若某个 shape 下 Sub-Tile 导致 segment 过碎，优先调
sub-tile 粒度、coalesce 规则或回落 V1 continuous segment；不能把多个 token owner 混写到同一远端连续地址，也不能
回到“先全量 `gmm2Out`，再 copy combine”的二阶段主路径。

### 13.6 Evidence matrix

M2/M3 的 accepted 不能只靠 final output pass。下面字段用于把 correctness、并发和 blocked 原因分开：

| 能力 | 证明字段 |
| --- | --- |
| single fused 主路径 | `stage_graph_mode=single_fused_mpmd`、`single_fused_payload_migrated=true`、`multi_launch_debug_only=true` |
| AIC tile ownership | `gmm_tile_is_aic_work_unit=true`、`gmm_tile_tasks`、`gmm_active_aic_blocks`、`gmm_l1/l0_tile_shape` |
| group/tile 正交 | `group_is_sync_boundary=true`、`gmm_all_aic_visit_all_groups=true` 或等价 tile assignment dump |
| dispatch payload 多 AIV | per-worker route/count/pack/gather rows、source rank ranges、processed/skipped counter |
| activation payload 多 AIV | `swiglu_group_tile_ranges`、activation tile owner、requant processed rows、pipe prefill/drain |
| combine payload 多 AIV | owner segment worker assignment、segment coalesce count、return `TPUT` count by worker |
| scoreboard async | dependency-domain `scoreboardMinStatus`、poll count、wait count、timeout locator |
| Sub-Tile/stride | `subtile_rows`、`subtile_owner_segments`、strided or multi-segment `TPUT` evidence、`subtile_syncall_count` |
| timeline | per-stage begin/end、ready/wait intervals、SyncAll/CV wait bubble、overlap on/off deltas |
| blocked 原因 | `primitive_gap=<name>`、`env_gap=<name>`、last producer/consumer/task id、affected rank/expert/tile |

### 13.7 禁止路径

后续 M2.8c/M3 开发禁止通过下面捷径“看起来完成”：

- 禁止把 mixed launch heartbeat、`mixed_aiv_blocks` 或 AIV branch enter counter 当成 payload-level 多 AIV 证据。
- 禁止把 expert group 固定绑定到某个 AIC/AIV core；group 是 ready 边界，tile/row/segment 才是 worker work item。
- 禁止在 M3 重写 M2 已验收的 row order、offset、owner segment、SwiGLU group 或 GMM tile policy。
- 禁止用 host multi-launch、host barrier 或 host copy/reorder 作为 accepted fused 主路径。
- 禁止为了 Sub-Tile/stride remote write 引入 AscendC/Catlass fallback；PTO 表达不了就标 primitive-gap blocked。
- 禁止用 small smoke shape 证明最终形态；small shape 只证明语义和风险，不证明调度效率。
- 禁止把 final restore 折叠进 combine 通信；topK weighted reduce 仍由 `RestoreOutput` 负责。

### 13.8 M2/M3 交接口径

M2.8c accepted 表示：最终 protocol、layout、row/order/offset、GMM tile policy、return segment schema 和 single fused
stage graph 已经固定，active full data path 能在 fused launch 内正确运行。它不表示：dispatch/activation/combine
已经完成 payload-level AIV 数据并行，也不表示 runtime overlap 已经有性能收益。

M3 accepted 表示：在 M2 固定边界上，至少按任务拆分证明了 async 或 worker 分摊的真实执行证据；无法完成的点有
明确 primitive-gap/env-gap、locator 和不影响 M4 correctness 的回退口径。M3 的核心验收不是“打开开关”，而是能用
counter/timeline 说明每个 group、tile、sub-tile 由谁生产、谁消费、等了多久、为什么等。M3/M3N accepted **不要求** overlap on
的 E2E 性能优于 off：首要目标是对标 `ffn.md` 把切分与并发形态做正确、证据完整；E2E delta 只需如实记录可对照，性能为正是
后续优化迭代目标而非 accept 门槛（见第 0 章性能门槛口径）。M3 的逐边/分段执行以 §14.4N 的 M3N 设计为准。

## 14. 任务分解与阶段验收标准

本章是 implementation agent 开发任务内容、依赖、文件范围和验收标准的单一事实源。`TASKS.md` 只跟踪
state、owner、report、Issue Log 和 Design Change Log；领取任务后必须回到本章对应小节读取实现要求。

本章开头的两张表只做开发导航，帮助 agent 先判断阶段目标和任务边界；每个 task 的依赖、文件范围、实现要求和
验收标准仍以对应 `M*. *` 小节为准。

阶段导航：

| 阶段 | 完成什么 | 关闭口径 |
| --- | --- | --- |
| M0 | 工程骨架、脚本、layout、host smoke。从 `gemm_ar` 裁剪 CMake/run.sh/main.cpp，不从空目录手写。 | M0.1-M0.6 全部 accepted；dry-run/smoke 路径可用；依赖扫描无禁用接口、build helper 或 fallback。 |
| M1 | PTO dispatch/combine protocol 闭环。只能把中间 `GMM1 -> SwiGLU/Quant -> GMM2` 专家计算整体 mock；routing、count、prefix、dispatch `TGET`、combine `TPUT`、signal、restore 必须真实落到 device path。 | M1.0-M1.11 全部 accepted；2 卡 NPU/mpirun 实跑通过；metadata、row order、signal、restore 正确；mock 只替代 expert compute。dry-run/reference-only 不能关闭 M1。 |
| M2 | A3 int8_int8 全路径功能，并前置 MegaMoE 必需数据布局和 `ffn.md` 五层切分事实：dispatch 融合、GMM1 contiguous input、runtime-shape GMM tile partition、GMM2 epilogue+combine return、soft-sync ledger、Swiglu sync-group metadata、tile-split return map 和 segment-driven return path。 | M2.0-M2.8 全部 accepted；active runtime 已归一到 `host/`、`kernel/`、`include/`，dispatch/activation/combine 合并点、GMM L1/L0 tile plan、multi-block work partition、soft-sync ledger、`swigluSyncGroups/dequantSum`、shape-derived tile-split return map、segment-driven return 和 single fused MPMD stage graph 已在最终布局中验收；accumulator 精确对齐，epilogue/final output 按 tolerance 对齐；M2 report 必须证明 group 是 sync boundary、GMM tile 是 AIC work unit，并如实记录各 AIV stage worker 事实；AIV 侧 dispatch/activation/return 数据并行不是 M2 硬门槛，不能把 `mixed_aiv_blocks>0` 误报为全阶段多核；M3 不需要重写 row/order/layout/stage graph。 |
| M3 | 在 M2 已固定的依赖边上打开或验证 runtime overlap、AIV 数据并行、scoreboard、Sub-Tile remote write 和 timeline；若 M2.8 仍是 `multi_launch_debug`，M3.0 首先补齐 single fused MPMD stage graph，而不是只打开开关。 | M3.0-M3.9 accepted 或明确 primitive-gap blocked；timeline/counter 能解释 Dispatch-GMM、GMM1-SwiGLU、GMM2-Combine 的 overlap、AIV worker 分摊、group/tile/sub-tile ownership、等待空泡或阻断原因。 |
| M4 | 最终 PTO 化回归与文档状态收口。 | M4.1-M4.2 全部 accepted；状态、report、设计一致，无 open `needs_user_decision`。 |

任务导航：

| Task | 完成什么 |
| --- | --- |
| M0.1 | 已完成：markdown 规则、目录骨架、`TASKS.md`/report 规则；开发 agent 不领取。 |
| M0.2 | 从 `gemm_ar/CMakeLists.txt` 裁剪 single fused mixed-kernel CMake target。 |
| M0.3 | 从 `gemm_ar/run.sh` 裁剪 `run_a3.sh`，加入显式 MoE/GMM tiling 参数、MPI rank 来源、dry-run、fixed warmup=3/measure=5。 |
| M0.4 | 定义 `ShapeConfig`、`RankConfig`、`WorkspaceLayout`、`PeerWindowLayout`、control/timeline 字段入口和关键 offset。 |
| M0.5 | 建 host reference、deterministic data、`CorrectnessReport` skeleton。 |
| M0.6 | 从 `gemm_ar/main.cpp` 裁剪 host main/runtime/smoke launch。 |
| M1.0 | 核验并固化 MegaMoE 主路径协议不变量和列入/不列入边界。 |
| M1.1 | 建 control metadata typed views。 |
| M1.2 | 固化 PTO primitive 直接调用约定，不做隐藏 PTO 调用的 helper wrapper。 |
| M1.3 | 实现 `RouteLocalTokens`，生成 count、`expandedRowIdx`、`dispatchOffset`。 |
| M1.4 | 实现 `RoutePackQuantLocal` 的 M1 mock/non-quant pack 形态，写本 rank peer-visible dispatch payload。 |
| M1.5 | host 侧 HCCL window bootstrap，把 window/context 传入 kernel。 |
| M1.6 | 实现 `PublishCounts` 和 `WaitCounts`。 |
| M1.6a | 构建 `cumsumMM`、`preSumBeforeRank`、`expertTokenNums`。 |
| M1.7 | 实现 `GatherDispatchToGmm1Input` 的真实 `TGET` 形态，expert owner 远端读 dispatch payload 到 M1 mock payload target。 |
| M1.8 | mock expert output，只替代 GMM 数值，不替代协议。 |
| M1.9 | 实现 `RunGmm2EpilogueAndReturn` 的真实 `TPUT` return 形态，expert owner 把 mock expert output 远端写回 return payload。 |
| M1.10 | 实现 `RestoreOutput`，按 `expandedRowIdx + probs` 加权恢复。 |
| M1.11 | M1 四类 case 集成回归，输出 correctness/perf 控制台摘要。 |
| M2.0 | 锁定 M2 active implementation baseline、路径迁移和 primitive/arch preflight。 |
| M2.1 | 定义 A3 int8 backend layout/interface 和主路径 dtype。 |
| M2.2 | host 侧 int8 weight、scale、reference accumulator。 |
| M2.2a | 实现 fused `RoutePackQuantLocal`，route/pack/quant 一次写 peer-visible payload。 |
| M2.2b | 实现 `GatherDispatchToGmm1Input`，远端读后直接落 GMM1 contiguous input。 |
| M2.2c | 固化 Dispatch-GMM soft-sync ledger、scoreboard task map 和 counter。 |
| M2.3 | PTO `int8 x int8 -> int32` GMM1，包含 runtime shape work partition、L1/L0 tile 切分和双缓冲 pipeline。 |
| M2.4 | GMM1 epilogue：scale dequant/cast，不加 bias。 |
| M2.5 | SwiGLU + per-token requant，生成 GMM2 int8 input/scale，并固定 `swigluSyncGroups/dequantSum`、group descriptor 和 activation tile plan。 |
| M2.6 | PTO `int8 x int8 -> int32` GMM2，复用同一 GMM tile/pipeline 契约并对接 activation row range。 |
| M2.7 | GMM2 epilogue + fused combine remote return。 |
| M2.7a | 固化 GMM-Combine tile/sub-tile return segment map，并让 M2 return 主路径按 segment plan 驱动。 |
| M2.8 | M2 全链路回归，证明 dispatch/combine 合并点、single fused correctness 和 AIV stage worker 事实；不要求 dispatch/activation/return 全量 AIV 数据并行。 |
| M3.0 | 打开 single-kernel MPMD 调度，AIV/AIC role 在同 kernel 内推进。 |
| M3.1 | 启用/校验 M2 已固定的 overlap signal、counter 和 `swigluSyncGroups/dequantSum` 分组语义。 |
| M3.2 | 打开/验证 Dispatch-GMM1 expert-group overlap，并把 init_routing-like route/count/pack/gather 从 M2 单 AIV correctness path 提升为多 AIV worker 分摊。 |
| M3.3 | 基于 M2.5 group descriptor 打开/验证 GMM1-Activation sync-group overlap，并启用 activation/SwiGLU/requant group 内多 AIV tile 分摊和 epilogue pipe 排空证据。 |
| M3.4 | 基于 M2.5 group row/tile range 打开/验证 Activation-GMM2 sync-group overlap。 |
| M3.5 | 打开/验证 GMM2-Combine expert/group 级 overlap，并把 GMM2 epilogue/continuous owner-segment return 从 M2 单 AIV correctness path 提升为多 AIV worker 分摊。 |
| M3.6 | 增加 timeout dump、overlap on/off 回归和 counter 证据。 |
| M3.7 | 基于 M2.2c 账本打开真实 Dispatch-GMM scoreboard soft sync。 |
| M3.8 | 基于 M2.7a segment map 验证并打开 Combine V2 风格 Sub-Tile/stride remote write；能力不足则 blocked；可在同一 segment schema 上继续细化 return worker 分摊。 |
| M3.9 | 增加 kernel timestamp/timeline 输出，证明 overlap、AIV worker 利用、group/tile/sub-tile ownership 或定位阻断原因。 |
| M4.1 | 最小最终回归脚本。 |
| M4.2 | 文档、`TASKS.md` 状态和 report 收口。 |

M3 只允许在 M2 已固定的 producer/consumer edge、signal、counter、row range 和 layout 上打开异步执行。若实现
M3 时发现必须重写 dispatch row order、combine offset、`scoreboardTaskMap`、`ReturnSegmentPlan` 或 stage graph，
这不是正常 M3 工作，必须按 issue/DCL 流程回到受影响的 M1/M2 任务处理。

通用执行规则：

- 每个 `M*. *` 小节本身就是一个 agent step，执行时逐条完成“任务”，逐条满足“验收标准”。
- 每个 task 交付前必须更新 `TASKS.md` 的 task state、owner、report 和 issue 状态。
- 每个 task 必须创建或更新 `reports/Mx.y.md`，只记录验收结论摘要、关键观察和 blocked 原因；不要粘贴编译日志、
  stdout 原文、counter 原文或 timeline 原文。
- 任何实现偏离本设计的 stage graph、layout、signal、precision、PTO-only 或验收口径，必须先在 `TASKS.md` 创建 issue；需要用户确认时不得交付为 accepted。
- 源码、CMake 和脚本不能引用、包装、转发或 fallback 到 Catlass/AscendC；每个阶段都必须保留依赖扫描证据。
  各 task 写的“依赖扫描无输出”只扫描本项目源码、CMake 和脚本文件，不扫描 `DESIGN.md`、`TASKS.md` 或
  `reports/`，因为这些 markdown 需要记录禁用依赖名和设计决策。CMake 里仅用于 CANN SDK/编译器搜索的
  系统 include 路径不算 AscendC 接口依赖；扫描重点是禁用 API include/call/wrap、build helper、target 和 fallback。

### 14.1 M0: 工程初始化

M0 的目标不是从空目录手写工程，而是从仓内已跑通过的 manual 工程裁剪出可维护骨架：

- `kernels/manual/a2a3/gemm_ar/CMakeLists.txt`：优先参考 CANN 环境探测、`bisheng`、host executable、kernel shared
  library、runtime/HCCL 链接和 `RUN_MODE/SOC_VERSION` 传参组织。
- `kernels/manual/a2a3/gemm_ar/run.sh`：优先参考 CANN env、MPI discovery、HCCL_BUFFSIZE 估算、build/run 串联、
  `mpirun` 启动和 first-device 传参组织。
- `kernels/manual/a2a3/gemm_ar/main.cpp`：优先参考 host 端数据生成、HCCL/ACL runtime 初始化、rank/device 选择、
  correctness/perf 结构化控制台输出和 fixed warmup/measure 组织。
- `kernels/manual/a2a3/moe_dispatch`、`kernels/manual/a2a3/dispatch_combine_tile`：只作为 PTO manual kernel 单 target
  命名、最小 kernel build 和显式 MoE shape 参数的补充参考。

裁剪规则：

- 允许复制文件骨架和命令组织；复制后必须改成 `moe_dispatch_combine_a8w8` 命名、单 fused kernel 目标和本项目显式
  MoE 参数。
- 不复制 `gemm_ar` 的双 kernel split 作为最终结构；本项目 host 只 launch single fused MPMD kernel 或 dry-run
  smoke。
- 不把 `gemm_ar` 的 HCCL 内部通信 helper 当成本项目算法依赖。M0 可以保留 host 侧 HCCL window/bootstrap 占位和必要
  runtime link 组织，真正 window protocol 在 M1.5 落地。
- 不能引入 Catlass/AscendC 接口、build helper 或 fallback。若从参考 CMake 复制了仅用于 CANN 编译器搜索的路径，
  也不能在本项目源码中 include/call/wrap `AscendC::`、`DataCopy`、Catlass 或 AscendC kernel helper。
- 当前 A3/CANN 8.5 实测单个 `dav-c220` raw mixed target 会让现有 PTO Vec/comm primitive 报 target feature
  不支持；这个负证据只排除“单 raw mixed target”路线。M2.8 的 fused 路线必须改为 PTO mixed ELF/metadata：
  AIC `dav-c220-cube` + AIV `dav-c220-vec` 分别编译，再通过 `.ascend.meta` / registration ELF 单 launch。
- M0.1 的 markdown bootstrap 已由设计 agent 在当前会话完成，开发 agent 不领取 M0.1；真正工程初始化从 M0.2 开始。

#### M0.1 设计 bootstrap 与项目规则（已完成，非开发任务）

状态：已完成并在 `TASKS.md` 标为 `accepted`。后续开发 agent 只读取 `DESIGN.md`、`TASKS.md`、
`reports/M0.1.md`，不得重做本任务。若发现 M0.1 的规则需要变化，按 `Issue Log`、`Design Change Log`
和用户确认门禁处理。

已完成内容：

- 建立 `host/`、`include/`、`kernel/`、`reports/`、`scripts/` 目录骨架。
- `DESIGN.md` 写清当前项目路线、MegaMoE 对齐点、PTO-only 依赖禁令、markdown 分工、任务拆解和验收口径。
- `TASKS.md` 写清 state enum、stage/task 状态表、Issue Log、Design Change Log 和 handoff rules。
- `reports/TEMPLATE.md` 只保留轻量验收摘要、blocked 原因、design issue 和 downstream handoff 字段，明确不记录编译日志、
  stdout 原文、counter 原文或 timeline 原文。

#### M0.2 CMake 最小 target

依赖任务：M0.1。

文件范围：

- 创建 `kernels/manual/a2a3/moe_dispatch_combine_a8w8/CMakeLists.txt`
- 创建 `kernels/manual/a2a3/moe_dispatch_combine_a8w8/kernel/moe_dispatch_combine_a8w8_kernel.cpp`
- 创建 `kernels/manual/a2a3/moe_dispatch_combine_a8w8/kernel/protocol_core.hpp`
- 创建 `kernels/manual/a2a3/moe_dispatch_combine_a8w8/kernel/control_metadata.hpp`
- 创建 `kernels/manual/a2a3/moe_dispatch_combine_a8w8/kernel/a3_int8_backend.hpp`

任务：

- 以 `kernels/manual/a2a3/gemm_ar/CMakeLists.txt` 为主参考，复制后裁剪成单 fused kernel 工程：
  `project(pto_moe_dispatch_combine_a8w8)`、`moe_dispatch_combine_a8w8_kernel` shared library、后续
  `moe_dispatch_combine_a8w8` host executable。
- 保留 `bisheng`、`ASCEND_HOME_PATH`/`ASCEND_DRIVER_PATH` 探测、runtime output 目录、`RUN_MODE/SOC_VERSION`
  传参、host link dirs、`runtime/runtime_camodel` 条件链接等工程骨架。
- 参考 `moe_dispatch` / `dispatch_combine_tile` 的单 kernel target 命名。M0/M1 active dispatch/combine target
  可使用 `dav-c220-vec`；M2 必须通过 M2.0/M2.3 记录 GMM cube primitive preflight 和 single raw mixed-target
  gate，M2.8 再通过 PTO mixed ELF/metadata spike 证明 single fused MPMD launch。
- CMake 使用 `bisheng + add_library(... SHARED ...)` 路线，不使用 `ascendc_library`、`ascendc.cmake`、
  `ascendc_kernel_cmake`、Catlass target 或外部融合算子 target。
- kernel 文件只放最小 launch 入口或空实现骨架。
- `protocol_core.hpp`、`control_metadata.hpp`、`a3_int8_backend.hpp` 只放 header guard、命名空间和阶段占位注释；
  真实协议/GMM 逻辑分别留给 M1/M2。
- kernel include 只使用 PTO public headers 和项目本地 header；从参考工程复制 include/link 组织时，不能把
  `AscendC::` API、`DataCopy` helper 或 Catlass include/call 引入源码。

验收标准：

- CMake target 能被构建系统发现。
- kernel 文件不包含禁用依赖。
- CMake 不包含 `ascendc_library`、`ascendc.cmake`、`ascendc_kernel_cmake`、Catlass target 或外部融合算子 target。
- CMake 中能看出 active M1 dispatch/combine target 是本项目 target，不是 `gemm_ar` 的 compute/comm 双 kernel
  直接复制；如果当前 compile arch 仍是 `dav-c220-vec`，M2.0 必须记录 mixed target gate 和 GMM primitive
  preflight 路线。
- task report 只记录 target 是否接入/可发现/可编译的结论摘要；若当前仓构建脚本尚未接入该 target，交付状态只能是
  `blocked` 或“target created, build integration pending”，不能声称已编译通过；不要粘贴编译日志。
- 依赖扫描无输出。

#### M0.3 显式参数与 run 脚本

依赖任务：M0.1、M0.2。

文件范围：

- 创建 `host/args.hpp`
- 创建 `scripts/run_a3.sh`

任务：

- 以 `kernels/manual/a2a3/gemm_ar/run.sh` 为主参考，复制后裁剪出本项目 `scripts/run_a3.sh`：CANN env 加载、
  MPI discovery、`FIRST_DEVICE`/device-base、`RUN_MODE/SOC_VERSION`、HCCL shared-memory 清理、CMake build/run
  串联、`mpirun -n` 启动。
- 结合 `moe_dispatch/run.sh` 的显式 MoE shape 参数风格，保留显式参数，不引入隐藏 case preset。
- `args.hpp` 定义第一批显式 shape 参数：`rankNum`、`rankId`、`expertPerRank`、`topK`、`M`、
  `hiddenSize`、`intermediateSize`、`maxTokensPerExpert`、`payloadTileCols`、`gmmBlockM`、`gmmBlockN`、
  `gmmBlockK`。
- `rankId` 的默认来源必须是 MPI rank。`run_a3.sh` 默认使用 `--rank-from-mpi 1` 或等价行为，不向所有
  `mpirun` 进程传同一个固定 `rankId`；`--rank` 只允许用于 single-process/debug，且必须和
  `--rank-from-mpi 0` 一起显式使用。
- `run_a3.sh` 提供 `--help`。
- `run_a3.sh` 提供 small、balanced、skewed、zero-token 四类显式参数模板。
- `run_a3.sh` 提供 `--timeline` 开关；correctness/perf 报告默认直接打印到 stdout。
- E2E bench 固定 `warmup_iters=3`、`measure_iters=5`，不暴露 warmup/measure 迭代次数参数。
- `run_a3.sh` 需要输出 workspace/peer window/HCCL_BUFFSIZE 估算摘要。M0.3 先按显式 shape 参数给出保守估算；
  M0.4/M0.6 接入 `WorkspaceLayout` / `PeerWindowLayout` 后，再替换为 layout total-bytes 计算。
- `run_a3.sh` 只输出控制台结果，不写 correctness/perf json 文件。
- 不实现隐藏 `case all`。

验收标准：

- `bash kernels/manual/a2a3/moe_dispatch_combine_a8w8/scripts/run_a3.sh --help` 能打印所有参数。
- help 文本中有 small、balanced、skewed、zero-token 四类用例说明。
- help 文本中说明 correctness/perf 默认打印到 stdout，且有 timeline 开关说明。
- help 或 dry-run 输出中明确 rank 来源：MPI 派生为默认，手动 `--rank` 为 debug override。
- `run_a3.sh` 能在 `--skip-run 1` 或 dry-run 模式完成 CANN/MPI/build 参数路径检查，并打印基于显式 shape 的
  HCCL_BUFFSIZE 保守估算摘要。
- 依赖扫描无输出。

#### M0.4 Workspace 与 window layout 类型

依赖任务：M0.1、M0.3。

文件范围：

- 创建 `include/moe_dispatch_combine_a8w8_types.hpp`
- 创建 `include/moe_dispatch_combine_a8w8_layout.hpp`
- 创建或修改 `host/workspace_layout.hpp`

任务：

- 定义 `ShapeConfig`、`RankConfig`、`WorkspaceLayout`、`PeerWindowLayout`。
- 分离 ordinary GM workspace 与 HCCL peer window。
- 每个 layout field 提供 byte offset、byte size、alignment。
- `ShapeConfig` 必须覆盖第 4.2 节列出的 launch-time shape / tiling 参数，包括 `gmmBlockM/N/K`。
- `PeerWindowLayout` header 必须包含 `dtype_in`、`dtype_out`、`dispatchPayloadRowBytes`、
  `returnPayloadRowBytes`。
- signal/counter 字段按 cache line 对齐。
- layout 必须显式覆盖主路径关键字段：`tokenPerExpertMatrix[rankNum][rankNum][expertPerRank]`、
  `cumsumMM[rankNum][expertPerRank]`、`preSumBeforeRank[rankNum][expertPerRank]`、`expertTokenNums[expertPerRank]`、
  `expandedRowIdx[M * topK]`、return payload `offsetD`。
- layout 需要预留第 7 章的 control/timeline 类字段入口：ready signal/counter、scoreboard、Sub-Tile return、
  `swigluSyncGroups/dequantSum`、timestamp/timeline；后续 task 可以填充具体语义，但不能重做 offset 解释体系。
- dispatch payload 必须有 `dispatchPayloadRowBytes` 字段。A3 int8 可以只用 hidden payload，但 row stride 仍必须显式，
  不能靠后续 stage 临时 reinterpret 改 layout。
- peer window 中 dispatch payload 语义必须是 token owner rank 本地发布、expert owner rank 远端读；字段名或注释要避免误解成
  token owner rank 直接远端写 expert owner rank 的 dispatch output。

验收标准：

- host 侧提供 `DumpLayout` / `PrintLayoutDump` 或等价接口，可打印所有 layout offset 和总 byte size；实际 dry-run
  stdout 证明由 M0.6 的 host main 完成，M0.4 不新增一次性临时 driver。
- 相同 shape 下每个 rank 计算出的 peer window layout 完全一致。
- 所有 peer-visible field 都在 `PeerWindowLayout`，local-only field 都在 `WorkspaceLayout`。
- layout dump 中能看见 `tokenPerExpertMatrix`、`cumsumMM`、`preSumBeforeRank`、`offsetD`、`dispatchPayloadRowBytes`。
- layout dump 中能看见 `dtype_in/dtype_out`、`returnPayloadRowBytes`、`gmmBlockM/N/K` 和 control/timeline 字段分组。
- 依赖扫描无输出。

#### M0.5 Host reference 与数据生成骨架

依赖任务：M0.3、M0.4。

文件范围：

- 创建 `host/reference.hpp`
- 创建 `scripts/gen_data.py`

任务：

- 参考 `gemm_ar/main.cpp` 的 deterministic host-side data generation 与 verification 组织；如果实现选择保留
  `scripts/gen_data.py`，Python 只生成输入/参考数据，不引入新的运行状态源。
- 提供 deterministic input、expertId、probs 生成规则。
- 提供 CPU reference 数据结构：routing counts、expanded row、dispatch rows、mock combine output。
- 提供 host-side `CorrectnessReport` 数据结构和 stdout print skeleton，字段按“Correctness 与 E2E 性能记录规则”。
- `host/reference.hpp` 提供构造和打印 `[CorrectnessReport]` skeleton 的接口；实际由 M0.6 host main 调用并打印，
  M0.5 不新增一次性临时 driver。
- `gen_data.py` 能按 small/balanced/skewed/zero-token 生成输入文件。

验收标准：

- 同一 seed 多次生成的输入完全一致。
- small case 的 CPU routing reference 能打印 tokenPerExpert 和 expandedRowIdx。
- zero-token expert case 至少包含一个 local expert count 为 0。
- host reference 能为 small case 构造 `[CorrectnessReport]` skeleton，protocol checksum 字段非空；实际 stdout
  打印归入 M0.6 验收。
- 依赖扫描无输出。

#### M0.6 Host main 与最小 launch smoke

依赖任务：M0.2、M0.3、M0.4、M0.5。

文件范围：

- 创建 `host/main.cpp`
- 创建 `host/hccl_window.hpp`
- 修改 `CMakeLists.txt`
- 修改 `scripts/run_a3.sh`

任务：

- 以 `gemm_ar/main.cpp` 为主参考，复制后裁剪 host runtime 骨架：rank/device 选择、ACL/HCCL 初始化入口、
  stream 管理、host data generation、correctness/perf 控制台报告、fixed warmup/measure。
- host main 完成参数解析、layout 打印、数据生成入口、最小 kernel launch 或 dry-run。
- host main 是 M0 阶段 layout dump、`[CorrectnessReport]` skeleton 和 `[PerfReport]` skeleton 的唯一 required
  executable proof；不得为了 M0.4/M0.5 验收保留一次性临时 driver。
- host main 提供 `PerfReport` 数据结构和 stdout print skeleton，字段按“Correctness 与 E2E 性能记录规则”。
- `CMakeLists.txt` 增加 host executable target，并链接 M0.2 的最小 kernel library；如果当前只能 dry-run，也必须让
  build target 和 run script 的路径一致。
- `host/hccl_window.hpp` 在 M0 只提供 host 侧接口占位和 window size fail-fast 函数声明；真实 HCCL window bootstrap
  由 M1.5 实现。
- host 侧 HCCL/ACL runtime 组织可参考 `gemm_ar`，但 M0 不落算法通信协议，不把参考工程内部 tiling 结构当成本项目
  device protocol。
- `run_a3.sh` 调用 host main；默认允许 dry-run，不要求真实通信和 compute。
- 当前阶段可以不做真实通信和 compute。

验收标准：

- small case dry-run 能打印 shape、layout、用例名。
- small case dry-run 能打印 `[CorrectnessReport]` skeleton，protocol checksum 字段非空。
- small case dry-run 能打印 `[PerfReport]` skeleton，`warmup_iters/measure_iters/e2e_us` 字段存在。
- `[PerfReport]` skeleton 中 `warmup_iters=3`、`measure_iters=5`。
- CMake host executable target 可发现；若编译已接通，run script 使用同一 build 输出路径。
- 若 kernel launch 已接入，最小 kernel 正常返回。
- 依赖扫描无输出。

### 14.2 M1: PTO dispatch/combine protocol mock

#### M1.0 MegaMoE 主路径协议不变量核验

依赖任务：M0.4、M0.5；不硬依赖 M0.6。

文件范围：

- 修改 `host/reference.hpp`
- 按需修改 `DESIGN.md`

任务：

- 核验第 0-12 章已经写出的 MegaMoE 主路径子目标表：`RoutingMetadata`、`RoutePackQuantLocal`、
  `GatherDispatchToGmm1Input`、`RunActivationAndQuant`、`RunGmm2EpilogueAndReturn`、`RestoreOutput`。如果发现
  缺口，按 issue/DCL 流程补 `DESIGN.md`，不能在实现里另立协议。
- 对每个子目标写清“列入理由”和“不列入的相邻内容”，避免后续 agent 按参考源文件数量拆任务。
- 明确 small shape 只作为 correctness smoke case：必须走同一主路径语义并验证 metadata/payload/scale，不作为独立
  full-load fast path 或性能子目标。
- 核验 MegaMoE 通信策略不变量：routing/quant 输出在 peer-visible Shmem、Dispatch 前同步 +
  远端读、Combine 远端写、final unpermute 从 return payload/`offsetD` 读取。
- 记录文章级优化不变量：Dispatch-GMM 使用 scoreboard soft sync，GMM-Combine 使用 Sub-Tile/stride remote write；
  如果 PTO A3 无法表达 stride remote copy，必须标记 blocked。
- 在 host reference 中固定 metadata 不变量：`tokenPerExpertMatrix[tokenOwnerRank][expertOwnerRank][localExpert]`、
  `cumsumMM[tokenOwnerRankPrefix][localExpert]`、`preSumBeforeRank[tokenOwnerRank][localExpert]`、
  `expandedRowIdx[token, topK]`。
- M1.0 的验收只要求 host reference 能构造/打印上述 metadata；M0.6 host main/run script 是可复用的
  executable proof 能力，但不是 M1.0 的硬门槛。M1 四类 case 的可执行 stdout 回归由 M1.11 负责。
- 明确 capacity 策略：M1/M2 默认 fail fast；M3 原始等价路径必须支持 `maxOutputSize` clamp 或在 `DESIGN.md` 中明确
  仍未等价。

验收标准：

- `DESIGN.md` 有“MegaMoE 主路径子目标 -> PTO stage/backend”的表格，且每个子目标都有列入/不列入理由。
- `DESIGN.md` 明确 small shape 只做 correctness smoke，不作为独立性能目标。
- `DESIGN.md` 有“MegaMoE 文章通信策略 -> PTO stage”的表格，并明确 Dispatch 是 remote read、Combine 是 remote write。
- host reference 能构造/打印 small case 的 `tokenPerExpertMatrix`、`cumsumMM`、`preSumBeforeRank`；本项不要求
  host main/run script executable stdout proof。
- 任一后续 agent 能只读该表判断自己的任务是否破坏原协议。
- 依赖扫描无输出。

#### M1.1 Control metadata 数据结构

依赖任务：M0.4。

文件范围：

- 修改/扩展 `kernel/control_metadata.hpp`
- 修改 `include/moe_dispatch_combine_a8w8_layout.hpp`

任务：

- 定义 `tokenPerExpert`、`blockTokenPerExpert`、`blockPrefixPerExpert`、`expandedRowIdx`、`dispatchOffset` 的 view。
- 定义 `cumsumMM`、`preSumBeforeRank`、`expertTokenNums` 的 view。
- 定义 `countReadySignal`、`combineDoneSignal`、debug counter 的 view。
- metadata view 只做 address/view 计算，不做 payload copy。

验收标准：

- typed view 能从 layout offset 构造 typed pointer 或 view。
- `tokenPerExpert` view 的索引函数必须显式接收 `(tokenOwnerRank, expertOwnerRank, localExpert)`。
- signal/counter 与 payload field 不共用 cache line。
- 依赖扫描无输出。

#### M1.2 PTO primitive 直接调用约定

依赖任务：M0.2、M0.4。

文件范围：

- 创建或修改 `kernel/protocol_core.hpp`
- 创建或修改 `kernel/a3_int8_backend.hpp`

任务：

- 规定 PTO primitive 是 stage/backend 主流程的一等调用，不新增 `pto_payload_ops.hpp`。
- 在 `protocol_core.hpp` 的 stage 伪实现或注释中直接展示 `TLOAD/TSTORE/TGET/TPUT/TNOTIFY/TWAIT/TTEST`
  的调用位置。
- 在 backend 文件中直接展示 `TMATMUL/TMATMUL_ACC/TQUANT/TCVT/TMUL/TADD` 的调用位置。
- 只允许 layout/view 层负责 offset、shape、stride 和 `GlobalTensor` 构造；不能把 PTO 指令藏进通用 copy/comm wrapper。
- 当前任务只建立 stage/backend 主流程骨架和注释约定，不要求实现真实 GMM。

验收标准：

- 项目中不存在 `kernel/pto_payload_ops.hpp`。
- `TLOAD/TSTORE/TGET/TPUT/TNOTIFY/TWAIT/TTEST` 出现在具体 stage/backend 代码中，而不是只出现在通用 wrapper 中。
- layout/view 代码不调用 PTO data movement 或 comm primitive。
- 依赖扫描无输出。

#### M1.3 RouteLocalTokens

依赖任务：M1.1、M1.2。

文件范围：

- 修改 `kernel/protocol_core.hpp`
- 修改 `host/reference.hpp`

任务：

- device 侧实现 local token routing：`expertId/topK -> tokenPerExpert`。
- 生成 `expandedRowIdx` 和初始 `dispatchOffset`。
- `tokenPerExpert` 写入本 rank 作为 `tokenOwnerRank` 的 count row，目标索引是
  `(tokenOwnerRank=rankId, expertOwnerRank, localExpert)`。
- host reference 实现同样逻辑。

验收标准：

- single-rank small case 下，device `tokenPerExpertMatrix` 与 host reference 一致。
- topK=2 case 下，`expandedRowIdx` 保留 token 和 topK slot 信息。
- zero-token expert case 不写越界。
- 依赖扫描无输出。

#### M1.4 RoutePackQuantLocal mock/non-quant pack

依赖任务：M1.2、M1.3。

文件范围：

- 修改 `kernel/protocol_core.hpp`
- 修改 `host/reference.hpp`

任务：

- 在最终 stage `RoutePackQuantLocal` 中，把本 rank token 按 `expertOwnerRank/localExpert/row` 打包到本 rank
  peer-visible dispatch payload。M1 暂不做 dynamic quant；payload 可以保持 mock/FP 形态或当前 M1 约定形态。
- row offset 必须使用与 M1.6a `preSumBeforeRank[tokenOwnerRank=rankId][localExpert]` 同源的 global-expert-order
  prefix 语义；M1.4 可以先生成 local source-row ledger，但该 ledger 必须能被 M1.6a 无损校验和映射，不能引入另一套
  local-only row order。
- 本地 pack 在 `RoutePackQuantLocal` 主流程中直接调用 `TLOAD/TSTORE`。
- M2.2a 只能把同一 stage 的 payload 形态升级为 int8 dispatch payload + routing per-token scale；不能改变 row
  order、source ledger、ready signal 或新增独立 pack/quant/reorder stage。
- host reference 生成相同 packed layout。

验收标准：

- single-rank case 下 packed payload 与 host reference bitwise 一致。
- two-rank dry reference 下，每个 expert owner rank 的 row count 与 `tokenPerExpert` 一致。
- 代码/注释能看出该 stage 不向目标 rank 远端写 dispatch payload；目标 rank 后续通过
  `GatherDispatchToGmm1Input` `TGET` 读取。
- source-row ledger 能用 M1.6a 的 `preSumBeforeRank` 语义解释；M2.2a 只升级 payload dtype/scale，不改变协议边。
- 依赖扫描无输出。

#### M1.5 HCCL window host bootstrap

依赖任务：M0.4、M0.6。

文件范围：

- 修改 `host/hccl_window.hpp`
- 修改 `host/main.cpp`
- 修改 `scripts/run_a3.sh`

任务：

- 参考本仓 A2/A3 HCCL window 示例建立 rank/device/window bootstrap。
- host 检查 `peerWindowOffset + PeerWindowLayout.totalBytes <= winSize`。
- 把 device-visible HCCL context 和 local peer window pointer 传给 kernel。

验收标准：

- two-rank small case 能启动并打印 rank/window base/offset/size。
- window size 不足时 host 直接报错，不进入 kernel。
- 依赖扫描无输出。

#### M1.6 PublishCounts 与 WaitCounts

依赖任务：M1.1、M1.2、M1.3、M1.5。

文件范围：

- 修改 `kernel/protocol_core.hpp`
- 修改 `kernel/control_metadata.hpp`

任务：

- 把本 rank count row 发布到所有 peer 的 `tokenPerExpertMatrix`，作为 dispatch 远端读前同步的一部分。
- 发布完成后用 `TNOTIFY` 通知 peer。
- consumer 用 `TWAIT/TTEST` 等待 count row ready。
- allgather 结果必须形成每个 rank 都可读的
  `tokenPerExpertMatrix[tokenOwnerRank][expertOwnerRank][localExpert]`，不是只有 `[tokenOwnerRank][localExpert]`
  或 `[expertOwnerRank][localExpert]` 的二维表。

验收标准：

- two-rank balanced case 下，每个 rank 收到完整 count matrix。
- `tokenPerExpertMatrix[tokenOwnerRank][expertOwnerRank][localExpert]` 与 host reference 完全一致。
- timeout dump 能打印缺失的 token owner rank。
- 不能用 host barrier 替代 device wait。
- 依赖扫描无输出。

#### M1.6a BuildCumsumAndPreSumBeforeRank

依赖任务：M1.6。

文件范围：

- 修改 `kernel/protocol_core.hpp`
- 修改 `kernel/control_metadata.hpp`
- 修改 `host/reference.hpp`

任务：

- 按 token-owner-rank 维度为每个 local expert 计算 `cumsumMM[tokenOwnerRankPrefix][localExpert]`。
- 计算 `preSumBeforeRank[tokenOwnerRank][localExpert]`。语义必须同时对齐两件事：Dispatch 从 token owner
  peer window 远端读当前 global expert 的起始 row；Combine 写回 token owner `offsetD` 的起始 row。
- 写出 `expertTokenNums[localExpert] = cumsumMM[rankNum - 1][localExpert]`。
- 对 zero-token expert 生成合法的 0 row range，不发布无意义 payload wait。

验收标准：

- two-rank balanced/skewed case 下 `cumsumMM`、`preSumBeforeRank`、`expertTokenNums` 与 host reference 一致。
- `cumsumMM[rankNum - 1][localExpert]` 等于该 local expert 从所有 token owner rank 收到的 row 总数。
- `preSumBeforeRank` 能解释 Dispatch 源 row range 和 `RunGmm2EpilogueAndReturn` 的每一个 destination row offset。
- 依赖扫描无输出。

#### M1.7 GatherDispatchToGmm1Input mock sink

依赖任务：M1.2、M1.4、M1.6a。

文件范围：

- 修改 `kernel/protocol_core.hpp`
- 修改 `host/reference.hpp`

任务：

- 当前 rank 等待 count row 和 prefix metadata 后，在最终 stage `GatherDispatchToGmm1Input` 中按
  local expert/token owner rank 用 `TGET` 拉取 dispatch payload。
- `TGET`、count/prefix wait 和 `dispatchGroupReady` 必须在 device kernel 中真实执行；host reference/dry-run
  只能作为 precheck，不能替代 M1.7 验收。
- M1 允许先写入 `workspace.dispatchedA`，但它只表示 M1 mock payload target / debug mirror；不是新的
  `GatherDispatch` stage，也不是 M2 可以保留的二次 reorder buffer。
- 记录 `tokenOwnerRankOffsets` 供 mock expert 和后续 GMM 使用。
- 每个 local expert 的 row 排列必须与 `cumsumMM` 的 token-owner-rank 累加顺序一致。
- zero-token expert 的 row range 必须显式跳过，不发布无效 `TGET` 或 payload wait。
- M2.2b 只能在同一 stage 中把 payload target 扩展或切换为 `gmm1InputInt8 + routingPerTokenScale`；
  不能改变 source offset、row order、ready signal、producer/consumer 边，也不能新增独立 reorder stage。
- M1.7 task report 必须记录：`dispatchedA` is M1 mock sink/debug mirror, not a second reorder buffer.

验收标准：

- two-rank NPU/mpirun case 必须真实 launch kernel，并在 device path 中执行 `TWAIT/TTEST + TGET`。
- two-rank balanced case 下 `dispatchedA` 与 host reference bitwise 一致。
- skewed experts case 下 row offsets 单调且不越界。
- zero-token expert case 不发无效 `TGET`，timeout counter 保持为 0。
- 代码和 report 能证明 M1.7 使用最终 `GatherDispatchToGmm1Input` stage 边界；M2.2b 只替换 payload dtype/buffer，
  不替换协议和数据流。
- 依赖扫描无输出。

#### M1.8 MockExpertOutput

依赖任务：M1.7。

文件范围：

- 修改 `kernel/protocol_core.hpp`
- 修改 `host/reference.hpp`

任务：

- 用 deterministic mock 替代 GMM：例如 `expertOut = dispatchedA * alpha + beta` 或 identity。
- mock output layout 与后续 backend output layout 一致。
- mock output 必须按 `cumsumMM[rankNum - 1][localExpert]` 的 expert-major row range 写出。

验收标准：

- single-rank 和 two-rank mock expert output 与 host reference bitwise 一致。
- empty expert segment 不产生输出 payload。
- 依赖扫描无输出。

#### M1.9 RunGmm2EpilogueAndReturn mock return

依赖任务：M1.2、M1.6a、M1.8。

文件范围：

- 修改 `kernel/protocol_core.hpp`
- 修改 `kernel/control_metadata.hpp`

任务：

- 在最终 stage `RunGmm2EpilogueAndReturn` 中，按 token owner rank 把 mock expert output 用 `TPUT` 写回
  token owner peer window return payload。M1 暂不做真实 GMM2 epilogue；mock output 是该 stage 的 M1 input。
- `TPUT`、`TNOTIFY` 和 token owner rank 的 `TWAIT` 必须在 device kernel 中真实执行；host reference/dry-run
  只能作为 precheck，不能替代 M1.9 验收。
- 写回 offset 必须由 `tokenPerExpertMatrix[tokenOwnerRank][expertOwnerRank=rankId][localExpert]` 和
  `preSumBeforeRank[tokenOwnerRank][localExpert]` 推导，不能按当前本地 row 顺序硬编码。
- 写完某 owner segment 后发布 `combineDoneSignal`。
- M2.7 只能在同一 stage 中把 mock output input 替换为 GMM2 accumulator epilogue/cast 后的 payload；不能新增
  独立 `ReturnCombine` copy stage，也不能先全量写 `gmm2Out` 再二次搬运。
- M1.9 的 owner segment 是 token-owner rank 维度的连续 return row segment，用来证明真实 remote-write combine
  协议闭环；它不声称实现 GMM-Combine Sub-Tile/stride 通信。Sub-Tile owner mapping 属于 M2.7a，stride 或
  多段 `TPUT` 能力验证属于 M3.8。

验收标准：

- two-rank NPU/mpirun case 必须真实 launch kernel，并在 device path 中执行 `TPUT/TNOTIFY/TWAIT`。
- two-rank balanced case 下 token owner rank 能收到所有 return segment。
- return payload row offset 与 host reference 的 `preSumBeforeRank` 映射一致。
- signal producer count 与 expected expert owner rank count 一致。
- 代码和 report 能证明 M1.9 使用最终 `RunGmm2EpilogueAndReturn` stage 边界；M2.7 只替换 numeric payload
  producer，不替换 return protocol。
- 依赖扫描无输出。

#### M1.10 RestoreOutput

依赖任务：M1.9。

文件范围：

- 修改 `kernel/protocol_core.hpp`
- 修改 `host/reference.hpp`

任务：

- token owner rank 等待所有 `combineDoneSignal`。
- 按 `expandedRowIdx + probs` 做 weighted restore。
- 输出 `c[M, hidden]`。

验收标准：

- topK=1 bitwise 对齐 mock reference。
- topK=2 在定义 tolerance 内对齐 mock reference。
- zero-token expert case 不死锁。
- 依赖扫描无输出。

#### M1.11 M1 集成回归

依赖任务：M1.0 至 M1.10。

文件范围：

- 修改 `scripts/run_a3.sh`

任务：

- 把 M1 small、balanced、skewed、zero-token 四类用例串到 run script。
- M1.11 的通过命令必须包含真实 2 卡 NPU/mpirun launch；`--dry-run 1`、`--rank-from-mpi 0` 的 direct-host
  reference suite 只允许作为 build/precheck，不能让 M1.11 到 `review_ready`。
- 每个用例打印 `[CorrectnessReport]` 和 `[PerfReport]`；M1 的 GMM checksum 字段允许为 `null`，protocol checksum 必须非空。
- 若实际验收命令或输出字段与本章设计不一致，先按 issue/DCL 流程修正 `DESIGN.md`，不能只在 `TASKS.md`
  追加说明。

验收标准：

- 至少 balanced 2 卡 NPU/mpirun case 真实实跑通过，且 dispatch/combine primitive counters 或 white-box dump
  能证明 `TGET/TPUT/TNOTIFY/TWAIT` 在 device path 中执行。
- M1 四类用例全部通过。
- 四类用例的 `[CorrectnessReport]` 中 `pass=true`，`[PerfReport]` 中 `correctness_pass=true`，E2E samples 非空。
- 任一失败能定位到 stage 和 rank。
- 依赖扫描无输出。

### 14.3 M2: A3 int8_int8 MegaMoE-ready data path

#### M2.0 Active runtime baseline 与 primitive preflight

依赖任务：M1.11。

文件范围：

- 修改 `CMakeLists.txt`
- 修改 `scripts/run_a3.sh`
- 修改 `host/main.cpp`
- 修改 `host/args.hpp`
- 修改 `host/reference.hpp`
- 修改 `host/comm_mpi.hpp`
- 修改 `host/hccl_context.hpp`
- 修改 `kernel/moe_dispatch_combine_a8w8_kernel.cpp`
- 修改 `kernel/kernel_launchers.hpp`
- 修改 `include/moe_dispatch_combine_a8w8_runtime_types.hpp`
- 修改 `include/moe_dispatch_combine_a8w8_m1_layout.hpp`

任务：

- 把 M1 已实跑的真实 device runtime 归一到 M2 预期主路径：host 入口在 `host/main.cpp`，kernel 入口在
  `kernel/moe_dispatch_combine_a8w8_kernel.cpp`，shared ABI/header 在 `include/`；`m1_runtime/` 不再作为 active
  source tree。
- CMake active target 必须编译上述 `host/`、`kernel/`、`include/` 文件；后续 M2 task 只能在这些 active path
  上开发，不能新增一套不参与构建的 runtime。
- 保持 M1 real dispatch/combine 行为不退化：routing、pack、count publish/wait、prefix、dispatch `TGET`、
  identity mock expert bridge、combine `TPUT`、notify/wait/test、restore 仍在 device path 执行。
- 记录 arch gate：当前 A3/CANN 8.5 下单个 `dav-c220` raw mixed target 会让 PTO Vec/comm primitive 编译失败，
  因此 M2 不把这条构建方式作为 acceptance 前提；dispatch/combine target 仍使用 `dav-c220-vec`，GMM primitive
  通过 M2.3/M2.6 cube target 进入主链。M2.8 的 single fused MPMD 必须走 PTO mixed ELF/metadata 双侧编译路线。
- 记录 `TMATMUL int8 x int8 -> int32` preflight 来源：优先用仓内
  `tests/npu/a2a3/src/st/testcase/tmatmul/tmatmul_kernel.cpp` 和
  `kernels/manual/a2a3/dispatch_gmm_combine_v2/op_kernel/compute/pto_gmm_block_int8.hpp`，先确认 A3 PTO
  tile dtype、valid shape、arch 和 `TSTORE`/`TSTORE_FP` 出口策略，再实现 M2.3/M2.6 主路径。

验收标准：

- `rg --files kernels/manual/a2a3/moe_dispatch_combine_a8w8/m1_runtime` 无 active 文件；非报告 markdown 中没有 active
  `m1_runtime` 引用。
- `CMakeLists.txt` 编译 `kernel/moe_dispatch_combine_a8w8_kernel.cpp` 和 `host/main.cpp`。
- `scripts/run_a3.sh --m1-suite 1 --dry-run 0 --skip-kernel-launch 0` 从 device 4 跑通 small、balanced、skewed、
  zero-token；两卡 case 的 `[CorrectnessReport]` 中 `two_rank_npu_run=true`、`pass=true`。
- M2 task report 或 TASKS DCL 记录 mixed target gate 和 GMM primitive preflight 结论；如果 `TMATMUL` 或
  `TSTORE_FP` 出口能力不足，M2.3/M2.6/M2.7 只能标 `primitive-gap`，不能引入 AscendC/Catlass fallback。

#### M2.1 Int8 backend layout 与接口

依赖任务：M2.0。

文件范围：

- 修改 `kernel/a3_int8_backend.hpp`
- 修改 `include/moe_dispatch_combine_a8w8_layout.hpp`
- 修改 `kernel/protocol_core.hpp`

任务：

- 定义 `A3Int8Backend` 接口和 M2 fused data path stage 名称。
- `include/moe_dispatch_combine_a8w8_layout.hpp` 是 M2/M3 canonical layout；迁移保留的
  `include/moe_dispatch_combine_a8w8_m1_layout.hpp` 只承载 M1 compatibility path。新增 M2 字段必须进入 canonical
  layout，并在 host dump/device typed view 中使用，不能在两套 layout 中各自演进。
- M2.1 必须在 active runtime 中建立明确的 backend/layout 选择边界：`m1-mock` 或默认兼容路径可以继续使用
  `DispatchCombineTileShape + moe_dispatch_combine_a8w8_m1_layout.hpp`；`int8` 后端必须把
  `DispatchCombineTileArgs` 映射到 `ShapeConfig/RankConfig`，并使用 `moe_dispatch_combine_a8w8_layout.hpp`
  计算 workspace、peer window、HCCL_BUFFSIZE、host dump 和 device typed view。不能只新增一套未参与构建/运行的
  canonical layout。
- 如果 M2.1 引入 `--backend m1-mock|int8` 或等价开关，M1 compatibility path 必须保持当前四类 M1 regression 可跑；
  M2.8 只负责把 `int8` 后端串成完整回归，不再重新定义 backend 选择语义。
- 增加 `gmm1InputInt8`、`gmm1WeightInt8`、`gmm1AccInt32`、`gmm1Out`、`swigluOut`、
  `gmm2InputInt8`、`gmm2WeightInt8`、`gmm2AccInt32` layout；`gmm2Out` 只能作为 debug mirror，不能成为
  M2 主 combine path 的必经中间结果。
- 增加 `routingPerTokenScale`、`gmm2PerTokenScale`、`scale1Uint64`、`scale2Uint64` layout 字段或 typed view。
- peer window layout 必须显式包含 int8 dispatch payload、routing per-token scale、`offsetD` return payload 和
  combine signal；这些字段在 M2 就按 M3 overlap 的最终 row/offset 语义使用。
- control layout 必须显式包含第 10/11 章需要的 `dispatchGroupReady`、`gmm1SyncGroupReady`、
  `activationSyncGroupReady`、`gmm2GroupReady`、`swigluSyncGroups`、`dequantSum`、`subTileReady` typed view
  或预留字段。M3 只能启用/校验这些字段，不能重新定义其 row range 或 offset 语义。
- 当前 M2 明确固定 `dtype_in` / `dtype_out`，例如 FP16 输入、FP16 输出；如果选择 BF16，必须在 M2.1 task
  report 摘要里写清并保持 M2.2-M2.8 一致。M2 最终输出不能是 int8。
- 当前 M2 weight layout 只接一种显式布局，建议先用 ND/row-major int8；遇到 FRACTAL_NZ 或其他 layout 必须
  fail-fast，不能静默 reinterpret。
- protocol core 通过 stage 名称调度，不直接写 int8 GMM 细节；`a3_int8_backend.hpp` 的
  `RunGmm1/RunGmm2/RunActivationAndQuant/RunGmm2EpilogueAndReturn` 主流程必须直接调用
  `TMATMUL/TQUANT/TPUT` 或等价 PTO primitive 序列。

验收标准：

- M1 mock backend 仍可编译运行。
- 切换 backend 不影响 dispatch/combine protocol API。
- `--backend int8` 或等价 runtime path 使用 canonical M2 layout 计算 workspace/window size；`--backend m1-mock`
  或默认兼容路径仍能走迁移后的真实 dispatch/combine M1 suite。
- layout dump 中能看见 peer-visible int8 dispatch payload、routing/GMM2 per-token scale、`offsetD` return payload、
  `scale1/scale2` typed view。
- layout dump 中能看见 `dispatchGroupReady/gmm1SyncGroupReady/activationSyncGroupReady/gmm2GroupReady`、
  `swigluSyncGroups/dequantSum`、`subTileReady` 的字段或预留字段。
- 依赖扫描无输出。

#### M2.2 Int8 weight/scale reference

依赖任务：M2.1。

文件范围：

- 修改 `host/args.hpp`
- 修改 `host/reference.hpp`
- 修改 `host/main.cpp`
- 修改 `include/moe_dispatch_combine_a8w8_runtime_types.hpp`
- 修改 `scripts/gen_data.py`

任务：

- 生成 int8 `weight1/weight2`。
- 生成 `scale1/scale2` per-channel dequant scale，数据语义按 A8W8 主路径的 `uint64_t` scale view 固定；
  reference 必须把 scale 的十六进制 bit pattern 和 float 解释都打印出来。
- 生成 routing dynamic quant 的 per-token scale reference：`scaleTemp = reduceMax(abs(row)) / 127.0f`，
  zero row 的 scale/quant 策略必须固定。
- 生成 GMM2 per-token scale reference，供 M2.5/M2.7 复用。
- CPU reference 计算 GMM1/GMM2 int32 accumulator，并显式模拟 accumulator -> scale dequant -> cast。
- `host/reference.hpp` 是 M2 runtime reference 真值；`scripts/gen_data.py` 只能生成同语义离线数据和白盒 dump。
  两者的 seed、rounding、zero row、scale bit pattern 必须一致，不能出现 Python 一套、C++ runtime 一套。
- active runtime 必须有 weight/scale 的显式 host 数据结构、device allocation/copy 生命周期和 checksum/report 字段；
  不能只在 `scripts/gen_data.py` 生成离线文件后让 device path 仍无权重输入。
- M2.2 固定的 weight/scale seed、rounding、zero-row 策略和 `uint64_t` scale bit pattern 必须被 `--backend int8`
  runtime 与离线脚本共用；如果某个 shape/布局不支持，host parse/validate 要 fail-fast。

验收标准：

- small shape 下 CPU int32 accumulator 可打印 checksum。
- 同 seed 下 weight、scale1/scale2 bit pattern、routing per-token scale 完全一致。
- task report 摘要记录 weight layout、scale layout、dtype_in、dtype_out；不允许隐藏 bias。
- 依赖扫描无输出。

#### M2.2a RoutePackQuantLocal fused dispatch input

依赖任务：M1.11、M2.1、M2.2。

文件范围：

- 修改 `kernel/a3_int8_backend.hpp`
- 修改 `kernel/protocol_core.hpp`
- 修改 `host/reference.hpp`
- 修改 `host/main.cpp`

任务：

- 实现 `RoutePackQuantLocal`，把 M1 的 `RouteLocalTokens + RoutePackQuantLocal mock/non-quant pack`
  扩展为 MegaMoE 输入侧合并路径：
  读取原始 `x/expertIdx/probs`，生成 `expandedRowIdx`、本 rank count row、peer-visible int8 dispatch payload 和
  `routingPerTokenScale`。
- 对齐主路径 routing dynamic quant 语义：按 expanded row 做 `reduceMax(abs(row))/127.0f`，
  写 int8 hidden row 到本 rank peer window 的 dispatch payload，写 float scale 到 peer-visible scale 区。
- 这个 stage 不能先生成一个独立 expanded FP payload 再二次转换成 int8 dispatch payload；允许 debug dump，但主路径必须是
  route/pack/quant 一次落入 peer-visible dispatch layout。
- row 顺序必须能由 `tokenPerExpertMatrix`、`cumsumMM`、`preSumBeforeRank` 解释，并和 M1 的 mock row-order 一致。
- 打印 small/topK=2/skewed case 的 peer dispatch payload checksum、routing per-token scale checksum、row-order dump。
- small shape 只作为 correctness smoke：必须复用同一 `RoutePackQuantLocal` 主路径，不新增独立 fast path 或性能验收。

验收标准：

- single-rank small case 下 peer-visible int8 dispatch payload 与 host routing-quant reference bitwise 一致。
- two-rank balanced/skewed case 下，peer dispatch row order 与 M1 row ledger / `cumsumMM` 一致。
- `routingPerTokenScale` 与 host reference 在固定 tolerance 内一致，zero-token expert 不产生非法 scale。
- white-box dump 能证明没有主路径二次 reorder workspace；若为了 debug 保留 dump buffer，必须在 report 标注为非主路径。
- small shape 的验收只看 metadata、payload、scale correctness，不要求单独 latency 优于 normal path。
- mock backend 仍可运行；protocol core API 不因为 int8 backend 改变。
- 依赖扫描无输出。

#### M2.2b GatherDispatchToGmm1Input

依赖任务：M1.6a、M2.2a。

文件范围：

- 修改 `kernel/protocol_core.hpp`
- 修改 `kernel/a3_int8_backend.hpp`
- 修改 `host/reference.hpp`
- 修改 `host/main.cpp`

任务：

- 实现 `GatherDispatchToGmm1Input`：expert owner rank 前同步后按 token owner rank 用 `TGET` 读取 peer-visible
  int8 dispatch payload，直接写入本 rank `gmm1InputInt8/gmA` 的 expert-major contiguous row range。
- 同一个 stage 同步读取/落位 `routingPerTokenScale` 到本 rank GMM1 epilogue 可消费的 per-token scale layout。
- rowStart 必须来自 `cumsumMM[tokenOwnerRankPrefix][localExpert]`，owner 源 offset 必须来自
  `preSumBeforeRank[tokenOwnerRank][localExpert]` 或 M2.2a 记录的等价 source row ledger。
- M2.2b 是 M1.7 `GatherDispatchToGmm1Input` 的 payload 形态升级：允许复用 M1 已固定的 row order、source offset、
  ready signal 和 producer/consumer 边，但必须把主 payload 直接落到 `gmm1InputInt8 + routingPerTokenScale`。
  不能把远端 payload 先拉到 `dispatchedA`，再由后续二次 GMM1-input stage 做 reorder；`dispatchedA` 只能作为
  M1 mock sink / debug mirror。

验收标准：

- two-rank balanced case 下 `gmm1InputInt8/gmA` 与 host reference bitwise 一致。
- skewed case 下每个 local expert 的 row range 单调、不越界，`routingPerTokenScale` row order 与 `gmm1InputInt8` 一致。
- zero-token expert 不发无效 `TGET`，timeout counter 为 0。
- white-box dump 记录 remote source offset、local `gmm1InputInt8` rowStart、scale rowStart 三者映射。
- 依赖扫描无输出。

#### M2.2c Dispatch-GMM soft-sync ledger

依赖任务：M2.2b。

文件范围：

- 修改 `include/moe_dispatch_combine_a8w8_layout.hpp`
- 修改 `kernel/control_metadata.hpp`
- 修改 `kernel/protocol_core.hpp`
- 修改 `kernel/a3_int8_backend.hpp`
- 修改 `host/reference.hpp`
- 修改 `host/main.cpp`

任务：

- 在 M2 固定 Dispatch-GMM soft-sync 的语义账本，而不是等 M3.7 再设计依赖粒度。
- 增加 `scoreboardTaskMap`、producer status、`scoreboardMinStatus`、worker wait counter、scoreboard timeout counter
  的 layout 和 host/device typed view。这里的 `scoreboardMinStatus` 表示 consumer dependency domain 的聚合 ready
  状态，不是一个全局递增 task id 的最小值。
- task id 必须能映射到 token owner rank segment、local expert group 或 row/tile range；不能只是无语义递增计数。
- M2 必须定义 dependency domain：GMM1 consumer 实际等待的是一个 local expert 或 `GmmTileTask`，该 domain
  依赖哪些 token owner segment 必须能从 `scoreboardTaskMap` 反查；zero-row segment 必须标记 skip/done，不能产生
  永远不可完成的依赖。
- producer status 的最终 publish 时机必须固定为：对应 segment 的 `TGET` payload 和 routing scale 已写入
  `gmm1InputInt8/routingPerTokenScale`，且满足 GM 可见性后才能写 done。M2 ledger-only 可以顺序记录该语义，但不能把
  `rows > 0` 本身当成最终 ready 条件。
- M2 可以保持 `overlap_mode=off`、BSP wait 或 soft-sync ledger-only 执行，但 `GatherDispatchToGmm1Input` 到
  `RunGmm1` 的依赖必须能由账本解释，并且 call site、signal/counter、task id 映射就是 M3.7 要打开的最终结构。
- 打印 small/balanced/skewed case 的 task map、producer status 初末值、dependency domain 到 producer segment
  的映射、`scoreboardMinStatus` 参考更新顺序和 worker wait plan；zero-token expert 必须有 skip 语义。
- 若发现 PTO public sync primitive 不能支撑后续 async scoreboard，只能在本 task report 摘要里写 M3.7
  primitive-gap gate，不能改成 AscendC flag 或 host barrier。

验收标准：

- white-box dump 能把每个 Dispatch-GMM task id 映射回 expert group 或 row/tile range。
- producer status、`scoreboardMinStatus`、worker wait counter、timeout counter 的 layout 与 cache-line 隔离规则可见。
- white-box dump 能证明每个 GMM1 consumer domain 只等待自身依赖的 producer segment；不能用全局 min task id
  阻塞无关 expert/tile。
- producer status 的说明必须区分 `rows > 0`、payload/scale copy done 和 skip/done；如果当前代码只是 ledger-only
  mirror，report 摘要必须写明 M3.7 需要把 publish 移到 payload 可见之后。
- counter-log 能说明 M2 当前是 `soft_sync_mode=off` 或 `soft_sync_mode=ledger_only`，并给出 M3.7 将要打开的
  scoreboard path；不能以另一套临时 barrier 代替该账本。
- zero-token expert 不创建不可完成 task；skewed case 的 task range 不越界。
- 依赖扫描无输出。

#### M2.GMM shared tile/pipeline contract

本小节约束 M2.3 和 M2.6 的共同 GMM 实现方式。共享 scheduler/preflight 由 M2.3a 单独领取和验收；
M2.3/M2.6 只能消费这套结构，不能各自实现一套 fixed-shape GMM path。

MC2 W4A8 代码里 GMM 的有效经验是两层切分：外层按 expert 内的 L1 block 分配 AIC work，内层把一个 L1 block
再切成 L0A/L0B/L0C 小矩阵，并用 L1/L0 ping-pong 覆盖搬运和 MMAD。当前 PTO A3 int8 版本不复制 Catlass/AscendC
接口，但必须吸收这个设计原则，不能只做固定 smoke shape 的单次 `TMATMUL`。

本项目的 M2 GMM 不是 arbitrary-shape GEMM library。M2 只服务 MegaMoE A8W8/int8 realistic model shapes，
不为极端小 K/N、非对齐 hidden size 或任意 odd shape 消耗主路径复杂度。small shape 只保留为 smoke/debug，
不能作为 M2.GMM cache-level tile/pipeline 验收依据。

M2.GMM 主切分必须基于 `kernels/manual/a2a3/gemm_ar` 的 cache-aware 规则，而不是临时 PTO micro-tile：

```text
gmm_tile_policy=gemm_ar_cache_level_int8
baseM=128
baseN=256
baseK=64
stepK=4
```

这些值的含义：

- 外层 AIC work item 是一个 `baseM x baseN` output tile，M/N tail 用 valid shape 控制。
- K 维以 `baseK=64` 为 L0 slice，L1 一次缓存 `stepK=4` 个 slice。
- L1 A panel 是 `baseM x (baseK * stepK)`；L1 B panel 是 `(baseK * stepK) x baseN`。
- L0A/L0B 分别是 `baseM x baseK` 和 `baseK x baseN`，二者都要 ping-pong。
- L0C 是 `baseM x baseN` int32 accumulator，A2/A3 下 `128 x 256 x 4 = 128 KiB`，刚好贴 L0C 可见容量，
  因此 L0C 主路径单 buffer 累加，不能再要求 L0C double buffer。

基于当前知识库的 PTO 可见容量表，A2/A3 下 `Mat/L1=512 KiB`、`Left/L0A=64 KiB`、`Right/L0B=64 KiB`、
`Acc/L0C=128 KiB`。int8 主路径预算为：

```text
L0A single = 128 * 64 * 1  = 8 KiB
L0B single = 64 * 256 * 1  = 16 KiB
L0A/L0B double buffer stays within their 64 KiB spaces
L0C single = 128 * 256 * 4 = 128 KiB
L1 A ping/pong = 2 * 128 * (64 * 4) * 1 = 64 KiB
L1 B ping/pong = 2 * (64 * 4) * 256 * 1 = 128 KiB
L1 total = 192 KiB <= 512 KiB
```

实现如果继续保留 `16x32x32` 或其他小 tile，只能作为 PTO `TMATMUL` primitive preflight、debug fallback 或
micro-tile probe，不能作为 M2 accepted GMM tile。accepted GMM 必须报告并使用上述 cache-level tile policy。

M2.GMM shape 约束：

- `hiddenSize` 必须是 `baseK=64` 的整数倍。
- `intermediateSize` 必须是 `baseK=64` 的整数倍。
- GMM1 的 `N = 2 * intermediateSize`，GMM2 的 `N = hiddenSize` 可以按 `baseN=256` 做 N-tail，但正式验收
  不以过小 N 作为主证据。
- M-tail 来自 expert rows，可以小于 `baseM=128`；这是 MoE row distribution 的正常 tail，必须用 valid shape 处理。
- 不支持的 hidden/intermediate shape 必须 fail-fast 并在结构化输出里标明 `gmm_shape_class=unsupported`，
  不能静默切到临时小 tile。

GMM tile plan 必须满足：

- 外层 work item 由 runtime shape 推导：`GmmTileTask{whichGmm, localExpert, mBegin, mCount, nBegin, nCount,
  kSize, tileId}`。`mBegin/mCount` 来自 `cumsumMM[rankNum - 1][localExpert]` 和 activation row range，
  `nBegin/nCount` 来自 GMM1 intermediate 或 GMM2 hidden 维度，`kSize` 来自 hidden 或 intermediate 维度。
- task 粒度按 `baseM=128/baseN=256` 的 cache-level output tile 表达。`gmmBlockM/N/K` 只能作为 requested
  policy 入参，device path 必须打印 requested/effective policy；若入参不匹配 M2 policy，则 fail-fast 或明确标成
  unsupported，不能让用户误以为 arbitrary block size 已受支持。
- 每个 AIC block 用 `blockIdx`/grid-stride 或等价方式遍历 `GmmTileTask`。不能用 `if (blockIdx != 0) return`
  屏蔽多 AIC；zero-token expert 可以 skip，但必须在 task/counter 中可见。
- L1 block 内按 K 维分块累加。参考 `gemm_ar` 的 `stepKa=stepKb=4` 形态：每 `stepK` 个 K-slice 做一次
  GM -> L1 `TLOAD`，后续逐 slice `TEXTRACT/TMOV` 到 L0A/L0B，再执行 `TMATMUL/TMATMUL_ACC`。
  结构化输出必须显式记录 `baseM/baseN/baseK/stepK`、L1/L0 bytes、`l1_stages`、`l0a_stages`、
  `l0b_stages`、`l0c_stages`。
- L1/L0 staging 必须有明确生命周期：GM -> L1/GM staging -> L0A/L0B -> `TMATMUL/TMATMUL_ACC` -> L0C/int32
  accumulator -> GM accumulator。PTO primitive 必须在 GMM stage 主流程直接出现；不能通过通用 helper 隐藏
  `TLOAD/TSTORE/TMATMUL`。
- 双缓冲首版至少要在计划和状态机上成立：L1 A/B payload 使用 2-stage ping-pong；L0A/L0B 使用 2-stage
  ping-pong；L0C 单 buffer 累加同一 output tile。若 PTO 当前只能表达其中一部分，必须把 M2.3/M2.6 标成
  `primitive-gap` 或保留 `needs_fix`，不能把单 buffer smoke 当成 accepted GMM。
- tail 必须由 valid shape 控制。M tail、N tail、K tail 的 padding 不能污染 int32 accumulator checksum；
  对 zero-token expert 不发无效 `TMATMUL`。K tail 不作为 M2 realistic-shape 主目标；hidden/intermediate 不满足
  `baseK=64` 对齐时 fail-fast。
- GMM1 和 GMM2 可以共用同一 tile scheduler/numeric launcher，但不能把 PTO 调用封装到看不见的 helper。允许
  使用小的 typed struct 描述 task 和 tile shape；不允许做隐藏 Catlass/AscendC fallback 的 wrapper。
- M2.GMM 当前只要求 correctness 和结构证据，不要求证明 L1/L0 双缓冲带来性能收益；性能 overlap 属于 M3。

控制台结构化输出至少要包含以下 GMM 结构字段，report 只摘要 pass/fail 和关键数值：

```text
gmm_tile_policy=gemm_ar_cache_level_int8
gmm_runtime_shape=true
gmm_multiblock=true
gmm_shape_class=realistic_model_shape|smoke_debug|unsupported
gmm_base_m=128
gmm_base_n=256
gmm_base_k=64
gmm_step_k=4
gmm_l1_a_bytes=<...>
gmm_l1_b_bytes=<...>
gmm_l0a_bytes=<...>
gmm_l0b_bytes=<...>
gmm_l0c_bytes=<...>
gmm_l1_stages=2
gmm_l0a_stages=2
gmm_l0b_stages=2
gmm_l0c_stages=1
gmm_tile_tasks=<count>
gmm_active_aic_blocks=<count>
gmm_tail_m/n/k=<covered|none>
gmm_micro_tile_debug_only=<true|false>
```

M2.3/M2.6 的验收不能只看 accumulator 数值。必须同时证明：

- runtime shape 驱动 task 生成；
- 多 expert 或多 tile case 下有多个 work item；
- 至少一个 4-rank realistic-shape balanced 或 skewed case 下多个 AIC block 参与；
- L1/L0 cache-level tile shape、bytes、stage 数和 tail 覆盖在控制台结构化输出中可见；
- int32 accumulator 与 host reference 精确对齐。

#### M2.3a GMM shared scheduler and primitive preflight

依赖任务：M2.1、M2.2、M2.2b、M2.2c。

文件范围：

- 修改 `kernel/a3_int8_backend.hpp`
- 修改 `kernel/kernel_launchers.hpp`
- 修改 `host/main.cpp`
- 修改 `host/reference.hpp`

任务：

- 定义 `GmmTileTask{whichGmm, localExpert, mBegin, mCount, nBegin, nCount, kSize, tileId}` 和共享
  GMM tile config；M2.3/M2.6 必须通过该 task schema 生成 work，不得各自硬编码 fixed smoke shape。
- 基于 runtime shape、`cumsumMM[rankNum - 1][localExpert]`、M2.5 activation group row range 和
  `gemm_ar` cache-level policy (`baseM=128/baseN=256/baseK=64/stepK=4`) 生成 GMM1/GMM2 task preview。
  `gmmBlockM/N/K` 作为 requested policy 只允许匹配该主策略；不匹配时必须 fail-fast 或报告 unsupported。
- 完成 PTO `TMATMUL<int32_t, int8_t, int8_t>` primitive preflight，记录可用 tile dtype、valid shape、
  accumulator store 出口和 `dav-c220-cube` 或等价目标；失败时标 `primitive-gap`，不能引入 AscendC/Catlass fallback。
- 定义 L1/L0 cache-level tile shape、bytes、stage 数和 ping-pong 状态字段，至少包括
  `baseM/baseN/baseK/stepK`、`l1_a_bytes/l1_b_bytes`、`l0a_bytes/l0b_bytes/l0c_bytes`、
  `l1_stages`、`l0a_stages`、`l0b_stages`、`l0c_stages`。
- 定义 multi-AIC work partition 规则：每个 AIC block 用 `blockIdx`/grid-stride 或等价方式遍历
  `GmmTileTask`；zero-token expert skip 必须可见。
- 输出 host/device 共享 dump 字段，供 M2.3/M2.6/M2.8 证明 `gmm_runtime_shape=true` 和
  `gmm_multiblock=true`。

验收标准：

- small case 只作为 smoke/debug；M2.3a accepted 证据必须包含至少一个 realistic-shape 4-rank case 的
  GMM1/GMM2 task preview，row/tile range 不越界。
- 至少一个 realistic-shape case 的 task preview 包含多个 cache-level tile 或多个 expert work item。
- preflight 指向实际编译过的 PTO `TMATMUL` call site 或明确 `primitive-gap`。
- 结构化输出包含 shared contract 要求的 cache-level tile policy、L1/L0 bytes、stage、task count、active block
  预算字段。
- 依赖扫描无输出。

#### M2.3 RunGmm1 PTO int8 matmul

依赖任务：M2.1、M2.2、M2.2b、M2.2c、M2.3a。

文件范围：

- 修改 `kernel/a3_int8_backend.hpp`
- 修改 `host/main.cpp`

任务：

- 用 PTO `TMATMUL` / `TMATMUL_ACC` 实现 GMM1 `int8 x int8 -> int32`。
- 实现前必须完成 M2.0/M2.3 preflight：确认 A3 `TMATMUL<int32_t, int8_t, int8_t>` 的 tile dtype、valid shape、
  `dav-c220-cube` 或等价 arch、以及 accumulator 出口策略。不能在 `dav-c220-vec` target 内直接加入
  `TMATMUL` 后假定可编译。
- 如果采用独立 cube target 承载 M2 numeric stage，该 target 必须属于本项目 CMake，使用 PTO-only
  `dav-c220-cube` 编译，并由 active host runtime 以 overlap-off numeric stage 方式调用；dispatch/combine 仍保持
  `dav-c220-vec` target。不能把它写成未接入主链路的 preflight demo，也不能退回 `dav-c220` mixed target。
- 仓内 `dispatch_gmm_combine_v2` 的 `pto_gmm_block_int8.hpp` 只能作为 PTO tile/shape/`TMATMUL` 写法参考；其中
  AscendC flag/helper 不能复制进本项目。需要同步时使用本项目允许的 PTO/public flag/event 写法。
- 支持 local expert-major segment。
- 必须实现并使用 `M2.GMM shared tile/pipeline contract` 的 GMM tile scheduler。GMM1 work item 的 M 维来自
  local expert row range，N 维来自 GMM1 output intermediate 维度，K 维来自 hidden size；不能只把当前
  `hiddenSize/intermediateSize` 写死进 kernel。
- GMM1 的 L1/L0 tile shape、bytes 和 stage 数必须由同一份 GMM config 暴露给 host dump 和 device path。
  M2 主策略固定为 `gemm_ar` cache-level policy (`baseM=128/baseN=256/baseK=64/stepK=4`)；PTO micro-tile
  只能是该策略内部的 primitive 实现细节，不能替代外层 cache-level task。
- GM -> tile staging -> `TMATMUL/TMATMUL_ACC` -> int32 accumulator store 的状态机必须体现 L1 A/B ping-pong、
  L0A/L0B ping-pong 和 L0C 累加 buffer。如果 PTO 当前只能表达其中一部分，必须记录 primitive-gap，不能用
  单次 fixed tile 关闭 M2.3。
- 每个 expert group 的 `currentM` 来自 `cumsumMM[rankNum - 1][localExpert]`；M1/M2 如采用 fail-fast capacity，
  必须在超过 `maxOutputSize` 时报告错误，不允许静默截断。
- `TMATMUL` 的 A/B/C tile dtype 必须显式是 `int8/int8/int32`，不能通过 helper 或 fallback 隐藏。
- 处理 M/N tail valid region；K 维 realistic shape 必须满足 `hiddenSize % 64 == 0`，不满足时 fail-fast。
- 固定 PTO `TMATMUL` micro-tile 尺寸可以作为底层实现选择，但 accepted 主路径不能退化为固定
  `hiddenSize/intermediateSize` smoke shape、小 micro-tile fallback 或单 AIC block demo；GMM1 的 M/K/N 必须由
  runtime shape 与 expert row range 驱动。
- 多 block launch 不能被 `get_block_idx() != 0` 之类 guard 直接屏蔽。M2.3 accepted 口径至少要按 expert/tile
  维度把 GMM1 work partition 到可用 AIC blocks；若 PTO primitive 或编译目标暂时无法支持，必须把 M2.3/M2 stage
  标成 `needs_fix` 或 `primitive-gap`，不能用单 block correctness smoke 关闭任务。

验收标准：

- single-rank small shape 下 GMM1 int32 accumulator 与 CPU reference 完全一致，但它只作为 smoke/debug。
- 4-rank realistic-shape balanced/skewed case 下每个 local expert 的 accumulator checksum 与 reference 一致。
- 至少一个 4-rank realistic-shape case 的 GMM1 device path 证明多个 AIC block 参与并覆盖全部有效 cache-level
  tile；若只能单 block 运行，
  M2.3 只能作为 preflight，不得 `review_ready`。
- `[CorrectnessReport]` 或等价结构化 stdout 必须打印 `gmm_runtime_shape=true`、`gmm_multiblock=true`、
  `gmm_tile_policy=gemm_ar_cache_level_int8`、base tile、L1/L0 bytes、stage 数、tile task count、active AIC
  block count、shape class 和 tail 覆盖结果。
- 至少一个 realistic-shape case 覆盖 N 或 K 方向超过单个 cache-level tile/stepK panel 的分块累加；不能只用
  small smoke。
- tail shape 不越界。
- preflight 记录必须指向实际编译过的 PTO `TMATMUL` call site 或仓内已验证 ST/reference；若失败，任务状态改为
  `blocked`/`primitive-gap`，不改用 AscendC/Catlass fallback。
- 依赖扫描无输出。

#### M2.4 GMM1 epilogue

依赖任务：M2.3。

文件范围：

- 修改 `kernel/a3_int8_backend.hpp`
- 修改 `host/reference.hpp`
- 修改 `host/main.cpp`

任务：

- 对 GMM1 int32 accumulator 按 `scale1` 做等价 dequant、cast 到 activation workspace；当前 int8 主路径不加 bias。
- scale index 必须对齐 A8W8 scale view：`arrayGroupIdx = listLen == 1 ? 0 : localExpert`，scale offset 随 GMM1 N tile
  移动；当前若只支持 `listLen == 1` 必须 fail-fast 并写进 report。
- device 主路径必须用 PTO Vec 表达 scale-dequant/cast：`TLOAD` int32 accumulator 和 scale tile，`TCVT/TMUL`
  或等价 PTO Vec 序列生成 FP32 activation workspace，再 `TSTORE`。逐元素 scalar GM loop 只能作为 correctness
  preflight/debug path，不能关闭 M2.4/M2.8 的 PTO data-plane gate。
- host reference 实现同样 epilogue。

验收标准：

- GMM1 epilogue output 与 host reference 在 dtype tolerance 内一致。
- scale 为 1 或 2 的 sanity case bitwise 或近似一致；reference 不生成 bias。
- `[CorrectnessReport]` 打印 `scale_dequant_checksum` 和首个 mismatch 的 expert/tile/row/col。
- `[CorrectnessReport]` 打印 `gmm1_epilogue_vec=true` 或等价字段；若只能 scalar preflight，M2.4 必须保持
  `needs_fix` 或标 `primitive-gap`。
- 依赖扫描无输出。

#### M2.5 SwiGLU 与 int8 requant

依赖任务：M2.4。

文件范围：

- 修改 `kernel/a3_int8_backend.hpp`
- 修改 `host/reference.hpp`
- 修改 `host/main.cpp`

任务：

- 实现 SwiGLU：GMM1 output 前半做 sigmoid，乘后半 hidden；row 宽度从 `intermediateSize * 2` 降到
  `intermediateSize`。
- 在 SwiGLU 前按 routing per-token scale 乘 GMM1 dequant output，语义对齐 A8W8 主路径的 GMM1 activation epilogue。
- 实现 dynamic/per-token quant 到 int8，生成 `gmm2InputInt8` 和 `gmm2PerTokenScale`；默认策略为
  `reduceMax(abs(row)) / 127.0f`，rounding/saturation 必须与 reference 固定。
- 生成并 dump `swigluSyncGroups` 与 `dequantSum` row range：M2.5 可以在 `overlap_mode=off` 下顺序处理，但
  activation 的 row range 必须已经按第 11 章的 sync group 语义切分，供 M3.3/M3.4 直接启用 signal。
- 生成第 11 章定义的 `SwigluGroup` descriptor：`syncIdx/expertBegin/expertEnd/rowBegin/rowEnd/tileBegin/tileEnd`。
  `rowBegin/rowEnd` 必须由 `cumsumMM[rankNum - 1][localExpert]` 的 expert prefix 推导，不能根据当前执行顺序临时累计。
- 定义并 dump activation tile plan：`activationTileRows`、可选 `activationTileCols/intermediateChunk`、
  每个 group 的 `tileBegin/tileEnd`、empty group 数量和 AIV worker 分摊计划。M2 可以顺序执行，但 tile plan
  必须已经能解释 M3.3 的 AIV grid-stride 分工。
- 生成 GMM2 row range handoff metadata：每个 `activationSyncGroupReady[syncIdx]` 对应的
  `dequantSum[syncIdx]..dequantSum[syncIdx + 1]` 必须能被 M2.GMM scheduler intersect 成 GMM2 tile tasks。
  M2.5 不运行 GMM2，但必须把这个映射所需的 row range 固定下来。
- PTO 实现进入 `TQUANT<INT8_*>` 前必须形成 FP32 Vec tile；如果改用等价 PTO Vec primitive 序列，task report
  只写策略摘要。
- M2.5 验收拆成两类证据：
  - metadata handoff：`swigluSyncGroups/dequantSum/SwigluGroup/gmm2TileTaskPlan` 必须生成并 host 比对；
  - numeric PTO path：SwiGLU 和 requant 必须由 PTO Vec primitive 直接表达。若 SwiGLU 已 Vec 化但 requant 仍用
    scalar rowmax/quant loop，M2.5 只能作为 metadata + correctness preflight，不能 accepted。

验收标准：

- small shape 下 `gmm2InputInt8` 与 host reference 完全一致，或 rounding 差异在明确策略内。
- `gmm2PerTokenScale` 与 host reference 在 tolerance 内一致。
- saturation 计数可打印。
- smoke shape 输出 `{1,1}` 分组 dump；host synthetic `expertPerRank=16` 输出 `{8,4,2,1,1}` 分组 dump；
  `dequantSum` 覆盖所有有效 expert rows 且不重复。
- 控制台结构化输出包含 `swiglu_group_count`、`swiglu_group_sizes`、`swiglu_group_row_ranges`、
  `swiglu_group_tile_ranges`、`activation_tile_rows`、`activation_aiv_workers` 和 `activation_empty_groups`。
- 控制台结构化输出包含 `activation_swiglu_vec=true` 和 `activation_requant_vec=true` 或等价字段；若 requant
  未走 `TABS/TROWMAX/TQUANT/TSTORE`，必须显式输出 `activation_requant_vec=false`，task 保持 `needs_fix`。
- `dequantSum[0] == 0`，`dequantSum` 单调不降，最后一个元素等于本 rank 所有 local expert 的有效 row 总数。
- zero-token expert 可以出现在 group 内，但 empty group 必须可见并 skip；不能产生后续永远等待的 ready event。
- M2.GMM scheduler 或 host reference 能用 M2.5 输出的 group row range 生成 GMM2 tile task preview；如果做不到，
  M2.5 不能交付为 accepted，必须记录 `task-gap`。
- zero row/zero-token expert 不产生 NaN/Inf scale。
- 依赖扫描无输出。

#### M2.6 RunGmm2 PTO int8 matmul

依赖任务：M2.5、M2.3a。

文件范围：

- 修改 `kernel/a3_int8_backend.hpp`
- 修改 `host/reference.hpp`
- 修改 `host/main.cpp`

任务：

- 用 PTO `TMATMUL` / `TMATMUL_ACC` 实现 GMM2 `int8 x int8 -> int32`。
- 复用 M2.3 的 `TMATMUL` preflight 结论；若 GMM2 K/N shape 与 GMM1 不同，必须补一个 GMM2 shape preflight，
  不能只凭 GMM1 编译成功推断。
- 如果 M2.3 采用独立 cube target，M2.6 必须复用同一 active numeric target/launcher 体系或新增同等约束的
  PTO-only cube target；不能让 GMM1 是 device PTO 而 GMM2 退回 host/mock。
- 支持 local expert-major segment。
- 必须复用 `M2.GMM shared tile/pipeline contract`，不能为 GMM2 另写一套 fixed-shape single-block path。
  GMM2 work item 的 M 维来自 `RunActivationAndQuant` 输出的 expert-major row range 或 sync group row range，
  N 维来自 output hidden size，K 维来自 SwiGLU 后 intermediate size。
- GMM2 的 tile scheduler 必须能把 activation sync group row range 投影到 local expert/tile work item。
  当前 M2 可顺序执行，但 task/row range 必须与 M3.4/M3.5 要打开的 `activationSyncGroupReady` 和
  `gmm2GroupReady/subTileReady` 对齐。
- GMM2 的 L1/L0 tile shape、stage 数、ping-pong 状态机和 tail 处理要求与 M2.3 相同；差异只能来自
  `K=intermediateSize`、`N=hiddenSize` 和 return segment 投影，不能降低为单 block smoke。
- `GMM2` 的 M 维 row range 必须与 `RunActivationAndQuant` 输出的 expert-major row range 一致。
- `TMATMUL` 的 A/B/C tile dtype 必须显式是 `int8/int8/int32`，不能通过 helper 或 fallback 隐藏。
- 固定 PTO `TMATMUL` micro-tile 尺寸可以复用 M2.3，但 accepted 主路径不能只覆盖固定
  `intermediateSize=32/hiddenSize=64` smoke shape、小 micro-tile fallback 或单 AIC block demo；GMM2 的 M/K/N
  必须由 runtime shape、activation row range 和 output hidden size 驱动，且 `intermediateSize % 64 == 0`。
- 多 block launch 必须按 expert/tile 维度分摊 GMM2 work，不能屏蔽除 block0 外的 AIC block；若暂时无法做到，
  M2.6/M2 stage 必须标成 `needs_fix` 或 `primitive-gap`。
- host reference 计算 GMM2 accumulator。

验收标准：

- single-rank small shape 下 GMM2 int32 accumulator 与 CPU reference 完全一致，但它只作为 smoke/debug。
- 4-rank realistic-shape balanced/skewed case 下每个 local expert 的 GMM2 accumulator checksum 与 reference 一致。
- skewed experts case 不越界。
- 至少一个 4-rank realistic-shape skewed 或 balanced case 的 GMM2 device path 证明多个 AIC block 参与并覆盖全部
  有效 cache-level tile；若只能单 block 运行，M2.6 只能作为 preflight，不得 `review_ready`。
- `[CorrectnessReport]` 或等价结构化 stdout 必须打印 GMM2 的 `gmm_runtime_shape=true`、`gmm_multiblock=true`、
  `gmm_tile_policy=gemm_ar_cache_level_int8`、base tile、L1/L0 bytes、stage 数、tile task count、active AIC
  block count、shape class、activation row range 覆盖和 tail 覆盖结果。
- 至少一个 realistic-shape case 覆盖 GMM2 K 或 N 方向超过单个 cache-level tile/stepK panel 的分块累加。
- preflight 记录必须覆盖 GMM2 的 `intermediateSize x hiddenSize` shape 或明确 blocked。
- 依赖扫描无输出。

#### M2.7 GMM2 epilogue 与 fused combine return

依赖任务：M2.6。

文件范围：

- 修改 `kernel/a3_int8_backend.hpp`
- 修改 `host/reference.hpp`
- 修改 `host/main.cpp`

任务：

- GMM2 accumulator 按 `scale2` 做等价 dequant，乘 per-token scale2 后 cast 到 output payload dtype；当前 int8
  主路径不加 bias。
- scale index 必须对齐 A8W8 scale view：`arrayGroupIdx = listLen == 1 ? 0 : localExpert`，scale offset 随 GMM2 N tile
  移动；当前若只支持 `listLen == 1` 必须 fail-fast 并写进 report。
- epilogue/cast 必须用 PTO Vec `TLOAD/TCVT/TMUL/TSTORE` 或等价序列表达，输出 per-segment return payload。
  scalar loop 逐元素 decode scale、cast half 只能作为 correctness preflight/debug，不能关闭 M2.7/M2.8。
- `RunGmm2EpilogueAndReturn` 在同一个 stage 中完成 epilogue 和 `TPUT` remote return：按
  `tokenPerExpertMatrix[tokenOwnerRank][expertOwnerRank][localExpert] + preSumBeforeRank[tokenOwnerRank][localExpert]`
  把结果写到 token owner rank 的 `offsetD` return payload。
- 主路径不能先把全量 expert output 写到 `gmm2Out` 再由独立 `ReturnCombine` 二次搬运；`gmm2Out` 只能作为 debug mirror。
- 允许 `returnSegmentStaging` 或等价 per-segment GM staging 作为 `TPUT` source；如果同 stage 同时写 `gmm2Out`
  mirror，必须证明 `gmm2Out` 不是 `TPUT/CopyRowHalf` 的 source，也不是 correctness verifier 的唯一数据源。
- 每个 owner segment 写完后发布 combine signal；M2 可以用 `overlap_mode=off` 或 BSP wait，但 signal/counter 语义
  和 call site 必须就是 M3 overlap 要用的最终结构。

验收标准：

- `offsetD` return payload 与 host reference 在 dtype tolerance 内一致。
- RestoreOutput 可直接消费 `offsetD + expandedRowIdx + probs`，不需要额外 combine reorder。
- counter-log 能看出 producer count 等于实际写回的 owner segment 数。
- payload dtype 是 M2.1 固定的 `dtype_out`，不能输出 int8。
- `[CorrectnessReport]` 打印 `gmm2_epilogue_vec=true`、`gmm2_out_debug_mirror_only=true`、
  `gmm2_out_return_source=false` 或等价字段；若 mirror 强制写入但不参与 return，最多记录 P2 cleanup，
  不能误报为“全量 gmm2Out 二次 combine 主路径”。
- 依赖扫描无输出。

#### M2.7a GMM-Combine tile-split return map

依赖任务：M2.7。

文件范围：

- 修改 `include/moe_dispatch_combine_a8w8_layout.hpp`
- 修改 `kernel/control_metadata.hpp`
- 修改 `kernel/protocol_core.hpp`
- 修改 `kernel/a3_int8_backend.hpp`
- 修改 `host/reference.hpp`
- 修改 `host/main.cpp`

任务拆分：

- `M2.7a.1`：定义 shape-derived `ReturnSegmentPlan/OwnerSegment` schema、row+hidden chunk 字段和
  `subTileCap/ownerSegmentCap/subTileReady` 容量规则。
- `M2.7a.2`：实现 tile/sub-tile 到 owner segment 的 plan builder、coalesce、overflow 检查和 host/device dump。
- `M2.7a.3`：让 `RunGmm2EpilogueAndReturn` 消费 `ReturnSegmentPlan`，按 owner segment 直接发 PTO `TPUT`；
  `gmm2Out` 只能作为 debug mirror 或 per-segment staging。
- `M2.7a.4`：补齐 skewed row-tile 跨 owner、hidden/N chunk 拆分和 actual-write-from-plan 的验证。
- `M2.7a` 是 roll-up 验收项，只有 M2.7a.1-M2.7a.4 全部通过后才能接受。

任务：

- 在 M2 固定 GMM-Combine Tile 切分通信的语义映射，而不是等 M3.8 再重做 combine offset。
- 增加 tile/sub-tile owner mapping、source row range、N/hidden chunk range、destination `offsetD` segment view、
  multi-owner segment count、owner segment counter 和 M2.1 已预留的 `subTileReady` typed view / layout。
- `RunGmm2EpilogueAndReturn` 的 owner segment 必须由 `tokenPerExpertMatrix + preSumBeforeRank` 推导；如果一个
  GMM2 tile 跨多个 token owner rank，要拆成显式多段 segment plan。
- 构造 `ReturnSegmentPlan`：用 tile/sub-tile row range intersect tokenOwnerRank row range，再按 hidden/N chunk
  切分，生成 `OwnerSegment{tokenOwnerRank, srcRowBegin, rowCount, srcColBegin, colCount, dstOffsetD, dstColBegin,
  localExpert, tileId}`；相邻且 source/destination 的 row 和 col 都连续的 segment 必须 coalesce，避免退化成逐 row
  或逐 col 小包。
- `subTileCap`、`ownerSegmentCap`、`subTileReady` 容量必须由 shape 推导，至少覆盖
  `ceil(expertRows / returnTileRows) * ceil(hiddenSize / returnTileCols) * possibleOwnerSegments`。不能固定成
  `expertPerRank * 8` 后只靠 overflow counter 通过当前 small case。
- 每个 owner segment 必须能形成 direct PTO `GlobalTensor Shape/Stride` source/destination view，并在
  `RunGmm2EpilogueAndReturn` 主流程里按 `ReturnSegmentPlan -> OwnerSegment` 顺序直接 `TPUT`。多个离散目的地拆成
  多条 `TPUT`，不能引入 AscendC scatter fallback，也不能通过通用 helper 隐藏 PTO 调用。
- M2 可以 `overlap_mode=off` 顺序执行 segment return，但主路径必须已经由 segment plan 驱动；不能只生成
  `subTileReturnPlan/subTileOwnerSegments` 做 white-box dump，而实际 return 仍绕过 plan 按 tokenOwner/localExpert
  连续段循环写回。
- `gmm2Out` 只能是当前 segment 或 debug mirror。M2.7a 不能要求先 materialize 全量 `gmm2Out` 再按 segment copy；
  如果因为 PTO primitive 限制需要 segment staging buffer，必须在 report 摘要写清它是 per-segment staging，
  不是全量 combine copy path。
- 若 A3 PTO `GlobalTensor Shape/Stride + TPUT` 尚未验证，M2.7a 仍必须完成多段 direct `TPUT` 的 segment-driven
  correctness；M3.8 只验证 strided/async `TPUT` 是否能提升为文章级 overlap。若连多段 direct `TPUT` 都无法表达，
  M2.7a/M2.8 必须标 `primitive-gap`，不能推迟到 M3 才发现。

验收标准：

- small/balanced/skewed case 的 tile/sub-tile mapping 能还原到 token owner rank、source row range 和
  destination `offsetD` segment。
- skewed case 至少覆盖一个 tile 对多个 owner segment 的 mapping；若固定 shape 不能自然触发，task report 摘要必须说明
  使用的 synthetic mapping case。
- white-box dump 记录 `tile_count`、`segment_count`、`coalesced_segment_count`、capacity、overflow、每个 segment 的
  `tokenOwnerRank/srcRowBegin/rowCount/srcColBegin/colCount/dstOffsetD/dstColBegin/localExpert/tileId`，并能证明 segment 边界来自
  `tokenPerExpertMatrix + preSumBeforeRank`。
- return payload 与 host reference 在 dtype tolerance 内一致；actual return counter 必须按 owner segment 计数，
  并能证明每个 written segment 来自 `ReturnSegmentPlan`，不是旁路连续段写回。
- skewed case 至少覆盖 row tile 跨两个 tokenOwnerRank；另一个 synthetic 或 real case 至少覆盖 hidden/N chunk
  拆分。如果固定 shape 做不到，M2.7a 不能 accepted，必须记录 `test-gap`。
- `return_map_overflow` 必须为 0；容量不足是 M2 layout bug，不是 M3 async 问题。
- primitive-gap gate 情况下，task report 摘要写清待 M3.8 验证的 PTO primitive、最小 shape、预期 direct PTO 调用位置。
- 依赖扫描无输出。

#### M2.8 M2 集成回归

依赖任务：M2.1 至 M2.7a，包含 M2.2a、M2.2b、M2.2c、M2.7a。

文件范围：

- 修改 `CMakeLists.txt`
- 创建 `scripts/make_m2_mixed_register_elf.py` 或复用同等本目录 helper
- 修改 `scripts/run_a3.sh`
- 修改 `host/main.cpp`
- 修改 `kernel/kernel_launchers.hpp`
- 修改 `kernel/moe_dispatch_combine_a8w8_kernel.cpp`
- 修改 `kernel/moe_dispatch_combine_a8w8_gmm_kernel.cpp`
- 修改 `DESIGN.md`

任务：

- 增加 `--backend int8` 运行路径。
- 跑通 single-rank 和 two-rank 完整链路；链路必须是
  `RoutePackQuantLocal -> GatherDispatchToGmm1Input -> GMM1 -> ActivationQuant -> GMM2 -> RunGmm2EpilogueAndReturn -> RestoreOutput`。
- 同时汇总 M2.2c 的 soft-sync ledger dump 和 M2.7a 的 tile-split return map dump；M2 不要求 async speedup，
  但这些结构必须已经是 M3.7/M3.8 要消费的结构。
- M2.8 的 accepted 主路径必须是最终 single fused MPMD kernel stage graph，或明确输出
  `stage_graph_mode=multi_launch_debug` 并保持 `needs_fix`。当前 host 侧多 kernel launch + `MpiBarrier`
  只能作为 correctness preflight，不能作为 M2 关闭证据。
- M2.8 single fused MPMD 的构建方式必须是 PTO mixed ELF/metadata 双侧编译：AIC side 用 `dav-c220-cube`
  承载 GMM1/GMM2，AIV side 用 `dav-c220-vec` 承载 dispatch、epilogue、activation/requant、return、restore。
  禁止再把 `FUSED_KERNEL_ARCH=dav-c220` 单 target 探测失败当成 accepted blocker 或 final route。
- M2.8a 先交付 minimal mixed launch spike：同一 ABI、同一 launch handle、AIC/AIV heartbeat/counter 均可见，
  并在 report 中记录 `mixed_elf_register=true`、`mixed_aic_heartbeat=true`、`mixed_aiv_heartbeat=true`。
- M2.8b 再交付 fused stage graph skeleton：host 对 M2 主链只 launch 一次，stage 边界由 device-side ledger /
  `SYNCALL` / scoreboard 表达；允许 overlap-off，但不允许 stage 间 host `aclrtSynchronizeStream` 或 `MpiBarrier`。
- M2.8c 最后把现有 M2 full-chain data path 迁入 fused launch，保留 `--stage-graph multi_launch_debug` 或等价开关作为
  对照回归路径。
- M2.8c 的 AIV 口径是“fused graph 中 AIV side 参与并如实报告 worker 事实”，不是“所有 AIV stage 已经数据并行”。
  `mixed_aiv_blocks>0` 只能证明 mixed ELF/metadata launch 中 AIV side 存在；不能据此声称
  `RoutePackQuantLocal/GatherDispatchToGmm1Input`、GMM1 epilogue、ActivationQuant 或
  `RunGmm2EpilogueAndReturn` 已由多个 AIV 共同处理 payload。若这些 stage 仍为 single-worker correctness path，
  M2.8c 必须打印 `dispatch_aiv_workers=1`、`activation_aiv_workers=1`、
  `combine_return_aiv_workers=1` 或等价字段，并标注 `aiv_data_parallel_deferred_to_m3=true`。
  这些值为 1 不阻塞 M2.8c；M3.2/M3.3/M3.5/M3.8 才负责把对应 stage 提升为多 AIV worker 分摊。
- RestoreOutput 若已按 token shard 使用多个 AIV，可以在 M2.8c 记录为 partial AIV data parallel；但硬编码 worker 数
  不能作为最终能力声明，M3 需要按 runtime logical AIV count 和 token range 继续收口。
- 每个 M2 回归用例打印 final output correctness report 和 overlap-off E2E perf report。
- 覆盖第 6.4 节的 drop 与 `xActiveMask` 语义：除 small/balanced/skewed/zero-token 外，必须新增
  (a) 一个 over-capacity case（某 expert token 数超过其在 `maxOutputSize` 预算内可占额度，触发位置序前缀和截断）和
  (b) 一个含 inactive token 的 `xActiveMask` case。两者的 host golden 必须用同一套截断/sentinel/no-renorm restore
  规则生成；不传 mask 时行为与无 mask 完全一致。drop/mask 复用既有 routing + sentinel 路径，不新增独立分支。
- `DESIGN.md` 记录 M2 tolerance、scale/dequant 顺序、dispatch/quant 合并点、combine/epilogue 合并点、
  Dispatch-GMM soft-sync ledger、GMM-Combine tile-split return map，以及当前 A8W8/int8 主路径的精度边界。
- 若 M2.3/M2.6 的 cube primitive preflight 失败，M2.8 不允许以 host numeric、identity mock、旧 M1
  `workspace.dispatchedA` mock payload 或跳过 GMM 的方式交付；必须把受影响任务和 M2 stage 标成
  `blocked`/`primitive-gap`，并在 report 记录最小失败形态。
- 若 GMM1/GMM2 仍是固定 smoke shape、单 AIC block、无 runtime-shape work partition 的实现，M2.8 不允许交付为
  `review_ready`。这类实现只能作为 `TMATMUL` primitive preflight 或 debug path，不能作为 M2 full-chain
  acceptance path。

验收标准：

- small、balanced、skewed、zero-token 四类用例全部通过。
- over-capacity drop case 通过：被丢行在 gather/GMM1/SwiGLU/GMM2/return 一致截断，dropped `(token,slot)` 的
  `expandedRowIdx >= num_out_tokens`，final output 与 golden 在 tolerance 内一致，不死锁、不越界；结构化输出含
  `drop_triggered=true`、`dropped_rows=<n>` 或等价字段。
- inactive `xActiveMask` case 通过：inactive token 在路由阶段被 drop（expertId 改写为越界 invalid），restore 跳过其
  slot 且不重新归一，final output 与 golden 一致；不传 mask 的回归结果与无 mask 基线 bitwise 一致。
- GMM accumulator checksum 与 reference 精确对齐；scale/dequant checksum 和 final output 按 M2.8 记录的 tolerance 对齐。
- `[CorrectnessReport]` 或等价结构化输出必须证明 `gmm_runtime_shape=true`、`gmm_multiblock=true`；如果某 case
  因零 token 退化为单 block，不得作为该证明来源。
- `[CorrectnessReport]` 必须包含 `dispatch_merge=true`、`combine_merge=true` 或等价字段，证明 M2 没有走旧的
  “先二次 reorder，再单独 combine copy” 主路径。
- `[CorrectnessReport]` 必须包含 `soft_sync_ledger=true`、`swiglu_sync_groups=true`、
  `tile_split_return_map=true` 或等价字段，证明 M2 已经前置 Dispatch-GMM soft-sync 账本、Swiglu sync group row range
  和 GMM-Combine Tile 切分 return 映射。
- `[CorrectnessReport]` 必须包含 `stage_graph_mode=single_fused_mpmd` 才能关闭 M2.8；如果输出
  `stage_graph_mode=multi_launch_debug`，即使 final output 正确也只能作为 preflight。
- `[CorrectnessReport]` 必须包含 `mixed_elf_register=true`、`mixed_aic_blocks>0`、`mixed_aiv_blocks>0` 或等价字段，
  证明 fused path 来自 PTO mixed ELF/metadata，而不是 host multi-launch 串联。
- `[CorrectnessReport]` 必须包含 AIV stage worker 事实字段，例如 `dispatch_aiv_workers`、
  `gmm1_epilogue_aiv_workers`、`activation_aiv_workers`、`combine_return_aiv_workers`、
  `restore_aiv_workers` 和 `aiv_data_parallel_deferred_to_m3`。M2.8 不要求这些 worker 数都大于 1；
  验收重点是不能把 launch-level `mixed_aiv_blocks` 误读成 payload-level 多 AIV 数据并行。
- `[CorrectnessReport]` 必须包含 FFN 切分口径字段，例如 `ffn_partition_model=rank_core_group_tile_l1l0`、
  `group_is_sync_boundary=true`、`gmm_tile_is_aic_work_unit=true`、`dispatch_ready_grain=expert_group`、
  `activation_ready_grain=swiglu_sync_group`、`combine_ready_grain=owner_segment_or_group`。这些字段不要求
  M2 打开 async overlap，但必须证明 M2/M3 共享同一切分语义。
- `[CorrectnessReport]` 必须包含 `gmm1_epilogue_vec=true`、`activation_requant_vec=true`、
  `gmm2_epilogue_vec=true` 或等价字段，证明剩余 numeric epilogue/requant 不再由 scalar GM loop 主导。
- `[CorrectnessReport]` 记录 GMM1/GMM2 accumulator checksum、scale/dequant checksum、final output checksum、tolerance/max_abs_diff/max_rel_diff，且 `pass=true`。
- `[PerfReport]` 记录 overlap-off E2E `samples/avg/min/max/stddev`；M2 不要求 speedup。
- 依赖扫描无输出。

### 14.4 M3: overlap 与 ready queue 备选

> **执行口径（2026-05-30 重构，以此为准）**：本节 `M3.0`/`M3.1` 已 `accepted` 并继续复用；下面的 `M3.2`-`M3.9`
> 是最初"逐边 overlap + 多 AIV 捆绑"的线性设计，已被 **§14.4N 的 M3N 分段设计（先切分、再执行骨架、后逐边轮动）取代**。
> 后续执行**一律以 M3N 为准**，`M3.2`-`M3.9` 仅作历史保留、不再领取。旧 `M3.x` 与 `M3N.x` 的概念对应见 §14.4N 开头的对照表。
> M3N 把每条 overlap 边的"切分（stage 内部多 AIV）"和"轮动（stage 间时序重叠）"拆开，并在两者之间插入独立的
> "BSP→流水执行骨架"任务，使分段开发、分段验证更均匀。

#### M3.0 Single-kernel MPMD 调度开关

依赖任务：M2.8。

文件范围：

- 修改 `kernel/moe_dispatch_combine_a8w8_kernel.cpp`
- 修改 `kernel/protocol_core.hpp`
- 修改 `kernel/a3_int8_backend.hpp`

任务：

- 在 M2 已合入的最终 stage 图上打开 single-kernel MPMD 或等价 device-side producer-consumer 调度。
- AIV-like 分支负责 `RoutePackQuantLocal`、`GatherDispatchToGmm1Input`、activation、fused combine return、restore；
  AIC-like 分支负责 GMM1/GMM2。
- host multi-launch 只保留为 debug/offline path，不能作为 M3 overlap 完成口径；本任务不能重写 M2 的 layout、
  producer/consumer edge、signal/counter 名称或 row range 粒度。
- 本地 AIC/AIV handoff 使用 PTO public sync primitive 或 local GM signal + `TNOTIFY/TWAIT/TTEST`；不能直接用
  `AscendC::CrossCoreWaitFlag`。
- 若 M2.8 已经交付 single fused overlap-off correctness，但 AIV side 的 dispatch、activation 或 combine return
  仍是 single-worker correctness path，M3.0 不能把它误记为 blocked；后续 M3.2/M3.3/M3.5/M3.8 分别打开这些
  stage 的多 AIV worker 分摊和 overlap 证据。

验收标准：

- overlap off 时，single-kernel path 与 M2 reference 结果一致。
- kernel 中能清楚看到 AIC/AIV role split 或等价 stage 分支。
- 控制台或 report 能区分 launch-level AIV participation 与 payload-level AIV worker 数；不能只用
  `mixed_aiv_blocks>0` 证明 dispatch/activation/combine 已经多核。
- 依赖扫描无输出。

#### M3.1 Sync-group signal 与 counter 启用

依赖任务：M3.0。

文件范围：

- 修改 `kernel/control_metadata.hpp`
- 修改 `host/main.cpp` 或 M0.4 生成的 layout dump helper

任务：

- 启用并校验 M2.1/M2.5 已定义的 overlap 必需 signal：`dispatchGroupReady[expert]`、
  `gmm1SyncGroupReady[syncIdx]`、`activationSyncGroupReady[syncIdx]`、`gmm2GroupReady[expert]`。
- 校验 M2.5 已生成的 `swigluSyncGroups` 和 `syncGroupOffsets/dequantSum`：量化路径中
  `swigluSyncGroups` 是 `IsSyncTask(groupIdx, expertPerRank)` 的具体分组计划，`dequantSum` 给出每个 group
  覆盖的连续 row range。
- 校验 M2.5 已生成的 `SwigluGroup` descriptor 和 activation tile plan：`syncIdx/expertBegin/expertEnd`、
  `rowBegin/rowEnd`、`tileBegin/tileEnd`、`activationTileRows` 必须和第 11 章规则一致。
- 固定 smoke shape 的 `expertPerRank=2` 分组应退化为 `{1,1}`；同时复用 M2.5 的 host synthetic dump 验证
  `expertPerRank=16` 时生成 `{8,4,2,1,1}`，该数字表示 expert group size，不表示 AIV core 数。
- 每类 signal 有 producer counter、consumer counter、timeout counter；如果实现发现 M2 未预留对应字段，必须
  按 issue/DCL 回补 M2.1/M2.5，而不能在 M3.1 私自重新定义 layout 语义。

验收标准：

- counters cache-line aligned。
- `swigluSyncGroups` 的 group size 之和等于 `expertPerRank`，每个 expert 只属于一个 group。
- `dequantSum` 与 host reference 对同一 expert token 分布生成一致 row range，且能从
  `cumsumMM[rankNum - 1][localExpert]` 推导出来。
- `SwigluGroup` descriptor 的 row range 和 tile range 单调、无重叠；empty group 不创建永久等待。
- `activationSyncGroupReady` 的 row range 能被 GMM2 tile task preview 消费；不能只生成 activation 信号而无法映射到 GMM2。
- M2 的 `overlap_mode=off` path 仍能运行。
- 依赖扫描无输出。

#### M3.2 Dispatch/GMM1 group overlap

依赖任务：M3.1。

文件范围：

- 修改 `kernel/protocol_core.hpp`
- 修改 `kernel/a3_int8_backend.hpp`

任务：

- `GatherDispatchToGmm1Input` 完成某 expert group 的所有 token-owner-rank payload，并直接写好 GMM1 input 后发布
  `dispatchGroupReady[expert]`。
- `RunGmm1` 按 group ready 消费，不等所有 expert group 的 dispatch 完成。
- source-segment 级细化只允许作为后续优化，不是本任务验收口径。
- 本任务不能只把最终 `TGET` loop 改成多核。按 `ffn.md` 的 init_routing 口径，route/count/pack/gather 至少要在
  counter/timeline 上分开记录：route/pack/quant token shard、count publish/wait、prefix/cumsum 构建、
  dispatch remote gather。若某个子阶段仍为 single-worker 或 BSP 收口，必须标为 preflight/gap，不能把
  `dispatch_aiv_workers>1` 泛化成全 dispatch 子系统已多核。
- 在不改变 M2.2a/M2.2b row order、source offset、count/prefix 语义和 scoreboard task map 的前提下，
  将 dispatch AIV work 从 M2 single-worker path 提升为多 AIV worker 分摊：route/count/pack 可以按 token shard
  或 block prefix 分工，`GatherDispatchToGmm1Input` 可以按 local expert、token owner segment 或 row tile 分工。
  分工只改变 worker ownership，不改变 peer-visible dispatch layout 或 `gmm1InputInt8` contiguous layout。
- 增加 `dispatch_aiv_workers`、每个 worker 处理的 token/segment/row 计数、skip 计数和 overlap wait 计数。
  zero-token expert 和 empty segment 必须由 worker counter 证明 skip，不得产生永久等待。

验收标准：

- M2 correctness 不退化。
- timeline 或 counter 能证明 GMM1 在全量 expert dispatch 完成前开始。
- 对一个 4-rank balanced 或 skewed case，counter 能证明 dispatch payload work 至少由两个 AIV worker 处理；
  若 PTO sync/调度能力不足，只能标 `primitive-gap` 或 blocked，不能继续报告 payload-level dispatch 多核已完成。
- counter/timeline 能分别输出 `route_pack_workers`、`count_sync_workers`、`dispatch_gather_workers` 或等价字段；
  其中任何字段为 1 时，report 必须说明该子阶段仍未完成多 AIV 化。
- 依赖扫描无输出。

#### M3.3 GMM1/Activation sync-group overlap

依赖任务：M3.2。

文件范围：

- 修改 `kernel/control_metadata.hpp`
- 修改 `kernel/protocol_core.hpp`
- 修改 `kernel/a3_int8_backend.hpp`

任务：

- GMM1 在当前 `swigluSyncGroups[syncIdx]` 覆盖的 expert rows 完成时发布
  `gmm1SyncGroupReady[syncIdx]`，而不是等全部 GMM1 完成。
- Activation 消费 `dequantSum[syncIdx]..dequantSum[syncIdx+1]` row range，完成该 group 的 per-token
  scale、dequant、SwiGLU、requant 和 cast。
- Activation 必须按 M2.5 固定的 `SwigluGroup.tileBegin/tileEnd` 或等价 tile plan 消费，组内 tile 可由多个 AIV
  以 grid-stride 分摊；不能在 M3.3 重新按当前执行顺序生成 row range。
- 组内 row/tile 可由多个 AIV 分摊，但同步事件按 `swigluSyncGroups` 发布；不能把 `{8,4,2,1,1}` 理解成
  AIV 数量分配。
- 本任务负责把 M2.5 的 activation/SwiGLU/requant correctness path 提升为 payload-level 多 AIV worker 分摊。
  worker 分摊只能消费 M2.5 已固定的 `SwigluGroup.tileBegin/tileEnd` 和 `dequantSum` row range，不能重排
  `gmm2InputInt8/gmm2PerTokenScale` 或改变 dynamic requant 数值顺序。
- Activation/SwiGLU/requant 内部若使用 UB ping-pong 或 event pipe，必须显式记录开场预置、尾部排空和 store
  visibility 证据；不能在 `gmm1SyncGroupReady -> activationSyncGroupReady` 之间省略 `Finalize`/排空语义后
  提前放行 GMM2。没有 pipe 化实现时，report 也要写明 `activation_pipe_stages=1`，避免误报 overlap。

验收标准：

- M2 correctness 不退化。
- producer/consumer counter 一致。
- `swigluSyncGroups` 与 `dequantSum` row range 覆盖所有有效 expert rows，且没有重复消费。
- fixed smoke 输出 `{1,1}` 分组 dump；synthetic `expertPerRank=16` 输出 `{8,4,2,1,1}` 分组 dump。
- counter 或 timeline 能显示每个 non-empty `syncIdx` 的 `activation_tile_begin/end`、worker tile count 和
  empty group skip 计数。
- 对至少一个 multi-group 或 synthetic activation case，`activation_aiv_workers>1` 且各 worker tile range 无重叠、
  无缺口；如果实际 shape 只有一个 worker 有效，report 必须说明该 case 不能证明 activation 多核。
- counter 或 timeline 包含 `activation_pipe_stages`、`activation_pipe_prefill`、`activation_pipe_drain` 或等价字段；
  若当前实现是单缓冲顺序路径，这些字段必须诚实记录为 1/false/false。
- sync event 数不超过 `swigluSyncGroups.size()`；若 event 资源或 PTO sync primitive 不足，按
  `primitive-gap`/`needs_user_decision` 规则处理，不能退回全量 activation barrier 后仍声称 M3.3 通过。
- 依赖扫描无输出。

#### M3.4 Activation/GMM2 sync-group overlap

依赖任务：M3.3。

文件范围：

- 修改 `kernel/control_metadata.hpp`
- 修改 `kernel/protocol_core.hpp`
- 修改 `kernel/a3_int8_backend.hpp`

任务：

- Activation quant 完成某 `swigluSyncGroups[syncIdx]` 后发布 `activationSyncGroupReady[syncIdx]`。
- GMM2 按 `activationSyncGroupReady[syncIdx]` 消费 `dequantSum[syncIdx]..dequantSum[syncIdx+1]`
  row range，不等全量 activation 完成。
- GMM2 scheduler 必须把这个 sync group row range intersect 到 M2.GMM `GmmTileTask`；对跨 expert 的 group，
  需要拆成一个或多个 expert/tile tasks，不能把整段 row range 当成无 expert 边界的平铺矩阵后丢失
  `gmm2GroupReady/subTileReady` 投影。
- `activationSyncGroupReady` 不能直接驱动 combine return；GMM2 完成后仍必须发布 `gmm2GroupReady[expert]`
  或 `subTileReady[aicTile]`，供 M3.5/M3.8 消费。
- 记录每个 group 的 `activation_start/activation_ready/gmm2_consume_start` counter；如果 shape 中只有一个
  非空 group，则必须在 report 写明为什么该 case 不能证明提前消费，并用 synthetic dump 补足分组证明。

验收标准：

- M2 correctness 不退化。
- producer/consumer counter 一致。
- GMM2 tile task preview/actual dump 能从每个 consumed `syncIdx` 追溯到 `expertBegin/expertEnd`、
  `rowBegin/rowEnd` 和 GMM2 tile id。
- 对多 group 非空 case，counter 或 timeline 能证明 GMM2 第一组消费早于最后一组 activation ready；若被
  PTO sync/调度能力阻断，按 `primitive-gap` 或 blocked 记录最小复现，不能放宽为全量等待。
- 后段 SwiGLU/quant 如果推迟了 Combine，必须在 task report 摘要记录为 overlap gap；correctness tolerance 不因此放宽。
- 依赖扫描无输出。

#### M3.5 GMM2/Combine overlap

依赖任务：M3.4。

文件范围：

- 修改 `kernel/a3_int8_backend.hpp`
- 修改 `kernel/protocol_core.hpp`

任务：

- GMM2 完成某 expert group 后发布 `gmm2GroupReady[expert]`。
- `RunGmm2EpilogueAndReturn` 等待对应 expert group，按 `tokenPerExpertMatrix + preSumBeforeRank` 切分到各 token owner
  rank 并在 epilogue 后直接 `TPUT`。
- 本任务对应 `ffn.md` 的 Combine V1/continuous-owner-segment 方向：以 expert group ready 为输入，按 token owner
  rank/owner segment 发送连续 payload。它不要求完成文章中的 Sub-Tile 非连续通信，但必须记录是否仍存在
  all-AIV `SyncAll` 或等价全核栅栏；若仍存在，report 将其列为 M3.8/M3.9 的同步空泡证据，而不能声称已达到
  Combine V2 小 shape 行为。
- 在 M2.7a `ReturnSegmentPlan/OwnerSegment` schema 不变的前提下，把 GMM2 epilogue + continuous owner-segment
  return 从 M2 single-worker correctness path 提升为多 AIV worker 分摊。worker ownership 可以按 expert、
  owner segment 或 hidden chunk 切分，但每个 actual write 必须仍能追溯到 M2.7a segment plan。
- 增加 `combine_return_aiv_workers`、每 worker segment count、actual write count 和 skipped segment count。
  本任务只要求连续段/专家组级 overlap；Sub-Tile/stride async 能力仍归 M3.8。

验收标准：

- M2 correctness 不退化。
- counter 能证明 `RunGmm2EpilogueAndReturn` 不等所有 expert 的 GMM2 完成。
- 对一个 two-rank 或 4-rank skewed case，counter 能证明至少两个 AIV worker 处理 return segment 或 hidden chunk；
  如果只完成 expert-group overlap 而 return 仍单 AIV，任务不能 `accepted`，只能作为 M3.5 preflight。
- report 必须记录 `combine_mode=continuous_segment` 或等价字段、`combine_syncall_count`、`combine_cv_wait_count`
  和 `combine_owner_segment_workers`；这些字段用于和 M3.8 的 Sub-Tile/stride path 区分。
- task report 摘要明确：M3.5 不是文章级 GMM-Combine full async，只是连续段/专家组级 overlap。
- 依赖扫描无输出。

#### M3.6 Timeout dump 与 overlap 回归

依赖任务：M3.2 至 M3.5。

文件范围：

- 修改 `kernel/control_metadata.hpp`
- 修改 `host/main.cpp`
- 修改 `scripts/run_a3.sh`
- 修改 `DESIGN.md`

任务：

- 增加 timeout dump。
- run script 增加 overlap on/off 开关。
- run script 增加 overlap on/off E2E bench 输出，复用 `[PerfReport]` 字段。
- 若 overlap 验收方式需要变化，先按 issue/DCL 流程修正本章对应验收口径。

验收标准：

- overlap off 和 overlap on 都通过 M2 四类用例。
- overlap off/on 的 `[CorrectnessReport]` 都 `pass=true`，`[PerfReport]` 都有 E2E samples；same case/seed/shape 可对比。
- timeout 人为触发时能打印 rank、expert、token owner rank、expert owner rank、stage、signal id。
- 完成 M3.6 后只能声称“fused overlap skeleton”；若仍依赖 host 多 launch，则 M3.6 不通过。
- 依赖扫描无输出。

#### M3.7 Dispatch-GMM scoreboard soft sync

依赖任务：M3.6。

文件范围：

- 修改 `kernel/control_metadata.hpp`
- 修改 `kernel/protocol_core.hpp`
- 修改 `kernel/a3_int8_backend.hpp`
- 修改 `host/reference.hpp`

任务：

- 基于 M2.2c 已验收的 `scoreboardTaskMap`、producer status、`scoreboardMinStatus`、worker wait counter 和
  timeout counter 开启 async scoreboard 行为；本任务不能重新定义 Dispatch-GMM task 语义。
- 保持 producer 在 segment payload/scale 可见后直接写 GM status；增加一个 AIV Ctrl 角色轮询 producer status，
  并按 M2.2c 固定的 local expert 或 GMM tile dependency domain 计算聚合 ready。
- worker AIV/AIC 只轮询自己要消费的 `scoreboardMinStatus[dependencyDomain]`，不能每个 worker 都轮询所有 producer。
- `scoreboardMinStatus` 不能实现为单个全局 min task id；人为慢 producer 只能阻塞依赖该 producer segment 的
  expert/tile，不能把无关 expert/tile 拉齐。
- Dispatch-GMM 的 task id 必须引用 M2.2c 已固定的 expert group 或 tile row range 映射，不能只做无语义递增计数。
- 保留 M3.2 group-ready path 作为 fallback/debug，但 article-level 开关必须走 scoreboard path。
- 如果需要新增计数器，只能放入 M2.2c 已定义的 scoreboard counter 区域；不能改变 task id、row range 或
  `scoreboardTaskMap` offset 语义。

验收标准：

- M2 四类 correctness 用例继续通过。
- counter dump 至少包含 `producerPollCount`、`workerPollCount`、`scoreboardUpdateCount`、`scoreboardMinStatus`；
  结构上能证明 worker 不再逐个轮询所有 producer，不要求精确性能收益。
- 人为制造慢 producer 时，快 worker 不被无关 worker 或无关 producer 拉齐，只等待自身 dependency domain
  对应的 `scoreboardMinStatus`。
- timeout dump 包含 producer status 数组和 `scoreboardMinStatus`。
- 不直接使用 `AscendC::CrossCoreWaitFlag` / `SyncAll` 实现该同步。
- 依赖扫描无输出。

#### M3.8 GMM-Combine Sub-Tile/stride remote write

依赖任务：M3.6。

文件范围：

- 修改 `kernel/control_metadata.hpp`
- 修改 `kernel/protocol_core.hpp`
- 修改 `kernel/a3_int8_backend.hpp`
- 修改 `host/reference.hpp`
- 修改 `DESIGN.md`

任务：

- 基于 M2.7a 已验收的 tile/sub-tile owner mapping、source row/hidden chunk range 和 destination `offsetD`
  segment view 开启 Sub-Tile/stride remote write；本任务不能重新定义 combine offset、segment schema、capacity
  或 row/hidden chunk 语义。
- 本任务对应 `ffn.md` 的 Combine V2/small-shape 方向：目标不是让单条 DMA 更大，而是让 AIV consumer 尽量消费
  自己对应 AIC tile/sub-tile，减少或消除 Combine V1 中因为生产者 tile 分散而引入的全 AIV `SyncAll` 空泡。
  首版推荐以 16-row 或实现可证明的固定 row sub-tile 为最小 owner-segment refinement；若 shape 或 PTO primitive
  只能支持多段 direct `TPUT`，也必须在同一 `OwnerSegment` schema 上表达。
- 启用 M2.7a 已定义或预留的 `subTileReady[aicTile]`，作为 GMM2 tile 完成到
  `RunGmm2EpilogueAndReturn` 的细粒度 ready。
- 验证 A3 PTO `GlobalTensor Shape/Stride + TPUT` 能否表达 `N x 256` 级带 stride 的远端写；PTO primitive 调用必须直接出现在
  `RunGmm2EpilogueAndReturn` 主流程中。
- 如果一个 compute tile 覆盖多个 token owner rank 的 token slice，拆成显式多段 `TPUT` 或 strided `TPUT`，每段 offset
  由 M2.7a 的 `ReturnSegmentPlan/OwnerSegment` 提供。
- 如果 M2.7a 已经使用多段 direct `TPUT` 顺序执行，M3.8 的工作是把同一 segment plan 切到 strided/async path
  并证明能和 GMM2 overlap；不能把 M2.7a 未完成的 segment-driven return 主体搬到 M3.8。
- 如果 M3.5 已经完成 continuous owner-segment 多 AIV return，M3.8 只能在同一 worker ownership / segment schema
  上细化到 Sub-Tile 或 stride remote write；不能重新定义 `OwnerSegment` 字段、capacity 或 destination offset。
- 若 PTO A3 无法表达所需 stride/non-contiguous pattern，本任务以 `blocked` 交付，并保留 M3.5 连续段 correctness path；
  不允许引入 AscendC `DataCopy` fallback。
- 如果需要新增观测计数器，只能放入 M2.7a 已定义的 Sub-Tile counter 区域；不能改变
  `ReturnSegmentPlan/OwnerSegment` offset 语义。

验收标准：

- enabled 情况：two-rank skewed case 中同一个 GMM2 tile 可拆到至少两个 token owner rank，return payload 与 host reference 一致。
- enabled 情况：counter/timeline 能证明 `RunGmm2EpilogueAndReturn` 消费对应 AIC tile，不等所有 AIC/GMM2 完成。
- enabled 情况：counter/timeline 能区分 `combine_return_aiv_workers` 和 `subtile_return_workers`，并证明 worker
  使用的是 M2.7a segment schema 而不是旁路 copy。
- enabled 情况：报告 `combine_mode=subtile_stride` 或等价字段，并输出 `subtile_rows`、`subtile_stride_width`、
  `subtile_syncall_count`、`subtile_cv_wait_count`。若 `subtile_syncall_count` 未低于 M3.5 continuous path，
  只能称为 Sub-Tile correctness preflight，不能称为去同步空泡。
- blocked 情况：`DESIGN.md` 和 task report 摘要写明缺失的 PTO primitive 能力、最小复现 shape、需要补的 public primitive，不声称 full async combine。
- 依赖扫描无输出。

#### M3.9 Kernel timestamp 与 timeline 输出

依赖任务：M3.6，可与 M3.7/M3.8 并行。

文件范围：

- 修改 `include/moe_dispatch_combine_a8w8_layout.hpp`
- 修改 `kernel/control_metadata.hpp`
- 修改 `kernel/protocol_core.hpp`
- 修改 `host/main.cpp`
- 修改 `scripts/run_a3.sh`
- 修改 `DESIGN.md`

任务：

- 为 route、count sync、dispatch gather、GMM1、SwiGLU/quant group、GMM2、combine、restore 设置 kernel
  timestamp slot。
- timestamp slot 只能使用 M0.4/M2.1 已预留的 timeline/timestamp 区域；如果需要调整容量，只能扩展该区域的容量参数，
  不能改变 payload、row order、scoreboard 或 Sub-Tile segment offset 语义。
- host 读取 timestamp buffer，向 stdout 输出 rank/core/stage 维度的 CSV 或 JSON-style timeline。
- `SwiGLU/quant` timeline 行必须带 `syncIdx/groupId/rowBegin/rowEnd`，能和 `swigluSyncGroups`、
  `dequantSum` dump 对上。
- dispatch、activation、combine return 和 restore 的 timeline 行必须带 `workerId/logicalAiv`、owned row/segment/tile
  range、processed count 和 skipped count；这样才能区分 launch-level AIV participation、payload-level AIV 数据并行和
  async overlap。
- GMM timeline 行必须带 `groupIdx`、`tileId`、`mTile/nTile/kLoop`、logical AIC 和 wait source；dispatch/activation/
  combine 行必须能追溯到相同的 group、sync group、tile 或 owner segment。timeline 不能只输出 stage 总耗时。
- timeline 必须记录 `cross_rank_barrier_count`、`syncall_count`、`cv_wait_count` 或等价字段，用来解释
  `ffn.md` 中 Combine V1/V2 的同步差异和小 shape 空泡。
- run script 增加 `--timeline` 开关；本章写清如何对比 BSP、M3.6 skeleton、M3.7 scoreboard、M3.8 sub-tile；
  task report 只记录对比结论摘要。
- timeline stdout 必须能用 `case_name/seed/shape/run_id` 关联同一轮 `[PerfReport]`。
- 记录 small-shape 前同步开销和 compute/comm 并行导致的 MTE/带宽抢占风险。

验收标准：

- small case 能输出每个 stage 的非零 timestamp 或明确的 skipped 标记。
- timeline 中能看到 SwiGLU/quant group interval；没有多 group 非空 interval 时，必须打印 skipped reason 和
  对应 `swigluSyncGroups` dump。
- overlap on/off 的 timeline 能看到至少两个阶段的时间区间重叠或等待空泡变化。
- timeline 或 counter 能证明 M3 已打开的 AIV data-parallel stage 实际由多个 logical AIV 处理 payload；如果某 stage
  仍是 single-worker，必须在 report 摘要中列为未完成或 blocked，不能用 `mixed_aiv_blocks` 代替 worker 证据。
- timeline 能同时展示 group ready、AIC tile compute、SwiGLU sync group、owner segment return 四种粒度；若只能展示
  stage-level 粗粒度，M3.9 不能 `accepted`。
- `[Timeline]` 和 `[PerfReport]` 的 `case_name/seed/shape/run_id` 一致。
- timeline 输出失败不能影响 correctness 回归；失败时有独立错误码/日志。
- 依赖扫描无输出。

### 14.4N M3N: overlap 先切分后轮动分段设计

本节取代 §14.4 的 `M3.2`-`M3.9`。`M3.0`（single fused MPMD 调度）与 `M3.1`（signal/counter 启用）已 `accepted`，
M3N 直接复用其 fused launch、`--overlap-mode off|on` / `--timeline` 开关，以及 M3.1 已在 device 侧发布的
`dispatchGroupReady/gmm1SyncGroupReady/activationSyncGroupReady/gmm2GroupReady` 信号槽，不重做。

**总目标**：在 M2 已固定的 row order / offset / group plan / GMM tile / return segment schema 上，把 FFN 五层切分的
"形态"做出来——先让每个 stage 内部用满多 AIV（切分），再把 stage 之间从 BSP 全栅栏改成按 signal 推进的流水（轮动）。
M3N 首要交付"对标 `ffn.md` 的切分与并发形态 + counter/timeline 证据"。

**三段式 + 推荐执行顺序**：

- **Phase 1 切分（M3N.1-M3N.3）**：在当前 `SYNCALL<Mix>` 硬栅栏框架**内**，把 dispatch / activation / combine 三个 stage
  内部从 single-worker 提升为多 AIV。三者相互独立、可并行开发，correctness 基线 = M2 worker=1 结果（数值不变）。
  GMM1/GMM2 的 AIC tile 切分已在 M2.3/M2.6 固定，不在 M3N 重做。
- **Phase 2 骨架 + 逐边轮动（M3N.4-M3N.8）**：先用 M3N.4 把"stage 顺序 + 全栅栏"的 BSP 执行模型重构为"AIC 流 / AIV 流
  各自按 signal 推进"的循环骨架（signal 先全开 = 等价 BSP，correctness 不变、不声称性能）；再用 M3N.5-M3N.8 逐边把某条边的
  signal 等待范围从"等全部"缩成"等本 group/syncIdx"，建立真正的 compute/comm 错峰。
- **Phase 3 收口（M3N.9-M3N.12）**：timeout/回归、scoreboard async、Combine V2 Sub-Tile/stride、timeline。

推荐顺序：`M3N.1 ∥ M3N.2 ∥ M3N.3`（可并行）→ `M3N.4`（骨架，必须先于所有轮动）→
`M3N.5 → M3N.6 → M3N.7 → M3N.8`（逐边轮动，串行、共享骨架）→ `M3N.9` → `M3N.10 ∥ M3N.11` → `M3N.12`。

**性能门槛口径**：M3N accept = correctness 不退化 + 切分/worker/segment 证据正确 + counter/timeline 证明形态。
**不要求 overlap on 的 E2E 快于 off**；E2E delta 只需如实记录可对照（可正/负/平），性能调优（K width、preload/swizzle、
worker 数、sub-tile 粒度）是形态完成后的后续迭代，不阻塞 M3N/M4。若要把某具体 shape 的性能为正设成门槛，需走 issue/DCL 追加。

#### M3N 旧 M3.x → M3N 概念对照（交叉引用按此映射）

文档其它章节（§5.2.1、§8、§10、§13.x 等）仍以旧 `M3.x` 编号引用 overlap 工作，读者按下表映射到 M3N 任务：

| 旧 M3.x | 语义 | M3N 对应 |
| --- | --- | --- |
| M3.2 | Dispatch/GMM1 overlap + dispatch 多 AIV | 切分→**M3N.1**；轮动→**M3N.5** |
| M3.3 | GMM1/Activation overlap + activation 多 AIV | 切分→**M3N.2**；轮动→**M3N.6** |
| M3.4 | Activation/GMM2 overlap | 轮动→**M3N.7** |
| M3.5 | GMM2/Combine overlap + combine 多 AIV | 切分→**M3N.3**；轮动→**M3N.8** |
| M3.6 | timeout dump + overlap 回归 | **M3N.9** |
| M3.7 | Dispatch-GMM scoreboard async | **M3N.10** |
| M3.8 | Combine V2 Sub-Tile/stride | **M3N.11** |
| M3.9 | timeline | **M3N.12** |
| （新增）| BSP→流水执行骨架 | **M3N.4** |

#### M3N PTO 同步接口映射（轮动统一使用）

ffn.md 的同步手段在本 PTO 项目用以下等价接口表达；轮动任务（M3N.5-M3N.8）的同步必须直接调用这些 PTO primitive，
不得包装隐藏，也不得引入 AscendC `CrossCoreWaitFlag`/`SyncAll` 之外的 fallback：

| ffn.md 同步手段 | PTO 等价接口 | 用途 |
| --- | --- | --- |
| `CrossCoreSetFlag/WaitFlag<0x2>`（AIC↔AIV 组级握手）| `pto::Event<SrcOp,DstOp>::Init<CrossCoreId>() / Wait<CrossCoreId>()`（封装 `ffts_cross_core_sync` / `wait_flag_dev`）或 `TSYNC_CVID` | M3N.5-M3N.8 四条边的轮动放行 |
| `SetFlag()/Finalize()`（epilogue UB pipe 开场/排空）| PTO Vec pipe `set_flag/wait_flag` + 显式 prefill/drain 记录 | M3N.2/M3N.3 内部 pipe 生命周期 |
| `SyncAll`（同核全核栅栏）| `SYNCALL<SyncCoreType::Mix>` | M3N.4 骨架的粗栅栏 / overlap-off correctness 基线 |
| 跨卡 `DataAsFlag` / 计数到达 | payload `TPUT` 落地后 `TTEST` 轮询 count 区 | dispatch 前跨卡就绪（可选微优化）|

**FFTS flag 约束**：物理 flag id 0-15，PTO `SYNCALL` 占 11-14，用户安全区 0-10；多 group/syncIdx 同时在飞超过物理 flag
数时，用计数信号量 + 序号折叠（`flagId = idx % N`）复用，见 §10。

**轮动粒度冻结（贯穿 M3N.5-M3N.8 的硬约束）**：

- dispatch→GMM1（M3N.5）、GMM2→combine（M3N.8）= **expert 粒度**（`dispatchGroupReady[expert]` / `gmm2GroupReady[expert]`）。
- GMM1→SwiGLU（M3N.6）、SwiGLU→GMM2（M3N.7）= **sync-group 粒度**（`gmm1SyncGroupReady[syncIdx]` /
  `activationSyncGroupReady[syncIdx]`，按 `swigluSyncGroups` 前粗后细分组，**不是 per-expert**）。
- ready 粒度由轮动任务建立后**冻结**；Phase 1 切分任务只在每个粒度单位**内部**加 worker，不改 signal 的数量/粒度/row range。
  ready 粒度 ⊥ 多 AIV 切分粒度（group 是 sync 边界，tile/row/segment 才是 worker work item，见 §5.2.1 group/tile 正交）。

#### M3N.1 Dispatch 内部多 AIV 切分

依赖任务：M3.1。可与 M3N.2/M3N.3 并行。

文件范围：修改 `kernel/protocol_core.hpp`、`kernel/a3_int8_backend.hpp`。

任务：

- 在当前 `SYNCALL<Mix>` 硬栅栏框架内，把 dispatch 从 single-worker correctness path 提升为多 AIV worker 分摊；
  本任务不打开 stage 间 overlap（那是 M3N.5）。
- 按 `ffn.md` init_routing 口径分别记录 route/pack/quant token shard、count publish/wait、prefix/cumsum 构建、
  dispatch remote gather 的 per-worker counter；某子阶段仍 single-worker 必须标 preflight/gap。
- worker ownership 可按 token shard / block prefix / local expert / token owner segment / row tile 切分；
  只改 worker ownership，不改 M2.2a/M2.2b row order、source offset、count/prefix 语义、scoreboard task map、
  peer-visible dispatch layout 和 `gmm1InputInt8` contiguous layout。

验收标准：

- M2 correctness 不退化（与 worker=1 逐位一致）。
- 4-rank balanced 或 skewed case 下，counter 证明 dispatch payload 至少两个 AIV worker 处理，各 worker row/segment range 无重叠无缺口。
- 输出 `route_pack_workers`、`count_sync_workers`、`dispatch_gather_workers` 或等价字段；为 1 时说明该子阶段未多核化。
- 不实现 multi-core merge sort（offset-table pack+gather 已等价替代，见 §6）。依赖扫描无输出。

#### M3N.2 Activation/SwiGLU/requant 内部多 AIV 切分 + pipe

依赖任务：M3.1。可与 M3N.1/M3N.3 并行。

文件范围：修改 `kernel/control_metadata.hpp`、`kernel/protocol_core.hpp`、`kernel/a3_int8_backend.hpp`。

任务：

- 在硬栅栏框架内，把 M2.5 的 activation/SwiGLU/requant correctness path 提升为组内多 AIV tile 分摊（grid-stride）；
  本任务不打开 stage 间 overlap（那是 M3N.6/M3N.7）。
- worker 只消费 M2.5 已固定的 `SwigluGroup.tileBegin/tileEnd` 和 `dequantSum` row range，不重排
  `gmm2InputInt8/gmm2PerTokenScale`，不改 dynamic requant 数值顺序，不把 `{8,4,2,1,1}` 当 AIV 数量。
- activation/SwiGLU/requant 内部 UB ping-pong / event pipe 必须显式记录开场预置、尾部排空、store visibility；
  无 pipe 化实现时 report 写明 `activation_pipe_stages=1`。

验收标准：

- M2 correctness 不退化。
- multi-group 或 synthetic case 下 `activation_aiv_workers>1`，各 worker tile range 无重叠无缺口；单 worker 有效时 report 说明。
- 输出 `swiglu_group_tile_ranges`、activation tile processed/skipped、`activation_pipe_stages/prefill/drain` 或等价字段（单缓冲则诚实记 1/false/false）。依赖扫描无输出。

#### M3N.3 Combine 内部多 AIV owner-segment 切分（V1 continuous）

依赖任务：M3.1。可与 M3N.1/M3N.2 并行。

文件范围：修改 `kernel/a3_int8_backend.hpp`、`kernel/protocol_core.hpp`。

任务：

- 在硬栅栏框架内，把 `RunGmm2EpilogueAndReturn` 的 GMM2 epilogue + continuous owner-segment return 从 single-worker
  提升为多 AIV worker 分摊；本任务不打开 GMM2/combine stage 间 overlap（那是 M3N.8），也不做 Sub-Tile/stride（M3N.11）。
- worker ownership 可按 expert / owner segment / hidden chunk 切分，但每个 actual write 必须追溯到 M2.7a
  `ReturnSegmentPlan/OwnerSegment` schema，不改 capacity、destination offset 或 row/hidden chunk 语义。

验收标准：

- M2 correctness 不退化。
- two-rank / 4-rank skewed case 下 counter 证明至少两个 AIV worker 处理 return segment 或 hidden chunk，无重叠无缺口。
- 输出 `combine_return_aiv_workers`、per-worker segment count、actual write count、skipped segment count。依赖扫描无输出。

#### M3N.4 BSP→流水执行骨架（signal 全开 = 等价 BSP）

依赖任务：M3N.1、M3N.2、M3N.3（三个 stage 已多核后再转骨架，避免转换后再回头改分活）。

文件范围：修改 `kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`、`kernel/protocol_core.hpp`、`host/main.cpp`。

任务：

- 把当前"所有核做完一个 stage 一起过 `SYNCALL<Mix>` 全栅栏、再一起做下一个 stage"的 BSP 结构，重构为
  "AIC 流 / AIV 流各自按 per-group/per-syncIdx signal 推进"的循环骨架：AIC 主循环负责 GMM1/GMM2，AIV 主循环负责
  route/gather/activation/return/restore，stage 间用 §"PTO 同步接口映射"中的 `pto::Event<CrossCoreId>` / `TSYNC_CVID` 握手位点。
- **本任务 signal 先全开（等价 BSP）**：每个握手位点等待"全部上游 group 完成"，使行为与 overlap-off 逐位一致；
  只搭好"按 signal 推进"的结构和握手位点，不缩小等待范围（缩范围是 M3N.5-M3N.8）。
- 不改 M2 固定的 row order / offset / group plan / GMM tile / return segment；不改 M3N.1-M3N.3 的 stage 内部 worker 分活逻辑，
  只改 stage 间调度结构。

验收标准：

- M2 correctness 不退化（overlap-off 与 overlap-on=全开 signal 两种路径都逐位一致）。
- 结构证据：report 打印 `exec_model=aic_aiv_stream`（区别于 `bsp_syncall`），握手位点用 `pto::Event`/`TSYNC_CVID` 而非全核 `SYNCALL<Mix>` 分隔相邻 stage；保留的 `SYNCALL<Mix>` 仅作粗栅栏/基线并计数上报。
- 不声称任何性能收益（`overlap_on_payload_async_claim=false`，signal 全开）。依赖扫描无输出。

#### M3N.5 dispatch→GMM1 轮动（expert 粒度）

依赖任务：M3N.4。

**轮动粒度（冻结）**：**expert 粒度**，ready = `dispatchGroupReady[expert]`。
**同步接口**：`GatherDispatchToGmm1Input` 完成某 expert 的 payload 后用 `pto::Event`/`TSYNC_CVID` 发布；GMM1 侧 `Wait` 对应 expert。

文件范围：修改 `kernel/protocol_core.hpp`、`kernel/a3_int8_backend.hpp`。

任务：

- 在 M3N.4 骨架上，把 dispatch→GMM1 握手位点的等待范围从"等全部 expert dispatch"缩成"等本 expert"：
  `GatherDispatchToGmm1Input` 完成某 expert group 后发布 `dispatchGroupReady[expert]`，`RunGmm1` 按 expert ready 提前消费。
- 不改 M3N.1 的 dispatch worker 分活；不改 expert ready 粒度/row range。

验收标准：

- M2 correctness 不退化。
- timeline/counter 证明至少一个 expert 的 GMM1 早于最后一个 expert 的 dispatch ready（真错峰，非假开关）。
- `dispatch_overlap_granularity=expert`；zero-token expert 由 counter 证明 skip 无永久等待。依赖扫描无输出。

#### M3N.6 GMM1→SwiGLU 轮动（sync-group 粒度）

依赖任务：M3N.5。

**轮动粒度（冻结）**：**sync-group 粒度**，ready = `gmm1SyncGroupReady[syncIdx]`（按 `swigluSyncGroups`，非 per-expert）。
**同步接口**：GMM1（AIC）完成某 syncIdx 覆盖的 expert rows 后用 `pto::Event`/`TSYNC_CVID` 发布；activation（AIV）`Wait` 对应 syncIdx。

文件范围：修改 `kernel/control_metadata.hpp`、`kernel/protocol_core.hpp`、`kernel/a3_int8_backend.hpp`。

任务：

- 把 GMM1→activation 握手等待范围从"等全部 GMM1"缩成"等本 syncIdx"：GMM1 完成 `swigluSyncGroups[syncIdx]` 覆盖的
  expert rows 后发布 `gmm1SyncGroupReady[syncIdx]`，activation 按该 syncIdx 消费 `dequantSum[syncIdx]..dequantSum[syncIdx+1]`。
- 不改 M3N.2 的 activation 组内 worker 分活；不改 sync-group ready 粒度；sync event 数不超过 `swigluSyncGroups.size()`。

验收标准：

- M2 correctness 不退化；producer/consumer counter 一致。
- 多 group 非空 case 下 counter/timeline 证明 activation 第一组消费早于最后一组 GMM1 ready。
- `gmm1_activation_overlap_granularity=sync_group`；event 不足按 primitive-gap 处理，不退回全量 barrier。依赖扫描无输出。

#### M3N.7 SwiGLU→GMM2 轮动（sync-group 粒度 + intersect 回 GMM2 tile）

依赖任务：M3N.6。

**轮动粒度（冻结）**：**sync-group 粒度**，ready = `activationSyncGroupReady[syncIdx]`。
**同步接口**：activation（AIV）完成某 syncIdx 后发布 `activationSyncGroupReady[syncIdx]`。由于 DCL-115
已登记 M3N.5/M3N.6 共用 cross-core flag id 6-10，M3N.7 首版用 GM-poll ready 消费该槽，避免第三条边继续复用
同一 flag id；后续若扩展物理 flag 资源或改成可证明无 stale flag 的序号协议，再切回 `pto::Event`/`TSYNC_CVID`。

文件范围：修改 `kernel/control_metadata.hpp`、`kernel/protocol_core.hpp`、`kernel/a3_int8_backend.hpp`。

任务：

- activation 完成某 `swigluSyncGroups[syncIdx]` 后发布 `activationSyncGroupReady[syncIdx]`；GMM2 按该 syncIdx 消费
  对应 row range，不等全量 activation。
- **关键**：GMM2 scheduler 必须把 sync-group row range intersect 回 M2.GMM `GmmTileTask`；跨 expert 的 group 拆成
  一个或多个 expert/tile task，GMM2 完成后仍发布 `gmm2GroupReady[expert]` / `subTileReady[aicTile]` 供 M3N.8/M3N.11 消费，
  不能把整段 row range 当无 expert 边界的平铺矩阵后丢失投影。

验收标准：

- M2 correctness 不退化；GMM2 tile task preview/actual dump 能从每个 consumed syncIdx 追溯到 expertBegin/End、rowBegin/End、tile id。
- 多 group 非空 case 下 counter/timeline 证明 GMM2 第一组消费早于最后一组 activation ready；被 PTO 调度阻断按 primitive-gap 记录。
- 后段 SwiGLU 推迟 Combine 的部分在 report 记为 overlap gap，tolerance 不放宽。依赖扫描无输出。

#### M3N.8 GMM2→combine 轮动（expert 粒度）

依赖任务：M3N.7。

**轮动粒度（冻结）**：**expert 粒度**，ready = `gmm2GroupReady[expert]`。
**同步接口**：GMM2（AIC）完成某 expert 后发布 `gmm2GroupReady[expert]`；由于 DCL-115/DCL-117 已确认
cross-core flag id 6-10 不能再新增独立边，M3N.8 首版必须沿用 M3N.7 的 GM-poll ready 路线消费该槽，而不是新增
`pto::Event`/`TSYNC_CVID` 边。后续若扩展物理 flag 资源或引入可证明无 stale flag 的序号协议，再切回 event。

文件范围：修改 `kernel/a3_int8_backend.hpp`、`kernel/protocol_core.hpp`。

任务：

- GMM2 完成某 expert group 后发布 `gmm2GroupReady[expert]`；`RunGmm2EpilogueAndReturn` 等对应 expert，按
  `tokenPerExpertMatrix + preSumBeforeRank` 切到各 token owner rank 后直接 `TPUT`（Combine V1 continuous owner-segment 方向）。
- 不改 M3N.3 的 combine worker 分活；不改 expert ready 粒度。记录是否仍存在 all-AIV `SyncAll` 空泡，若有列为 M3N.11/M3N.12 证据。

验收标准：

- M2 correctness 不退化；counter 证明 return 不等所有 expert 的 GMM2 完成。
- `combine_mode=continuous_segment`、`combine_syncall_count`、`combine_cv_wait_count`、`combine_owner_segment_workers` 等字段输出。
- 必须输出 `m3n8_sync_transport=gm_poll_ready`、`m3n8_flag_reuse_risk_avoided=true`，并说明 DCL-115 flag-id 复用风险没有扩大。
- report 摘要明确：M3N.8 只是连续段/专家组级 overlap，非文章级 full async。依赖扫描无输出。

#### M3N.9 Timeout dump 与 overlap 回归

依赖任务：M3N.5 至 M3N.8。

文件范围：修改 `kernel/control_metadata.hpp`、`kernel/moe_dispatch_combine_a8w8_kernel.cpp`、
`kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`、`host/main.cpp`、`scripts/run_a3.sh`、`DESIGN.md`。

任务：增加 timeout dump；run script 增加 overlap on/off 开关与 E2E bench（复用 `[PerfReport]`）；overlap 验收口径若变先走 issue/DCL。

验收标准：

- overlap off/on 都通过 M2 四类用例，`[CorrectnessReport]` 都 `pass=true`，`[PerfReport]` 都有 E2E samples、可对比。
- 人为触发 timeout 能打印 rank、expert、token owner rank、expert owner rank、stage、signal id。
- 回归必须覆盖 DCL-115：M3N.9 不新增 cross-core flag id 6-10 的 `pto::Event` 边；M3N.7/M3N.8 仍使用
  GM-poll ready，`--overlap-mode off` 必须禁用 M3N.5-M3N.8 轮动并回到 M3N.4 full-open 骨架。
- 完成后只能声称"fused overlap skeleton"；仍依赖 host 多 launch 则不通过。依赖扫描无输出。

#### M3N.10 Dispatch-GMM scoreboard async

依赖任务：M3N.9。

文件范围：修改 `kernel/control_metadata.hpp`、`kernel/protocol_core.hpp`、`kernel/a3_int8_backend.hpp`、`host/reference.hpp`。

任务：

- 在 M2.2c 已固定的 `scoreboardTaskMap/producerStatus/scoreboardMinStatus`、producer publish 时机和 consumer
  dependency domain 上开启 async scoreboard；不重定义 task 语义。增加 AIV Ctrl 角色轮询 producer status，按 local expert /
  GMM tile dependency domain 聚合 ready；worker 只轮询自身 `scoreboardMinStatus[dependencyDomain]`。
- `scoreboardMinStatus` 不能实现为全局 min task id；保留 M3N.5 expert-ready path 作 fallback/debug。

验收标准：

- M2 四类 correctness 通过；counter 含 `producerPollCount/workerPollCount/scoreboardUpdateCount/scoreboardMinStatus`，结构证明 worker 不再轮询所有 producer。
- 人为慢 producer 时快 worker 不被无关 expert/tile 拉齐；timeout dump 含 producer status 数组和 `scoreboardMinStatus`。
- 不用 `AscendC::CrossCoreWaitFlag`/`SyncAll` 实现该同步。依赖扫描无输出。

#### M3N.11 Combine V2 Sub-Tile/stride remote write

依赖任务：M3N.9。可与 M3N.10 并行。

文件范围：修改 `kernel/control_metadata.hpp`、`kernel/protocol_core.hpp`、`kernel/a3_int8_backend.hpp`、`host/reference.hpp`、`DESIGN.md`。

任务：

- 在 M2.7a segment schema 与 M3N.3/M3N.8 的 owner-segment return 上，用**同步 `TPUT`** 做 Sub-Tile/stride return：
  已从 `TPut.hpp` 确认 src/dst 各自 5 维独立 stride 可表达 strided/多段非连续远端写（多 owner 写同一行用 `AtomicType::AtomicAdd`），
  故 sub-tile/stride return **功能形态可 enabled 交付**，PTO primitive 直接出现在 `RunGmm2EpilogueAndReturn` 主流程。
- 启用 `subTileReady[aicTile]` 作为 GMM2 tile→return 的细粒度 ready；首版以 16-row 固定 sub-tile 为最小 refinement，
  不改 `OwnerSegment` 字段/capacity/destination offset。
- **唯一 primitive-gap（blocked）**：sub-tile return 与 GMM2 计算的**异步 SDMA overlap**——`TPUT_ASYNC` 在 A3 硬性要求
  flat-contiguous-1D（`TPutAsyncCommonDetail.hpp`），无法承载 strided/多段；该子能力以 `blocked` 记录、
  locator 写 `TPUT_ASYNC requires flat-contiguous-1d`，保留同步 sub-tile return 与 M3N.8 连续段 path，不引入 AscendC `DataCopy` fallback。

验收标准：

- enabled：two-rank skewed case 中一个 GMM2 tile 可拆到至少两个 token owner rank，return payload 与 host reference 一致；
  counter 证明消费对应 AIC tile 不等所有 GMM2 完成；输出 `combine_mode=subtile_stride`、`subtile_rows`、`subtile_stride_width`、`subtile_syncall_count`、`subtile_cv_wait_count`。
- 异步 overlap：标 `primitive_gap=TPUT_ASYNC_flat_contiguous_1d`，不声称 full async combine，不放宽 correctness。依赖扫描无输出。

#### M3N.12 Kernel timestamp 与 timeline 输出

依赖任务：M3N.9，且 M3N.10/M3N.11 已完成或明确 blocked。

文件范围：修改 `include/moe_dispatch_combine_a8w8_layout.hpp`、`kernel/control_metadata.hpp`、`kernel/protocol_core.hpp`、`host/main.cpp`、`scripts/run_a3.sh`、`DESIGN.md`。

任务：

- 为 route、count sync、dispatch gather、GMM1、SwiGLU/quant group、GMM2、combine、restore 设 timestamp slot（只用 M0.4/M2.1 预留 timeline 区域）。
- host 输出 rank/core/stage 维度 timeline；SwiGLU 行带 `syncIdx/groupId/rowBegin/rowEnd`；dispatch/activation/combine/restore 行带
  `workerId/logicalAiv`、owned row/segment/tile range、processed/skipped；GMM 行带 `groupIdx/tileId/mTile/nTile/kLoop/logicalAic/waitSource`。
- 记录 `cross_rank_barrier_count/syncall_count/cv_wait_count` 解释 V1/V2 同步差异；run script 加 `--timeline`，对比 BSP / M3N.4 骨架 / M3N.10 scoreboard / M3N.11 sub-tile。

验收标准：

- small case 每 stage 非零 timestamp 或明确 skipped；overlap on/off timeline 能看到至少两阶段重叠或等待空泡变化。
- timeline 能同时展示 group ready、AIC tile compute、SwiGLU sync group、owner segment return 四种粒度；只能 stage-level 粗粒度则不 accepted。
- `[Timeline]` 与 `[PerfReport]` 的 `case_name/seed/shape/run_id` 一致；timeline 失败不影响 correctness 回归。依赖扫描无输出。

### 14.5 M4: 最终 PTO 化收口

#### M4.1 最小回归脚本

依赖任务：M3N.12（timeline），且 M3N.10/M3N.11 已完成或明确 blocked。

文件范围：

- 修改 `scripts/run_a3.sh`

任务：

- 固化 small、balanced、skewed、zero-token 回归。
- 固化依赖扫描。
- 固化一条 E2E bench 命令，执行 warmup/measure 并打印 `[CorrectnessReport]`、`[PerfReport]`。
- 固化 `--timeline` 运行开关和最小 timestamp dump。

验收标准：

- 一条命令能跑 A3 int8 最小回归。
- 一条命令能记录 A3 int8 E2E bench，并且只在 correctness pass 后采信性能数据。
- 一条命令能输出 timeline 或 stage timestamp dump。
- 依赖扫描无输出。

#### M4.2 文档与状态收口

依赖任务：M4.1。

文件范围：

- 修改 `DESIGN.md`
- 修改 `TASKS.md`

任务：

- 标注 A3 A8W8 backend 当前状态。
- 标注每个 MegaMoE article-level overlap 点当前状态：Dispatch remote read、scoreboard soft sync、Sub-Tile/stride
  combine、timeline。
- 清理已过期任务描述。
- 保留最终验收命令和已知限制。
- 关闭或转交 `Issue Log` 中所有 open P0/P1 issue。

验收标准：

- DESIGN/TASKS 不声称未完成能力已完成。
- `Task 状态` 中所有 accepted task 都有 report；所有 blocked task 都有 issue ID。
- 依赖扫描无输出。

## 15. 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| 从 `gemm_ar` 裁剪时复制过多工程结构 | 引入 compute/comm 双 kernel、参考工程内部通信 helper 或不属于本项目的 tiling 结构 | M0 只裁剪 CMake/run.sh/host runtime 骨架；本项目只接受 single fused kernel 或 dry-run smoke，真实 protocol 从 M1/M2 落地 |
| CMake/源码/脚本误引入 Catlass/AscendC | 破坏 PTO-only 目标，后续实现无法作为 PTO 化版本验收 | 每个 task 扫描本项目源码、CMake、脚本；若 PTO public primitive 不足，记录 `primitive-gap` 或 blocked，不能写 fallback |
| 依赖扫描误扫 markdown | `DESIGN.md`/`TASKS.md` 必须记录禁用依赖名，误扫会产生假阳性 | “依赖扫描无输出”只针对源码、CMake 和脚本；报告只记录扫描结论摘要 |
| M0.3 参数与 M0.4 layout 脱节 | run 脚本估算和真实 workspace/window 大小不一致，HCCL_BUFFSIZE 或 host check 不可信 | M0.3 先按显式 shape 做保守估算；M0.4/M0.6 接入 `WorkspaceLayout/PeerWindowLayout.totalBytes` 后替换为真实 layout 计算 |
| M1 mock 掩盖 protocol 错误 | GMM mock 通过但 row order、offset、signal 或 restore 错，M2/M3 返工 | mock 只能替代 expert compute 数值；routing、prefix、dispatch gather、combine return、restore 和 signal 必须走最终 protocol |
| M1.0 变成重新设计任务 | 后续 agent 可能另立协议，和第 0-12 章冲突 | M1.0 只核验既有设计不变量并固化 host reference；发现缺口必须走 issue/DCL，不在实现中私自改协议 |
| A8W8 主路径的半精度 epilogue 由 PTO int32 accumulator 显式展开 | 若 scale/dequant 顺序不一致，会出现系统性误差 | reference 显式模拟 `TMATMUL int32 -> scale1/scale2 dequant -> cast`，按中间 checksum 分段验收 |
| int8 主路径没有 bias，但实现误加 bias | reference 会对错目标 | 当前阶段禁止生成 bias |
| metadata 过度 Tile 化 | 设计复杂、同步难读 | payload/control 分层，metadata typed view 集中 |
| signal 发布顺序不严谨 | 跨 rank 偶现错误或死锁 | payload completion -> visibility -> notify，debug counter + timeout |
| Dispatch 算法退回后同步远端写 | GMM 前拿不到连续子矩阵，无法实现 MegaMoE overlap | 固定前同步 + 远端读；`RoutePackQuantLocal` 只写本 rank peer-visible window，`GatherDispatchToGmm1Input` 直接形成 GMM1 input |
| M2 未前置 MegaMoE 合并点 | M3 需要重写 dispatch/combine 数据布局，overlap gap 过大 | M2 必须完成 route/pack/quant 合并、GMM1 contiguous input、GMM2 epilogue/return 合并；M3 只改调度和 ready 粒度 |
| `swigluSyncGroups/dequantSum` 到 M3 才定义 | Activation/GMM2 overlap 需要重做 row range，M3.3/M3.4 无法只打开 signal | M2.1 预留字段，M2.5 固定 group plan 和 row range；M3.1-M3.4 只启用/校验 signal 和 counter |
| Dispatch-GMM scoreboard 到 M3 才设计 | task id、producer status、worker wait 粒度后改会影响 GMM1 启动条件 | M2.2c 固定 `scoreboardTaskMap/producerStatus/scoreboardMinStatus`、producer publish 时机和 consumer dependency domain；M3.7 只打开 async scoreboard 行为 |
| GMM-Combine owner segment 到 M3 才设计 | combine offset 或 return payload layout 被重写，M2 correctness 不能证明 M3 path | M2.7a 固定 `ReturnSegmentPlan/OwnerSegment/subTileReady` 语义；M3.8 只验证 stride/multi-segment remote write 能力 |
| PTO A3 remote stride 能力不足 | 无法实现文章级 Sub-Tile 非连续 combine | M3 单独做 `GlobalTensor Shape/Stride + TPUT` 能力验证；不足则 blocked，不写 AscendC fallback |
| timeline 改动污染 payload/control layout | 为观测新增字段时破坏 row/order/offset，导致 correctness 和 timeline 互相影响 | M0.4/M2.1 预留 timeline/timestamp 区域；M3.9 只使用或扩容该区域，不改 payload、scoreboard 或 Sub-Tile offset |
| small shape 前同步开销 | 性能可能差于独立通信加计算的基线 | 验收区分 correctness/overlap/性能；small shape 不作为性能收益门槛 |
| MTE/带宽抢占 | compute/comm 并行时 GMM 耗时膨胀 | timeline 打点拆解，记录 standalone 与 overlap 耗时差 |
| capacity overflow | 写越界或 combine 丢 token | host precheck + device counter + fail fast |
| empty expert segment | wait 永不满足 | count 为 0 的 segment 显式 skip，同时保证 rank-level signal 仍发布 |
| HCCL window offset 不一致 | remote pointer 错误 | 所有 rank 使用相同 layout，host 检查 winSize |
| async 调度过早引入 | correctness 难定位 | M1-M2 先用最终 stage 图的 `overlap_mode=off` 稳定，M3 在同一 stage graph 上验证或启用既有 producer/consumer 边 |
| report 或 TASKS 记录过多运行日志 | 状态台账膨胀，后续 agent 难以判断真实状态 | stdout 保持运行时证据；`reports/Mx.y.md` 只写 pass/fail、关键摘要、blocked 原因和下游提示 |
