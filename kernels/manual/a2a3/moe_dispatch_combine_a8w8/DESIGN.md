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
- Dispatch-GMM 同步：当前 active 路线按原始 FFN 保持 expert-ready，同步对象是 `dispatchGroupReady[expert]`。
  M3N.10 scoreboard async 已丢弃；后续 M3O 只优化 expert 内 dispatch/count/prefix worker 分活、ordinal/prefix
  计算和 GMM tile 调度，不恢复 producer status / min-status 类 scoreboard runtime。
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
- 当前边界：correctness baseline 固定为 A3 A8W8/int8 主路径；M3O 先证明 active worker/tile 分摊和 report
  真实性，再按 expert-ready、sync-group、owner-segment 粒度重新打开 overlap。外部实现只作为行为、协议、精度和性能对照，
  不能把 Catlass/AscendC 或外部融合算子依赖搬进本 PTO 项目源码。

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
  M2 使用 `overlap_mode=off`，E2E perf report 仍按固定 `warmup_iters=0`、`measure_iters=1` 输出
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
| dispatch gather -> GMM1 | `GatherDispatchToGmm1Input` | `Gmm1` | contiguous `gmm1InputInt8/gmA` row range + routing scale | `dispatchGroupReady[expert]` | M3O.2 |
| GMM1 -> activation | `Gmm1` | `ActivationQuant` | int32 accumulator dequant row range | `swigluSyncGroups/dequantSum` + `gmm1SyncGroupReady[syncIdx]` | M2.1/M2.5 metadata, M3O.6 signal |
| activation -> GMM2 | `ActivationQuant` | `Gmm2` | `gmm2InputInt8` + per-token scale2 row range | `swigluSyncGroups/dequantSum` + `activationSyncGroupReady[syncIdx]` | M2.1/M2.5 metadata, M3O.6 signal |
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
- M2.8/M3O 的正确实现路径是：AIC side 以 `dav-c220-cube` 编译，AIV side 以 `dav-c220-vec` 编译，通过
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
| Dispatch-GMM expert-ready | 保持原始 FFN expert 粒度 `dispatchGroupReady[expert]`，不恢复 scoreboard；M3O 优化 expert 内 worker/prefix/tile 分活 | M3O.2/M3O.3 report + counter |
| AIC/AIV 同 kernel 并行推进 | single-kernel MPMD stage functions | M3O.1-M3O.6 black-box-run + counter-log |
| `IsSyncTask/dequantSum` 粗粒度同步 | `gmm1SyncGroupReady` / `activationSyncGroupReady` | M2.5 固定 group row range，M3O.4/M3O.6 启用 signal/counter |
| Swiglu/quant 前粗后细分组 | `swigluSyncGroups` + `dequantSum` | M2.5 group plan dump，M3O.4/M3O.6 counter-log |
| GMM-Combine Sub-Tile/stride return | M2 固定 tile owner mapping/source row/destination `offsetD` segment plan，M3O 验证 V1/V2 worker policy 和同步 strided `TPUT` 能力 | M3O.5/M3O.6 evidence |
| kernel 内 timestamp 转 timeline | `debugCounters` / `timeline` export | M3O.7 timeline |

### 5.2.1 FFN 五层切分与并发契约

`ffn.md` 对本项目的直接约束是：FFN 融合不是把 dispatch、GMM、SwiGLU、combine 串到一个文件里，而是把
空间切分和时间依赖拆开表达。M2/M3 后续实现必须遵守下面的五层契约：

| 层级 | 本项目语义 | M2 必须固定 | M3 才打开或验证 |
| --- | --- | --- | --- |
| rank / expert ownership | `tokenOwnerRank` 与 `expertOwnerRank/localExpert` 决定 dispatch 读源、GMM group、combine 目的和 restore order | `tokenPerExpertMatrix/cumsumMM/preSumBeforeRank/expandedRowIdx` 同源，host/device dump 可追溯 | 不重排 owner 语义，只在已固定 owner segment 上打开 async |
| core worker | AIV 负责 route/count/gather/activation/return/restore，AIC 负责 GMM1/GMM2 | M2.8c 如实打印每个 AIV stage worker facts；worker=1 不阻塞 M2 | M3O.2/M3O.4/M3O.5 把对应 payload stage 提升为 stage-local worker，并用 counter/timeline 证明 |
| expert group / sync group | group 是 ready/sync 边界，不是“某个核负责某个 expert”的核分配单位 | `dispatchGroupReady`、`swigluSyncGroups/dequantSum`、`gmm2GroupReady` 的 row range 和 producer/consumer 边固定 | M3 只改变这些边的执行时序和 worker 分摊，不重新定义 group row order |
| tile / sub-tile | AIC 负载按 GMM tile 分配；combine 可把 GMM2 tile 投影成 owner segment 或 sub-tile | GMM tile task、`ReturnSegmentPlan/OwnerSegment/subTileReady` schema 固定，segment correctness 已验证 | M3O.3 做 GMM 多 AIC tile；M3O.5/M3O.6 做 V1/V2 combine 和 overlap |
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
- **各 stage 的 AIV worker 数不是设计常量**，而是 §5.2.1 core-worker 层和 M3O.2/M3O.4/M3O.5 负责的调优量：M2 可以是 worker=1，
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
3. Dispatch-GMM expert-ready 账本：M2/M3 已固定 `dispatchGroupReady[expert]`，M3O 不恢复 scoreboard。后续只在
   expert 内定义 worker/expert count、worker prefix、local ordinal、ready counter 和 timeout locator，不能到优化阶段
   重新发明 row order 或 task 粒度。
4. Swiglu/quant sync-group 元数据：M2 必须生成 `swigluSyncGroups/dequantSum`，并能把每个 group 映射到
   `cumsumMM[rankNum - 1][localExpert]` 推导出的连续 row range。M2 可按 `overlap_mode=off` 顺序执行
   activation/quant，但 row range 和 group plan 必须已经是 M3O.4/M3O.6 要启用的结构。
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
   M3O.5/M3O.6 再验证 V1/V2 worker policy、同步 strided `TPUT` 和可证明的 overlap 路径。
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
折叠进 gather 地址映射"的结论，offset-table 已经承担了排序职责，不需要独立 sort 子系统。因此 M3O.2 的 dispatch
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
- M2/M3 必须已经 dump `dispatchGroupReady[expert]`、`swigluSyncGroups/dequantSum` 和
  `subTileReturnPlan/subTileOwnerSegments`；M3O 只验证 stage-local worker、GMM tile、V1/V2 combine 和 overlap
  能否真实执行，不能重新定义 row order、offset 或 segment。
- timeline 里至少应能区分 `dispatch`、`GMM1`、`ActivationQuant`、`GMM2`、`RunGmm2EpilogueAndReturn`、
  `RestoreOutput` 的区间；若某个 overlap 点 blocked，task report 只写 blocked 的 primitive 或调度原因摘要。

同步 overlap 调度粒度：

| Edge | baseline ready 粒度 | 高性能方向 | consumer 规则 |
| --- | --- | --- | --- |
| Dispatch -> GMM1 | local expert / expert group；该 expert 的所有 token-owner rows 收齐并写入 contiguous `gmm1InputInt8/gmA` | 可选细化只能使用 M2.2c 已预留的 source segment 或 tile row range task，不允许 M3 重新定义 row order | `GMM1` 可以启动该 expert，不等其他 expert 的 dispatch |
| GMM1 -> Swiglu/quant | `swigluSyncGroups` / `dequantSum` 描述的连续 row range；一个 sync group 可以覆盖多个 expert，例如 `{8,4,2,1,1}` | 组内 row/tile 可由多个 AIV 分摊，但同步事件仍按 sync group 发布 | `ActivationQuant` 等当前 sync group 覆盖的 expert rows 都完成 GMM1，不要求单 expert 逐个同步，也不等全量 GMM1 |
| Swiglu/quant -> GMM2 | 同一个 `swigluSyncGroups` / `dequantSum` row range | GMM2 内部可按 GMM block 调度，但不能改变 sync group row range 语义 | `GMM2` 消费已完成 activation/quant 的 row range，不等全量 activation |
| GMM2 -> Combine | baseline 是 expert group；该 expert 的 GMM2 output ready 后可做连续段 return | M3O.5/M3O.6 目标是 owner-segment 或 tile/sub-tile；一个 GMM2 tile 可拆成多个 `OwnerSegment` 远端写；ready queue 仅作为 AIC/AIV 1:1 或静态 N:1 handoff 的备选 | `RunGmm2EpilogueAndReturn` baseline 不等所有 expert；Sub-Tile 模式不等整个 expert/GMM2，只等对应 AIC tile 或 owner segment |

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
4. Dispatch-GMM 的等待不能靠全核 BSP 拉齐。当前 active 路线保留 FFN expert-ready：
   dispatch 完成本 expert 的 contiguous rows 后发布 `dispatchGroupReady[expert]`，GMM1 只等待当前 expert。
   M3O 不恢复 scoreboard；优化集中在 expert 内 worker/prefix/tile 分活和 active counter 证据。
5. GMM-Combine 的 return path 在 M2 必须先按 tile/sub-tile 规划 owner segment、source row range 和
   destination `offsetD` view；M3O.5/M3O.6 再验证这些 segment 能否用 strided `TPUT` 或多段 `TPUT` 执行并 overlap。
6. `TGET/TPUT` 可作为本项目跨 rank payload 的 public PTO 表达；普通连续段必须在 M1/M2 使用 direct PTO
   调用落地。Sub-Tile 非连续 combine 是否能用 `GlobalTensor Shape/Stride + TPUT` 或多段 direct `TPUT`
   完整表达原文章的 stride remote copy，需要作为 M3O.5/M3O.6 独立验收项；若必须用 AscendC `DataCopy` 才能做到，
   则 PTO 版该点 blocked。

同步验收分层：

- M1/M2 的 correctness 可以在最终 stage graph 内用 `overlap_mode=off`、BSP wait 或 ledger-only 执行，但 signal、
  counter、row range 和 producer/consumer 边必须与 `6.1` 的运行视图一致。
- M3O.1-M3O.6 验证 fused overlap skeleton：`dispatchGroupReady`、`gmm1SyncGroupReady`、
  `activationSyncGroupReady`、`gmm2GroupReady` 能驱动 single-kernel MPMD 或等价 device-side producer-consumer 调度。
- M3O 在 M2/M3 已验收的 `dispatchGroupReady`、`swigluSyncGroups/dequantSum` 和
  `subTileReturnPlan/subTileOwnerSegments/subTileReady` 上验证 stage-local worker、GMM tile、V1/V2 return 和 overlap；
  不能重新定义 row order、offset、task id 或 owner segment。
- M3O.7 用 timeline 证明 `dispatch`、`GMM1`、`ActivationQuant`、`GMM2`、`RunGmm2EpilogueAndReturn`、
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
    dispatchGroupReady         // expert-ready；M3O 不恢复 scoreboard
    workerReadyCounters
    timeoutCounters
    subTileReturnPlan          // M2 定义 tile/sub-tile 到 owner segment 的映射；M3O.5/M3O.6 验证执行能力
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
8. `dispatchGroupReady`、`subTileReturnPlan/subTileOwnerSegments/subTileReady`
   是 M2/M3O overlap 共享账本，不能在 M3O 重新定义 offset、row range 或 owner segment 语义。
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
| ready/timeline/counter | raw GM typed view + explicit publish order | `dispatchGroupReady/subTileReady/timeline` 是控制面账本；ready publish 必须记录 producer/consumer 和 timeout locator，不通过 helper 隐藏 readiness 或 remote copy |

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
| AIC↔AIV per-group/per-tile 握手 | `pto::Event<SrcOp, DstOp>::Init<CrossCoreId>() / Wait<CrossCoreId>()`（封装 `ffts_cross_core_sync` / `wait_flag_dev`），或 `TSYNC_CVID` | `CrossCoreSetFlag/WaitFlag<0x2>(flagId)` | M3O.6 的 `GMM1↔Activation↔GMM2` sync-group 放行 |
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
  stage 边界做 overlap-off correctness；但 M3O.6 起，`GMM1 -> ActivationQuant -> GMM2` 等 overlap 边必须改用
  per-`CrossCoreId` 的 `pto::Event` 握手或 `TSYNC_CVID`，否则属于"串行未 overlap"，不能声称 M3 overlap 完成。
- 只有当某个 overlap 点确实超出上述 PTO 能力时，才标记 primitive-gap blocked，不能把 AscendC flag API 藏进本项目代码。

## 11. Swiglu/quant 粗细粒度同步策略

Swiglu 是必做计算，优化点不是减少计算量，而是避免 AIV 的 Swiglu/quant 工作推迟 GMM2 和 Combine 的关键路径。
在量化路径中，Swiglu 前后还有 per-token scale、dequant、requant、cast 等 AIV 工作；如果每个 expert 都独立同步，
会消耗大量同步事件并打碎 AIV 调度；如果一次等待太多 expert，又会让前面已经 ready 的 row range 无法及时进入 GMM2。

本项目采用三层粒度：

1. GMM/AlltoAll 仍按细粒度展开，因为这是融合收益来源：Dispatch-GMM 至少按 expert group ready，GMM-Combine
   在 M3O.5/M3O.6 进一步按 owner segment 或 tile/sub-tile ready。
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
M3O.4/M3O.6 才启用 group 内 AIV 并行和 `gmm1SyncGroupReady` 消费。

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
本项目当前固定 `warmup_iters=0`、`measure_iters=1`，不暴露成 run script 参数。

1. 每个性能用例默认不执行 warmup；如 host 端手动覆盖 warmup，warmup 不进入统计。
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
  warmup_iters = 0
  measure_iters = 1
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
| init_routing 在 `ffn.md` 是 route/sort/count/srcToDst/gather+quant 子系统；本项目用 offset-table pack+gather 等价替代 sort | M3O.2 的 dispatch 多 AIV 覆盖 route/count/pack/gather worker 分摊证据，不实现 multi-core merge sort | 把 PTO 简化误读成"必须补 sort 子系统"，或只把 `TGET` 循环并行化就声称 dispatch 多核 |
| count 同步是 dispatch 的前置点对点协议 | `tokenPerExpertMatrix`、count ready、prefix/cumsum 必须先对齐，再 remote gather | 把 count/prefix 做成 host barrier 或临时 host copy |
| SwiGLU 是 AIV 工作，按 sync group 粗细结合 | M2.5 固定 `swigluSyncGroups/dequantSum`，M3O.4/M3O.6 只打开 worker 分摊和 group overlap | 到 M3O 重排 activation row layout |
| epilogue pipe 有 prefill/drain 生命周期 | M3O.4/M3O.5/M3O.7 要记录 SetFlag/Finalize 或等价 PTO pipe lifecycle evidence | 只看 final output pass，不证明 pipe 没有悬空/脏 flag |
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
| 数据路径 | `RoutePackQuantLocal -> GatherDispatchToGmm1Input -> GMM1 -> ActivationQuant -> GMM2 -> RunGmm2EpilogueAndReturn -> RestoreOutput` 全链在 fused graph 内 | M3O 级 active worker/tile/overlap 证据 |
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

### 13.4 当前优化入口

历史 M3/M3N 任务列表已从 active 任务入口清空。后续并发和多核优化只按第 14 章的 M3O.1-M3O.7 推进：
先修 report 真实性，再做 dispatch/count/prefix、GMM tile、activation/restore、combine worker policy，最后重新打开
sync-group / expert 粒度 overlap 并固化回归矩阵。

M3O 不恢复 M3R/M3S，也不恢复 M3N.10 scoreboard async。dispatch -> GMM1 继续使用 expert-ready；更细粒度优化只发生在
expert 内部 work item 分配、GMM tile 调度、activation/combine worker policy 和既有 ready 边上的 overlap。

### 13.5 Combine V1/V2 与 Sub-Tile 的本项目落点

本项目已经在 M2 固定 `ReturnSegmentPlan/OwnerSegment/subTileReady` schema，因此 M3 不再重新设计 combine offset。
后续只在同一 schema 上区分两类执行形态：

| 形态 | 适用目标 | 本项目任务 | 验收重点 |
| --- | --- | --- | --- |
| V1 continuous owner segment | 大 shape / prefill 方向，尽量合并连续目的 rank、source rows、destination rows | M3O.5 | 多 AIV 分摊连续 owner segment，减少单 worker return 尾巴，记录 SyncAll/CV wait 空泡 |
| V2 Sub-Tile/stride | 小 shape / decode 方向，AIV consumer 与对应 AIC tile 对齐，降低全核同步空泡 | M3O.5/M3O.6 | 同步 strided/multi-segment `TPUT` 已确认可在 A3 表达，功能可 enabled；仅"异步 SDMA stride overlap"是 primitive-gap blocked |

Sub-Tile 的价值是减少同步等待，不是追求更大的 DMA。若某个 shape 下 Sub-Tile 导致 segment 过碎，优先调
sub-tile 粒度、coalesce 规则或回落 V1 continuous segment；不能把多个 token owner 混写到同一远端连续地址，也不能
回到“先全量 `gmm2Out`，再 copy combine”的二阶段主路径。

### 13.6 Evidence matrix

M2/M3O 的 accepted 不能只靠 final output pass。下面字段用于把 correctness、并发和 blocked 原因分开：

| 能力 | 证明字段 |
| --- | --- |
| single fused 主路径 | `stage_graph_mode=single_fused_mpmd`、`single_fused_payload_migrated=true`、`multi_launch_debug_only=true` |
| AIC tile ownership | `gmm_tile_is_aic_work_unit=true`、`gmm_tile_tasks`、`gmm_active_aic_blocks`、`gmm_l1/l0_tile_shape` |
| group/tile 正交 | `group_is_sync_boundary=true`、`gmm_all_aic_visit_all_groups=true` 或等价 tile assignment dump |
| dispatch payload 多 AIV | per-worker route/count/pack/gather rows、source rank ranges、processed/skipped counter |
| activation payload 多 AIV | `swiglu_group_tile_ranges`、activation tile owner、requant processed rows、pipe prefill/drain |
| combine payload 多 AIV | owner segment worker assignment、segment coalesce count、return `TPUT` count by worker |
| dispatch -> GMM1 expert ready | `dispatchGroupReady[expert]` producer/consumer count、empty expert skip、timeout locator |
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

M3O accepted 表示：在 M2/M3 已固定边界上，重新证明 active worker/tile 分摊和 overlap 的真实执行证据；无法完成的点有
明确 primitive-gap/env-gap、locator 和不影响 M4 correctness 的回退口径。M3O 的核心验收不是“打开开关”，而是能用
counter/timeline 说明每个 expert、sync-group、tile、sub-tile 由谁生产、谁消费、等了多久、为什么等。M3O accepted
不要求 overlap on 的 E2E 性能立即优于 off；E2E delta 只需在 correctness pass 后如实记录。

## 14. 当前任务分解与阶段验收标准

本章只保留当前待执行任务。历史 M0/M1/M2/M3/M3N 任务列表已按用户要求从 active design task list 清空；历史验收事实以
`reports/`、git history 和当前代码为准，不再作为后续 agent 领取任务的入口。

当前重新编号后的任务只覆盖 M3O 优化收口和 M4 最终收口。M3O 的输入依据是
`ffn_original_code_split_report.md` 的原始 FFN 代码级调研，以及当前 a8w8 active source 的实际行为。

阶段导航：

| 阶段 | 完成什么 | 关闭口径 |
| --- | --- | --- |
| M3O | 对照原始 FFN，把 a8w8 各计算阶段的 active 多核切分、同步粒度、report 证据和 overlap 差距收口。 | M3O.1-M3O.7 全部 accepted；每个任务都跑通统一 small/large case；report 不再把 launch/resource claim 当作 active payload 并行。 |
| M4 | 最终回归命令、文档和 TASKS 状态收口。 | M4.1 accepted；DESIGN/TASKS 不保留过期任务入口，无 open P0/P1 或 `needs_user_decision`。 |

硬约束：

- M3R/M3S 两条实验路径已丢弃，不恢复其代码、脚本或验收入口。
- M3N.10 scoreboard async 已丢弃，不作为优化路线；dispatch -> GMM1 保持原始 FFN 的 expert 粒度 ready 即可。
- 不恢复 producer status、scoreboard min-status、scoreboard wait/timeout 等 runtime 代码或 report 字段。
- 不改变 M2/M3 已固定的 row order、capacity/drop、owner segment schema、host reference 和 final output 语义。
- `TPUT_ASYNC` stride 仍按 primitive gap 记录；不能引入 Catlass/AscendC fallback。
- 当前 bench 口径固定 `warmup_iters=0`、`measure_iters=1`。

统一验收用例：

| Case | 参数 | 要求 |
| --- | --- | --- |
| `ffn-v3-small` | `m=16, k=128, n=128, topk=2, experts=2, max-output-size=32` | 每个 M3O task 必跑；correctness pass 后才采信 counter/timeline/perf。 |
| `ffn-v3-4097` | `m=4097, k=128, n=128, topk=2, experts=2, max-output-size=8194` | 每个 M3O task 必跑；`maxOutputSize=8194` 合法，不能改小规避容量边界。 |

### 14.1 当前阶段差距总表

| 阶段 | FFN 原始代码事实 | a8w8 当前 active 事实 | 差距 / 风险 | 对应任务 |
| --- | --- | --- | --- | --- |
| route/pack/quant | init routing 是 AIV-only 多 stage 子系统；sort/count/srcToDst/gather+quant 分阶段复用 UB。 | a8w8 走 offset-table route/pack/quant；dispatch worker count 有 64B route-shard 对齐约束，large 容易退到 1 worker；active fused path 仍有 scalar helper。 | large route/quant 控制面容易串行；shard 内前序 route 回扫会放大成本；report 可能高估 PTO Vec active path。 | M3O.1/M3O.2 |
| count/prefix | count matrix 跨 rank DataAsFlag；prefix/cumsum 是后续 offset 事实源。 | 已有 metadata 和 ready，但 count merge/prefix 多处仍 main AIV 收口。 | 只优化 payload copy 不拆 count/prefix，large 仍可能卡在控制面串行点。 | M3O.2 |
| dispatch gather -> GMM1 | 每个 local expert gather 完发布 ready，GMM1 只等当前 expert。 | expert-ready 路线保留；scoreboard/M3N10 runtime 已废弃；gather 分活仍受 dispatch worker policy 限制。 | 同步粒度正确，问题在 expert 内 gather worker、ordinal/prefix 和 report 证据。 | M3O.2 |
| GMM1 | 所有 AIC 遍历 expert，tile 由 scheduler 分发，`loopIdx += coreNum`；跨 expert 用 `startCoreIdx` 接力。 | standalone GMM kernel 有 multi-AIC task loop；active fused `M2FusedRunInt8GmmTileScalar` 仍 block0-only。 | `mixed_aic_blocks` / `gmm_multiblock=true` 只是 launch 或意图，large E2E 主要被 block0 scalar GMM 限住。 | M3O.3 |
| activation / SwiGLU / requant | AIV 等 GMM1 sync flag，按 row range 做 dequant、SwiGLU、dynamic quant。 | `M3N6Gmm1ActivationOverlapEnabled` 返回 false，M3N7/M3N8 链式 disabled；activation worker 复用 dispatch assignment。 | M3N.2 的历史 worker 概念不能代表当前 active overlap；需要 stage-local worker 和真实 ready 证据。 | M3O.4/M3O.6 |
| GMM2 | 与 GMM1 一样按 tile 多 AIC 分发，消费 activation ready。 | active fused GMM2 同样调用 block0 scalar runner；sync-group intersect 代码存在但 enable 被 false 链挡住。 | 没有 active 多 AIC tile 并行；`syncIdx -> expert -> tile task` 证据不能只停留在 disabled 代码。 | M3O.3/M3O.6 |
| combine return | V1 连续 owner segment 适合大 shape；V2 sub-tile/stride 适合小 shape 和减少空泡。 | 同步 strided sub-tile 能力存在；`TPUT_ASYNC` 仍是 primitive gap；combine worker 仍复用 dispatch assignment。 | 有同步 sub-tile 功能，但没有 full async combine；large 优先 V1，小 shape 可 V2。 | M3O.5/M3O.6 |
| restore | combine 后按 token 切分 topK weighted reduce。 | restore 当前按 8 AIV token shard；per-worker token/route 证据不足。 | restore 已有多 AIV 轮廓，但不是主耗时根因；补证据即可。 | M3O.4/M3O.7 |
| report/evidence | 判断依据应是实际 producer/consumer 粒度和 worker/tile ownership。 | host 仍可能打印 `route_quant_impl=pto_vec...`、`gmm_multiblock=true`、`mixed_aic_blocks` 等易误读字段。 | 先修 report truth，否则无法判断优化是否真实生效。 | M3O.1 |

### 14.2 M3O 任务列表

#### M3O.1 Report 真实性与 stale claim 清理

目标：把 report 从“launch 资源/设计意图”改成“requested / available / active 三层事实”，先解决证据可信度。

文件范围：`host/main.cpp`、`scripts/run_a3.sh`、必要时 `include/moe_dispatch_combine_a8w8_types.hpp`。

任务：

- 将 `mixed_aic_blocks`、`mixed_aiv_blocks`、`gmm_multiblock`、`route_quant_impl` 等字段改为不误导的 requested/claim 字段。
- 新增或修正 active 证据字段：`gmm1_active_aic_blocks`、`gmm2_active_aic_blocks`、`dispatch_active_aiv_workers`、
  `activation_active_aiv_workers`、`combine_active_aiv_workers`、`restore_active_aiv_workers`。
- rank-local structured report 要避免 mpirun stdout interleave 破坏机器判读。
- 删除或禁用 scoreboard/M3N10 report 字段，不保留“未来可能恢复”的 active claim。

验收：

- 统一 small/large 两个 case 都 correctness pass。
- 当前 GMM 若仍 block0-only，report 必须显示 `gmm*_active_aic_blocks=1`，不能因 launch 多 block 误报 active 多核。
- 当前 worker 若退到 1，payload parallel 字段必须为 false 或 active worker=1。
- 依赖扫描只覆盖源码/CMake/scripts，不能有 Catlass/AscendC fallback。

#### M3O.2 Dispatch/count/prefix stage-local 分活

目标：保留 dispatch -> GMM1 expert-ready 同步粒度，修正 dispatch 内部 worker policy、count/prefix 串行收口和 ordinal 计算热点。

文件范围：`kernel/moe_dispatch_combine_a8w8_kernel.cpp`、`kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`、
`host/main.cpp`、必要时 `host/reference.hpp`。

任务：

- 拆出 dispatch 专属 worker policy；scale false-sharing fallback 只影响 dispatch，不再拖累 activation/combine。
- 为每个 worker/expert 生成 `workerExpertCount`，再生成 `workerExpertPrefix`。
- packed row 计算改为 `expertBase + workerExpertPrefix + localOrdinalInWorkerExpert`，移除 shard 内前序 route 回扫热点。
- count/prefix publish/wait 和 reducer 证据按 rank/expert/worker 打点，避免 main AIV 隐式串行被 report 掩盖。
- invalid/inactive/over-capacity route 的 `expandedRowIdx`、drop 计数和 reference 语义不变。

验收：

- 统一 small/large 两个 case 都 correctness pass，large 使用 `maxOutputSize=8194`。
- large route/pack 阶段不再依赖 shard 内前序 route 回扫。
- worker=1 fallback 与 multi-worker path 的 `expandedRowIdx`、`tokenPerExpertMatrix`、payload/scale 和 reference 一致。
- zero-token、inactive-mask、over-capacity 语义保持可回归。

#### M3O.3 GMM1/GMM2 active 多 AIC tile scheduler

目标：把 active fused GMM 从 block0 scalar correctness path 推进到真实多 AIC tile work item。

文件范围：`kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`、`kernel/moe_dispatch_combine_a8w8_gmm_kernel.cpp`、
`kernel/moe_dispatch_combine_a8w8_kernel.cpp`、`host/main.cpp`。

任务：

- 复用 standalone GMM 的 `for (taskId = blockIdx; taskId < taskCount; taskId += blockNum)` 基础分发。
- 移除 `M2FusedRunInt8GmmTileScalar` 内部 block0-only 过滤，把过滤/skip 放到 scheduler。
- 每个 task 只写唯一 `(expert,row tile,n tile)` accumulator 区域，避免 atomic 或重复写。
- GMM1 ready 发布仍受 dispatch expert-ready 控制；GMM1 -> activation 和 activation -> GMM2 的细粒度 ready 等 M3O.6 再打开。
- 第一阶段只要求 round-robin 正确和 active 多 AIC 证据；FFN `startCoreIdx`/swizzle 接力可作为后续细化。

验收：

- 统一 small/large 两个 case 都 correctness pass。
- multi-tile large case 中 `gmm1_active_aic_blocks>1` 或 `gmm2_active_aic_blocks>1`。
- timeline/report 能 dump 至少首个和末个 GMM task 的 `blockIdx/taskId/expert/rowBegin/nBase`。
- large E2E 记录但不作为本任务唯一 pass/fail。

#### M3O.4 Activation/SwiGLU/requant 与 restore 证据修正

目标：把 activation/requant 的 active work 放回 AIV stage-local worker，并补齐 restore worker 证据。

文件范围：`kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`、`kernel/moe_dispatch_combine_a8w8_kernel.cpp`、
`host/main.cpp`。

任务：

- 拆出 activation worker policy，按 `swigluGroupDesc` row range 或 sync group 做 grid-stride 分活，不复用 dispatch assignment。
- AIV worker 只消费自己负责的 row range，不改变 `gmm2InputInt8`、`gmm2PerTokenScale`、`swigluOut` layout。
- activation ready 只能在对应 row/sync group 全完成后发布；未打开 overlap 时也要如实记录 coarse/barrier path。
- restore 保持 8 AIV token shard，但新增 per-worker token count、route count、skipped route count 和 non-overlap 证据。

验收：

- 统一 small/large 两个 case 都 correctness pass。
- activation active worker range 无重叠/无缺口；empty range 可 skipped 但不能造成等待不满足。
- `activation_pipe_stages`、prefill/drain 字段只按 active 实现如实输出，不能把 scalar path 写成 pipe 化。
- restore report 包含 active worker、token range 和 processed/skipped route 证据。

#### M3O.5 Combine stage-local worker policy 与 V1/V2 选择

目标：让 combine 的 worker policy 独立于 dispatch，并按 shape 选择 V1 continuous 或 V2 sub-tile/stride。

文件范围：`kernel/moe_dispatch_combine_a8w8_kernel.cpp`、`kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`、
`host/main.cpp`。

任务：

- V1 large/default：按 `OwnerSegment` 或 `(expert, tokenOwner, hidden chunk)` 分 worker。
- V2 small/sub-tile：按 `subTileReturnPlan.tileId` 或 `subTileOwnerSegments.segmentId` 分 worker。
- `M3N11TransferHalfSubtileStride` 保持同步 strided `TPUT` 语义；`TPUT_ASYNC_flat_contiguous_1d` 继续记录为 primitive gap。
- combine notify 必须在 payload/segment counter visibility 之后发布，不能为了性能提前发 done。

验收：

- 统一 small/large 两个 case 都 correctness pass。
- small 或 synthetic case 中能看到 `combine_active_aiv_workers>1`，且 segment 不全落在 worker0。
- large case 明确 report `combine_mode=continuous_segment`、`segment_stride` 或 `subtile_stride`，并写明选择原因。
- return segment counter checksum 与 final correctness report 一致。

#### M3O.6 重新打开 sync-group / combine overlap

目标：在 M3O.1-M3O.5 的 active worker/tile 证据可信后，再按 FFN 粒度重新打开 overlap。

文件范围：`kernel/protocol_core.hpp`、`kernel/moe_dispatch_combine_a8w8_mixed_spike.cpp`、`host/main.cpp`。

任务：

- dispatch -> GMM1 不改粒度，继续 expert-ready；不恢复 scoreboard。
- GMM1 -> activation：sync-group 粒度，优先 GM-poll ready；只有 flag 资源和时序能证明安全时才使用 `pto::Event`。
- activation -> GMM2：sync-group 粒度，必须 intersect 回 GMM2 tile task，不允许丢 expert 投影。
- GMM2 -> combine：expert 粒度；combine 可用 V1 owner segment 或 V2 sub-tile refinement。
- 每条边输出 producer/consumer counter、first-consume-before-last-ready 证据和 timeout dump 信息。

验收：

- 统一 small/large 两个 case 都 correctness pass。
- 任一 overlap enabled 前，上游 active worker/tile counter 必须可信。
- report 中 `gmm1_activation_overlap_granularity=sync_group`、`activation_gmm2_overlap_granularity=sync_group`、
  `gmm2_combine_overlap_granularity=expert` 与实际路径一致。
- `full_async_combine_claim=false` 保留，直到 PTO primitive gap 关闭。

#### M3O.7 回归矩阵、性能拆解与交付口径

目标：把 M3O 优化串成稳定回归入口，明确 correctness、active evidence、timeline 和 E2E 的采信顺序。

文件范围：`scripts/run_a3.sh`、`host/main.cpp`、`TASKS.md`、`reports/M3O.md`。

任务：

- 固化 M3O 最小回归命令，默认 `warmup_iters=0`、`measure_iters=1`。
- 必跑 case 包含统一 small/large，外加 small、balanced、skewed、zero-token、inactive-mask、over-capacity。
- large 若仍为秒级，timeline 必须拆出 route、GMM1、activation、GMM2、combine、restore 的主耗时。
- `reports/M3O.md` 只记录 pass/fail、关键 metric 摘要、blocked 原因和下游提示，不粘 stdout 原文。

验收：

- 统一 small/large 两个 case 和补充回归 case 全部 correctness pass。
- 每个 active optimization 都能在 report/timeline 中证明 worker/tile payload 实际生效。
- 性能数据只在 correctness pass 后采信；M3O.6 以后至少一条 overlap 边有真实 producer/consumer 错峰证据。
- 依赖扫描无输出。

### 14.3 M4 最终收口

#### M4.1 文档、状态和最终回归命令收口

依赖任务：M3O.7。

文件范围：`DESIGN.md`、`TASKS.md`、`scripts/run_a3.sh`、`reports/M4.md`。

任务：

- 保留最终验收命令和当前 known limits。
- 确认 DESIGN/TASKS 只保留当前任务入口，不再列历史 M0-M3N 任务清单。
- 标注当前 A8W8 backend 的真实状态：哪些阶段 active 多核，哪些只是 launch/resource claim，哪些仍是 primitive gap。
- 关闭或转交 open P0/P1 issue。

验收：

- M3O.7 的统一 small/large 和补充回归可通过一条最终命令复现。
- DESIGN/TASKS 不声称未完成能力已完成。
- `reports/M4.md` 是轻量摘要，不复制长日志。
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
| M2 未前置 MegaMoE 合并点 | M3O 需要重写 dispatch/combine 数据布局，overlap gap 过大 | M2 必须完成 route/pack/quant 合并、GMM1 contiguous input、GMM2 epilogue/return 合并；M3O 只改调度和 ready 粒度 |
| `swigluSyncGroups/dequantSum` 到 M3O 才定义 | Activation/GMM2 overlap 需要重做 row range，M3O.4/M3O.6 无法只打开 signal | M2.1 预留字段，M2.5 固定 group plan 和 row range；M3O 只启用/校验 signal 和 counter |
| Dispatch-GMM 同步粒度后改 | 若在优化中恢复细粒度依赖，会影响 GMM1 启动条件和 timeout 定位 | 当前固定为 `dispatchGroupReady[expert]` expert-ready 语义；scoreboard runtime/layout/report 不恢复 |
| GMM-Combine owner segment 到 M3O 才设计 | combine offset 或 return payload layout 被重写，M2 correctness 不能证明 M3O path | M2.7a 固定 `ReturnSegmentPlan/OwnerSegment/subTileReady` 语义；M3O.5/M3O.6 只验证 stride/multi-segment remote write 能力 |
| PTO A3 remote stride 能力不足 | 无法实现文章级 Sub-Tile 非连续 combine | M3O 单独做 `GlobalTensor Shape/Stride + TPUT` 能力验证；不足则 blocked，不写 AscendC fallback |
| timeline 改动污染 payload/control layout | 为观测新增字段时破坏 row/order/offset，导致 correctness 和 timeline 互相影响 | M0.4/M2.1 预留 timeline/timestamp 区域；M3O.7 只使用或扩容该区域，不改 payload 或 Sub-Tile offset |
| small shape 前同步开销 | 性能可能差于独立通信加计算的基线 | 验收区分 correctness/overlap/性能；small shape 不作为性能收益门槛 |
| MTE/带宽抢占 | compute/comm 并行时 GMM 耗时膨胀 | timeline 打点拆解，记录 standalone 与 overlap 耗时差 |
| capacity overflow | 写越界或 combine 丢 token | host precheck + device counter + fail fast |
| empty expert segment | wait 永不满足 | count 为 0 的 segment 显式 skip，同时保证 rank-level signal 仍发布 |
| HCCL window offset 不一致 | remote pointer 错误 | 所有 rank 使用相同 layout，host 检查 winSize |
| async 调度过早引入 | correctness 难定位 | M1-M2 先用最终 stage 图的 `overlap_mode=off` 稳定，M3O 在同一 stage graph 上验证或启用既有 producer/consumer 边 |
| report 或 TASKS 记录过多运行日志 | 状态台账膨胀，后续 agent 难以判断真实状态 | stdout 保持运行时证据；`reports/Mx.y.md` 只写 pass/fail、关键摘要、blocked 原因和下游提示 |
