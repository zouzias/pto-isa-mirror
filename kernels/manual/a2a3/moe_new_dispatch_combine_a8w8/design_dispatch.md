# moe_new_dispatch_combine_a8w8 dispatch 阶段设计

本文档只定义 `moe_new_dispatch_combine_a8w8` 的 dispatch 阶段：前重排/initquant 已经把本 rank 的 token
按 global expert 密排、量化，并写入 peer-visible window；dispatch 阶段要把属于本 rank 本地 expert 的 rows 从
各 token owner rank 拉过来，直接形成 GMM1 输入。

设计原则是按 FFN `dispatch_ffn_combine` 的业务语义重写，不搬 AscendC 类和临时串行实现。PTO 版本只替换数据搬运、
metadata 构造和同步表达方式。

## 0. 性能定位和硬目标

dispatch 阶段不是 demo/toy correctness kernel。`dispatch-only` small/large 全 0 mismatch 只是进入下一轮开发的
最低门槛，不能作为生产化完成标准。最终目标是对标 FFN `dispatch_ffn_combine` 的商用性能能力：同 shape、同 rank
数、同设备、同数据分布下，PTO dispatch 的 D0-D4 e2e 不能出现明显劣化，并且要具备 FFN dispatch 的并行调度能力。

商用性能验收必须同时满足：

- 建立 FFN baseline：用 FFN 原实现或等价 stop stage 记录同口径 dispatch e2e，包含 count publish/wait、
  metadata、payload/scale gather、expert ready。
- PTO dispatch e2e 对标 FFN：默认硬门槛是 `PTO_dispatch_e2e <= 1.1 * FFN_dispatch_e2e`；如果 FFN 没有独立
  dispatch 计时，必须先补 baseline 计时或用同一 stop stage 派生，不能用 correctness 时间替代。
- 多 AIV gather 是生产化要求：不能长期停留在单 AIV 遍历所有 expert/tokenOwner 的路径。
- hot expert 必须 rowBlock 化：`task = (localExpert, tokenOwnerRank, rowBlock)`，避免单个大 expert 把 dispatch
  拖成串行长尾。
- D3 payload/scale 搬运必须保持 PTO row-block `TGET`/local tile copy，不能为了通过性能或定位问题退回 scalar
  byte loop。
- GMM1 overlap 是性能路径的一部分：correctness 阶段允许 D4 后临时全同步收口计时；商用性能验收时必须删除这个
  临时全同步，用 expert-ready 或等价分组 ready 驱动 GMM1 消费。
- 常规日志只保留 summary、first mismatch 和 e2e；性能定位只看同步点摘要、worker 分工、rowBlock 负载和带宽，
  不加 token/row 级 dump。

因此，本文后续的 D0-D4 correctness 设计只说明业务正确性落点；性能完成必须额外关闭“单 AIV / 无 rowBlock /
无 overlap / 无 FFN baseline”的缺口。

## 1. 阶段边界

### 1.1 输入

来自前重排阶段：

- `peerWindow.dispatchPayload`：本 rank 作为 token owner 时生成的 int8 packed rows。
- `peerWindow.dispatchScale`：每个 packed row 对应的 dynamic quant scale。
- `peerWindow.tokenPerExpertMatrix[myRank][expertOwnerRank][localExpert]`：本 rank 发往每个 expert 的 route 数。
- `expandedRowIdx` / `packedRowToRouteIndex`：后续 combine/restore 验收需要的映射，不在 dispatch 内重写。

来自 host/layout：

- `rankNum` / `myRank` / `expertPerRank` / `maxOutputSize`。
- `rowBytes = AlignOrPadded(K)`，payload 行跨度必须和前重排写入 peer window 的行跨度一致。
- `peerWindowLayout`、`workspaceLayout`、`HcclDeviceContext`。

### 1.2 输出

dispatch 阶段输出给 GMM1 和 combine 使用：

- `workspace.gmm1InputInt8`：本 rank 本地 expert-major 连续 rows，直接作为 GMM1 A 矩阵输入。
- `workspace.routingPerTokenScale`：与 `gmm1InputInt8` 行号一致的 scale。
- `workspace.dispatchOffset[localExpert]`：本地 expert 在 `gmm1InputInt8` 中的起始行。
- `workspace.cumsumMM[tokenOwnerRank][localExpert]`：当前 local expert 内，按 token owner rank 累加后的目的端前缀。
- `workspace.preSumBeforeRank[tokenOwnerRank][localExpert]`：FFN 语义的源端 packed row 起点。
- `workspace.expertTokenNums[localExpert]`：当前 local expert 有效 rows，供 GMM1 的 M 维使用。
- `workspace.dispatchGroupReady[localExpert]`：当前 local expert 的 dispatch rows 已经写完，可以交给 GMM1。

`preSumBeforeRank` 这个名字必须保持 FFN 语义：它表示在 token owner rank 的 packed buffer 里，当前
`globalExpert = myRank * expertPerRank + localExpert` 前面有多少 rows。目的端写入位置不要用它表达，目的端位置
由 `dispatchOffset + cumsumMM prefix` 表达。

## 2. FFN dispatch 业务逻辑

FFN 的 dispatch 不是重新路由，也不是排序。前面的 `moe_init_routing_quant_v2` 已经完成密排和量化。dispatch 做的
事情可以直接写成下面的业务循环：

```text
for localExpert in [0, expertPerRank):
  for tokenOwnerRank in [0, rankNum):
    rows = tokenPerExpertMatrix[tokenOwnerRank][myRank][localExpert]
    srcStart = preSumBeforeRank[tokenOwnerRank][localExpert]
    dstStart = dispatchOffset[localExpert] +
               prefix_before_tokenOwner_in_this_localExpert
    copy rows:
      remote(tokenOwnerRank).dispatchPayload[srcStart : srcStart + rows]
        -> workspace.gmm1InputInt8[dstStart : dstStart + rows]
      remote(tokenOwnerRank).dispatchScale[srcStart : srcStart + rows]
        -> workspace.routingPerTokenScale[dstStart : dstStart + rows]
  dispatchGroupReady[localExpert] = 1
```

原 FFN 代码里的对应关系：

- `RunRoutingImpl`：
  - 调 `moe_init_routing_quant_v2` 生成本 rank 的 packed payload、scale、local count row。
  - `CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2` 把 count row 发布到各 rank，并构造
    `preSumBeforeRank`。
  - `GetCumsumForMMAIV` 构造 GMM 需要的 local expert prefix。
- `RunDispatchGatherImpl`：
  - 外层遍历 `groupIdx`，也就是本地 expert。
  - 内层遍历 `dstEpIdx`，实际是 token owner rank。
  - 用 `tokenPerExpert[dstEpIdx][rank][groupIdx]` 决定拉多少 rows。
  - 用 `cumsumMM` 和 `prevGroupSum1` 决定写到本地 GMM1 哪一段。
  - 用 `preSumBeforeRank` 等价的源端前缀决定从 token owner 的 packed buffer 哪一段开始拉。
  - 每个 expert rows 完成后置 ready flag，GMM1 可以按 expert 消费。

PTO 版保留这条业务逻辑，只把 `CopyGMToGMPerToken` 换成 PTO row-block `TGET` / local PTO copy，把 AscendC 同步
换成 PTO-COMM notify/wait 和本项目已有 AIV/AIC 同步封装。

## 3. 核心 metadata 定义

统一使用下面的 owner 术语，避免 `src/dst rank` 混用：

- `tokenOwnerRank`：token 原始所在 rank，也是前重排 packed payload 的生产者。
- `expertOwnerRank`：expert 权重所在 rank，也是 GMM1 的执行 rank。
- `localExpert`：`expertOwnerRank` 内部 expert 编号。
- `globalExpert = expertOwnerRank * expertPerRank + localExpert`。

### 3.1 count matrix

```text
tokenPerExpertMatrix[tokenOwnerRank][expertOwnerRank][localExpert]
```

含义：`tokenOwnerRank` 有多少 route 发往 `expertOwnerRank` 的 `localExpert`。

前重排阶段只生产本 rank 的一行：

```text
tokenPerExpertMatrix[myRank][*][*]
```

dispatch 开始后，每个 rank 都需要看到所有 token owner 的 count row，所以需要 count publish/wait。

### 3.2 源端 packed row 前缀

```text
preSumBeforeRank[tokenOwnerRank][localExpert] =
  sum_{globalExpert' < myRank * expertPerRank + localExpert}
      tokenPerExpertMatrix[tokenOwnerRank]
                          [owner(globalExpert')]
                          [local(globalExpert')]
```

它是从 `tokenOwnerRank` 的 `dispatchPayload/dispatchScale` 读取当前 global expert rows 的 `srcStart`。

这个值来自 FFN `CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2`，不能改成目的端 prefix。combine
阶段写回 token owner 时也要靠同一个源端/global-expert 顺序解释 row order。

### 3.3 目的端 GMM1 前缀

```text
cumsumMM[tokenOwnerRank][localExpert] =
  sum_{r <= tokenOwnerRank} effectiveRows(r, myRank, localExpert)

dispatchOffset[localExpert] =
  sum_{e < localExpert} expertTokenNums[e]

dstStart =
  dispatchOffset[localExpert] +
  (tokenOwnerRank == 0 ? 0 : cumsumMM[tokenOwnerRank - 1][localExpert])
```

`effectiveRows` 按 FFN 的 `maxOutputSize` 目的端容量规则裁剪。源端 packed row 前缀仍按 raw count matrix 计算，
目的端只少拉被裁掉的 tail rows。

`expertTokenNums[localExpert] = cumsumMM[rankNum - 1][localExpert]`，是 GMM1 当前 group 的 M。

## 4. PTO 阶段拆分

dispatch 阶段拆成 5 个小阶段。它们是业务顺序，不要求写成 5 个 kernel。

| 阶段 | 主要核 | 职责 | PTO 实现 |
| --- | --- | --- | --- |
| D0 `PublishCounts` | AIV | 把本 rank count row 发布给所有 expert owner rank | `GlobalTensor<int32_t>` + `comm::TPUT`，完成后 `TNOTIFY`/ready signal |
| D1 `WaitCounts` | AIV | 等待所有 token owner count row 在本 rank 可见 | `TWAIT/TTEST`，等待后才能读 `tokenPerExpertMatrix` |
| D2 `BuildDispatchMetadata` | AIV | 构造 `preSumBeforeRank`、`cumsumMM`、`dispatchOffset`、`expertTokenNums` | PTO Vec tile 批量 load/add/store；small 可由 owner core 做，large 可按 localExpert 分片 |
| D3 `GatherDispatchToGmm1Input` | AIV | 遍历本地 expert 和 token owner rank，拉 payload/scale 到 GMM1 输入 | remote 用 `comm::TGET` row-block，local 用 PTO tile copy |
| D4 `PublishDispatchReadyAndTiming` | AIV | expert ready，记录 dispatch e2e 结束 | 每个 expert ready flag；当前阶段末尾临时全同步后记录 e2e |

### 4.1 D0/D1 count publish/wait

按 FFN 的方向，count row 由 token owner rank 发布到每个 expert owner rank：

```text
for expertOwnerRank assigned to this AIV:
  if expertOwnerRank != myRank:
    TPUT local tokenPerExpertMatrix[myRank][*][*]
      -> remote(expertOwnerRank).tokenPerExpertMatrix[myRank][*][*]
  notify countReadySignal[myRank] on expertOwnerRank
```

本 rank 等待所有 token owner：

```text
for tokenOwnerRank in [0, rankNum):
  wait countReadySignal[tokenOwnerRank]
```

等待完成后，`tokenPerExpertMatrix[*][myRank][*]` 和构造 `preSumBeforeRank` 所需的完整 count row 都可读。

### 4.2 D2 metadata 构造

metadata 分两类构造：

1. 源端前缀 `preSumBeforeRank`：按每个 `tokenOwnerRank` 的完整 global expert count row 做前缀和。
2. 目的端前缀 `cumsumMM/dispatchOffset/expertTokenNums`：只看 `expertOwnerRank == myRank` 的 local experts，并应用
   `maxOutputSize` 裁剪。

伪代码：

```text
dispatchCursor = 0
for localExpert in [0, expertPerRank):
  dispatchOffset[localExpert] = dispatchCursor
  expertRows = 0

  for tokenOwnerRank in [0, rankNum):
    rawRows = tokenPerExpertMatrix[tokenOwnerRank][myRank][localExpert]
    rows = clip_to_max_output(dispatchCursor + expertRows, rawRows, maxOutputSize)
    expertRows += rows
    cumsumMM[tokenOwnerRank][localExpert] = expertRows

  expertTokenNums[localExpert] = expertRows
  dispatchCursor += expertRows

for tokenOwnerRank in [0, rankNum):
  prefix = 0
  for globalExpert in [0, rankNum * expertPerRank):
    if owner(globalExpert) == myRank:
      preSumBeforeRank[tokenOwnerRank][local(globalExpert)] = prefix
    prefix += tokenPerExpertMatrix[tokenOwnerRank]
                                [owner(globalExpert)]
                                [local(globalExpert)]
```

D2 完成后做一次 AIV-only 同步，保证所有 gather worker 看到一致 metadata。

### 4.3 D3 payload/scale gather

D3 是 dispatch 的核心，业务上就是 AIV workers 拉数据：

```text
for localExpert assigned to this worker:
  globalExpert = myRank * expertPerRank + localExpert
  for tokenOwnerRank in [0, rankNum):
    current = cumsumMM[tokenOwnerRank][localExpert]
    previous = tokenOwnerRank == 0 ? 0 : cumsumMM[tokenOwnerRank - 1][localExpert]
    rows = current - previous
    if rows == 0:
      continue

    srcStart = preSumBeforeRank[tokenOwnerRank][localExpert]
    dstStart = dispatchOffset[localExpert] + previous

    if tokenOwnerRank == myRank:
      local PTO tile copy payload and scale
    else:
      TGET remote(tokenOwnerRank).dispatchPayload[srcStart, K]
        -> workspace.gmm1InputInt8[dstStart, K]
      TGET remote(tokenOwnerRank).dispatchScale[srcStart]
        -> workspace.routingPerTokenScale[dstStart]
```

实现上以 row-block 为搬运单位：

- payload：`GlobalTensor<int8_t>` 描述二维 `[rows, rowBytes]`，用 `TGET` 按 tile cols 分块。
- scale：`GlobalTensor<float>` 描述二维 `[rows, 1]`，用小 tile 一次搬多行。
- local rows 不走 scalar byte loop，也用 PTO tile load/store 或统一的 local copy helper。
- remote rows 不用 scalar 逐 byte 读；当前代码里的 scalar remote copy 只能作为待替换参考，不作为验收路径。

worker 分工第一版可以按 `(localExpert, tokenOwnerRank)` 分给 AIV：

```text
taskId = localExpert * rankNum + tokenOwnerRank
worker handles taskId % workerCount == workerId
```

如果出现 hot expert，后续只把 `rows` 再切成 `rowBlock`：

```text
task = (localExpert, tokenOwnerRank, rowBlock)
```

这不是改业务逻辑，只是避免某个 expert rows 很大时单个 AIV 拉太久。

生产化路径不得停留在“第一版可以”。实际验收时必须根据 FFN baseline 和本项目 AIV 数量打开多 worker：

- 基础分工：`(localExpert, tokenOwnerRank)` 粒度覆盖所有 owner/expert。
- 大 rows 分工：当 `rows` 超过 rowBlock 阈值时，继续拆成 `(localExpert, tokenOwnerRank, rowBlock)`。
- ready 协议：rowBlock 并行后必须用 per-expert 完成计数或 AIV-only 收口，确保最后一个 rowBlock 完成后再置 ready。
- 负载均衡：任务分配要按 rows 或 rowBlock 数均衡，不能只按 expert 个数平均导致 hot expert 长尾。
- 性能报告：输出 active worker 数、rowBlock 数、最大/最小 worker rows、dispatch e2e，并和 FFN baseline 对比。

### 4.4 D4 ready 和 e2e timing

每个 `localExpert` 的所有 `(tokenOwnerRank, rowBlock)` 都完成后，才能置：

```text
dispatchGroupReady[localExpert] = 1
```

如果第一版每个 expert 只由一个 worker 负责，可以由该 worker 直接置 ready。如果后续按 rowBlock 分工，必须增加
per-expert 完成计数或 AIV-only 收口，不能让第一个完成的 worker 提前置 ready。

dispatch e2e 时间记录规则：

- 开始点：D0 `PublishCounts` 之前，也就是前重排完成、dispatch 即将开始时。
- 结束点：D3 所有 rows 搬完、D4 所有 `dispatchGroupReady` 已发布后。
- 记录者：每个 rank 只由一个固定 AIV owner core 记录，例如 `workerId == 0`。不要每个核都写一份 e2e。
- 当前验收版：D4 后可以临时加一次全同步，再由 owner core 记录 end cycle。这样 e2e 覆盖完整 dispatch 结束。
- 下一阶段启用 GMM1 overlap 后，这个临时全同步要删除，改为用 expert-ready 时间线或 host 侧总耗时证明 overlap。

host 报告可以输出每 rank dispatch cycles，并取 `max(rankCycles)` 作为跨 rank dispatch wall time。

## 5. 同步设计

dispatch 只有三个必要同步点：

1. count row 可见：`PublishCounts` 完成远端写后才能 notify；`WaitCounts` 等到 notify 后才能读 count matrix。
2. metadata 可见：`BuildDispatchMetadata` 完成后，gather workers 才能读 `cumsumMM/preSumBeforeRank/dispatchOffset`。
3. expert rows 可见：当前 expert 所有 payload/scale rows 搬完后，才能置 `dispatchGroupReady[localExpert]`。

不要为了定位卡住把并发改成串行。同步问题按同步点原理查：

- 参与者是否一致：producer/consumer 的 rank、AIV worker 集合是否一致。
- 地址是否一致：signal 地址、count matrix row 地址、peer window rank 偏移是否一致。
- 值协议是否一致：ready 初值、递增值、等待值是否一致；是否复用了上一轮旧值。
- 顺序是否一致：远端写完成后再 notify；wait 之后再读；ready 之前 rows 必须全部写完。
- 可见性是否一致：`TPUT/TGET` 完成、必要的 pipe/event/barrier、GM cache invalidate 是否在正确位置。

定位时只加同步点前后的摘要计数或 timeout summary，不加 token/row 级大 dump。卡住时看哪一个同步点的 producer
和 consumer 对不上，而不是改执行并发模式试结果。

卡住定位必须先做同步信号关系审计，再改代码。审计内容至少包括：

- 信号图：列出 `countReadySignal[tokenOwner]`、metadata AIV-only sync、`dispatchGroupReady[localExpert]`、
  GMM1 consume wait 之间的依赖方向。
- 生产者：哪个 rank、哪个 worker、在什么条件下写数据和置 signal。
- 消费者：哪个 rank、哪个 worker、等待哪个 signal 值，wait 后读取哪些地址。
- 地址关系：local peer window、remote peer window、workspace 字段 offset 是否由同一 layout 计算，host/device
  offset 是否一致。
- 值协议：初值、目标值、是否每轮递增、是否可能读到上一轮旧值。
- 顺序关系：数据写完成 -> 可见性 fence/event -> signal；signal wait 完成 -> invalidate/读数据；rows 完成 ->
  expert ready。
- 并发关系：哪些 worker 可以同时执行，哪些同步是 AIV-only，哪些同步跨 AIC/AIV 或跨 rank。

禁止的定位方式：

- 为了看是否不卡，把多 AIV 改成单 AIV、把并行改串行、关闭某个 worker 分支。
- 随机加 sleep、随机换 signal 值、随机挪 barrier。
- 加 token/row 级大 dump，用日志量掩盖同步问题。

允许的临时信息只有同步点摘要：producer/consumer id、signal 地址和值、stage id、ready count、timeout summary、
worker rowBlock 汇总。任何临时日志都必须能对应上面的同步信号图。

## 6. FFN 到 PTO 的实现映射

| FFN 代码 | 业务含义 | PTO 设计 |
| --- | --- | --- |
| `moe_init_routing_quant_v2` | 本地路由、密排、量化、count row | 已由前重排/initquant PTO 阶段完成 |
| `CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2` | 发布 count row，等待所有 rank，构造源端前缀 | D0/D1 + D2 的 `preSumBeforeRank` |
| `GetCumsumForMMAIV` | 构造本 rank local expert 的 GMM M 前缀 | D2 的 `cumsumMM/expertTokenNums/dispatchOffset` |
| `RunDispatchGatherImpl` | 本 rank 遍历 local expert，从各 token owner 拉 rows | D3 `GatherDispatchToGmm1Input` |
| `CopyGMToGMPerToken` | payload rows + scale 搬到 GMM1 输入 | PTO `TGET` row-block + local PTO tile copy |
| `CrossCoreSetFlag` 给 GMM1 | expert dispatch 完成 | D4 `dispatchGroupReady[localExpert]` |

当前新算子中已有同名或相近 helper 可以作为落点参考：

- `M3NPublishCountRowsShard` / `M3NWaitCountRowsShard`
- `M2BuildPrefixMetadata`
- `M2TGetRowsInt8` / `M2TGetRowsFloat`
- `M2GatherDispatchToGmm1Input` / `M3NGatherDispatchToGmm1InputShard`
- `M3NPublishAllDispatchExpertsReadyAfterGather`

这些 helper 的生产化要求是：metadata 命名必须对齐 FFN 语义，remote payload 不走 scalar byte loop，ready 不能早于所有
row-block 完成。

## 7. 验收条件

### 7.1 metadata

- `tokenPerExpertMatrix[*][*][*]` 与 CPU reference 一致。
- `preSumBeforeRank[tokenOwnerRank][localExpert]` 等于当前 global expert 在 token owner packed buffer 中的 raw 起点。
- `cumsumMM`、`dispatchOffset`、`expertTokenNums` 能解释 GMM1 输入每一行属于哪个 `(tokenOwnerRank, localExpert)`。
- `maxOutputSize` 裁剪后，dispatch 写入行数、GMM1 消费行数和 `expertTokenNums` 一致。

### 7.2 payload/scale

- `gmm1InputInt8[dstStart + i]` 等于对应 `remote(tokenOwnerRank).dispatchPayload[srcStart + i]`。
- `routingPerTokenScale[dstStart + i]` 等于对应 `remote(tokenOwnerRank).dispatchScale[srcStart + i]`。
- 本地 token owner 和远端 token owner 都覆盖。
- rows 为 0、tail row-block、capacity clip 都覆盖。

### 7.3 同步和 ready

- 所有 rank 都能通过 D0/D1，不出现 count wait 卡住。
- D2 后 gather 读到的 metadata 不含旧值。
- 每个 local expert 的 ready 只在该 expert 所有 rows 搬完后置位。
- GMM1 只消费 ready expert 的 row range。

### 7.4 e2e timing 和性能验收

- 输出 dispatch start/end/cycles，覆盖 D0 开始到 D4 结束。
- 每 rank 一个 owner core 记录，不接受每个核多份 e2e 混在一起。
- 当前验收版允许 D4 后临时全同步，报告中标明该同步是 dispatch 计时收口同步。
- 商用性能验收必须和 FFN 同口径 baseline 对比，默认要求 `PTO_dispatch_e2e <= 1.1 * FFN_dispatch_e2e`。
- 商用性能验收必须打开多 AIV gather、hot expert rowBlock 和 GMM1 overlap；只跑 standalone 单 AIV dispatch
  不能声明 dispatch 性能完成。

### 7.5 日志

常规验收日志只保留：

- shape/rank summary。
- metadata pass/fail。
- payload/scale pass/fail 和 first mismatch。
- ready/e2e summary。

不要保留 token/row 级 debug dump。卡住时按第 5 节同步点定位。

## 8. 任务列表

| 任务 | 内容 | 验收 |
| --- | --- | --- |
| D0.1 | 固定 FFN metadata 语义，明确 `preSumBeforeRank` 是源端 packed row prefix | 文档和代码注释/字段使用不再混用源端、目的端 prefix |
| D0.2 | 接通 `PublishCounts` / `WaitCounts` 的 PTO count row publish | 多 rank 下 count matrix 全量一致 |
| D1.1 | 实现 `BuildDispatchMetadata`，构造 `preSumBeforeRank/cumsumMM/dispatchOffset/expertTokenNums` | metadata 与 CPU reference 一致，含 capacity clip |
| D2.1 | 用 PTO row-block 实现 payload `TGET`，替换 remote scalar copy | remote payload 与 reference 一致 |
| D2.2 | 用 PTO row-block 实现 scale `TGET` 和 local tile copy | scale 与 reference 一致，本地/远端都覆盖 |
| D3.1 | 实现 expert ready：每个 expert rows 完成后置 `dispatchGroupReady` | GMM1 只读 ready expert，不卡住 |
| D3.2 | 加 dispatch e2e timing，D4 后临时全同步收口 | 报告 dispatch start/end/cycles |
| D4.1 | small/large correctness 验收 | metadata、payload、scale、ready 全 pass |
| D4.2 | 同步问题定位标准落地 | 若卡住，报告具体同步点、producer/consumer、signal 地址和值，不做串行降级实验 |
| D5.1 | 建立 FFN dispatch baseline | 同 shape/同 rank/同设备输出 FFN dispatch e2e，形成对比表 |
| D5.2 | 多 AIV gather 分工 | active AIV 不再固定为 1；worker rows 分布可解释，无长尾 |
| D5.3 | hot expert rowBlock 分工 | 不改变业务 row order，提升大 rows 并行度，ready 晚于所有 rowBlock 完成 |
| D5.4 | GMM1 overlap 性能路径 | 删除 correctness 临时全同步，用 expert-ready 驱动 GMM1，e2e 不明显劣化 |
| D5.5 | 商用性能验收 | PTO dispatch e2e 默认不超过 FFN baseline 1.1x；若超过，必须给出瓶颈和优化计划 |

第一轮开发顺序按 D0.1 -> D0.2 -> D1.1 -> D2.1/D2.2 -> D3.1 -> D3.2 -> D4.1。D5.* 不阻塞
dispatch correctness，但阻塞“对标 FFN 商用性能完成”的结论。没有 D5.* 验收，不能把 dispatch 阶段描述成生产化完成。
