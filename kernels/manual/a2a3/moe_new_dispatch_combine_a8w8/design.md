# moe_new_dispatch_combine_a8w8 前重排 initquant PTO 重写设计

本文档定义 `moe_new_dispatch_combine_a8w8` 的第一个前重排阶段，也就是把原 FFN 项目
`/home/ntlab/zy/code/zhangyuan/vllm-ascend-zy/csrc/mc2/dispatch_ffn_combine` 中的
`moe_init_routing_quant_v2` 用 PTO 重写。本阶段在 dispatch 通信和 GMM1 之前运行，只使用 AIV，产出后续
dispatch/GMM1 需要的密排 token、计数和映射元数据。

“重写”的边界是：device 侧不继续搬运原 AscendC kernel 类、AscendC buffer/queue/tensor 接口或 Catlass 接口；
前重排的数据搬运、向量计算、量化和同步都落到 PTO `Tile`、`GlobalTensor` 和 `T*` 指令上。原 FFN 代码只作为
功能、shape、容量语义和性能拆分的对标输入。

## 1. 目标

### 1.1 功能目标

输入本 rank 的 `inputA[M, K]`、`expertIdx[M, topK]`、可选 `xActiveMask[M]`，输出：

- `tokenPerExpertMatrix[rankId, globalExpert]`：本 rank 发往每个 global expert 的 route 数。
- `expandedRowIdx[M * topK]`：每个原始 route 对应的本地 packed row；失效、非法或容量截断 route 写 `maxOutputSize`。
- `peerWindow.dispatchPayload[packedRow, K]`：按 global expert 连续密排后的 int8 token。
- `peerWindow.dispatchScale[packedRow]`：每个 packed row 的 dynamic quant scale。
- `packedRowToRouteIndex[packedRow]`：反查原始 route index，供 debug、combine/unpermute 验收使用。

这里的 packed row 是本 rank 本地前重排 row，不是 GMM1 的最终本地专家 row。跨 rank count 同步和 gather
会在后续 dispatch 阶段把它转换为 GMM1 的 `gmm1InputInt8`。

### 1.2 性能目标

对标原始 FFN 的生产路径，而不是做单核调试版：

- device 主路径只使用 PTO 编程模型；不能引入 `AscendC::*`、`TQue/TPipe/TBuf/LocalTensor` 或 Catlass GEMM/
  epilogue 接口。
- route 计数、前缀、scatter+quant 都按 AIV 多核切分。
- 主路径避免对 `M * topK` 做全量 comparison sort。MoE expert 数通常远小于 route 数，前重排按 expert 分桶更适合
  `count -> prefix -> stable scatter`。
- 片上缓存和搬运流水必须按生产 FFN 的思路设计：UB 能容纳的场景使用 UB resident/full-load 融合，`R=M*topK`
  较大的场景使用多 AIV 分片、UB 双缓冲/ping-pong、token-centric quant 和批量化 metadata 写回。small/large
  只是验收锚点，不能在实现里特化成只支持这两个 shape 或固定 `topK=2`。
- worker 数、full-load 进入条件、large-token 路径和 row/column tiling 都必须由 `M/topK/K/globalExpertNum`、
  route 分布和 UB 容量共同决定，不能由 `caseName`、固定 `M=4097`、固定 `topK=2` 或验收脚本名称决定。
- 每个 stage 只做必要同步。跨 AIV 全局同步控制在 3 次以内：
  1. local count 完成后同步；
  2. prefix/cursor 初始化完成后同步；
  3. scatter+quant 完成后同步。
- 量化按 row/column tile 做 double buffer，支持 K=64 到 K=7168 的生产 shape，不因整行装不进 UB 退化成单核。
- 日志验收能直接从 host 输出判断本阶段结果是否正确，不依赖人工 dump 二进制后离线比对。常规日志只保留
  pass/fail 汇总和 first mismatch；临时 debug probe/debug stop 用完必须删除，避免代码和日志持续膨胀。

### 1.3 非目标

- 不在本阶段做跨 rank AllToAll、GMM1、SwiGLU、GMM2 或 combine。
- 不在第一版实现 drop-pad 的复杂容量语义；只支持 capacity clip 到 `maxOutputSize`，被截断 route 写哨兵。
- 不要求 bitwise 对齐原始 CANN `moe_init_routing_quant_v2` 的内部排序顺序；要求对下游可见的 count、packed row、
  int8 payload、scale 和 capacity 语义一致。

## 2. 生产 FFN 对标

### 2.1 原 FFN initquant 拆解

原始 FFN 的 initquant 慢路径是：

```text
sort -> expertTokenOut -> srcToDst -> gather+quant
```

其中 sort 大 shape 下拆成 VBS/VMS/SortOut：

```text
VBS: 每个 AIV 排本地分片
VMS: 4 路归并，多轮 GM 乒乓
SortOut: 0 号核做最后归并并拆出 expert 和 src row
```

这条路径通用性强，能处理普通 init routing 算子，但在当前融合场景里有两个成本：

- 全量排序对 `M * topK` 做 O(R log R) 比较，R 为 route 数；MoE 的目标只是按 expert 分桶。
- VMS/SortOut 需要多轮 SyncAll 和两块 GM sort workspace，容易成为 GMM 前不可 overlap 的固定开销。

本项目主路径改成分桶式前重排：

```text
Stage A: route count
Stage B: expert prefix + per-worker cursor base
Stage C: stable scatter + dynamic quant
```

在 expert 数很大、`globalExpertNum` 超过 debug scratch/UB 分片能力，或后续要求完全复刻 CANN 排序顺序时，
可以保留 VBS/VMS/SortOut 的算法形态作为 fallback，但实现也必须 PTO 化，不能直接搬原 AscendC 类。第一版优先
实现分桶主路径。

### 2.1.1 small/large 验收 case 的原 FFN 实际分支

原 FFN host tiling 对当前 A8W8/int8 路径固定使用：

```text
quantMode = 1                  // dynamic quant
dropPadMode = 0                // dropless
scaleDim0 = 0                  // no smooth
expertTokensCountOrCumsumFlag = 2
aivNumInitRouting = 2 * BLOCK_NUM = 40
ubSize = 196352
sortLoopMaxElement = floor(196352 / (sizeof(int32) * 2 * 4) / 32) * 32 = 6112
```

small case:

```text
worldSize=2, M=16, K=128, N=128, topK=2, experts=2, maxOutputSize=32
R = M * topK = 32
```

满足原 FFN full-load dynamic quant 条件：

- `R <= sortLoopMaxElement`；
- `K <= MAX_COLS_ONE_LOOP_QUANT(8192)`；
- dropless；
- sort、expert count、dynamic quant 所需 UB 空间能放下。

因此原 FFN `initRoutingQuantTilingKey=21000`，走 `MoeV2FullLoadDynamicQuant`。该路径把 sort、
`expandedRowIdx`、expert cumsum、dynamic quant 和输出写回放在一个 full-load kernel path 内完成，不再走独立的
`SortMultiCore -> ExpertTokenOut -> SrcToDst -> GatherDynamicQuant`。它的主要收益是减少 GM 中间结果和阶段级
`SyncAll`；代价是为了服务 per-row quant 分片，小 shape 下部分元数据计算会在多个 AIV 上重复，但 `R=32` 时成本很小。

large case:

```text
worldSize=2, M=4097, K=128, N=128, topK=2, experts=2, maxOutputSize=8194
R = M * topK = 8194
```

`R > sortLoopMaxElement`，所以不走 full-load。原 FFN `initRoutingQuantTilingKey=11010`，含义是：

```text
dropless + dynamic quant + sort multi-core
```

large 的原 FFN 关键 tiling：

```text
VBS sort needCoreNum = ceil(8194 / 6112) -> 2 -> round_up_pow4 -> 4
VBS perCoreElements = 2048
VBS lastCoreElements = 2050
VMS middle needCoreNum = 0       // 4 个 list 可由最终 SortOut 直接归并
SortOut = core0 final 4-way merge
srcToDst/gather needCoreNum = 40
srcToDst/gather perCoreRows = ceil(8194 / 40) = 205
srcToDst/gather lastCoreRows = 199
gather dynamic quant colLoops = 1 // K=128 整行进 UB，不触发列切分
```

实际优化点是：

- 4 个 AIV 做 VBS UB sort；
- 4 路最终归并，无中间 VMS 多轮 GM ping-pong；
- 40 个 AIV 做 `srcToDst` 和 `gather dynamic quant`；
- `K=128` 走 no-smooth 1H 路径，按 source token row 量化一次，再写到同一 token 的多个 route packed row；
- gather dynamic quant 配了 `BUFFER_NUM=2`，但这个 case 每核 row-loop 很少，双缓冲收益受限；
- 多阶段之间有若干 `SyncAll`，最后 caller 在 initquant 后还有一次全量同步。

L1/L0/swizzle/preload 属于后续 Catlass GMM 路径，不属于 `moe_init_routing_quant_v2` 前重排本身。前重排的性能
对标重点是 full-load 融合、sort/分桶策略、多 AIV row quant、GM 中间结果和同步次数。

### 2.1.2 PTO full-load fast path 设计

PTO 不直接复刻原 full-load 里的 AscendC sort 类。small/full-load path 使用和主路径一致的分桶语义，但在 UB 内
融合 count、prefix、scatter 元数据和 row quant，目标是降低小 shape 的固定同步和 GM workspace 成本。

进入 full-load fast path 的建议条件：

```text
dropless
dynamic quant
R = M * topK <= fullLoadRouteThreshold
K <= fullLoadColsThreshold
globalExpertNum <= fullLoadExpertThreshold
UB fits:
  count[globalExpertNum]
  prefix[globalExpertNum]
  localOrdinal[globalExpertNum]
  routePackedRow[R] or route cursor scratch if needed
  one/two row quant tiles: half + fp32 + abs + int8 + scale
```

第一版阈值先取保守值，但选择条件仍按 `R`、`K`、`globalExpertNum` 和 UB fits 判断，不按 case name 或固定
`topK` 判断：

```text
fullLoadRouteThreshold = 64 or 128
fullLoadColsThreshold = 1024
fullLoadExpertThreshold = 16
```

后续可按原 FFN 条件扩展到：

```text
R <= 6112
K <= 8192
fullLoadUbFits == true
```

full-load 是 UB fits 参数化路径，不是 small case 的专用分支。`topK` 增大后只要 `R`、`K`、`globalExpertNum`
和 quant tile scratch 仍满足 UB fits，就继续走 full-load；超过阈值时自然进入 large-token 路径。device 侧不能判断
`caseName == small`、`M == 16` 或 `topK == 2` 来选择路径。

PTO full-load 流程：

```text
main AIV:
  zero count/localOrdinal in UB
  pass0 over token/topK:
    validate active/expert
    count[expert]++
  prefix in UB:
    expertBase[expert], effectiveCount[expert], tokenPerExpertMatrix[rankId, expert]
  pass1 over token/topK:
    validate active/expert
    packedRow = expertBase[expert] + localOrdinal[expert]++
    expandedRowIdx[route] = packedRow or maxOutputSize
    packedRowToRouteIndex[packedRow] = route
    collect token's valid packed rows
    quantize inputA[token, :] once
    store quant payload/scale to all valid packed rows for this token
```

任意 `topK` 下都必须保持 token-centric quant：同一个 `inputA[token, :]` 不应为多个有效 route 重复做
`load -> absmax -> quant`。实现先遍历 `slot in [0, topK)` 得到所有有效 packed row，再量化一次，并把同一份
payload/scale 写到该 token 的所有有效 packed row。

full-load 同步策略：

- 单 AIV full-load：前重排内部不需要 AIV phase sync，只在 ready flag 前做最终可见性收口。
- 多 AIV full-load：仍按 `count done -> prefix done -> scatter done` 三个 AIV-only phase sync，但只有当
  `R` 大到单 AIV quant 不是最低延迟时才启用。
- 不使用 `SYNCALL<Mix>`，不引入中间 stop 日志。

full-load 输出合同与通用路径一致：

- `tokenPerExpertMatrix[rankId,*]`；
- capacity-clipped `expandedRowIdx`；
- `packedRowToRouteIndex`；
- `dispatchPayload`、`dispatchScale`；
- stop17 最终验收字段不增加。

T19 已落地的第一版实现采用保守 UB-fit 阈值：

```text
R <= 128, K <= 1024, globalExpertNum <= 16, localRows >= R
```

进入条件只读 shape/capacity，不读 `caseName` 或固定 `topK`。实现为 main AIV full-load fused path：
UB 中保留 `count[expert]` 和 `cursor[expert]`，一次生成 count/prefix/scatter 元数据，随后用
`M2QuantizeTokenToPackedRows` 对每个 token 做一次 PTO Vec quant 并写多个有效 packed row。该路径不使用
per-worker count/prefix scratch，不经过 count/prefix/scatter 三个 AIV phase sync；后置 count publish/wait、
prefix/gather 和 stop17 final sync 仍复用现有同步合同。

### 2.1.3 PTO large-token 优化路径设计

large-token path 面向 `R=M*topK` 较大、full-load 不再合适的通用场景，不是只面向
`M=4097,K=128,topK=2,maxOutputSize=8194`。PTO 不走原 FFN 的全量 sort，而使用
`count -> prefix -> stable scatter`，复杂度从 sort 的 `O(R log R)` 变成 `O(R + E * W)`。这里
`E=globalExpertNum`，`W=workerCount`。

large-token path 的优化目标不是仅让某个验收 case 的 4 个 worker 可用，而是根据 `R`、`topK`、`K`、专家数和
路由分布把 quant/scatter 并发度拉到接近 AIV 数，同时保持 prefix 和 packed row 顺序确定。

推荐分层 worker：

```text
workerCount:
  不应被 debug counter scratch 限制
  使用 workspace 正式字段 initQuantWorkerTokenPerExpert/initQuantWorkerPrefixPerExpert 保存 per-worker count/prefix
  上限为 kInitQuantMaxDispatchWorkers(40)，再按 R/topK/K/route shard alignment/有效 route 数动态收敛
  K=128/R=8194 只是其中一个覆盖点，目标是接近 logicalAivCount，而不是固定 4 worker
```

如果 count/scatter worker 数不同，必须重新定义 prefix：

```text
prefixBase[quantWorker, expert] =
  sum_{worker < quantWorker} countByQuantWorker[worker, expert]
```

因此生产方案保留 `blockTokenPerExpert` 和 `blockPrefixPerExpert` 作为 per-expert 汇总输出，并新增正式
`initQuantWorkerTokenPerExpert/initQuantWorkerPrefixPerExpert` workspace，布局为
`[kInitQuantMaxDispatchWorkers, align_up(globalExpertNum, 16)]`。count、scatter、quant 使用同一组
`workerCount`。debug counter 只记录摘要，不能作为 worker scratch 容量上限。

large Stage A 优化：

- 每个 AIV worker 按 token shard 遍历；
- `globalExpertNum` 小时，`count[expert]` 常驻 UB，结束后一次连续写 GM；
- `xActiveMask` 和 `expertIdx` 按 token/topK 连续读，避免每 route 反复读取 token 级状态；
- 不用 atomic，worker 独占 `initQuantWorkerTokenPerExpert[worker,*]`。

large Stage B 优化：

- main AIV 对 `E * W` 做 prefix，large case 为 `4 * W`，成本很低；
- 写 `tokenPerExpertMatrix`、`expertBase`、`blockPrefixPerExpert`；
- 同步后 worker 只读自己的 prefix row；
- 后续可把 prefix 计算扩成 int32 tile load/store，减少 scalar GM 指令。

large Stage C 优化：

```text
for token in worker token range:
  packedRows[0:topK] = compute packed row for each valid route
  if no valid packed row:
      continue
  quantize inputA[token, :] once
  for each valid packed row:
      store same int8 row and scale
```

这与原 FFN no-smooth 1H gather dynamic quant 的核心优化一致：按 source token row 量化一次，而不是按 route
重复量化。`topK` 增大时计算量应按 token 数增长，payload/scale store 才按有效 route 数增长；不能用固定长度数组
或 `topK==2` 分支做特化。

large quant pipeline：

- `K=128` 时整行一个 tile，无列切分；
- 使用两个 UB buffer 组做 ping-pong：
  - `halfTile[2]`
  - `fp32Tile[2]`
  - `absTile[2]`
  - `int8Tile[2]`
  - `scaleTile[2]`
- loop i 处理当前 token 时，提前发起 token i+1 的 GM->UB load；
- 当前 tile 完成 `half2float -> abs -> rowmax -> scale -> int8` 后，MTE3 store payload/scale 到一个或多个
  packed row；
- 复用 tile 前只等待对应 store event，不做全核同步。

T20 已落地的第一版 large-token quant pipeline：

- token-centric quant 主路径不再对 `validRowCount == 1` 退回单 packed row helper；任意 `topK` 都先收集本 token 的
  有效 packed row，再对 source token row 量化一次。
- 有效 packed row 写入 UB cache，容量为 256 个 route；常见 topK sweep 不再在 scale store、payload chunk store
  和 cache invalidation 阶段反复从 GM 读取 `expandedRowIdx`。超过 UB cache 容量时回退到循环扫描 `topK`，语义不变。
- row quant 使用 ping/pong 两套 UB buffer：`half/fp32/abs/int8` 各两组；`M2RawFloatToInt8` 使用独立 s32/f16
  scratch，避免覆盖 ping/pong 的预取输入。
- 当前实现按 column chunk 预取下一块 GM->UB load，再处理当前块。`K=128` 是单 chunk；更大 K 仍按
  `kM2RouteQuantTileCols` 切列，不要求整行常驻 UB。完整 large-K 验收留给 T24。
- 新增的 store 成对同步使用 PTO `PtoSetWaitFlag`，延迟预取的 load-ready record/wait 使用 PTO
  `Event<Op::TLOAD, Op::VECTOR>`。raw copy helper 后续仍应继续向 `TLOAD/TSTORE` 收敛，但本任务没有新增
  AscendC/Catlass API 依赖。

large metadata 优化：

- `expertBase`、`workerPrefix`、`localOrdinal` 缓存在 UB；
- `expandedRowIdx` 和 `packedRowToRouteIndex` 可按 route 连续范围批量写，不能每写一个元素都做 cache line flush；
- `dispatchPayload`/`dispatchScale` store 后只在发布 ready 前做必要 GM 可见性收口；
- count/prefix/scatter 三个 AIV phase sync 之外不增加中间验收日志。

T21 已落地第一版 worker/cache 调度优化：

- worker 选择收敛到共享 `M3NDispatchTargetWorkerCount`：输入为 `M/topK/K/globalExpertNum`、logical AIV 数和
  fused lane slot 上限。selector 同时考虑 token-centric quant 的 token 侧工作量、`M*topK` 的 route store fanout
  和 `globalExpertNum * workerCount` 的 prefix merge 成本，避免 worker 数固定为验收 large 的 4 核。
- large-token scatter 进入每个 worker 后，按 `globalExpertNum` 初始化 UB expert cache。cache 中保存
  `blockPrefixPerExpert[expert] + initQuantWorkerPrefixPerExpert[worker, expert]` 和
  `localOrdinal[expert]`，后续每个 valid route 只做 UB cursor 更新，不再在 route 热循环里反复读写 GM cursor。
- UB expert cache 容量按 expert 数控制，当前支持到 1024 个 global expert。超过容量时保留 GM cursor fallback，
  输出合同不变，仍不退 AscendC/Catlass。
- large/more-token/topK/skew/more-expert case 均保持 `m3n_multi_worker`，没有按 `small/large`、固定 `M` 或固定
  `topK` 分支。
- T23 继续负责 `expandedRowIdx`、`packedRowToRouteIndex` 以及 payload/scale 可见性收口的更大粒度批量化；T21
  只移除 per-expert base/cursor 的 GM 热点，不新增中间日志。

T23 已落地 metadata GM 批量化第一版：

- route scatter 写 `expandedRowIdx/packedRowToRouteIndex` 时，同步把当前 token 的 valid packed row push 到 UB
  cache；后续 token-centric quant 直接使用该 cache，不再在同一 token 内刚写 `expandedRowIdx` 后又从 GM 读回。
- `packedRowToRouteIndex` 的可见性按 expert 连续段收口：full-load/单 worker 按
  `blockPrefixPerExpert + blockTokenPerExpert` 分段 flush；multi-worker 按
  `blockPrefixPerExpert + initQuantWorkerPrefixPerExpert[worker, expert]` 和
  `initQuantWorkerTokenPerExpert[worker, expert]` 分段 flush，不再每个 worker flush 全 `R`。
- `expandedRowIdx` 仍按 token shard 连续范围 flush；capacity clip 的非连续 route 更新保留最终全量 flush，避免为
  稀疏 clip 路径增加长期 debug/索引膨胀。
- T23 没有增加中间日志或 debug stop。验收仍只看 stop17 最终结构体摘要和 `init_quant_e2e_us`。

### 2.1.3.1 前重排 fence / GM 可见性优化设计

当前 large anchor 的前重排耗时约 580us，和 small 差距异常大。主要嫌疑不是三次 AIV phase sync，而是
route quant 热循环在每个 packed row 后调用 `InvalidateGmCacheLines(dispatchPayload row)` 和
`InvalidateGmCacheLines(dispatchScale row)`。该 helper 内部包含 `pipe_barrier(PIPE_ALL)`、逐 cache line `dcci`
和 `dsb(DSB_DDR)`；large 有 8194 个 packed row，`K=128` 时会放大成数千次 heavy fence。

fence 优化必须先定义输出可见性合同，不能简单删除 `dcci/dsb`：

- `SYNCALL` 只解决 AIV 间阶段顺序，不保证 scalar/MTE3 GM store 已经对 host、其他 AIV 或远端 rank 可见。
- 前重排输出分两类处理：
  - metadata：`expandedRowIdx`、`packedRowToRouteIndex`、`blockTokenPerExpert`、`blockPrefixPerExpert`、
    `initQuantWorker*`、`tokenPerExpertMatrix` 等由 scalar GM store 或小块 store 写入，发布前需要按连续 range 做
    cache-line publish；
  - payload/scale：`dispatchPayload`、`dispatchScale` 由 route quant 的 MTE3/TSTORE 类路径写入，热循环内只需要
    用 PTO event/flag 保证 UB buffer 可复用；阶段末先做一次 store drain，随后按硬件可见性需求做 segment 级
    publish，不能在每个 packed row 后 `dsb`。
- ready/count publish 只能发生在 metadata 和 payload/scale 都完成可见性收口之后。之前 naive batch flush 曾出现
  `expandedRowIdx` stale/mismatch，说明 metadata 发布顺序不能被 payload fence 优化绕开。

建议把现有单一 `InvalidateGmCacheLines` 拆成三个 PTO helper，底层 `dcci/dsb` 只允许在 helper 内出现：

```text
PtoGmStoreDrain()
  pipe_barrier(PIPE_ALL) + dsb(DSB_DDR)
  目的：排空本核已发出的 scalar/MTE3 GM store，不遍历 cache line。

PtoGmPublishRangeNoFinalDsb(ptr, bytes)
  只按 64B 对齐 range 做 dcci，不做 pipe_barrier/dsb。
  目的：同一个阶段批量登记多个 metadata/payload range。

PtoGmPublishBatchFinish()
  dsb(DSB_DDR)
  目的：一次性提交本批 range publish。
```

业务流程只调用更高层的 front-reorder helper：

```text
PtoPublishInitQuantMetadataShard(workerId):
  publish expandedRowIdx[tokenBegin * topK, (tokenEnd-tokenBegin) * topK]
  publish packedRowToRouteIndex[expertBase + workerPrefix, workerRows] for each expert
  publish worker count/prefix rows when the current phase produced them
  batch finish once

PtoPublishInitQuantPayloadShard(workerId):
  for each expert with workerRows > 0:
    rowBegin = blockPrefixPerExpert[expert] + initQuantWorkerPrefixPerExpert[workerId, expert]
    rows = initQuantWorkerTokenPerExpert[workerId, expert]
    drain/publish dispatchPayload[rowBegin, rows, rowBytes]
    drain/publish dispatchScale[rowBegin, rows]
  batch finish once
```

large-token Stage C 的目标顺序：

```text
for each worker:
  token loop:
    scatter metadata scalar stores
    quant payload/scale stores
    only wait per-buffer PTO event before reusing UB ping/pong buffer
    no dcci, no dsb, no full pipe barrier for every packed row

  PtoGmStoreDrain()
  PtoPublishInitQuantMetadataShard(workerId)
  PtoPublishInitQuantPayloadShard(workerId)

AIV-only phase_sync_scatter_done
record init_quant_e2e end
```

payload/scale 的实现分两档推进：

- 首选路径：MTE3/TSTORE payload/scale 在 `PtoGmStoreDrain()` 后视为 DDR 可见，不额外逐行 `dcci`；只保留
  metadata range publish。该路径最接近 20us 目标。
- 如果 stop17 或后续 dispatch gather 证明远端读取 payload/scale 仍可能 stale，则只给 payload/scale 增加
  expert/worker 连续 segment 级 publish，并且所有 segment 共用一次 batch finish；禁止恢复 per-row
  `InvalidateGmCacheLines`。

full-load path 使用同一合同：单 AIV 写完后 `PtoGmStoreDrain()`，metadata 按 full expert 连续段 publish，
payload/scale 按 `blockPrefixPerExpert + blockTokenPerExpert` 连续段处理。small case 不允许新增额外 debug 日志。

验收和性能门禁：

- `scripts/run_initquant_acceptance.sh` small/large stop17 必须 pass，尤其是 `expandedRowIdx`、
  `packedRowToRouteIndex`、payload sample、scale、GMM1 input 和 dispatch ready。
- large anchor 的 `init_quant_e2e_us` 目标先压到 30us 内，stretch 目标 20us 量级；若未达到，需要在 report 中用
  coarse device cycles 说明剩余耗时来自 quant 计算、range publish 还是 phase sync。coarse 计数不进入常规日志。
- T26 参数化矩阵中 `M>4097`、`topK>2`、`K=1024/7168` 继续 pass，证明不是 large anchor 特化。
- 常规输出仍只保留 stop17 最终摘要、first mismatch 和 `init_quant_e2e_us`；临时 fence counter/probe 用完删除。

T27 已落地的 large anchor 优化结论：

- route quant 热循环不再对每个 packed row 做 payload/scale `dcci+dsb`；metadata 按 token range 和
  expert/worker packed-row segment batch publish。
- `prepare` 阶段不再清完整 `gmm1InputInt8`。该 buffer 由 dispatch gather 按有效 row 覆盖；前重排只保留
  `routingPerTokenScale` 默认值和 initquant 控制面轻量 clear。
- `routingPerTokenScale` 默认值不再在前重排入口按 `localRows` 全量清理；prefix 生成本 rank 有效输出行数后，只清
  `[validRows, localRows)` tail，并在后续 prefix/dispatch 元数据发布前保证可见。
- production/stop17 的 dispatch worker assignment 改为每个 AIV 本地 dense assignment，去掉 debug lane-slot
  `dcci+dsb` 和对应两轮 mixed sync；历史 debug stop 继续保留旧 lane-slot 分配路径。
- `expandedRowIdx` 的正确性边界保持 64-token batch publish。阶段末一次性 publish 和 128-token batch 在 large
  anchor 中会出现 stale/mismatch；去掉每批 final dsb 的实验没有稳定收益，因此不作为正式路径保留。
- stop17 常规输出新增四个阶段边界耗时字段，但不增加 token/row 级 timeline 或 counter dump。
- small stop17 e2e 约 3.2us；large stop17 e2e 约 20.4-25.1us，其中 large `prepare` 约 0.6us，
  `route_pack_quant` 约 15.4-20.1us。

large-token 路径验收：

```text
large stop17:
  init_quant_active_workers >= 2    // 验收下限，实际 worker 数由 R/topK/K/route shard alignment 决定
  target: active_workers 接近 logicalAivCount 和 kInitQuantMaxDispatchWorkers 上限
  init_quant_route_count_match=true
  init_quant_expanded_row_match=true
  init_quant_payload_sample_match=true
  init_quant_token_matrix_full_match=true
  init_quant_prefix_match=true
  init_quant_gmm1_input_match=true
  init_quant_dispatch_ready_match=true
  route_quant_path=pto_vec
  route_quant_scalar_fallback_reason=none
  init_quant_e2e_us=<nonzero>
```

性能验收必须加原 FFN baseline。没有同 shape/seed 的原 FFN `moe_init_routing_quant_v2` e2e 对比前，只能说
large-token path 已具备多核分桶和 PTO Vec quant 正确性，不能宣称达到商用性能。

### 2.1.4 非特化扩展设计

small/large 是固定验收锚点，但不能覆盖生产 shape 空间。后续 T19-T24 的实现和报告必须额外覆盖下面的参数族，
以证明路径是对标 FFN 的通用设计，而不是针对两个 case 的特化优化。

| 参数族 | 推荐 shape | 覆盖目的 | 任务门禁 |
| --- | --- | --- | --- |
| `anchor_small` | `M=16,K=128,topK=2,globalExpertNum=4,maxOutputSize=32` | 对标原 FFN full-load dynamic quant | T19 必跑，stop17 pass，`init_quant_e2e_us` 非 0 |
| `anchor_large` | `M=4097,K=128,topK=2,globalExpertNum=4,maxOutputSize=8194` | 对标原 FFN large sort+gather dynamic quant | T20/T21 必跑，active workers >= 2，stop17 pass |
| `token_scale_sweep` | `M=512/2048/4097/8192/16384,K=128,topK=2,globalExpertNum=4,maxOutputSize=M*topK` | 验证 token 数量连续扩展，不只适配 small/large 两个点 | T20/T21/T23 必跑至少一个大于 large 的点和一个非 2 次幂点 |
| `more_tokens` | `M=16384,K=128,topK=2,globalExpertNum=4,maxOutputSize=32768` | 验证 worker 数随 token 数扩展，metadata store 不成为瓶颈 | T20/T21/T23 必跑，worker 不被 4 核或 debug scratch 限死 |
| `topk_sweep` | `M=1024/2048,K=128,topK=1/2/4/8,globalExpertNum>=4,maxOutputSize=M*topK` | 验证任意 `topK` 下 token-centric quant，只量化 token row 一次 | T20/T21/T24 必跑，代码不得出现 `slot < 2` 或固定 packed row 数组 |
| `topk_expand` | `M=2048,K=128,topK=4 或 8,globalExpertNum>=4,maxOutputSize=M*topK` | 覆盖比 anchor 更大的 route fanout | T20/T21 必跑，代码不得出现 `topK==2` 特化 |
| `skew_topk` | `M=4097,K=128,topK=4,caseName=skewed,maxOutputSize=M*topK` | 验证专家倾斜时 per-worker prefix 和 capacity 仍确定 | T21/T23 必跑，count/prefix/expanded row 全匹配 |
| `large_k` | `M=1024,K=1024 或 7168,topK=2/4,maxOutputSize=M*topK` | 验证大 K column chunk、row tile、ping-pong，不退 scalar fallback | T24 必跑，`route_quant_path=pto_vec` |
| `capacity_clip` | `maxOutputSize < M*topK`，专家倾斜或 over-capacity | 验证 clip 哨兵和 packed row 上界 | T12/T23 回归必跑，clipped route 与 golden 一致 |

参数族使用显式 shape 参数驱动。`caseName` 只允许作为 host 数据生成模式，例如 balanced/skewed/zero-token/
over-capacity；device 侧路径选择只能读取 shape、workspace capacity、worker count 和 runtime config。

禁止作为性能路径选择条件的信号：

```text
case_name == small|large
M == 16
M == 4097
topK == 2
slot < 2
固定长度 packedRows[2]
```

实现要求：

- 所有 `slot` 循环都使用 `slot < shape.topK`，中间 packed row 缓存需要按 `topK` tile 或可循环 flush，不能写死 2。
- token-centric quant 的复杂度目标是 `O(activeToken * K + validRoute * storeBytes)`；`topK` 增大时不能退化成
  `O(validRoute * K)` 的重复 row quant。
- worker 调度按 token range 切分，但 worker 数要随 `R=M*topK`、`K`、有效 route 数和专家倾斜调整；更多 token
  时应接近 `logicalAivCount/kInitQuantMaxDispatchWorkers`，不能停在验收 large 的 4 worker。
- metadata 写回按 route/expert 连续片段批量化。更多 token 或更大 `topK` 时，不允许每 route 增加独立 cache flush
  或中间日志。
- full-load、large-token、large-K 三条路径共享同一输出合同和 stop17 验收字段，不能为了扩展 case 增加新的常规
  中间 dump。

共享验收脚本需要支持参数化矩阵，而不是在脚本里写死 small/large。最小矩阵：

```text
token scale: M in {16, 512, 2048, 4097, 8192, 16384}, topK=2, K=128
topK scale:  M in {1024, 2048}, topK in {1, 2, 4, 8}, K=128
K scale:     K in {128, 1024, 7168}, topK in {2, 4}
distribution: balanced + skewed + capacity_clip
```

报告必须打印每个 case 的 `M/topK/K/maxOutputSize/activeWorkers/e2e_us/path/reason/pass` 汇总。性能结论按
参数族给出，不允许用单个 `caseName` 推导通用结论。

交付前做 device 侧非特化审计。下面的模式在前重排 device 主路径中应返回空，host 数据生成脚本可使用
`caseName` 选择数据分布，但不能把它传入 device 性能分支：

```bash
rg -n "caseName|case_name|M == 16|M == 4097|topK == 2|slot < 2|packedRows\\[2\\]" \
  kernels/manual/a2a3/moe_new_dispatch_combine_a8w8/kernel
```

每个扩展 shape 的常规验收仍只读 stop17 最终摘要：

```text
init_quant_route_count_match=true
init_quant_expanded_row_match=true
init_quant_payload_sample_match=true
init_quant_token_matrix_full_match=true
init_quant_prefix_match=true
init_quant_gmm1_input_match=true
init_quant_dispatch_ready_match=true
route_quant_path=pto_vec
route_quant_scalar_fallback_reason=none
init_quant_e2e_us=<nonzero>
pass=true
```

性能结论必须按参数族表达。例如只能说“topK expansion 下仍保持 token-centric PTO Vec 主路径”，不能用 small/large
两个 anchor 推导“任意 token/topK 已达到 FFN 商用性能”。

### 2.1.5 商用性能对标门禁

本阶段不能只按 correctness bring-up 设计。前重排要对标原 FFN initquant 的商用优化方式，后续 GMM 阶段还要对标
Catlass/Catcoc 的片上缓存和 swizzle 设计。实现和验收按下面的门禁推进。

| FFN 商用优化点 | 原 FFN 使用位置 | PTO 使用要求 | 验收口径 |
| --- | --- | --- | --- |
| full-load fast path | small/UB fits 场景 `MoeV2FullLoadDynamicQuant` | UB fits shape 用 UB resident count/prefix/scatter，token-centric quant 一次服务多个 topK route | small stop17 pass，`init_quant_e2e_us` 对比原 FFN baseline 不显著退化 |
| UB resident metadata | full-load sort/count、gather tiling | `count/prefix/localOrdinal/expertBase` 优先驻留 UB，结束后批量写 GM | large metadata 阶段不能成为 e2e 主瓶颈；无 per-route cache flush |
| 多 AIV 分片 | large VBS 4 核 sort、srcToDst/gather 40 核 | count/scatter/quant 使用正式 workspace 支持更多 worker，不能被 debug scratch 限死 | active workers 按 `R/topK/K` 动态扩展，large stop17 仍 pass |
| 4 路归并/分桶并发 | VBS/VMS/SortOut | PTO 主路径用 `count -> prefix -> stable scatter` 替换全量 sort，但必须保持 expert 分桶确定性 | `expandedRowIdx`、count matrix、GMM1 input 与 host golden 一致 |
| token-centric quant | no-smooth 1H gather dynamic quant | 同一 token 的多个有效 topK route 只做一次 absmax/quant，payload store 多份 | 增大 `topK` 时 quant 计算量不随 route 数线性翻倍 |
| UB ping-pong | `BUFFER_NUM=2` queue、gather quant pipeline | row quant 用双 buffer 交叠 GM load、Vec compute、GM store；复用 tile 前只等对应 event | large `init_quant_e2e_us` 较当前 T15 基线下降或瓶颈解释清楚 |
| 行/列 tiling | gather dynamic quant 的 row/col tiling | `K=128` 整行 tile；大 K 走 column chunk，不能退成单核或整行超 UB | K 扩展 case 不触发 fallback，payload/scale pass |
| 同步收敛 | VBS/VMS/SortOut、srcToDst 后 SyncAll | 前重排只保留 count/prefix/scatter 三个 AIV-only phase sync；禁止中间 debug sync 长期保留 | sync 点数量、scope 可审计；卡顿定位 probe 用完删除 |
| GM workspace ping-pong | sort workspace 0/1 | 前重排主路径避免 sort workspace；必要 workspace 只存正式 per-worker count/prefix 和 payload | workspace layout 不覆盖，GM traffic 有明确用途 |
| L1/L0/swizzle/preload | 后续 Catlass GMM，不属于 initquant | PTO 重写整体必须在 GMM1/GMM2 阶段用 PTO `TMATMUL` tiling、L1/L0 tile、preload 和 swizzle 替代 | GMM 设计任务单独验收，不能把 initquant pass 当成整体商用性能完成 |

关键判断：

- initquant 的片上缓存核心是 UB，不是 L1/L0。原 FFN 的 L1/L0/swizzle 出现在 GMM/Catlass 计算阶段，PTO
  重写整体必须考虑，但不能误算到前重排完成项里。
- UB-fits path 要像原 FFN full-load 一样减少 GM 中间结果和阶段同步。
- large-token path 要像原 FFN large path 一样用多 AIV 和流水，而不是 correctness 版长期停留。
- 任何性能声明必须带原 FFN baseline、PTO e2e、worker 数、sync scope 和 route quant path；没有 baseline 不宣称商用达标。

T22 边界结论：

- 前重排的商用性能门禁只覆盖 initquant 子系统：UB resident、AIV worker、AIV-only phase sync、
  token-centric quant、metadata 批量化和 row/column tiled quant。
- 原 FFN 中 `GemmIdentityBlockSwizzle`、L1/L0 tile、preload async、`SynchronizeBlock()` 排空、
  小 group L2 cache hint 都属于 GMM1/GMM2 的 Catlass 路径。PTO 重写必须在后续 GMM 任务里用
  `TMATMUL`、PTO tile scheduler 和 cross-core event 替代，不能把 stop17 initquant pass 当成整算子
  商用性能完成。
- GMM 设计必须继承 initquant 的真实输出规模：`packedRows <= maxOutputSize`、每 expert 的
  `currentM`、capacity clip、skew 分布，以及 `M/topK/K/globalExpertNum` 参数族。不能只按
  `M=16` 或 `M=4097,topK=2,K=128` 两个 anchor 估算 tile 和 worker。
- T23/T24 是前重排内继续对标 FFN initquant 的任务；T25 才开始承接 FFN GMM 的 L1/L0/swizzle/preload
  性能设计。

### 2.1.6 更多 token 的扩展设计

后续优化不能只对 `small` 和 `large` 两个 anchor 做特化。原 FFN 的 initquant tiling 由
`totalLength = M * topK` 驱动，token 数继续增大后仍按同一套规则扩展：

- full-load 只由 `R <= sortLoopMaxElement`、`K <= MAX_COLS_ONE_LOOP_QUANT`、dropless 和 UB fit 决定；
- sort large path 的 VBS worker 由 `ceil(R / sortLoopMaxElement)` 决定，并向 4 的幂次对齐后受 AIV 数上限约束；
- VMS 只有当 VBS list 数超过 4 时才进入中间归并；
- `srcToDst` 和 `gather dynamic quant` 的 worker 由 `perCoreRows = ceil(R / aivNum)` 推到接近 AIV 数；
- row loop 和 col loop 由 `perCoreRows`、`K`、UB 中 row metadata/quant tile/scale buffer 的 fit 情况决定；
- dynamic quant 使用 `BUFFER_NUM=2` 队列化 GM load、Vec compute 和 GM store，`topK` 增大时应复用同一 source token
  的 row quant 结果。

PTO 路径的扩展规则必须对应这些生产分支，但不复刻原 FFN 的 AscendC sort 类：

```text
routeCount = M * topK
fullLoadFits = dropless && dynamicQuant && ubFits(R, K, globalExpertNum)

if fullLoadFits:
    use UB resident count/prefix/scatter + token-centric quant
else:
    workerCount = select_by(routeCount, topK, K, validRouteEstimate, logicalAivCount, kInitQuantMaxDispatchWorkers)
    use count -> prefix -> stable scatter
    use row/col tiled PTO Vec quant with ping-pong buffers
```

`large` 的 `M=4097` 只是非 2 次幂 anchor。`M=8192/16384`、更大的 `topK`、更多 expert 和倾斜分布必须使用同一
worker/cache/metadata 策略：

- worker 数不能固定为 4。`R` 足够大且 workspace fit 时应继续提升到 logical AIV 上限或
  `kInitQuantMaxDispatchWorkers`；
- token shard 仍按完整 token 切分，避免一个 token 的 `topK` route 分到多个 worker 后重复量化；
- `topK` 较大时 packed row 列表可分批处理，但 quant 只对 `inputA[token, :]` 做一次；
- metadata 写回量随 `R` 线性增长，必须按连续 route/expert tile 批量写回，不能用 per-route flush 或中间日志维持正确性；
- 大 `K` 下 worker 选择要考虑每 token quant 成本，不能只按 route 数切分；row/col tile 和 ping-pong 是生产路径要求。

验收结论也按参数族给出：small/large pass 只能证明 anchor 正确；只有 token scale、topK scale、large K、skew 和
capacity clip 都通过，才能说该阶段没有做 anchor 特化。

从 T20 开始，每个性能 hardening 任务交付时都必须带一个超出 large anchor 的参数族验证，不能把 T26 当成最后
补测。T26 是共享矩阵和脚本化门禁，T20-T24 自己的报告也要证明实现没有围绕 `M=4097,topK=2,K=128` 做特化。

| 任务 | 必带的非 anchor 验证 | 设计约束 |
| --- | --- | --- |
| T20 large-token quant pipeline | `M>4097` token scale 至少一个；`topK=4/8` 至少一个 | quant 按 token row 复用，`topK` 增大只增加 payload/scale store，不重复 row absmax/quant |
| T21 worker/cache 调度 | `M=8192/16384`、skew、more experts 至少覆盖两类 | worker 数由 `R/topK/K/validRoute/globalExpertNum` 选择，不能固定在 large anchor 的 4 核 |
| T23 metadata GM 批量化 | `M>4097` 和 `topK>2` 都要覆盖 | metadata 写回按连续 route/expert tile 批量化，不用 per-route flush 或中间日志维持正确性 |
| T24 大 K 行/列 tiling | `K>=1024` 且包含 `topK>2` | row/col tiling 仍走 PTO Vec 主路径，不因整行放不下 UB 退回 scalar 或单核 |
| T25 GMM PTO 性能设计 | 设计中继承 `M/topK/K` 参数族 | GMM 设计使用 initquant 的真实 packed row 规模，不只按 small/large 输出规模估算 |

报告里的性能描述必须使用参数族表述，例如“token scale 到 `M=16384` 时 worker 选择仍随 `R` 扩展”，而不是用
single shape 的 pass 推导“已对标所有 FFN 生产 shape”。如果某个参数族暂时不能跑通，报告要写清楚瓶颈是
同步、worker 调度、metadata store、row/col tiling 还是 PTO primitive gap。

### 2.1.7 后续 GMM PTO 性能设计

T25 只做设计和任务拆分，不把 Catlass 代码拷进来改。后续 GMM1/GMM2 的 PTO 化需要参考原 FFN 的并发、
缓存和同步设计，但 device 主路径用 PTO `TMATMUL`、PTO Vec epilogue 和 PTO cross-core event 重写。

#### 2.1.7.1 原 FFN GMM 事实

原 FFN GMM 配置来自 `dispatch_ffn_combine.h` 和 `dispatch_ffn_combine_kernel.hpp`：

| 维度 | 原 FFN 事实 | PTO 设计含义 |
| --- | --- | --- |
| L1 tile | `L1TileShape = GemmShape<128, 256, 512>` | GMM tile 的分配单位是 `M=128,N=256`，L1 一次预取 512 个 K 元素 |
| L0 tile | `L0TileShape = GemmShape<128, 256, 128>` | L1 tile 内继续按 K=128 做 L0A/L0B/TMATMUL 分片 |
| stages | `preloadStages=1,l1Stages=2,l0A/BStages=2,l0CStages=1` | A/B L1 与 L0 需要 ping-pong，结束前必须 drain outstanding preload |
| scheduler | `GemmIdentityBlockSwizzle<9,1>` | tile 遍历不是简单 row-major；hot expert 多 M tile 时要提高 weight L2 复用 |
| group shape | `currentM = cumsumMM[...]`，再按 `preCurrentmSum/maxOutputSize` 截断 | GMM shape 来自 initquant/dispatch 的真实 expert rows，不由 small/large anchor 推导 |
| core assignment | `startLoopIdx=((coreIdx<startCoreIdx)?coreIdx+coreNum:coreIdx)-startCoreIdx`，`loopIdx+=coreNum` | 所有 AIC 都遍历 expert，真正分给 AIC 的是 tile，不是固定 expert-to-core |
| cross-expert balance | `startCoreIdx=(startCoreIdx+coreLoops)%coreNum` | 下一个 expert 从上一个 expert 结束位置继续轮转，避免每个 expert 都从 AIC0 起步 |
| cache hint | `currentM <= L1TileShape::M` 时禁 L2 cache | 冷 expert 只读一次 weight，旁路 L2；hot expert 保留复用 |
| preload drain | `BlockMmad::SynchronizeBlock()` 在 `Finalize` 前执行 | 发布 `GMM1 -> SwiGLU` 或 `GMM2 -> combine` ready 前，所有 delayed tile 必须已经写出 |
| AIC/AIV sync | `CrossCoreWaitFlag/SetFlag<0x2>` | group ready 是 producer-consumer 语义，不能用全局 `SYNCALL<Mix>` 代替 |

当前新工程的 `moe_new_dispatch_combine_a8w8_gmm_kernel.cpp` 已有 PTO GMM 雏形：`kGmmBaseM=128`、
`kGmmBaseN=256`、`kGmmBaseK=64`、`kGmmStepK=4`，即 L1-K=256、L0-K=64，并使用
`TLOAD/TEXTRACT/TMATMUL/TMATMUL_ACC/TSTORE`。这证明 PTO TMATMUL 路径可承接 GMM，但它还不是原 FFN
L1-K=512/L0-K=128 的性能等价实现；T25 后续任务必须把该差异作为 tile sizing 或 primitive gap 明确记录。

#### 2.1.7.2 PTO GMM shape 和 task contract

GMM1/GMM2 的 shape 只来自前重排和 dispatch 产出的正式结构：

```text
localExpertRows[expert] = expertTokenNums[expert]
rowBegin[expert] = dispatchOffset[expert]
gmm1:
  A = gmm1InputInt8[rowBegin, hiddenSize]
  B = gmm1WeightInt8[globalExpert, hiddenSize, 2 * intermediateSize]
  C = gmm1AccInt32[rowBegin, 2 * intermediateSize]
gmm2:
  A = gmm2InputInt8[rowBegin, intermediateSize]
  B = gmm2WeightInt8[globalExpert, intermediateSize, hiddenSize]
  C = gmm2AccInt32[rowBegin, hiddenSize]
```

capacity clip 必须在同一位置序上保持一致：若 `dispatchOffset + expertTokenNums` 超过 `maxOutputSize`，
后续 GMM/activation/return 都只能消费被保留的 row range。不能让某个 stage 自行重新截断或重新排序。

task plan 不按 `M=16`、`M=4097` 或 `topK=2` 决定，而按每个 local expert 的 `currentM`、`N`、`K` 和
tile 容量生成：

```text
for localExpert in expertPerRank:
  rowBegin = dispatchOffset[localExpert]
  currentM = expertTokenNums[localExpert]
  for mTile in ceil(currentM / tileM):
    for nTile in swizzled_n_order(ceil(N / tileN)):
      task = {stage, expert, rowBegin + mTile * tileM, mValid, nBase, nValid, kSize, syncGroup}
```

`syncGroup` 对 GMM1 来自 `swigluSyncGroups/dequantSum` 对 expert row range 的覆盖；对 GMM2 来自
`activationSyncGroupReady[syncIdx]` 对 row range 的放行。GMM task 可以按 tile 细分，但 ready 边界仍是
expert group 或 sync group，不能把 tile ownership 误写成 group ownership。

#### 2.1.7.3 PTO AIC scheduler 和 swizzle

PTO scheduler 需要保留原 FFN 的两层调度：

- 外层遍历 local expert group，等待本 expert 的 `dispatchGroupReady[expert]` 后才可跑 GMM1；
- 内层按 `tileM x tileN` 生成 tile work item，AIC 通过 `loopIdx += aicCount` 轮转消费；
- 每个 expert 结束后更新 `startCoreIdx=(startCoreIdx+coreLoops)%aicCount`，让下一 expert 接力；
- tile 坐标映射使用 N 聚簇 swizzle 或等价顺序，避免简单 row-major 造成同一 weight tile 反复从 GM 读取。

验收时需要记录结构化调度摘要，而不是中间日志 dump：

```text
gmm_stage=1/2
gmm_tile_task_count=<n>
gmm_active_aic_workers=<n>
gmm_start_core_carry=true
gmm_swizzle_policy=n_cluster_or_equivalent
gmm_zero_row_experts=<n>
```

这些字段只作为最终验收摘要，不在常规路径打印每个 tile 的原始 counter。

#### 2.1.7.4 PTO L1/L0 和 preload 设计

目标路径使用 PTO tile 表达 Catlass `BlockMmad` 的片上缓存层级：

| 层级 | PTO 对应 | 要求 |
| --- | --- | --- |
| GM -> L1 | `TLOAD` 到 `Tile<TileType::Mat,int8>` A/B panel | A/B L1 至少双缓冲；K panel 优先对齐原 FFN 512，若 PTO/L1 容量不支持则记录 gap |
| L1 -> L0 | `TEXTRACT/TMOV` 到 `TileLeft/TileRight` | L0 K 优先对齐原 FFN 128；当前 `kGmmBaseK=64` 只能作为保守 bring-up |
| L0 compute | `TMATMUL/TMATMUL_ACC` | int8 x int8 -> int32，K loop 首次清 acc，后续累加 |
| L0C -> GM | `TSTORE` 或 PTO fixpipe/epilogue store | GMM1/GMM2 publish ready 前必须等待 store 完成 |
| preload drain | PTO helper `GmmDrainPreload()` | 对应 `SynchronizeBlock()`，drain 后才能发布 `gmm1SyncGroupReady/gmm2GroupReady` |

preload pipeline 的设计原则：

```text
issue TLOAD for K panel i + preloadDepth
compute K panel i
advance ping/pong stage
on expert/sync-group boundary:
  drain outstanding panels
  wait final C store
  publish ready
```

裸 `set_flag/wait_flag/pipe_barrier/dsb` 只能出现在 PTO GMM helper 内；业务主流程通过 `GmmLoadPanel`、
`GmmComputePanel`、`GmmStoreTile`、`GmmDrainPreload` 表达语义。若 PTO 目前缺少 L2 cache hint 或更大 L1-K
tile 能力，报告为 primitive/capacity gap，不恢复 AscendC/Catlass fallback。

#### 2.1.7.5 GMM 和 AIV epilogue 的同步接口

GMM 的同步对象固定为当前 workspace 已有账本：

| Producer | Consumer | Ready object | 粒度 | 发布前条件 |
| --- | --- | --- | --- | --- |
| dispatch gather | GMM1 | `dispatchGroupReady[expert]` | local expert | `gmm1InputInt8/routingPerTokenScale` 对该 expert 连续 row range 全部可见 |
| GMM1 | activation/SwiGLU | `gmm1SyncGroupReady[syncIdx]` | `swigluSyncGroups/dequantSum` 覆盖的 row range | sync group 内所有 GMM1 tile 已 drain/store |
| activation/SwiGLU/quant | GMM2 | `activationSyncGroupReady[syncIdx]` | 同一 sync group row range | `gmm2InputInt8/gmm2PerTokenScale` 已写完 |
| GMM2 | return/combine | `gmm2GroupReady[expert]` 或 `subTileReady[tile]` | expert 或 return sub-tile | GMM2 tile/segment 已 drain/store |

PTO 实现应使用 cross-core event 或封装的 GM ready helper。`SYNCALL<Mix>` 只允许在 overlap-off bring-up 或 debug
stop 收口中出现；生产路径不能等所有 expert 完成后再启动 activation/GMM2/return。

#### 2.1.7.6 非特化验收和 baseline

GMM 性能报告必须使用参数族，而不是 anchor 结论。最低覆盖：

```text
token scale: M in {2048, 4097, 8192, 16384}, topK in {2, 4}
K/N scale: K in {128, 1024, 7168}, N in {128, 4096}
distribution: balanced + skewed + capacity_clip
```

未来 GMM PTO 实现的验收字段至少包括：

```text
gmm1_path=pto_tmatmul
gmm2_path=pto_tmatmul
gmm_tile_task_count>0 for nonempty experts
gmm_active_aic_workers>=2 for large/hot expert cases
gmm_swizzle_policy!=row_major_default for hot expert cases
gmm_preload_drained=true before ready publish
gmm_ready_scope=expert_or_sync_group
gmm_no_catlass_fallback=true
```

只有 T23/T24/T25/T26 都通过，并且有原 FFN baseline 对比，才能讨论“PTO 重写对标 FFN 商用性能”。

### 2.2 PTO 重写边界

前重排阶段的实现文件可以复用当前工程已有 host/test/log 框架，但 device 侧入口和 helper 需要按 PTO 风格重写：

| 原 FFN/AscendC/Catlass 依赖 | PTO 重写目标 | 说明 |
| --- | --- | --- |
| `using namespace AscendC` | `#include <pto/pto-inst.hpp>`、`using namespace pto` | 新 kernel 文件不引入 AscendC namespace |
| `AscendC::GlobalTensor<T>` | `pto::GlobalTensor<T, Shape, Stride>` 或当前工程 `GlobalNd<T>` wrapper | GM 访问通过 PTO tensor 描述 |
| `LocalTensor<T>` | `pto::Tile<TileType::Vec, T, Rows, Cols>` | UB buffer 显式 `TASSIGN`，不走 AscendC queue 分配 |
| `TPipe/TQue/TQueBind/TBuf` | 固定 PTO Tile + `TASSIGN` offset 规划 | 第一版不用动态 queue；double buffer 用两个 Tile offset |
| `DataCopy/DataCopyPad` | `TLOAD/TSTORE`；mixed AIV 中 public tile wrapper 若触发 `__cce_get_tile_ptr` 卡住，则用本工程固定 UB PTO-lowered load/store helper | payload row 尾部按 64B 对齐补 0 |
| `Abs/ReduceMax/Max` | PTO Vec 语义；mixed AIV route quant 使用固定 UB pointer helper 调 PTO lower-level/vector 指令 | dynamic quant 的 row absmax 全部走 Vec，不退 scalar |
| `Cast/Duplicate/Muls/Div` | `TCVT/TEXPANDS/TMUL/TMULS/TDIV/TDIVS` 语义；mixed AIV route quant 用固定 UB raw helper 避开 public wrapper 指针转换 | 标量参数写固定 UB 参数区 |
| AscendC dynamic quant 手写链 | `load -> half2float -> abs -> rowmax -> scale/invScale -> quant -> store` 的 PTO Vec 语义 | `dispatchScale` 存 dequant scale，量化输入 invScale |
| `AscendC::SyncAll()` | `pto::SYNCALL<SyncAllMode::Soft, SyncCoreType::AIVOnly>(...)` 或硬同步封装 | 前重排只需要 AIV 参与 |
| Catlass `BlockMmad/MatmulEpilogue/TileCopy` | 不在本阶段使用；后续 GMM 用 PTO `TMATMUL`/tile epilogue 重写 | 前重排是 AIV route+quant，不触碰 Catlass GMM |
| `platform_ascendc::PlatformAscendC` / `Mc2CcTilingConfig` | 本工程 host shape/layout 计算和 PTO kernel launch 参数 | host 只保留运行时、内存、日志和 golden 生成 |

接口约束：

- 新增前重排 device 代码中不能出现 `AscendC::`、`Catlass::`、`catlass/` include、`TQue`、`TPipe`、`TBuf`、
  `LocalTensor`、`DataCopyPad`。
- 如果必须使用底层 CCE intrinsic 做 PTO 当前未暴露的缓存或 fence 操作，必须包在本工程 PTO helper 内，并在调用点
  用 helper 名表达语义；业务逻辑不能散落原 AscendC API。
- fallback 方案也必须是 PTO fallback。不能把原 VBS/VMS/SortOut AscendC 类直接拷进新工程作为 fallback。

### 2.3 AscendC/Catlass 逻辑的 PTO 实现方案

原 FFN 的 device 逻辑可以按“控制面、数据搬运、向量计算、矩阵计算、通信、同步”六类迁移。前重排阶段只落地
控制面、数据搬运、向量计算和同步，但设计必须和后续 GMM/dispatch/combine 的 PTO 化口径一致。

| 原逻辑类别 | 原实现形态 | PTO 实现方案 | 前重排阶段落地方式 |
| --- | --- | --- | --- |
| UB buffer/queue 生命周期 | `TPipe::InitBuffer` + `TQue::AllocTensor/EnQue/DeQue/FreeTensor` | 编译期固定 `Tile<TileType::Vec, ...>` 类型，运行期用 `TASSIGN(tile, ubOffset)` 绑定 UB；双缓冲显式分配 `tile0/tile1` offset | count/localOrdinal 用固定 UB slice；quant 用 half/fp32/abs/int8/scale tile offsets |
| GM tensor 描述 | `AscendC::GlobalTensor<T>::SetGlobalBuffer` | `pto::GlobalTensor<T, Shape, Stride>` 或现有 `GlobalNd<T>`，动态 shape 用构造参数/`SetShape` | `inputA`、`dispatchPayload`、`dispatchScale` 用 2D row-major view；metadata 可用 typed GM view |
| 连续 GM 搬运 | `DataCopy/DataCopyPad` | `TLOAD/TSTORE`；尾部 padding 用 `TEXPANDS` zero tile + `TSTORE`，或 `TFILLPAD` 能覆盖时使用 | 输入 row chunk `TLOAD`，payload/scale `TSTORE`，`K -> rowBytes` 尾部写 0 |
| metadata 标量读写 | raw pointer + scalar load/store 或 `GlobalTensor<int32_t>` | 低频控制面允许 typed GM scalar accessor；批量 metadata 走 `TLOAD/TSTORE` int32 tile | count/prefix/cursor 可先用 typed GM scalar，稳定后把连续 count row 改成 int32 tile store |
| sort/routing | AscendC sort、VBS/VMS/SortOut、`MoeV2ExpertTokenOut`、`MoeV2SrcToDst*` | 主路径不用全 sort，改为 PTO 语义的 `count -> prefix -> stable scatter`；若保留 sort fallback，要用 `TSORT32/TMRGSORT/TEXTRACT/TSTORE` 等 PTO 指令重写 | 第一版不实现 PTO sort fallback，只实现稳定分桶 |
| dynamic quant | `Abs/ReduceMax/Cast/Duplicate/DataCopyPad` 手写链 | PTO Vec 语义：GM->UB、half2float、abs、rowmax、scale/invScale、int8 quant、UB->GM。mixed AIV 当前用固定 UB pointer helper 调 PTO lower-level/vector 指令，避免 public tile wrapper 的 `__cce_get_tile_ptr` 卡顿 | `dispatchScale` 写 dequant scale，量化参数写 invScale |
| gather/scatter 类搬运 | `DataCopyPad` + scalar index loop | 连续行优先 `TLOAD/TSTORE`；稀疏索引可用 PTO `MGATHER/MSCATTER` 或显式按 packed row 做 row tile | 前重排 scatter 目标是 packed row，按 row tile store；不引入 AscendC gather 类 |
| Catlass GMM | `Catlass::BlockMmad`、`MatmulEpilogue`、`TileCopy` | 后续 GMM1/GMM2 用 PTO `TMATMUL/TMATMUL_ACC`，Vec epilogue 用 `TCVT/TMUL/TADD/TQUANT/TSTORE` | 本阶段只输出 GMM1 输入所需 int8 payload/scale，不调用 Catlass |
| Catlass epilogue/activation | Catlass epilogue tile copy/elemwise | PTO Vec tile epilogue：`TLOAD/TCVT/TMUL/TADD/TABS/TROWMAX/TQUANT/TSTORE` | 前重排只覆盖 route quant；SwiGLU/requant 后续沿同一 Vec 方案 |
| HCCL/MC2 notify 账本 | `CrossCoreSetFlag/WaitFlag`、HCCL signal、DataAsFlag | 卡内用 PTO `Event`/`SYNCALL`；跨 rank payload 用 `TGET/TPUT`，ready 用 `TNOTIFY/TWAIT/TTEST` 或数据即信号轮询 | 本阶段只发布本 rank metadata；跨 rank wait 在 dispatch count sync 阶段 |

迁移顺序按依赖推进：

1. 先替换控制面和 UB/GM 表达，保证新文件没有 AscendC/Catlass include。
2. 再实现 `count -> prefix -> scatter`，用日志对齐 host golden。
3. 最后把 scalar bring-up quant 收敛到 PTO Vec quant，关闭 scalar fallback。

### 2.4 PTO 同步接口设计

同步按作用域分层，不能把所有场景都降成一个 `SYNCALL<Mix>`。前重排阶段只需要 AIV 内同步和单核 pipe 依赖；
后续接 GMM/dispatch/combine 时再打开 AIC/AIV 和跨 rank 同步。

| 同步场景 | 原 FFN/AscendC 形态 | PTO 接口 | 使用规则 |
| --- | --- | --- | --- |
| 单核指令链依赖 | `SetWaitFlag<HardEvent::MTE2_V>`、`PipeBarrier<PIPE_V>` | 优先使用 PTO `RecordEvent` 返回值和 `TSYNC(event)`；必要时 `pto::PtoSetWaitFlag<SrcPipe, DstPipe>()` | `TLOAD -> TCVT`、`TQUANT -> TSTORE` 等 producer/consumer 明确时用事件串接，少写裸 flag |
| 单 pipe 排空 | `AscendC::PipeBarrier<PIPE_V>` | `pto::TSYNC<pto::Op::TABS>()`、`pto::TSYNC<pto::Op::TROWMAX>()` 等单 Op barrier，或 PTO 指令内部 barrier | 只有同一 tile 被原地复用或 scalar 读取 vector 结果前才需要显式排空 |
| Vector 结果给 Scalar | `SetWaitFlag<HardEvent::V_S>` | `pto::PtoSetWaitFlag<PIPE_V, PIPE_S>()`，封装成 `WaitVecToScalar()` | 读取 `rowMaxTile.GetValue(0)` 前必须有 V->S 可见性 |
| Scalar 参数给 Vector | `SetWaitFlag<HardEvent::S_V>` | `pto::PtoSetWaitFlag<PIPE_S, PIPE_V>()`，封装成 `WaitScalarToVec()` | scalar 计算 `scale/invScale` 后，用 `TEXPANDS` 写 parameter tile 前同步 |
| MTE2 load 给 Vector | `SetWaitFlag<HardEvent::MTE2_V>` | `auto ev=TLOAD(...); TCVT(..., ev)`；或 `PtoSetWaitFlag<PIPE_MTE2, PIPE_V>()` | PTO 指令支持 event 参数时必须用 event 参数 |
| Vector 结果给 MTE3 store | `SetWaitFlag<HardEvent::V_MTE3>` | `auto ev=TQUANT(...); TSTORE(dst, tile, ev)`；或 `PtoSetWaitFlag<PIPE_V, PIPE_MTE3>()` | store 前只同步当前 tile，不做全核 barrier |
| MTE3 store 后复用 tile/保证 GM 可见 | `SetWaitFlag<HardEvent::MTE3_MTE2>`、`SyncAll` | store event 后等待，必要时封装 `WaitStoreTileReusable()`；跨核消费前再 `SYNCALL` | 复用 UB tile 和发布 ready 是两个不同边界，不混用 |
| AIV worker 阶段 barrier | `AscendC::SyncAll()` | `pto::SYNCALL<pto::SyncAllMode::Soft, pto::SyncCoreType::AIVOnly>(gm, ub, activeWorkers)` | 前重排 count/prefix/scatter 三个 phase 用 AIVOnly，不让 AIC 参与 |
| bring-up 全核硬 barrier | mixed kernel `SyncAll` | `pto::SYNCALL<pto::SyncCoreType::Mix>()`，或带 workspace 的 `SYNCALL<SyncAllMode::Hard, SyncCoreType::Mix>(...)` | 只允许 overlap-off bring-up 或 debug stop 收口；生产路径逐步替换为细粒度事件 |
| AIC -> AIV ready | `CrossCoreSetFlag(SYNCFLAGC2V)` / `CrossCoreWaitFlag` | `pto::Event<pto::Op::TSTORE_ACC, pto::Op::TLOAD, false, EVENT_IDx>().Init<CrossCoreId>() / Wait<CrossCoreId>()` 或项目封装 | GMM1/GMM2 tile 或 sync group ready 用 cross-core id；不能用全局 `SYNCALL<Mix>` 代替 |
| AIV -> AIC ready | `CrossCoreSetFlag(SYNCFLAGV2C)` / `CrossCoreWaitFlag` | `pto::Event<pto::Op::TSTORE_VEC, pto::Op::TLOAD, false, EVENT_IDx>().Init<CrossCoreId>() / Wait<CrossCoreId>()` 或项目封装 | dispatch/GMM1 input ready、activation/GMM2 input ready 用 per-group/per-tile handoff |
| rank 间 metadata ready | HCCL signal / DataAsFlag | `TSTORE/TPUT` 写 count row，`TNOTIFY/TWAIT/TTEST` 或 DataAsFlag polling | count row 写完再 publish ready；wait 不能用 host barrier 替代 |
| rank 间 payload 搬运 | SHMEM/Catcoc/MC2 data copy | `pto::comm::TGET/TPUT`；能证明 flat-contiguous 时才用 `TPUT_ASYNC/TGET_ASYNC` | dispatch gather 用 `TGET`，combine return 用 `TPUT`；strided async 不支持时记录 primitive gap，不退回 AscendC |

前重排三个 phase 的同步接口固定如下：

| Phase | Producer | Consumer | PTO 同步 |
| --- | --- | --- | --- |
| count done | 每个 AIV worker 写 `initQuantWorkerTokenPerExpert[worker,*]` | main AIV merge count | `SYNCALL<Soft, AIVOnly>(..., activeWorkers)` |
| prefix done | main AIV 写 `tokenPerExpertMatrix`、`expertBase`、`blockPrefixPerExpert` | 每个 AIV worker scatter | `SYNCALL<Soft, AIVOnly>(..., activeWorkers)` |
| scatter+quant done | 每个 AIV worker 写 `expandedRowIdx/payload/scale` | 后续 count sync / debug stop | `SYNCALL<Soft, AIVOnly>(..., activeWorkers)` |

PTO 同步封装建议：

```cpp
template <typename GlobalSync, typename UbSync>
AICORE inline void InitQuantAivPhaseSync(GlobalSync& gmSync, UbSync& ubSync, int32_t activeWorkers)
{
    pto::SYNCALL<pto::SyncAllMode::Soft, pto::SyncCoreType::AIVOnly>(gmSync, ubSync, activeWorkers);
}

AICORE inline void WaitVecToScalar()
{
    pto::PtoSetWaitFlag<PIPE_V, PIPE_S>();
}

AICORE inline void WaitScalarToVec()
{
    pto::PtoSetWaitFlag<PIPE_S, PIPE_V>();
}
```

调用约束：

- 业务代码优先写 PTO 指令事件依赖，例如 `auto ev = TLOAD(tile, gm); TCVT(fp, tile, ev);`。
- 只有 PTO 指令接口暂时无法表达的 V/S 标量交互，才调用 `WaitVecToScalar/WaitScalarToVec`。
- 所有 `set_flag/wait_flag/pipe_barrier/dsb` 裸调用只能出现在 PTO helper 内；前重排主流程不直接写。
- `SYNCALL<Mix>` 不能出现在前重排生产路径。若 debug 需要，日志必须标记 `sync_scope=mix_debug`。
- AIC/AIV cross-core event 不属于前重排阶段，但本设计预留同一套接口，后续 GMM 接入时不得恢复 Catlass
  `CrossCoreSetFlag/WaitFlag`。

## 3. 数据契约

### 3.1 Shape 符号

```text
rankNum = EP rank 数
expertPerRank = 每 rank 本地专家数
globalExpertNum = rankNum * expertPerRank
M = token 数
topK = 每 token route 数
R = M * topK
K = hiddenSize
rowBytes = align_up(K, 64) for int8 dispatch payload
capacity = maxOutputSize
```

### 3.2 输入输出布局

复用现有 `WorkspaceLayout` 和 `PeerWindowLayout`：

| 字段 | 用途 |
| --- | --- |
| `tokenPerExpertMatrix` | 本 rank 的 local count 写到 `rankId` 行；后续 count sync 扩展到所有 rank |
| `blockTokenPerExpert` | 每个 global expert 的汇总 count，供 prefix/GMM metadata 使用 |
| `blockPrefixPerExpert` | 每个 global expert 的 packed row 起点，供 scatter 和后续 metadata 使用 |
| `initQuantWorkerTokenPerExpert` | 每 worker 的局部 count，布局为 `[kInitQuantMaxDispatchWorkers, align_up(globalExpertNum, 16)]` |
| `initQuantWorkerPrefixPerExpert` | 每 worker 在每个 expert 内的 scatter base，布局同 worker count |
| `expandedRowIdx` | route index 到 packed row 的映射 |
| `packedRowToRouteIndex` | packed row 到 route index 的映射 |
| `dispatchOffset` | `[expertPerRank]`，本地 expert 在 `gmm1InputInt8` 中的 row 起点，供 GMM task plan 和后续 gather 使用 |
| `tokenOwnerRankOffsets` | Stage C 使用的 per-worker cursor 或 main AIV 临时 cursor |
| `peerWindow.dispatchPayload` | int8 packed rows |
| `peerWindow.dispatchScale` | dynamic quant scale |
| `peerWindow.debugCounters` | 日志验收 counters |
| `workspace.stageStatus` | debug stop marker |
| `workspace.timelineScratch` | 每 worker stage 时间线 |

`expandedRowIdx[route]` 的合同：

- active 且 expert 合法且未超过 capacity：写 packed row。
- inactive token、非法 expert、capacity clip 后丢弃：写 `maxOutputSize`。
- 不允许保留未初始化值。

packed row 顺序采用稳定分桶：

```text
packedRow = expertBase[globalExpert] + perWorkerBase[worker, globalExpert] + localOrdinalInWorker
```

同一 expert 内按 worker 的 token range 顺序拼接；worker 内按 route index 升序。只要 token 分片边界按 token 对齐，
这个顺序等价于按 route index 稳定遍历后分桶。

## 4. 多核切分

### 4.1 Worker 数选择

AIV 逻辑 worker 数：

```text
workerCount = min(logicalAivCount, kInitQuantMaxDispatchWorkers)
workerCount <= ceil(R / minRoutesPerWorker)
workerCount satisfies route-shard cacheline alignment when required
```

`kInitQuantMaxDispatchWorkers` 当前为 40，对标原 FFN `aivNumInitRouting=40`。debug counter 不再作为 worker
scratch 容量上限；per-worker count/prefix 使用正式 workspace。每个 worker 处理一段完整 token：

```text
tokenBegin = floor(M * workerId / workerCount)
tokenEnd = floor(M * (workerId + 1) / workerCount)
routeBegin = tokenBegin * topK
routeEnd = tokenEnd * topK
```

按 token 而不是按 route 切分，是为了 `xActiveMask[token]`、topK 访问和后续日志更直观；同时避免一个 token 的
topK 被切到两个 worker。

### 4.2 Stage A: route count

每个 worker 遍历自己的 token range：

```text
if xActiveMask[token] == 0: skip
expert = expertIdx[token, slot]
if expert not in [0, globalExpertNum): skip
localCount[expert]++
```

`globalExpertNum` 小时使用 UB resident `localCount`，结束后一次写
`initQuantWorkerTokenPerExpert[worker, expert]`。
`globalExpertNum` 大时使用 GM scratch 的 cache-line 对齐 counter，worker 独占行，无原子冲突。

Stage A 结束后做一次 AIV phase sync。main AIV 归并：

```text
count[expert] = sum_worker initQuantWorkerTokenPerExpert[worker, expert]
tokenPerExpertMatrix[rankId, expert] = count[expert]
```

同时记录：

- `debugCounters[kInitQuantCounterBase + 0] = activeWorkers`
- `debugCounters[kInitQuantCounterBase + 1] = workerCount`
- `debugCounters[kInitQuantCounterBase + 2] = totalValidRoutes`
- `debugCounters[kInitQuantCounterBase + 3] = invalidRoutes`

### 4.3 Stage B: prefix 和 cursor base

main AIV 对 global expert 做串行 prefix。global expert 数通常是 `rankNum * expertPerRank`，相对 R 很小，串行
prefix 的成本低于多核 prefix 的同步成本。

```text
running = 0
for expert in [0, globalExpertNum):
    expertBase[expert] = running
    clipped = min(count[expert], capacity - running)
    effectiveCount[expert] = max(clipped, 0)
    running += effectiveCount[expert]
```

然后计算每个 worker 在每个 expert 内的 base：

```text
workerRunning = 0
for worker in [0, workerCount):
    initQuantWorkerPrefixPerExpert[worker, expert] = workerRunning
    workerRunning += initQuantWorkerTokenPerExpert[worker, expert]
```

capacity clip 在 Stage C 生效。若 `expertBase + workerBase + localOrdinal >= capacity`，该 route 丢弃。

Stage B 输出：

- `initQuantWorkerPrefixPerExpert[worker, expert]`
- `initQuantWorkerTokenPerExpert[worker, expert]`
- `blockPrefixPerExpert[expert]`
- `blockTokenPerExpert[expert]`
- `tokenOwnerRankOffsets[expert] = expertBase[expert]`

Stage B 后做第二次 AIV phase sync，保证所有 worker 能读到 prefix。

### 4.4 Stage C: stable scatter + dynamic quant

每个 worker 再次遍历自己的 token range，维护 UB/GM cursor：

```text
localOrdinal[expert] = 0
for token in token range:
  for slot in topK:
    validate route
    packedRow = blockPrefixPerExpert[expert] + initQuantWorkerPrefixPerExpert[worker, expert] +
                localOrdinal[expert]++
    if packedRow >= capacity:
        expandedRowIdx[route] = maxOutputSize
        continue
    expandedRowIdx[route] = packedRow
    packedRowToRouteIndex[packedRow] = route
    quantize inputA[token, :] -> dispatchPayload[packedRow, :]
    dispatchScale[packedRow] = scale
```

量化策略使用 per-row dynamic quant。日志和下游保存 dequant scale：

```text
absMax = max(abs(x[token, col])) over K
scale = max(absMax / 127, eps)       // 写入 dispatchScale[packedRow]
invScale = 1 / scale                 // 传给 PTO TQUANT
q = clamp(round(x * invScale), -127, 127)
```

PTO Vec 主路径语义：

```text
GM->UB fixed-UB load half input tile
PTO-lowered half2float into fp32 tile
PTO-lowered abs into abs tile
PTO rowmax lower helper -> chunk max
scalar UB update -> row max
fixed-UB scale/invScale parameter tiles
fixed-UB store dispatchScale
PTO-lowered fp32*invScale -> int8 payload
fixed-UB store dispatchPayload
```

`K <= routeQuantTileCols` 时整行处理。`K > routeQuantTileCols` 时按列 chunk：

1. 第一遍按 chunk 求 row absMax，并在 UB 中归约为 row scale。
2. 第二遍按相同 chunk 生成 int8 payload。

为了避免大 K 下读两次 GM 成为瓶颈，第二版可以把 row 的 chunk max 写到 `timelineScratch` 或 dedicated scratch，
但第一版优先保持实现简单和可验收。

Stage C 结束后做第三次 AIV phase sync。main AIV 设置：

- `debugCounters[kInitQuantCounterBase + 4] = packedRows`
- `debugCounters[kInitQuantCounterBase + 5] = clippedRoutes`
- `debugCounters[kInitQuantCounterBase + 6] = quantizedRows`
- `stageStatus[stageBase + 5] = 103`

## 5. UB 与 Buffer 规划

### 5.1 Count/Prefix

小 expert 数主路径：

```text
localCount: align_up(globalExpertNum, 16) * int32
localOrdinal: align_up(globalExpertNum, 16) * int32
```

`globalExpertNum <= 64` 时这两块小于 512B。即使扩展到 256 expert，也远小于 UB 预算。

### 5.2 Quant

使用固定 tile cols：

```text
routeQuantTileCols = min(1024 或 2048, align_down(UB budget / bufferFactor))
bufferFactor 约为 half input + float tmp + abs tmp + int8 output + reduce tmp
```

推荐第一版沿用现有 `kM2RouteQuantTileCols = 1024`，对应 PTO tile offsets 已存在。每个 worker 只处理一行的一个
col chunk，避免多 row tile 带来的动态 packed row scatter 复杂度。后续优化可以扩成 2-4 行一组，提高 MTE 连续搬运效率。

### 5.3 GM workspace

第一版不新增大块 workspace。需要新增或复用：

- `blockTokenPerExpert`: `[globalExpertNum] int32`，per-expert 汇总 count
- `blockPrefixPerExpert`: `[globalExpertNum] int32`，per-expert packed row base
- `initQuantWorkerTokenPerExpert`: `[kInitQuantMaxDispatchWorkers, align_up(globalExpertNum, 16)] int32`
- `initQuantWorkerPrefixPerExpert`: `[kInitQuantMaxDispatchWorkers, align_up(globalExpertNum, 16)] int32`
- `tokenOwnerRankOffsets`: `[globalExpertNum] int32`，作为 expert base
- `dispatchOffset`: `[expertPerRank] int32`，保存本地 expert 的 row 起点。旧实现若按 `R/expandedRows` 分配，会和后续
  workspace 字段的 host/device layout 不一致；本阶段按实际消费方收敛为本地 expert 数。

如果 per-worker scratch 继续复用 debug counter 或 `blockTokenPerExpert/blockPrefixPerExpert` 汇总字段，多 worker
count 会互相覆盖或被 debug scratch 上限限制。这是实现和 review 必须检查的 layout 改动点。

## 6. 同步设计

本阶段只在 AIV 内同步，不让 AIC 参与：

| 同步点 | 参与者 | 目的 |
| --- | --- | --- |
| `phase_sync_count_done` | active AIV + main AIV | 所有 worker count 写完，main AIV 才能归并 |
| `phase_sync_prefix_done` | active AIV + main AIV | prefix 和 cursor base 可见，worker 才能 scatter |
| `phase_sync_scatter_done` | active AIV + main AIV | dispatch payload/scale 全部写完，后续 count sync 才能发布 |

同步实现使用 PTO 同步入口，优先采用：

```cpp
pto::SYNCALL<pto::SyncAllMode::Soft, pto::SyncCoreType::AIVOnly>(syncWorkspaceGlobal, syncWorkspaceTile, activeWorkers);
```

如果第一版为了接入现有 mixed kernel 暂时复用 `M3NDispatchHardPhaseSync()`，该 helper 必须收敛成 PTO wrapper，
调用点不能直接出现 `AscendC::SyncAll()`。后续要拆成 AIV-only soft sync，减少 AIC 等待。

当前 mixed kernel 的 T18 实现只替换这三个 phase sync 为 AIV-only。worker assignment、state clear、worker
scratch clear、count publish/wait、prefix/gather 和 stop17 final sync 仍保留 hard/mix 配对。AIC 侧
`M3NDispatchAicWaitForAivSubphases` 只在 production/stop17 路径扣减这 3 个 AIV-only phase；stop13、
stop18-stop21 等调试停止路径保持原 mixed 轮次，避免再次出现 AIC/AIV `SYNCALL<Mix>` 配对不一致。

- Stage A/B/C 只需要 AIV side 可见性，不使用 Catlass/AscendC 的 mixed barrier。
- PTO 指令依赖优先用 `RecordEvent` 串接和 `TSYNC(event)` 表达；显式 fence 只允许出现在 PTO helper 内。
- 不使用跨 AIC/AIV flag。该阶段完成后才发布给后续 dispatch/GMM1。

## 7. 日志验收设计

### 7.1 Debug stop 阶段

正式验收只保留 stop17。早期用于定位的 stop11/12/14/15/16 不作为常规验收入口；临时 debug 点用完删除，
不要通过新增开关保留中间阶段日志。

| debug stop | 停止点 | 必验输出 |
| --- | --- | --- |
| 17 | 前重排 ready flag 后 | local route count、capacity-clipped `expandedRowIdx`、dispatch payload/scale、full `tokenPerExpertMatrix`、`cumsumMM`、`preSumBeforeRank`、`expertTokenNums`、`gmm1InputInt8`、`routingPerTokenScale`、`dispatchGroupReady` |

stop17 的结构体验收点在 dispatch ready 后；`init_quant_e2e_us` 的结束点在 dispatch 阶段之前，
即前重排 route-pack/quant 最终同步之后。host 只输出最终结构体验收摘要、前重排 e2e 和阶段边界耗时，
不打印中间 token/row timeline 明细。

### 7.2 必须打印的日志字段

host 在 `[CorrectnessReport]` 下打印必要汇总字段，不为了日志完整性新增大量一次性 debug 输出：

```text
init_quant_final_stop=17
init_quant_e2e_us=<device_cycles_converted_to_us>
init_quant_prepare_us=<device_cycles_converted_to_us>
init_quant_count_us=<device_cycles_converted_to_us>
init_quant_prefix_us=<device_cycles_converted_to_us>
init_quant_route_pack_quant_us=<device_cycles_converted_to_us>
init_quant_e2e_after_prepare_us=<device_cycles_converted_to_us>
init_quant_e2e_after_count_us=<device_cycles_converted_to_us>
init_quant_e2e_after_prefix_us=<device_cycles_converted_to_us>
init_quant_e2e_after_route_pack_quant_us=<device_cycles_converted_to_us>
init_quant_worker_count=<workerCount>
init_quant_active_workers=<activeWorkers>
init_quant_worker_mask=<bitmask>
init_quant_route_count_match=true
init_quant_expanded_row_match=true
init_quant_payload_sample_match=true
init_quant_token_matrix_full_match=true
init_quant_prefix_match=true
init_quant_gmm1_input_match=true
init_quant_dispatch_ready_match=true
route_quant_path=pto_vec
route_quant_scalar_fallback_reason=none
init_quant_dispatch_parallel_path=m3n_multi_worker|pto_main_aiv
init_quant_dispatch_parallel_fallback_reason=none|invalid_global_expert_num
```

失败时必须同时打印 first mismatch：

```text
buffer=init_quant.expandedRowIdx mismatches=<n> first_index=<route> actual=<a> expected=<e>
buffer=init_quant.dispatchPayloadInt8 mismatches=<n> first_index=<elem> actual=<a> expected=<e>
buffer=init_quant.dispatchScale mismatches=<n> first_index=<row> actual=<a> expected=<e>
```

最终验收点只保留结构体/输出匹配摘要和 first mismatch。中间阶段日志、临时 counter 原文和 probe dump 不进入
常规验收输出。

### 7.4 耗时记录

前重排 e2e 使用 device 侧 syscnt 记录，不使用 host launch/sync 时间替代：

- begin：进入前重排主入口后，第一次前重排同步前；
- end：route-pack/quant 完成后的最后一次前重排同步之后；后续 count publish/wait、prefix metadata、
  dispatch gather 和 stop17 final mix sync 不计入 `init_quant_e2e_us`；
- stage boundary：同一计时窗口内额外记录 `prepare/count/prefix/route_pack_quant` 四个粗粒度耗时。large 路径中
  `count` 覆盖 worker route count 到 count phase sync 完成，`prefix` 覆盖 count merge/prefix 到 prefix phase
  sync 完成，`route_pack_quant` 覆盖 scatter+dynamic quant 到最终前重排 phase sync 完成；full-load 路径把融合的
  count/prefix/scatter+quant 记录在 `route_pack_quant`，`count/prefix` 可为 0。
- host：从 workspace 固定 slot 读取 begin/end，按 A3 `timeline_syscnt_cycles_per_us=1850` 换算并打印
  `init_quant_e2e_us`、单阶段耗时和 `init_quant_e2e_after_*_us` 阶段边界累计耗时。

### 7.3 样例验收命令

第一阶段验收固定跑 small 和 large 两个 shape：

```bash
bash scripts/run_a3.sh --backend int8 --m2-fused-full 1 --m2-fused-debug-stop-stage 17 \
  --case-name small -pes 2 -M 16 -K 128 -N 128 -topK 2 -expertPerPe 2 \
  --max-output-size 32 --dry-run 0 --skip-kernel-launch 0
```

期望日志至少包含：

```text
init_quant_route_count_match=true
init_quant_expanded_row_match=true
init_quant_payload_sample_match=true
init_quant_token_matrix_full_match=true
init_quant_prefix_match=true
init_quant_gmm1_input_match=true
init_quant_dispatch_ready_match=true
init_quant_e2e_us=<nonzero>
pass=true
```

large case 覆盖非 2 次幂 token 数和更大 route 数：

```bash
bash scripts/run_a3.sh --backend int8 --m2-fused-full 1 --m2-fused-debug-stop-stage 17 \
  --case-name large -pes 2 -M 4097 -K 128 -N 128 -topK 2 -expertPerPe 2 \
  --max-output-size 8194 --dry-run 0 --skip-kernel-launch 0
```

期望：

```text
init_quant_worker_count>=2
init_quant_active_workers>=2
init_quant_route_count_match=true
init_quant_expanded_row_match=true
init_quant_payload_sample_match=true
init_quant_token_matrix_full_match=true
init_quant_prefix_match=true
init_quant_gmm1_input_match=true
init_quant_dispatch_ready_match=true
init_quant_e2e_us=<nonzero>
pass=true
```

也可以使用：

```bash
bash scripts/run_initquant_acceptance.sh
```

small/large 只是 anchor。生产 hardening 任务还要跑参数族扩展 case，命令仍使用显式 shape 参数，不新增中间日志：

```bash
# more tokens: 验证 worker 数和 metadata 写回随 R 扩展
bash scripts/run_a3.sh --backend int8 --m2-fused-full 1 --m2-fused-debug-stop-stage 17 \
  --case-name more-tokens -pes 2 -M 16384 -K 128 -N 128 -topK 2 -expertPerPe 2 \
  --max-output-size 32768 --dry-run 0 --skip-kernel-launch 0

# topK expansion: 验证任意 topK token-centric quant，不做 topK=2 特化
bash scripts/run_a3.sh --backend int8 --m2-fused-full 1 --m2-fused-debug-stop-stage 17 \
  --case-name topk-expansion -pes 2 -M 2048 -K 128 -N 128 -topK 4 -expertPerPe 2 \
  --max-output-size 8192 --dry-run 0 --skip-kernel-launch 0

# skew + topK: 验证专家倾斜下 per-worker prefix/capacity 仍确定
bash scripts/run_a3.sh --backend int8 --m2-fused-full 1 --m2-fused-debug-stop-stage 17 \
  --case-name skewed -pes 2 -M 4097 -K 128 -N 128 -topK 4 -expertPerPe 2 \
  --max-output-size 16388 --dry-run 0 --skip-kernel-launch 0

# large K: 验证 column chunk/row tile，不退 scalar fallback
bash scripts/run_a3.sh --backend int8 --m2-fused-full 1 --m2-fused-debug-stop-stage 17 \
  --case-name large-k -pes 2 -M 1024 -K 1024 -N 128 -topK 4 -expertPerPe 2 \
  --max-output-size 4096 --dry-run 0 --skip-kernel-launch 0

# very large K: 验证整行无法常驻 UB 时仍按 column chunk 走 PTO Vec
bash scripts/run_a3.sh --backend int8 --m2-fused-full 1 --m2-fused-debug-stop-stage 17 \
  --case-name large-k-7168 -pes 2 -M 1024 -K 7168 -N 128 -topK 4 -expertPerPe 2 \
  --max-output-size 4096 --dry-run 0 --skip-kernel-launch 0
```

这些扩展 case 的期望字段和 small/large 一致：结构体匹配为 true、`route_quant_path=pto_vec`、
`route_quant_scalar_fallback_reason=none`、`init_quant_e2e_us` 非 0。验收脚本可以汇总这些字段，但不得增加
Stage A/B/C 中间 dump。

## 8. 边界场景

| 场景 | 处理 |
| --- | --- |
| `xActiveMask=0` | 所有 topK route 写哨兵，不参与 count 和 quant |
| expert id 非法 | 写哨兵，`invalidRoutes++` |
| `count` 超过 capacity | 超出部分写哨兵，`clippedRoutes++` |
| 某 expert 0 token | count 为 0，prefix 不留空洞 |
| `K` 非 64 对齐 | payload row 按 64B 对齐，尾部填 0 |
| `M < workerCount` | worker 自动 inactive，active mask 不应误报 |
| `globalExpertNum` 较大 | 使用正式 per-worker workspace，worker 数按 `kInitQuantMaxDispatchWorkers` 和有效 work 收敛；不因 debug scratch 超限 fallback |

## 9. 实施切片

### Slice 1: host 和 layout 准备

- 新增正式 `initQuantWorkerTokenPerExpert/initQuantWorkerPrefixPerExpert` worker scratch；
  `blockTokenPerExpert/blockPrefixPerExpert` 保留 per-expert 汇总语义。
- host `PrintInitQuantDebugStopReport` 只保留 stop17 最终匹配摘要、first mismatch 和 `init_quant_e2e_us`。
- 增加 device API 审计脚本或 `rg` 检查，确保新前重排文件不包含 `AscendC::`、`Catlass::`、`catlass/`、
  `TQue`、`TPipe`、`TBuf`、`LocalTensor`、`DataCopyPad`。

### Slice 2: Stage A/B count+prefix

- 实现 worker assignment、local count、main AIV merge 和 prefix。
- stop17 最终验收覆盖 `tokenPerExpertMatrix[rankId]`。
- count/prefix 的 GM 访问使用 PTO `GlobalTensor`/`TLOAD`/`TSTORE` 或封装在 PTO helper 中的 scalar GM accessor；
  不移植原 AscendC sort/count 类。

### Slice 3: Stage C scalar scatter+quant

- 实现 stable scatter、expandedRowIdx、packedRowToRouteIndex。
- 先用 scalar row quant，stop17 最终验收覆盖 payload/scale。
- scalar 版本只作为 bring-up fallback，接口仍然不能用 AscendC/Catlass；能用 PTO Tile 的部分先用 PTO Tile。

### Slice 4: PTO vec quant 路径

- 用 PTO tile 替换 row quant 内部的 load/max/quant/store：`TLOAD/TCVT/TABS/TROWMAX/TMAX/TEXPANDS/TQUANT/TSTORE`。
- `route_quant_path=pto_vec`，fallback 原因必须为 `none`。
- 验收 `dispatchScale` 是 `absMax / 127`，`TQUANT` 参数 tile 是 `127 / absMax`。

### Slice 5: 多 rank 和 capacity 集成

- stop17 最终验收 full token matrix、clipped expandedRowIdx、prefix、GMM1 input 和 ready flag。

## 10. 任务列表

任务按可提交的最小闭环拆分。每个任务都要满足 PTO 重写边界：新增 device 主路径不能出现 `AscendC::`、
`Catlass::`、`catlass/` include、`TQue`、`TPipe`、`TBuf`、`LocalTensor`、`DataCopyPad`。

| ID | 任务 | 依赖 | 主要产物 | 验收 |
| --- | --- | --- | --- | --- |
| T0 | 固化 API 审计规则 | 无 | host 脚本或 README 命令，扫描新前重排 device 文件中的非 PTO API | `rg` 扫描返回空；日志可打印 `device_api_family=pto` |
| T1 | layout/workspace 扩容 | T0 | 新增 `initQuantWorkerTokenPerExpert/initQuantWorkerPrefixPerExpert` 正式 worker scratch；host/AIV/AIC layout 同步 | dry-run 打印 workspace offsets 不重叠；多 worker case 不覆盖 counter |
| T2 | initquant 验收摘要 | T1 | final stop17 输出匹配摘要和 first mismatch | stop17 日志字段完整，常规输出无中间阶段 dump |
| T3 | host golden 稳定分桶 | T1 | host 侧按 worker token range 拼接的 expected count、expandedRowIdx、packedRowToRouteIndex | 单 rank `expanded_row_match=true` |
| T4 | PTO AIV worker 调度 | T1 | `workerCount/activeWorkers/tokenBegin/tokenEnd` helper | `M < workerCount`、`M >= workerCount` 日志符合预期 |
| T5 | Stage A PTO count | T2,T4 | worker local count 写 `initQuantWorkerTokenPerExpert[worker, expert]`，汇总写 `blockTokenPerExpert[expert]` | stop17: `init_quant_route_count_match=true` |
| T6 | Stage B prefix/cursor | T5 | main AIV 归并 `tokenPerExpertMatrix[rankId]`，生成 expert base 和 per-worker base | stop17: count 矩阵无 mismatch |
| T7 | AIV-only PTO phase sync | T5,T6 | count/prefix/scatter 三个 phase sync helper，优先 `pto::SYNCALL<Soft, AIVOnly>` | 多 worker 不随机错；sync 次数日志为 3 |
| T8 | Stage C stable scatter 元数据 | T6,T7 | `expandedRowIdx`、`packedRowToRouteIndex`、invalid/clipped 哨兵写入 | stop17: `init_quant_expanded_row_match=true` |
| T9 | PTO scalar/bring-up quant | T8 | 用 PTO tile 能力完成最小 row quant，或只保留明确标记的 PTO scalar fallback | stop17: payload/scale sample 可对齐；fallback reason 非空 |
| T10 | PTO Vec dynamic quant | T9 | mixed AIV route quant 使用固定 UB pointer 的 PTO-lowered Vec 主路径，避开 public tile wrapper 卡顿 | `route_quant_path=pto_vec`，`route_quant_scalar_fallback_reason=none` |
| T11 | payload row tail padding | T10 | `K` 非 64B 对齐时尾部补 0 | sample dump 中 `[K, rowBytes)` 全 0 |
| T12 | capacity clip 集成 | T8,T10 | `packedRow >= maxOutputSize` 写哨兵，统计 clipped routes | stop17: clipped 数和 expandedRowIdx 匹配 |
| T13 | 多 rank count matrix 集成 | T12 | 本 rank count 与后续 count sync/dispatch 元数据衔接 | stop17: `init_quant_token_matrix_full_match=true` |
| T14 | 后续阶段回归 | T13 | 继续接 prefix、gather、ready flag | stop17: prefix/GMM1 input/ready 均 `pass=true` |
| T15 | 性能基线日志 | T10,T13 | 前重排入口和最后全量同步后记录 e2e cycles/us | 大 shape 下 activeWorkers >= 2，stop17 打印 `init_quant_e2e_us` |
| T16 | PTO fallback 决策 | T15 | 明确异常 shape/workspace 条件下 fallback 到 `pto_main_aiv`；count publish/wait/gather 与 ready flag 仍走 PTO helper | fallback 日志包含原因，不静默走非 PTO |
| T17 | baseline 对标脚本 | T15 | 同 shape/seed 下 PTO stop17 与原 FFN initquant e2e 对比脚本；原 FFN 命令缺失时显式报告 `missing_command` | small/large 记录 PTO `init_quant_e2e_us`、worker 数、path/reason；不宣称无 baseline 的商用达标 |
| T18 | sync 优化 | T17 | 将前重排 hard sync 收敛为必要 AIV-only phase sync，审计 count/prefix/scatter 三个阶段 | stop17 pass，sync 点数量和 scope 可审计；不新增中间验收日志 |
| T19 | UB-fits fast path | T17 | full-load/UB-fits 路径使用 UB resident count/prefix/scatter 和任意 topK token-centric quant | small anchor、UB-fits non-small、UB-fits `topK>2` stop17 pass；不得按 case name、`M==16` 或 `topK==2` 特化 |
| T20 | large-token quant pipeline | T17 | `R=M*topK` 大场景使用 token-centric quant、UB ping-pong，重叠 load/compute/store | large anchor、`M=8192/16384` token scale、`topK=4/8` sweep stop17 pass；不得每 route 重复 quant |
| T21 | worker/cache 调度 | T17 | 根据 `M/topK/K/globalExpertNum/route distribution` 选择 worker 数，正式 worker scratch 支持到 40 | skew route、`M>4097` more-token、topK sweep/more experts case activeWorkers 有效，metadata 不成为主瓶颈 |
| T22 | GMM cache 边界 | T17 | 明确 L1/L0/swizzle/preload 属于后续 GMM PTO 化，不计入 initquant 完成项 | design.md 边界清晰，GMM 阶段单独拆任务 |
| T23 | metadata GM 批量化 | T17,T21 | `expandedRowIdx`、`packedRowToRouteIndex`、count/prefix 写回按连续 route/expert tile 批量化 | small/large、`M=8192/16384` token scale、topK sweep 和 more experts case stop17 pass；无中间 dump 日志 |
| T24 | 大 K 行/列 tiling | T20,T23 | quant 支持大 K column chunk、row tile/ping-pong，不能因整行放不下 UB 退成单核 | `K=1024/7168` large K 与 `topK=4` case payload/scale pass；small/large、`M=16384`、`topK=8` 回归 pass；`route_quant_path=pto_vec` |
| T25 | GMM PTO 性能设计 | T22 | GMM1/GMM2 用 PTO `TMATMUL`、L1/L0 ping-pong、preload、swizzle 替代 Catlass/Catcoc 的任务设计 | 新 GMM design/task 拆分完成；不把 initquant pass 当整体商用性能完成 |
| T26 | 参数化非特化验收矩阵 | T19,T20,T21,T23,T24 | stop17 脚本支持 token/topK/K/distribution sweep，并做 device 非特化代码审计 | `M={16,512,2048,4097,8192,16384}`、`topK={1,2,4,8}`、`K={128,1024,7168}`、skew、capacity clip 均 pass；device 主路径无 small/large/topK==2 写死 |

推荐提交顺序：

1. `T0-T3`：先把验收和 golden 固化，避免后续实现变成不可判定。
2. `T4-T8`：完成多核稳定分桶，不做高性能量化也能先验收 mapping。
3. `T9-T12`：补齐 PTO quant、padding 和 capacity。
4. `T13-T16`：接入多 rank 和后续阶段，再看性能基线。
5. `T17-T26`：生产性能 hardening，覆盖 baseline、sync、UB/full-load、large-token、大 `topK`、大 `K`、参数矩阵和后续 GMM 边界。

阶段开发时可以短期增加局部定位点，但交付前必须删除中间验收日志。前重排主验收统一跑 stop17。

### 10.1 任务与 initquant 阶段对应关系

| Task | initquant 阶段 | 作用 |
| --- | --- | --- |
| T0 设计约束固化 | 全局前置 | 固定 PTO-only，不拷贝 AscendC/Catlass |
| T1 layout/workspace 扩容 | 全局前置 | 准备 initquant 输入输出、workspace、peer window |
| T2 initquant 验收摘要 | 全阶段验收前置 | 让 stop17 能直接判定 pass/fail |
| T3 host golden 稳定分桶 | 全阶段验收前置 | 给 count/prefix/scatter/quant 生成 expected |
| T4 PTO AIV worker 调度 | Stage A/B/C 共用 | 确定多核并发分工 |
| T5 Stage A PTO count | Stage A: route count | 每 worker 统计 expert route 数 |
| T6 Stage B prefix/cursor | Stage B: expert prefix | 生成 expertBase 和 per-worker scatter base |
| T7 AIV-only PTO phase sync | Stage A/B/C 共用同步 | 固定 count/prefix/scatter 三个 AIV-only barrier |
| T8 Stage C stable scatter 元数据 | Stage C-1: scatter metadata | 写 `expandedRowIdx`、`packedRowToRouteIndex` |
| T9 PTO scalar/bring-up quant | Stage C-2: quant bring-up | 先打通 payload/scale 输出 |
| T10 PTO Vec dynamic quant | Stage C-2: production quant | 替换成 PTO Vec 高性能量化 |
| T11 payload row tail padding | Stage C-3: padding | 处理 `K -> rowBytes` 尾部 0 |
| T12 capacity clip 集成 | Stage C-4: capacity | 处理 clip 和哨兵 |
| T13 多 rank count matrix 集成 | initquant 后置衔接 | 把本地 count 接到后续 count sync |
| T14 后续阶段回归 | initquant 后置衔接 | 验证不破坏 prefix/GMM1 ready 等后续阶段 |
| T15 性能基线日志 | 全阶段性能验收 | 验证多 worker 和生产性能方向；stop17 打印前重排 e2e 时间 |
| T16 PTO fallback 决策 | 全局异常路径 | 明确异常 shape/workspace 条件下仍走 `pto_main_aiv` fallback |
| T17 baseline 对标脚本 | 全阶段性能验收 | 建立原 FFN 与 PTO 的 e2e 对比口径 |
| T18 sync 优化 | Stage A/B/C 共用同步 | 收敛为必要 AIV-only phase sync |
| T19 UB-fits fast path | Stage A/B/C 融合路径 | 对标原 FFN full-load，减少小/UB-fits 场景固定开销 |
| T20 large-token quant pipeline | Stage C-2: production quant | `R=M*topK` 大场景下任意 topK token-centric quant 和 ping-pong，覆盖 `M>4097` |
| T21 worker/cache 调度 | Stage A/B/C 共用 | 按 `M/topK/K/专家分布` 动态选择 worker 和缓存策略，worker 数不能固定为 large anchor 的 4 |
| T22 GMM cache 边界 | 后续 GMM 前置设计 | 明确 L1/L0/swizzle/preload 不属于 initquant |
| T23 metadata GM 批量化 | Stage A/B/C metadata | tok/topK 更多时减少 metadata GM store/flush 开销，覆盖 `M=8192/16384` |
| T24 大 K 行/列 tiling | Stage C-2: production quant | hiddenSize 较大时按 column chunk 和 row tile 保持 PTO Vec 主路径，覆盖 `K=1024/7168` |
| T25 GMM PTO 性能设计 | 后续 GMM PTO 化 | 设计 TMATMUL、L1/L0、preload、swizzle 替代 Catlass/Catcoc |
| T26 参数化非特化验收矩阵 | 全阶段生产验收 | 用 token/topK/K/distribution sweep 约束 T19-T24 不能按 anchor case 特化，small/large 只作为 anchor |

按阶段推进：

```text
前置准备:
  T0, T1, T2, T3, T4

Stage A: route count
  T5, T7(count done)

Stage B: expert prefix + per-worker cursor base
  T6, T7(prefix done)

Stage C: stable scatter + dynamic quant
  T8, T9, T10, T11, T12, T7(scatter+quant done)

后置集成:
  T13, T14

性能与异常路径:
  T15, T16, T17, T18, T19, T20, T21, T22, T23, T24, T25, T26
```

与原始 FFN initquant 子阶段的对应：

| 原 FFN 子阶段 | PTO 重写任务 | 替换关系 |
| --- | --- | --- |
| `sort / VBS / VMS / SortOut` | T5 + T6 + T8 | 用 `count -> prefix -> stable scatter` 替代全量 sort |
| `expertTokenOut` | T5 + T6 | count 和 expert prefix 直接生成 |
| `srcToDst` | T8 | 生成 `expandedRowIdx` 和 `packedRowToRouteIndex` |
| `gather + dynamic quant` | T9 + T10 + T11 | 直接 PTO quant 到 `dispatchPayload/dispatchScale` |
| capacity/drop 处理 | T12 | packed row 超 `maxOutputSize` 写哨兵 |

历史定位点与最终验收的对应：

| 定位点 | 覆盖任务 | stop17 最终验收覆盖内容 |
| --- | --- | --- |
| Stage A | T5 + T6 + T7 | route count、worker count、prefix 初步事实源 |
| Stage C | T8 + T9/T10 + T11 | mapping、payload、scale、padding |
| capacity/multi-rank | T12 + T13 | capacity clip、多 rank/full count matrix |
| post-dispatch metadata | T14 | 后续 prefix、GMM1 input、ready flag 回归 |

中间定位点不进入常规脚本和常规日志；交付验收只跑 stop17。

## 10.3 卡顿定位经验

- 同步卡住时 kernel 通常不能正常返回，host 侧也读不到卡住后的 counter；外层必须用 `timeout` 防止验收挂死。
- 临时计数只放在疑似 wait/notify 之前，或配合早退 stop 定位，确认后删除，不保留为开关。
- 优先用最终 stop17 的结构体验收和 e2e 耗时判断是否回归；中间 timeline/counter 原文不长期写入文档或脚本输出。

## 11. 风险和约束

- 分桶顺序必须和 host golden 一致。若 host golden 仍按简单 route 遍历分桶，需要明确 worker 拼接顺序并同步更新 golden。
- 多 worker cursor 不能用共享 atomic。每 worker 使用 `blockPrefixPerExpert` 的独占 base，避免 GM 原子和 nondeterminism。
- PTO vec quant 必须先保证 `dispatchScale = absMax / 127` 与 host golden 容差一致，再追求吞吐；`TQUANT` 使用的是
  `invScale = 127 / absMax`。
- 如果后续要完全复用生产 `moe_init_routing_quant_v2` 的 drop-pad 语义，当前分桶主路径需要补一个 drop-pad 分支或切 fallback。
- `blockTokenPerExpert`/`blockPrefixPerExpert` layout 扩容会影响 workspace size，host/device layout 必须同步修改。
- 设计目标是 PTO 重写，不是把原 FFN AscendC/Catlass 文件改名后接入。任何临时 fallback 都要有明确的
  `route_quant_scalar_fallback_reason` 或 `init_quant_dispatch_parallel_fallback_reason`，并通过 API 审计避免生产路径静默退回非 PTO 实现。
