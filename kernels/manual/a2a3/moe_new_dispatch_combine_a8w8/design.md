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
| count done | 每个 AIV worker 写 `blockTokenPerExpert[worker,*]` | main AIV merge count | `SYNCALL<Soft, AIVOnly>(..., activeWorkers)` |
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
| `blockTokenPerExpert` | 每 worker 的局部 count，布局为 `[workerCount, globalExpertNum]` |
| `blockPrefixPerExpert` | 每 worker 在每个 expert 内的 scatter base |
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
workerCount = min(logicalAivCount, 8)
workerCount <= floor(dispatchScratchSlots / (2 * align_up(globalExpertNum, 16)))
```

现有 `M3NDispatchWorkerCount` 已经使用类似约束。前重排第一版沿用最多 8 个 AIV worker，避免 debug counter 和
scratch 扩容过大。每个 worker 处理一段完整 token：

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

`globalExpertNum` 小时使用 UB resident `localCount`，结束后一次写 `blockTokenPerExpert[worker, expert]`。
`globalExpertNum` 大时使用 GM scratch 的 cache-line 对齐 counter，worker 独占行，无原子冲突。

Stage A 结束后做一次 AIV phase sync。main AIV 归并：

```text
count[expert] = sum_worker blockTokenPerExpert[worker, expert]
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
    blockPrefixPerExpert[worker, expert] = workerRunning
    workerRunning += blockTokenPerExpert[worker, expert]
```

capacity clip 在 Stage C 生效。若 `expertBase + workerBase + localOrdinal >= capacity`，该 route 丢弃。

Stage B 输出：

- `blockPrefixPerExpert[worker, expert]`
- `blockTokenPerExpert[worker, expert]`
- `tokenOwnerRankOffsets[expert] = expertBase[expert]`

Stage B 后做第二次 AIV phase sync，保证所有 worker 能读到 prefix。

### 4.4 Stage C: stable scatter + dynamic quant

每个 worker 再次遍历自己的 token range，维护 UB/GM cursor：

```text
localOrdinal[expert] = 0
for token in token range:
  for slot in topK:
    validate route
    packedRow = expertBase[expert] + blockPrefixPerExpert[worker, expert] + localOrdinal[expert]++
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

- `blockTokenPerExpert`: `[workerCount, globalExpertNum] int32`
- `blockPrefixPerExpert`: `[workerCount, globalExpertNum] int32`
- `tokenOwnerRankOffsets`: `[globalExpertNum] int32`，作为 expert base
- `dispatchOffset`: `[expertPerRank] int32`，保存本地 expert 的 row 起点。旧实现若按 `R/expandedRows` 分配，会和后续
  workspace 字段的 host/device layout 不一致；本阶段按实际消费方收敛为本地 expert 数。

如果现有 `blockTokenPerExpert` 和 `blockPrefixPerExpert` 当前只按 `globalExpertNum` 分配，需要扩成
`maxDispatchWorkers * globalExpertNum`，否则多 worker count 会互相覆盖。这是实现前必须检查的 layout 改动点。

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

stop17 的开始时间在前重排入口函数记录，结束时间在最后一次前重排全量同步之后记录。host 只输出最终
`init_quant_e2e_us`，不打印中间阶段 timeline 明细。

### 7.2 必须打印的日志字段

host 在 `[CorrectnessReport]` 下打印必要汇总字段，不为了日志完整性新增大量一次性 debug 输出：

```text
init_quant_final_stop=17
init_quant_e2e_us=<device_cycles_converted_to_us>
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
init_quant_dispatch_parallel_fallback_reason=none|global_expert_scratch_limit
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
- end：ready flag 发布后的最后一次前重排全量同步之后；
- host：从 workspace 固定 slot 读取 begin/end，按 A3 `timeline_syscnt_cycles_per_us=1850` 换算并打印
  `init_quant_e2e_us`。

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

## 8. 边界场景

| 场景 | 处理 |
| --- | --- |
| `xActiveMask=0` | 所有 topK route 写哨兵，不参与 count 和 quant |
| expert id 非法 | 写哨兵，`invalidRoutes++` |
| `count` 超过 capacity | 超出部分写哨兵，`clippedRoutes++` |
| 某 expert 0 token | count 为 0，prefix 不留空洞 |
| `K` 非 64 对齐 | payload row 按 64B 对齐，尾部填 0 |
| `M < workerCount` | worker 自动 inactive，active mask 不应误报 |
| `globalExpertNum` 超出 scratch | fallback 到 `pto_main_aiv`，count publish/wait/gather 仍使用 PTO 侧 scalar GM/Shard helper，不退回 AscendC/Catlass |

## 9. 实施切片

### Slice 1: host 和 layout 准备

- 扩容 `blockTokenPerExpert` 和 `blockPrefixPerExpert` 到 `[maxWorkers, globalExpertNum]`。
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
| T1 | layout/workspace 扩容 | T0 | `blockTokenPerExpert`、`blockPrefixPerExpert` 扩成 `[maxWorkers, globalExpertNum]`；host/device size 同步 | dry-run 打印 workspace offsets 不重叠；多 worker case 不覆盖 counter |
| T2 | initquant 验收摘要 | T1 | final stop17 输出匹配摘要和 first mismatch | stop17 日志字段完整，常规输出无中间阶段 dump |
| T3 | host golden 稳定分桶 | T1 | host 侧按 worker token range 拼接的 expected count、expandedRowIdx、packedRowToRouteIndex | 单 rank `expanded_row_match=true` |
| T4 | PTO AIV worker 调度 | T1 | `workerCount/activeWorkers/tokenBegin/tokenEnd` helper | `M < workerCount`、`M >= workerCount` 日志符合预期 |
| T5 | Stage A PTO count | T2,T4 | worker local count 写 `blockTokenPerExpert[worker, expert]` | stop17: `init_quant_route_count_match=true` |
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
| T16 | PTO fallback 决策 | T15 | 明确 `globalExpertNum` 超出 scratch 时 fallback 到 `pto_main_aiv`；count publish/wait/gather 与 ready flag 仍走 PTO helper | fallback 日志包含原因，不静默走非 PTO |

推荐提交顺序：

1. `T0-T3`：先把验收和 golden 固化，避免后续实现变成不可判定。
2. `T4-T8`：完成多核稳定分桶，不做高性能量化也能先验收 mapping。
3. `T9-T12`：补齐 PTO quant、padding 和 capacity。
4. `T13-T16`：接入多 rank 和后续阶段，再看性能基线。

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
| T16 PTO fallback 决策 | 全局异常路径 | 明确超出 scratch/shape 时仍走 `pto_main_aiv` fallback |

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
  T15, T16
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
