# dispatch_ffn_combine_v3 A5 设计说明

## 1. 这份文档解决什么问题

本文面向第一次阅读 `kernels/manual/a5/dispatch_ffn_combine_v3` 的开发者，目标是把三个问题讲清楚：

1. 这个项目要解决什么 MoE 场景问题。
2. MegaMoE / MC2 背景里的算法思想，如何落到当前 A5 PTO 项目里。
3. 当前代码从 host 到 device 的主要调用链、数据布局、同步关系和关键文件在哪里。

本文描述的是当前 A5 standalone 项目，不展开其他代码仓的工程结构。MegaMoE 算法资料和 `megamoe理解.md` 只作为背景材料；当前仓内事实以本目录代码为准。

## 2. 背景：MoE 的 dispatch、FFN、combine 是什么

MoE 模型里，每个 token 不会经过所有专家，而是先由 gating/top-k 算出它要去哪些 expert：

```text
expert_idx[token, k] = token 的第 k 个目标专家
probs[token, k]      = 这个专家输出在最终结果里的权重
```

随后 routed expert 路径大致分成三段：

```text
1. dispatch：把 token 发送到目标 expert 所在 rank
2. FFN：目标 rank 上执行专家前馈网络，通常是 GMM1 -> SwiGLU -> GMM2
3. combine：把专家结果送回 token 原始 rank，按 probs 做加权求和并恢复原 token 顺序
```

如果这三段拆成多个算子或多个通信步骤，会有几个成本：

- 通信和计算之间有多次全局同步。
- 中间结果频繁落到 GM，workspace 和 remote window 压力大。
- token routing、量化、GMM、SwiGLU、combine/unpermute 很难形成流水。
- AIC 计算核和 AIV 搬运/通信核的职责不清晰，容易互相等待。

`dispatch_ffn_combine_v3` 的目标就是把这些步骤收进一个 mixed AIC/AIV kernel，用 PTO 通信和计算原语显式表达跨 rank 搬运、GMM、vector epilogue 和同步，让通信、GMM 和后处理尽量贴在一起执行。

## 3. 先分清几个容易混淆的概念

### 3.1 gating 和 routing 不是一回事

- **gating**：模型已经算好的选择结果，告诉每个 token 去哪些 expert，以及每个 expert 的权重。
- **routing**：kernel 内根据 `expert_idx` 和 `probs` 把 token 重新组织成 dispatch / GMM 可以消费的布局。

当前项目里的 routing 会做这些事：

1. 按 `topK` 把 token 展开成多份。
2. 按 global expert id 排列展开后的 token。
3. 记录每个展开行来自哪个原始 token，也就是 `expandedRowIdx`。
4. 统计每个 expert 收到多少 token，也就是 `localTokenPerExpert`。
5. 在 int8 路径里顺便做 per-token 动态量化，生成 per-token scale。

一句话：**gating 决定去哪个 expert，routing 决定 token 在内存里怎么摆，才能给通信和 GMM 用。**

### 3.2 “通信后重排”不是恢复原始 token 顺序

以 `EP=2`、每卡 2 个 expert 为例：

```text
rank0 experts: A, B
rank1 experts: C, D
```

普通 AlltoAllV 收到的数据常常是 source-rank-major：

```text
rank1 recv buffer:
  来自 rank0: C0, D0
  来自 rank1: C1, D1

内存里可能是: C0, D0, C1, D1
```

但 GMM 希望同一个 local expert 的 token 连续：

```text
expert-major:
  C: C0, C1
  D: D0, D1

内存里希望是: C0, C1, D0, D1
```

所以这里的“通信后重排”指的是：**把 source-rank-major 或稀疏接收布局，压紧成 expert-major 的 dense GMM 输入矩阵。**

它不是为了恢复 token 的语义顺序。真正关心原始 token index 和 top-k 权重的是最后的 restore/combine 阶段，所以 kernel 需要保存 `expandedRowIdx` 和 `probs`。

### 3.3 MegaMoE 的核心优化不是“不做重排”

更准确地说，MegaMoE 的思路是：

> 保留通信前的本地 routing / quant / count，但把通信后的“稀疏到密排重排”改写成通信地址计算的一部分。

流程可以理解为：

```text
本地 routing/quant/count
  -> 交换每个 source rank、每个目标 expert 的 token 数
  -> 每个目标 rank 算 cumsum，知道每段 token 的最终落点
  -> 通信直接写/读到 expert-major 的最终位置
  -> 某个 expert 的连续 token 到齐后，对应 GMM 就能开始
```

也就是说，通信不是先乱放再整理，而是借助 `tokenPerExpert` 和 `cumsum` 直接落到 GMM 能消费的位置。

当前 A5 PTO 项目对应该思想的代码入口是：

- routing / quant：`moe_init_routing_quant_v2(...)`
- 全局 token count 同步：`CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(...)`
- cumsum 地址表：`GetCumsumForMMAIV(...)`
- remote gather 到 GMM 输入：`CopyGMToGMPerToken(...)`

### 3.4 Data / Count / Flag 和 DataAsFlag

在 dispatch 通信里，发送方通常要告诉目标方三类信息：

```text
Data  = token payload
Count = 每个 expert 实际收到多少 token
Flag  = 数据已经写完、可以读取的同步信号
```

DataAsFlag 的背景思想是把 Data、Count、Flag 尽量合并成一次投递：目标端看到 Flag 时，Data 和 Count 也已经可见，从而减少额外往返同步。

当前 A5 项目没有一个叫 `DataAsFlag` 的独立模块；它更直接可见的机制是：

- `tokenPerExpert` 保存 token count。
- `TNOTIFY/TWAIT` 做跨 rank ready/wait。
- `CrossRankSync()` 做跨 rank barrier。
- `TGET/TPUT` 做 remote window 数据搬运。

因此，本文把 DataAsFlag 当作算法背景，不把它写成当前代码里的同名实现。

## 4. 为什么要区分远端读和远端写

跨卡通信可以按两个维度拆开看：

1. 数据依赖方向
   - Dispatch-GMM：先通信，后 GMM。
   - GMM-Combine：先 GMM，后通信。
2. 通信发起方式
   - 远端读：我去别的 rank 的 window 读数据。
   - 远端写：我把数据写到别的 rank 的 window。

于是会有四种组合：

| 场景 | 远端读 | 远端写 |
| --- | --- | --- |
| Dispatch-GMM | 目标 expert rank 主动读取源 rank token，然后本 rank AIC 计算 | 源 rank 主动写 token 到目标 rank，目标 rank 等数据再计算 |
| GMM-Combine | 原 token rank 主动读取 expert rank 的输出 | expert rank 算完后主动写回原 token rank |

MegaMoE 文章强调的关键不是机械列举四种模式，而是让通信发起者尽量和计算消费者/生产者在同一张卡上：

- Dispatch-GMM 更适合目标 rank 主动远端读：目标 rank 的 AIV 读数据，本 rank 的 AIC 消费数据。
- GMM-Combine 更适合 expert rank 主动远端写：expert rank 的 AIC 产出数据，本 rank 的 AIV 负责写回。

这样细粒度流水主要变成卡内 AIV/AIC 同步，而不是每个 tile 都做跨卡握手。

当前 A5 PTO 项目中，dispatch gather 侧主要通过 `TGET` 从 peer window 拉数据到本 rank，combine 侧通过 `TPUT` 把结果写回 owner rank。

## 5. 项目目录和文件分工

```text
kernels/manual/a5/dispatch_ffn_combine_v3/
├── run.sh                               # 生成数据、构建、mpirun 启动
├── main.cpp                             # Host 主程序：MPI/ACL/HCCL 初始化、launch、计时、校验
├── runtime_context.{hpp,cpp}            # HCCL comm/resource/window context 初始化
├── tiling_builder.{hpp,cpp}             # Host tiling、workspace bytes、block_dim、window 容量检查
├── data_utils.{hpp,cpp}                 # case.json、rank 文件路径、FP16 compare
├── kernel_launch.hpp                    # Host launch 参数结构
├── comm_mpi.h                           # dlopen MPI，避免硬链接 MPI
├── scripts/gen_data.py                  # 随机输入和 CPU golden 生成
├── op_kernel/
│   ├── dispatch_ffn_combine.cpp         # device kernel 入口 + host launch stub
│   ├── dispatch_ffn_combine.h           # device 参数整理、A5 policy 和 kernel params 构造
│   ├── dispatch_ffn_combine_kernel.hpp  # 主 kernel 类，真实 stage 实现集中在这里
│   ├── dispatch_ffn_combine_tiling.h    # tiling/runtime/launch config 结构
│   ├── stages/                          # AIC/AIV 主流程 facade
│   ├── utils/                           # PTO helper、HCCL window、layout、MMAD/epilogue policy
│   ├── moe_init_routing_quant_v2/        # routing、sort、gather、dynamic quant
│   └── unpermute/                       # restore/unpermute
└── DESIGN.md                            # 本文档
```

读代码时建议先按这个顺序：

1. `run.sh` 和 `main.cpp`：理解这个 standalone 程序怎么跑。
2. `runtime_context.cpp`：理解 HCCL remote window 怎么变成 device 可读 context。
3. `tiling_builder.cpp`：理解 shape、workspace 和 tiling 怎么传给 device。
4. `op_kernel/dispatch_ffn_combine.cpp`：理解 kernel 入口。
5. `op_kernel/dispatch_ffn_combine.h`：理解 A5 policy 和 params 怎么组装。
6. `op_kernel/stages/kernel_context.hpp`：先看 AIC/AIV 大流程。
7. `op_kernel/dispatch_ffn_combine_kernel.hpp`：再下钻每个 stage 的实现。

## 6. Host 侧执行流程

Host 侧每个 MPI rank 对应一张 NPU。整体流程如下：

```text
run.sh
  -> scripts/gen_data.py 生成 out/case.json 和 rank*.bin
  -> cmake 构建 host exe 和 kernel so
  -> mpirun 启动 dispatch_ffn_combine_v3

main.cpp
  -> CommMpiInit
  -> aclInit / rtSetDevice / aclrtSetDevice
  -> rank0 HcclGetRootInfo，广播 root_info
  -> InitStandaloneRankRuntime
  -> LoadCaseConfig / BuildRankFileSet
  -> BuildDispatchFFNCombineTiling
  -> 分配 x/weight/scale/probs/output/workspace/tiling device buffer
  -> warmup loop
  -> measure loop
  -> verify loop
  -> D2H 拷回 out，与 expected_out 比较
```

几个关键点：

- `main.cpp` 每轮 launch 前都会清零 HCCL window、输出、`expert_token_nums` 和 workspace，避免上一轮残留影响结果。
- measure 使用 ACL event 统计 kernel 时间，再用 MPI gather 汇总每轮最大 rank 时间。
- verify 会再跑一轮，把 `out` 拷回 host，写 `output_rank*.bin`，并与 CPU golden 做 FP16 误差比较。

## 7. HCCL remote window 初始化

跨 rank 数据放在 HCCL 分配的 remote window 中，但 device kernel 不直接吃 HCCL 原始结构，而是使用项目自己的简化 context：

```cpp
struct PtoRemoteWindowContext {
    uint64_t workspaceBase;
    uint64_t workspaceBytes;
    uint32_t rank;
    uint32_t rankSize;
    uint64_t windowBytes;
    uint64_t windowIn[PTO_HCCL_MAX_RANKS];
    uint64_t windowOut[PTO_HCCL_MAX_RANKS];
};
```

`runtime_context.cpp` 的职责是：

1. 初始化 ACL stream 和 HCCL stream。
2. 通过 `HcclCommInitRootInfo` 建立 HCCL comm。
3. 通过 `HcclAllocComResourceByTiling` 分配通信资源。
4. 优先按 A5 direct context 解析 HCCL 返回的 `ctx_ptr`。
5. 解析成功后，把 `rank/rankSize/windowBytes/windowIn/windowOut` 规整成 `PtoRemoteWindowContext`。
6. 把这个 context 拷到 device，tiling 里只传 device pointer。
7. 如果 A5 direct context 解析失败，再 fallback 到 ring-style resource parser。

这里最近修复过的关键点是：A5 上 `HcclAllocComResourceByTiling` 返回的 `ctx_ptr` 不能直接当作 `PtoRemoteWindowContext` 使用，需要先按 A5 HCCL context layout 解析，再生成自己的 PTO context。

## 8. Tiling 和 workspace

Host tiling 由 `tiling_builder.cpp` 完成，核心输出是 `DispatchFFNCombineTilingData`，包含：

- shape：`M/K/N/topK/expertPerRank/worldSize/maxOutputSize`
- CoC tiling：`m0/k0/n0/ubMoveNum/commNpuSplit/...`
- routing quant tiling：来自 `MoeInitRoutingQuantV2TilingBase::DoTiling(...)`
- runtimeInfo：`remoteWindowContext`、`rank`、`rankSize`
- launchConfig：`blockDim`、`tilingKey`、`workspaceBytes`

Host 还会校验 HCCL remote window 是否够放这些区域：

```text
offsetA                   = 0
offsetPeerPerTokenScale   = AlignUp(windowBytes / 3, 512)
offsetD                   = offsetPeerPerTokenScale + 1 MiB
offsetPeerTokenPerExpert  = windowBytes - 2 MiB
signalBase                = windowBytes - 1 MiB
```

这些区域含义是：

| 区域 | 用途 |
| --- | --- |
| `offsetA` 起始区域 | dispatch/gather token payload |
| `offsetPeerPerTokenScale` | peer per-token scale 和部分 scratch |
| `offsetD` | dispatch output / GMM2 output / combine return payload |
| `offsetPeerTokenPerExpert` | 跨 rank token count 矩阵 |
| 最后 1 MiB | barrier counter、token-ready counter、PTO notify/wait signal |

普通输入、输出、tiling、workspace 仍然由 `aclrtMalloc` 分配；只有跨 rank 可见的数据和信号使用 HCCL remote window。

Device 侧 `WorkspaceInfo` 会把普通 workspace 切成这些逻辑区：

```text
expandedRowIdx
ptrcumsumMM
ptrPerTokenScale
ptrPerTokenScale2
ptrTokenPerExpert       # 结构里有字段，但 live path 会绑定到 remote window token-count 区域
ptrC                    # GMM1 output / SwiGLU input
ptrC2                   # GMM2 output
ptrA                    # dispatch gather 后的 GMM1 输入
ptrPermutedToken        # restore 前的 token buffer
ptrSumBeforeRank
ptrSoftFlagBase
```

## 9. Device kernel 入口

device 入口在 `op_kernel/dispatch_ffn_combine.cpp`：

```text
dispatch_ffn_combine<<<blockDim, nullptr, stream>>>(...)
  -> GET_TILING_DATA
  -> TILING_KEY_IS(1000010)
  -> KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)
  -> DispatchFFNCombine<int8_t, DTYPE_W1, DTYPE_OUT, false, true>
```

当前主路径固定为：

- int8 weight
- `transB=false`
- `weightNz=true`
- tiling key `1000010`
- mixed AIC/AIV task type
- A5 compile arch `dav-c310`

`op_kernel/dispatch_ffn_combine.h` 是 device wrapper：

1. `Init()` 从参数和 tiling 中取出 GM pointer、shape、rank、remoteWindowContext、CoC tiling、routing tiling。
2. `Process()` 选择 A5 policy、layout、GMM tile shape、epilogue policy。
3. 构造 `DispatchFFNCombineKernel::Params`。
4. 调用 `DispatchFFNCombineKernel`。

## 10. AIC / AIV 分工

这个 kernel 是 mixed AIC/AIV。可以把两类 core 理解成：

- AIC：更适合矩阵乘，主要跑 GMM1 和 GMM2。
- AIV：更适合搬运、routing、量化、通信、SwiGLU、combine、restore。

stage facade 在 `op_kernel/stages/kernel_context.hpp`，非常薄，但很适合作为阅读入口：

```text
AIC:
  RunGmm1Stage
  RunGmmInterlockStage
  RunGmm2Stage

AIV:
  RunRoutingStage
  RunDispatchGatherStage
  RunSwigluStage
  RunCombineStage
  RunRestoreStage
```

真实实现仍主要在 `op_kernel/dispatch_ffn_combine_kernel.hpp`。

## 11. AIV 主流程详解

### 11.1 Routing：本地展开、排序、量化、计数

入口：`RunRoutingImpl(...)`

主要动作：

1. `remoteWindow.Init(params.remoteWindowContext)` 初始化 remote window helper。
2. `ApplyXActiveMask(...)` 把 inactive token 的 expert id 改成 sentinel，避免参与真实 expert 计算。
3. 调用 `moe_init_routing_quant_v2(...)`：
   - 读取 `x`、`expert_idx`、`x_active_mask`。
   - 按 expert 排序/分组。
   - 写出 `expandedRowIdx`。
   - 写出本地 token count。
   - 生成动态量化后的 int8 token 和 per-token scale。
4. `CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(...)` 同步各 rank 的 token count。
5. `GetCumsumForMMAIV(...)` 生成 `cumsumMM`，用于后续 GMM 知道每个 expert 的 token row 范围。
6. 写出 `expert_token_nums` 给 host 校验/观察。
7. 通过 cross-core flag 通知 AIC：GMM1 可以开始消费部分 expert 数据。

### 11.2 Dispatch gather：把远端 token 拉成本地 expert-major GMM 输入

入口：`RunDispatchGatherImpl(...)`

它会遍历本地 expert 和各个 source rank，根据 `tokenPerExpert` / `cumsumMM` 算出：

- 从 peer remote window 的哪里读 token 和 per-token scale。
- 写到本 rank workspace 的 `ptrA` / `ptrPerTokenScale` 哪一段。

核心 helper 是 `CopyGMToGMPerToken(...)`，它使用 PTO `TGET` 做 remote read。读回来后，本地 workspace 中的 token 按 expert 连续排列，GMM1 可以直接按 expert 分组消费。

这个阶段体现了 MegaMoE 思路：不是通信完再密排，而是通信地址本身就按 expert-major 目标位置计算。

### 11.3 SwiGLU：GMM1 后处理和 GMM2 输入量化

入口：`RunSwigluImpl(...)`

GMM1 产出的是中间隐藏层，SwiGLU 会把 `N` 维拆成两半：

```text
x1 = first N/2
x2 = second N/2
SwiGLU(x) = silu(x1) * x2
```

当前 int8 路径里，SwiGLU epilogue 还会做：

- GMM1 dequant。
- per-token/per-channel scale 应用。
- SwiGLU vector 计算。
- 为 GMM2 重新做 per-token int8 quant。
- 生成 `ptrPerTokenScale2`。

实现主要在 `utils/block_epilogue_pertoken_swiglu.hpp`。

### 11.4 Combine：把 expert 输出送回 owner rank

入口：`RunCombineImpl(...)`

GMM2 完成后，每个 expert 的输出需要回到原 token 所在 rank。当前有两条 combine path：

- `CombineV1(...)`：按 expert group 做 row-wise return。
- `CombineV2(...)`：按 tile 做更细粒度的 return。

底层 epilogue 在：

- `utils/block_epilogue_pertoken_row.hpp`
- `utils/block_epilogue_pertoken_v2.hpp`

combine 阶段会根据 `preSumBeforeRank`、`tokenPerExpert`、`rank` 等信息算出结果应该写回哪个 owner rank 的 remote window 位置，并用 PTO `TPUT` 执行 remote write。

### 11.5 Restore：按原 token 顺序和 topK 权重聚合

入口：`RunRestoreImpl(...)`

restore 阶段做最后一步：

1. 等跨 rank combine 数据可见。
2. 根据 `expandedRowIdx` 找到每个输出 token 对应的多个 expert 输出。
3. 读取 `probs`。
4. 对 topK 专家输出做加权累加。
5. 写回最终 `out[M, K]`。

实现主要在 `unpermute/moe_token_unpermute.h`。

## 12. AIC 主流程详解

AIC 路径只管 GMM 主链：

```text
GMM1
  -> interlock
  -> GMM2
```

### 12.1 GMM1

入口：`GMM1(...)`

它等待 AIV 准备好 `cumsumMM` 和 dispatch gather 的 token 数据后，按 local expert 遍历：

```text
expert group -> 当前 expert 的 M 范围 -> BlockScheduler 切 tile -> BlockMmad 执行
```

GMM1 输入来自 `ptrA`，权重来自 `weight1`，输出写到 `ptrC`。执行完对应阶段后，通过 cross-core flag 通知 AIV 的 SwiGLU 阶段。

### 12.2 GMM2

入口：`GMM2(...)`

它等待 AIV 的 SwiGLU/quant 输出后，消费 `ptrPermutedToken` 和 `weight2`，输出写到 `ptrC2`。之后 combine 阶段会从 `ptrC2` 读取结果并发回 owner rank。

### 12.3 MMAD / epilogue policy

当前 A5 主路径使用：

- `ArchTag = pto_ext::Arch::AtlasA5`
- `MmadAtlasA5PreloadAsyncFixpipe`
- `EpilogueAtlasA5PerTokenDequantSwigluQuant`
- `EpilogueAtlasA5PerTokenDequant`
- `EpilogueAtlasA5PerTokenDequantV2`

PTO 化的重点不是把所有 AscendC substrate 都删除，而是把主计算表达成 PTO 视角：

- `TMATMUL` / `TMATMUL_ACC` 表达 matmul。
- `TSTORE_FP` / `TSTORE` 表达 accumulator/fixpipe store。
- `TLOAD` / `TSTORE` 表达规整 GM/UB 搬运。
- 无法安全替换的本地 pipe、buffer 生命周期、L1/L0/FIX substrate 继续保留。

## 13. PTO helper 和保留的 AscendC 边界

当前项目中 PTO 相关 helper 主要分三层：

| 文件 | 职责 |
| --- | --- |
| `utils/pto_global_view.hpp` | 把 raw GM pointer 组织成 PTO `GlobalTensor` view |
| `utils/pto_vector_ops.hpp` | 统一 vector load/store/cast/fill/add/mul/div/exp/reduce 等 PTO helper |
| `utils/hccl_window.hpp` | 把 HCCL remote window 包成 rank-aware PTO comm 地址和 signal helper |

常见 PTO primitive：

| 类型 | primitive |
| --- | --- |
| tile/view | `TASSIGN`, `pto::Tile`, `pto::GlobalTensor` |
| vector load/store | `TLOAD`, `TSTORE` |
| vector compute | `TCVT`, `TADD`, `TADDS`, `TMUL`, `TMULS`, `TDIV`, `TABS`, `TEXP`, `TROWMAX/TMAX` |
| matmul | `TMATMUL`, `TMATMUL_ACC` |
| remote data | `TGET`, `TPUT` |
| remote sync | `TNOTIFY`, `TWAIT` |
| sort/gather | `TSORT32`, `TMRGSORT`, `TGATHER` |

仍然保留的 AscendC / CANN substrate：

- `kernel_operator.h`, `__aicore__`, `__gm__`, `GM_ADDR`
- `TPipe`, `TQue`, `TBuf`, `LocalTensor` 生命周期
- `SetFlag`, `WaitFlag`, `PipeBarrier`, `SyncAll`
- `CrossCoreSetFlag`, `CrossCoreWaitFlag`
- `DataCacheCleanAndInvalid`, `dsb`
- 部分 L1/L0/FIX 执行骨架中的 `LoadData` / `Fixpipe` 类语义

这些保留项不是“没改完”的业务路径，而是 PTO 目前不替代的 kernel ABI、buffer 生命周期、本地 pipe 同步、cache 可见性和硬件执行 substrate。

## 14. 当前实现和 MegaMoE 理想形态的差距

这份代码已经具备 MegaMoE 风格的核心数据流：

```text
routing/quant/count
  -> token count sync
  -> cumsum
  -> remote gather 直接形成 expert-major GMM 输入
  -> GMM1
  -> SwiGLU/quant
  -> GMM2
  -> remote combine return
  -> restore/unpermute
```

但也要明确：MegaMoE 微信文章描述的是更完整的算法目标，当前 A5 PTO 项目是手写 kernel 示例，不应把文章里的所有设计点都说成已完全实现。当前实现里仍能看到较保守的同步方式，例如：

- 多处 `SyncAll<true>()`。
- AIC/AIV 之间主要通过 `CrossCoreSetFlag/WaitFlag` 协调。
- 部分 scoreboard / soft flag 思路不是最主要的 live path。
- 当前本机验证边界主要是 A5 compile-only；A5 runtime PASS 需要在 A5 环境运行确认。

因此阅读代码时可以这样理解：

- **算法方向**来自 MegaMoE：让通信直接服务 GMM 输入布局，减少通信后重排，并让通信/计算/后处理更容易流水。
- **工程骨架**来自 MC2 dispatch_ffn_combine：routing、GMM1、SwiGLU、GMM2、combine、unpermute 的大结构保留。
- **当前实现形态**是 A5 standalone PTO 示例：更强调 PTO primitive、HCCL remote window context、A5 policy 和可独立运行验证。

## 15. 关键文件速查

| 想看什么 | 入口文件 |
| --- | --- |
| 一键构建运行 | `run.sh` |
| host 主流程 | `main.cpp` |
| HCCL context / remote window 初始化 | `runtime_context.cpp`, `runtime_context.hpp` |
| tiling / workspace bytes / remote window 容量校验 | `tiling_builder.cpp`, `dispatch_ffn_combine_tiling.h` |
| host launch 参数 | `kernel_launch.hpp` |
| device kernel 入口 | `op_kernel/dispatch_ffn_combine.cpp` |
| device wrapper / A5 policy / params | `op_kernel/dispatch_ffn_combine.h` |
| AIC/AIV stage 总览 | `op_kernel/stages/kernel_context.hpp` |
| routing/count/cumsum/gather/GMM/combine/restore 主实现 | `op_kernel/dispatch_ffn_combine_kernel.hpp` |
| HCCL remote window device helper | `op_kernel/utils/hccl_window.hpp`, `op_kernel/utils/hccl_context.hpp` |
| PTO vector helper | `op_kernel/utils/pto_vector_ops.hpp` |
| PTO global view helper | `op_kernel/utils/pto_global_view.hpp` |
| routing/sort/quant 子模块 | `op_kernel/moe_init_routing_quant_v2/` |
| restore/unpermute 子模块 | `op_kernel/unpermute/` |
| GMM/SwiGLU/combine epilogue | `op_kernel/utils/block_*` |
| MC2 到 PTO 转换说明 | `mc2_2_pto.md` |
| MegaMoE 阅读笔记 | `megamoe理解.md` |

## 16. 阅读建议

如果只是想快速理解项目，可以按下面路线读：

```text
README.md
  -> DESIGN.md 第 2~4 节，理解算法背景
  -> main.cpp，理解 standalone 怎么跑
  -> runtime_context.cpp，理解 A5 HCCL window 修复和 context 传递
  -> tiling_builder.cpp，理解 shape/workspace/remote window 校验
  -> stages/kernel_context.hpp，理解 AIC/AIV 大流程
  -> dispatch_ffn_combine_kernel.hpp，按 stage 下钻
```

如果要调试 runtime 初始化，优先看 `runtime_context.cpp`。

如果要调试精度，优先看：

```text
scripts/gen_data.py CPU golden
main.cpp CompareFp16File
RunRoutingImpl
RunDispatchGatherImpl
GMM1 / RunSwigluImpl / GMM2
RunCombineImpl
RunRestoreImpl
```

如果要调试 hang，优先区分：

1. 是否卡在 host HCCL init / remote window context。
2. 是否卡在跨 rank token-ready / barrier。
3. 是否卡在 AIC/AIV cross-core flag。
4. 是否卡在 stream sync 后的 device kernel 内部。

## 17. 当前验证边界

当前仓库所在机器按项目约定不是 A5 runtime 验证环境。本项目可在本机做 A5 compile-only 验证：

```bash
source /home/ntlab/liulei/can/cann-9.0.0-beta.1/set_env.sh
export PATH=/home/ntlab/miniconda3/envs/ltr_pto/bin:$PATH
export LD_LIBRARY_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib:${LD_LIBRARY_PATH:-}
export MPI_LIB_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib/libmpi.so
cmake -S kernels/manual/a5/dispatch_ffn_combine_v3 -B /tmp/dispatch_ffn_combine_v3_a5_build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/dispatch_ffn_combine_v3_a5_build --target dispatch_ffn_combine_v3 -j1
```

A5 端到端正确性需要在 A5-capable 环境运行 `run.sh`，并看到每个 rank 输出 `PASS rank=<id>`。
