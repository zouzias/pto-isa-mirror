# Dispatch Combine MoE A5 融合算子示例

## 概览

本目录实现面向 A5 / Ascend950 形态的 MoE `dispatch -> FFN -> combine` 融合 kernel。它把跨 rank token dispatch、两段 int8 grouped matmul、GMM1 后的 SwiGLU、GMM2 后的概率加权 combine、以及最终 token restore 放进一个混合 AIC/AIV kernel 中，通过 PTO tile / vector / comm 原语和 HCCL RDMA window 显式组织计算通信流水。

当前项目的顶层 target、host 可执行文件、运行脚本和输出 profile 名称均为 `dispatch_combine_moe`。当前 device 编译单元和 kernel symbol 仍使用 `dispatch_combine_moe.*` / `dispatch_combine_moe` 这一组源码符号；README 中出现这些名字时仅指当前目录内的实际代码文件或 device symbol。

## 支持的 AI 处理器

- A5 / Ascend950 系列，kernel 编译目标为 `dav-c310`。
- 当前仓库所在机器按项目约定不是 A5 runtime 验证环境，本目录默认可做 A5 compile-only 验证；端到端运行需要在 A5-capable 环境执行。

## 当前目录结构

生成目录如 `build/`、`out/`、`.cache/` 不属于源码分层，下面只列当前手写代码和脚本：

```text
kernels/manual/a5/dispatch_combine_moe/
├── CMakeLists.txt                         # A5 kernel shared lib + standalone host exe 构建入口
├── run.sh                                 # 生成数据、构建、MPI 多 rank 运行的一键脚本
├── main.cpp                               # Host runner：MPI/ACL/HCCL 初始化、launch、计时、校验
├── kernel_launch.hpp                      # host launch 参数结构与 launchDispatchCombineMoe 声明
├── op_host/
│   ├── comm_mpi.h                         # dlopen/dlsym 加载 MPI，避免 host 二进制硬链接 MPI
│   ├── data_utils.{hpp,cpp}               # case.json、rank 输入输出文件、FP16 compare
│   ├── runtime_context.{hpp,cpp}          # ACL/HCCL runtime、remote-window context 解析
│   └── tiling_builder.{hpp,cpp}           # Host tiling、block_dim、workspace、window 容量校验
├── op_kernel/
│   ├── dispatch_combine_moe.cpp           # device kernel symbol 与 host launch stub
│   ├── dispatch_combine_moe.h             # op Init/Process，A5 policy、layout、params 装配
│   ├── dispatch_combine_moe_kernel.hpp    # AIC/AIV 主 orchestrator 与融合流水实现
│   ├── dispatch_combine_moe_tiling.h      # tiling/runtime/launch config 结构
│   ├── token_reorder/
│   │   ├── routing/                       # routing / sort / expand / quant / expert count
│   │   └── unpermute/                     # top-k 概率加权 restore / unpermute
│   └── utils/
│       ├── block_mmad_preload_async_fixpipe_quant.hpp # AIC MMAD 多级流水
│       ├── block_epilogue_pertoken_swiglu.hpp         # GMM1 后 dequant + SwiGLU + quant
│       ├── block_epilogue_pertoken_row.hpp            # CombineV1 row 级 dequant + 回写/TPUT
│       ├── block_epilogue_pertoken_v2.hpp             # CombineV2 tile 级 dequant + 分 rank 回写/TPUT
│       ├── pto_mmad_ops.hpp               # PTO matmul / GM-L1 / L1-L0 / fixpipe store 封装
│       ├── pto_vector_ops.hpp             # PTO vector TLOAD/TSTORE/TCVT/算术桥接
│       ├── moe_pto_utils.hpp              # shape/layout/arch/resource/sync 公共封装
│       ├── hccl_context.hpp               # device 侧 HCCL context 结构解析
│       ├── hccl_window.hpp                # PtoRemoteWindow 与跨 rank notify/wait
│       ├── layout3d.hpp                   # tokenPerExpert 三维布局辅助
│       ├── const_args.hpp                 # 常量、对齐、flag stride、window 单位
│       └── dispatch_policy_custom.hpp     # A5 MMAD / epilogue policy tag
├── scripts/gen_data.py                    # CPU golden 与 rank 输入文件生成器
├── DESIGN.md                              # 设计说明
├── mc2_2_pto.md                           # PTO 化过程中的本地笔记
├── megamoe理解.md                         # MegaMoE 流水理解笔记
├── pto_tile_programming_report.md         # 当前 PTO tile 化改写报告
├── README.md                              # 英文 README
└── README_zh.md                           # 中文 README
```

## 算子功能

每个 rank 持有本 rank 的输入 token、本地 expert 权重和 scale。`expert_idx` 使用全局 expert id，专家按 rank 分片：

```text
global_expert = dst_rank * expert_per_rank + local_expert
```

对每个 active token 和每个 top-k expert，kernel 计算：

$$
Y_i = \sum_{j=0}^{topK-1} prob_{i,j} \cdot FFN_{expert_{i,j}}(X_i)
$$

当前 FFN 数据路径为：

```text
BF16 X
  -> routing / expand / per-token quantize to int8
  -> GMM1: int8 X @ int8 W1
  -> per-token/per-channel dequant
  -> SwiGLU: split N into N/2 + N/2
  -> per-token quantize to int8
  -> GMM2: int8 hidden @ int8 W2
  -> per-token/per-channel dequant
  -> top-k probability weighted combine
  -> restore to original token order
```

## 规格与约束

| 项目 | 当前值 |
| ---- | ------ |
| Project / host target | `dispatch_combine_moe` |
| Kernel shared target | `dispatch_combine_moe_kernel` |
| Device kernel symbol | `dispatch_combine_moe` |
| Kernel 类型 | `KERNEL_TYPE_MIX_AIC_1_2` |
| Tiling key | `1000010` |
| 输入 `x` | `M×K`, BF16 bits，以 `uint16_t` 文件表示 |
| 输入 `weight1` | `expert_per_rank×K×N`, `int8`, Zn packed |
| 输入 `weight2` | `expert_per_rank×(N/2)×K`, `int8`, Zn packed |
| 输入 `expert_idx` | `M×topK`, `int32`, global expert id |
| 输入 `scale1/scale2` | FP32 scale packed into `int64` view |
| 输入 `probs` | `M×topK`, `float32` |
| 输入 `x_active_mask` | `M`, `uint8` |
| 输出 `out` | `M×K`, `float16` |
| 输出 `expert_token_nums` | `expert_per_rank`, `int32` |

约束：

- `N` 必须是偶数，因为 GMM1 输出会被 SwiGLU 拆成 `N/2 + N/2`。
- `expert_idx` 合法范围为 `[0, world_size * expert_per_rank)`；inactive token 会被 `ApplyXActiveMask()` 改写到哨兵 expert id。
- `max_output_size` 是单 rank routed token workspace 上限；过小会截断 CPU golden 和 device 路径中的 routed token。
- 当前 HCCL context 解析按 `PTO_HCCL_MAX_RANKS` 保存 `windowIn/windowOut`。

## Host 层分工

### Host 启动闭环

```text
run.sh
  -> scripts/gen_data.py 生成 case.json 与 rank*.bin
  -> cmake -S dispatch_combine_moe -B build_dir
  -> cmake --build build_dir --target dispatch_combine_moe
  -> mpirun -n world_size build_dir/dispatch_combine_moe
```

`main.cpp` 负责 standalone 运行闭环：

1. 通过 `comm_mpi.h` 初始化 MPI，并把每个 rank 绑定到同编号 NPU device。
2. Rank 0 生成 `HcclRootInfo`，广播给全部 rank。
3. `InitStandaloneRankRuntime()` 创建 ACL stream、HCCL stream、HCCL comm，并分配 HCCL remote window。
4. `LoadCaseConfig()` 读取 `case.json`，`BuildRankFileSet()` 定位每个 rank 的输入/golden 文件。
5. `BuildDispatchCombineMoeTiling()` 生成 `DispatchCombineMoeTilingData`、workspace bytes 和 block dim。
6. 每轮 warmup / measure 前清理 HCCL window、workspace、`out`、`expert_token_nums`。
7. 通过 `launchDispatchCombineMoe()` launch device kernel。
8. measure 阶段用 ACL event 计时，rank 0 汇总 max-rank latency 并打印 `[PROFILE] dispatch_combine_moe`。
9. verify 阶段 D2H 拷回 `out`，写 `output_rank*.bin`，与 `expected_out` 做 FP16 compare。

### Host tiling builder

`op_host/tiling_builder.cpp` 的顶层函数是 `BuildDispatchCombineMoeTiling()`，主要填四类信息：

1. `DispatchCombineMoeInfo`：`M/K/N/topK/expertPerRank/worldSize/maxOutputSize/listLen/aivNum`。
2. `CoCTiling`：`m0=128`、`k0=256`、`n0=256`、`ubMoveNum=16KiB`、`commNpuSplit=world_size`、routing quant tiling 等。
3. `DispatchCombineMoeRuntimeInfo`：rank id、rank size、device 侧 remote-window context 地址。
4. `DispatchCombineMoeLaunchConfig`：`blockDim`、`tilingKey=1000010`、`workspaceBytes`。

它还会在 host 侧提前校验 HCCL window 布局是否足够容纳 per-token scale、dispatch output 和 token-count 区域，避免 kernel 内访问越界。

## Device 顶层入口

### Kernel symbol 与 Process 装配

`op_kernel/dispatch_combine_moe.cpp` 定义 device kernel symbol：

```text
dispatch_combine_moe(..., workspaceGM, tilingGM)
  -> REGISTER_TILING_DEFAULT(DispatchCombineMoeTilingData)
  -> TILING_KEY_IS(1000010)
  -> KERNEL_TASK_TYPE(1000010, KERNEL_TYPE_MIX_AIC_1_2)
  -> DispatchCombineMoe<int8_t, DTYPE_W1, DTYPE_OUT, false, true>
  -> op.Init(...)
  -> op.Process()
```

`op_kernel/dispatch_combine_moe.h` 的 `DispatchCombineMoe::Process()` 负责把 host tiling 转成真正的 A5 执行策略：

- `ArchTag = pto_ext::Arch::AtlasA5`
- `L1TileShape = GemmShape<128, 256, 512>`
- `L0TileShape = GemmShape<128, 256, 128>`
- `MmadAtlasA5PreloadAsyncFixpipe<preloadStages=1, l1Stages=2, l0AStages=2, l0BStages=2, l0CStages=1, enableShuffleK=true>`
- `BlockEpilogue1 = EpilogueAtlasA5PerTokenDequantSwigluQuant`
- `BlockEpilogue2 = EpilogueAtlasA5PerTokenDequant`
- `BlockEpilogue3 = EpilogueAtlasA5PerTokenDequantV2`
- `BlockScheduler = GemmIdentityBlockSwizzle<9, 1>`

最后构造 `DispatchCombineMoeKernel::Params`，再交给 `DispatchCombineMoeKernel` 按 `g_coreType` 分发到 AIC 或 AIV。

## Kernel 主链分层

`op_kernel/dispatch_combine_moe_kernel.hpp` 是当前 device 侧主 orchestrator。它不是简单调用几个独立 op，而是把 AIV 数据组织、remote-window 通信、AIC grouped matmul、epilogue 和 restore 拉成一个融合流水。

### 顶层 AIC/AIV 分发

```text
DispatchCombineMoeKernel::operator()
├── AIC: RunGmm1Impl()
│        -> RunGmmInterlockImpl()
│        -> RunGmm2Impl()
└── AIV: RunRoutingImpl()
         -> RunDispatchGatherImpl()
         -> RunSwigluImpl()
         -> RunCombineImpl()
         -> RunRestoreImpl()
```

### 主数据流

```text
x / expert_idx / probs / x_active_mask
  -> AIV RunRoutingImpl
       ApplyXActiveMask
       moe_init_routing_quant
       token-count all-gather + preSumBeforeRank + cumsumMM
  -> AIV RunDispatchGatherImpl
       peer window packed rows -> local gmA expert-major input
  -> AIC RunGmm1Impl
       grouped matmul by local expert
  -> AIV RunSwigluImpl
       GMM1 C -> dequant -> SwiGLU -> dynamic quant -> gmPermutedToken
  -> AIC RunGmm2Impl
       grouped matmul by local expert
  -> AIV RunCombineImpl
       GMM2 C2 -> dequant -> local/remote offsetD
  -> AIV RunRestoreImpl
       unpermute + top-k weighted accumulation -> out[M, K]
```

### 从 README 到代码的阅读路线

如果第一次接手这个目录，不建议从 `utils/` 里任意打开文件开始看。按下面顺序读，能把顶层设计和代码实现对齐起来：

1. `op_kernel/dispatch_combine_moe.cpp`：确认 device kernel symbol、tiling key、mixed AIC/AIV task type 和 host launch stub。
2. `op_kernel/dispatch_combine_moe.h`：看 `DispatchCombineMoe::Init()` 如何从 tiling 取 shape/rank/window 信息，再看 `Process()` 如何装配 A5 policy、GMM layout、epilogue policy 和 `DispatchCombineMoeKernel::Params`。
3. `op_kernel/dispatch_combine_moe_kernel.hpp`：先看 `operator()<AIC/AIV>()` 和五个 AIV `Run*Impl()` / 三个 AIC `Run*Impl()`，这是当前算子的主线。
4. `op_kernel/token_reorder/routing/moe_init_routing_quant.cpp`：看 routing 子系统如何根据 tilingKey 选择 sort/count/gather/quant 分支。
5. `op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp`：看 AIC GMM 如何把 A/B/scale 从 GM 送进 L1/L0，并用 `Finalize()` 给 AIV 发同步点。
6. `op_kernel/utils/block_epilogue_pertoken_swiglu.hpp`：看 GMM1 输出如何变成 GMM2 输入。
7. `op_kernel/utils/block_epilogue_pertoken_row.hpp` 和 `op_kernel/utils/block_epilogue_pertoken_v2.hpp`：看 GMM2 输出如何按 row 或 tile 粒度写回各 source rank。
8. `op_kernel/token_reorder/unpermute/moe_token_unpermute.h`：看最终如何用 `expandedRowIdx` 和 `probs` 累加回 `out[M, K]`。

### 核心变量字典

| 变量 / 区域 | 语义 | 主要生产者 | 主要消费者 |
| ----------- | ---- | ---------- | ---------- |
| `problemShape = [M, N, K]` | `M` 是本 rank token 数，`N` 是 GMM1 输出/SwiGLU 前维度，`K` 是输入和最终输出 hidden size | host tiling | GMM1/GMM2/epilogue |
| `EP` | world size / expert parallel rank 数 | host tiling | count exchange、dispatch、combine |
| `expertPerRank` | 每 rank 本地 expert 数 | host tiling | groupIdx 循环、tokenPerExpert layout |
| `topK` | 每 token 路由 expert 数 | host tiling / input | routing、restore |
| `expandedRowIdx` | expanded/top-k 行到原 token/top-k 槽位的映射，restore 依赖它恢复语义顺序 | routing | unpermute/restore |
| `tokenPerExpert[dst][src][expert]` | 每个目标 rank、源 rank、本地 expert 的 routed token 数 | routing count exchange | dispatch-gather、combine |
| `preSumBeforeRank[dst][expert]` | 当前 source rank 在某个 dst/expert 段内的起始偏移 | count exchange | dispatch-gather、CombineV2 |
| `cumsumMM` | 沿 source rank 累加后的 expert-major GMM 行边界 | `GetCumsumForMMAIV()` | AIC GMM、dispatch-gather、combine |
| `gmA` | dispatch-gather 后的本地 expert-major int8 GMM1 输入 | `RunDispatchGatherImpl()` | `GMM1()` |
| `gmPerTokenScale1` | GMM1 输入的 per-token scale | routing / dispatch-gather | SwiGLU epilogue |
| `gmC` | GMM1 输出 workspace | `GMM1()` | `RunSwigluImpl()` |
| `gmPermutedToken` | SwiGLU 后重新 quant 的 int8 GMM2 输入 | `RunSwigluImpl()` | `GMM2()` |
| `gmPerTokenScale2` | GMM2 输入的 per-token scale | `RunSwigluImpl()` | combine epilogue |
| `gmC2` | GMM2 输出 workspace | `GMM2()` | `CombineV1()` / `CombineV2()` |
| remote `offsetD` | combine 后按 source rank 回传的 expanded output | combine epilogue | unpermute/restore |

### Workspace 分层

`WorkspaceInfo` 在 `ptrWorkspace` 内按顺序切出以下主要区域：

| 区域 | 用途 |
| ---- | ---- |
| `expandedRowIdx` | routing 后的 expanded row index，restore 阶段按它回到原 token |
| `ptrcumsumMM` | `tokenPerExpert` 沿 source rank 累加后的 GMM 行边界 |
| `ptrPerTokenScale` | dispatch/routing quant 后的 GMM1 per-token scale |
| `ptrPerTokenScale2` | SwiGLU 后重新 quant 得到的 GMM2 per-token scale |
| `ptrC` | GMM1 FP16/fixpipe 输出 workspace |
| `ptrC2` | GMM2 FP16/fixpipe 输出 workspace |
| `ptrA` | dispatch-gather 后本 rank expert-major GMM1 int8 输入 |
| `ptrPermutedToken` | SwiGLU 后 GMM2 int8 输入 |
| `ptrSumBeforeRank` | combine/restore 需要的 peer rank 前缀和 |
| `ptrSoftFlagBase` | soft-progress helper 使用的进度区 |

跨 rank 可见的数据不放在普通 workspace，而放在 HCCL remote window 中。

## Routing 子系统：`token_reorder/routing`

Routing 子系统的入口是 `token_reorder/routing/moe_init_routing_quant.cpp` 中的 `moe_init_routing_quant()`。它只在 AIV 路径执行；AIC 进入该函数会直接返回。

### Routing 输入输出

输入：

- `x`：原始 BF16 token。
- `expertIdx`：`M×topK` global expert id。
- `scale` / `offset`：dynamic quant 可选平滑参数入口；当前主链传入 scale，offset 保留。
- `workspace`：routing 内部排序、临时索引用 workspace。
- `tilingData` / `tilingKey`：host 侧 `MoeInitRoutingQuantTilingBase::DoTiling()` 生成。

输出：

- `expandedX`：按 routing 顺序展开并 per-token quant 后的 int8 payload，当前主链写到 `remoteWindow + offsetA`。
- `expandedRowIdx`：记录每个 expanded row 对应的原始 token/top-k 位置，当前主链写到 ordinary workspace。
- `expertTokensCountOrCumsum`：本 rank token-per-expert 统计，当前主链写到 remote-window token-count 区域。
- `dynamicQuantScale`：每个 expanded row 的 per-token scale，当前主链写到 `remoteWindow + offsetPeerPerTokenScale`。

### Routing tilingKey 分支

```text
moe_init_routing_quant()
├── tilingKey == 21000
│   └── MoeFullLoadDynamicQuant
│       sort + count/cumsum + gather/quant 在 full-load 路径内完成
├── tilingKey == 11000
│   ├── MoeSortOneCore
│   ├── MoeExpertTokenOut
│   ├── MoeSrcToDstOp
│   └── MoeGatherDynamicQuant
└── tilingKey == 11010
    ├── MoeSortMultiCore
    ├── MoeExpertTokenOut
    ├── MoeSrcToDstOp
    └── MoeGatherDynamicQuant
```

### Routing 文件职责

| 文件 | 职责 |
| ---- | ---- |
| `moe_init_routing_quant.cpp` | routing 子系统入口，根据 tilingKey 选择 full-load、one-core sort 或 multi-core sort 路径 |
| `moe_init_routing_quant_tiling.h` | quant routing tiling：生成 `tilingKey`、workspace size、full-load/gather 分支参数 |
| `moe_init_routing_tiling_common.h` | common routing tiling：`AiCoreParams`、`TilingBaseClass`、VBS/VMS/sort-out/src-to-dst/gather 子 tiling |
| `moe_init_routing_sort.h` | 本卡内 routing sort 编排，集中定义 `MoeSortBase`、`MoeSortOneCore`、`MoeSortMultiCore` |
| `moe_packed_sort_merge.h` | packed sort record 的 merge/extract：`MoeMrgsort`、`MoeMrgsortOut`、`PtoMergePackedSortRecords` |
| `moe_init_routing_expert_tokens.h` | `MoeExpertTokenOut` 和 `MoeSrcToDstOp`：生成 expert token count/cumsum、expandedRowIdx、source-to-destination row 映射 |
| `moe_init_routing_fullload_dynamic_quant.h` | `MoeFullLoadDynamicQuant` full-load 路径：排序、count/cumsum、逐行 quant、输出 int8 payload 和 scale |
| `moe_init_routing_gather_dynamic_quant.h` | `MoeGatherDynamicQuant` gather 路径：按 expandedRowIdx 读取原 token，逐行 dynamic quant，写 expandedX 和 scale |
| `moe_pto_sort.h` | PTO UB 内 int32 sort、packed sort、vector helper 和 AscendC sync bridge |
| `moe_common.h` | 常量、对齐、全局 memory init 等公共基础 |

### Routing 的关键数据语义

1. `ApplyXActiveMask()` 在 routing 前把 inactive token 的 expert id 改成哨兵值 `expertNum`，这样后续排序/count 不把它当作真实 expert。
2. 排序按 expert id 把 expanded token 聚到一起，同时保留 `expandedRowIdx`，用于最终 restore。
3. dynamic quant 对每个 token row 计算 abs max，得到 per-token scale，并把 BF16/FP 输入量化到 int8 payload。
4. `expertTokensCountOrCumsum` 先描述本 rank 的本地 expert token 分布，随后主 kernel 会把它 all-gather 到各 rank 可见的 `tokenPerExpert[dst_rank][src_rank][local_expert]` 视图。

## Dispatch-gather 与跨 rank count 同步

`RunRoutingImpl()` 在 routing 结束后做两件事：

1. `CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2()`：
   - 把本 rank 的 local token count 发布到 peer window。
   - 对 peer 执行 `TPUT`，并用 remote-window token-ready signal 做 notify/wait。
   - 读取完整 `tokenPerExpert[dst_rank][src_rank][local_expert]`。
   - 计算每个 dst rank 在当前 source rank 之前的 `preSumBeforeRank`。
2. `GetCumsumForMMAIV()`：
   - 对 `tokenPerExpert` 沿 source rank 做 cumsum。
   - 生成 AIV dispatch-gather 和 AIC GMM 都使用的 `cumsumMM`。

`RunDispatchGatherImpl()` 再按 `groupIdx` 遍历 local expert：

```text
for each local expert groupIdx:
  for each dstEpIdx assigned to this AIV core:
    rows = tokenPerExpert[dstEpIdx][rank][groupIdx]
    rowStart = cumsumMM prefix + previous expert-group sum
    rowSrc = preSumBeforeRank prefix
    TGET peer offsetA packed rows
    strip UB_ALIGN payload metadata
    write gmA[rowStart, :]
    write gmPerTokenScale1[rowStart]
  SyncAll
  CrossCoreSetFlag -> AIC GMM1 can consume this expert group
```

这里的底层抓手是 `CopyGMToGMPerToken()`：它用 `pto::comm::TGET` 从 peer window 拉取 packed token rows 到本 rank scratch，再用 vector load/store 把 token payload 和 per-token scale 拆到本地 GMM workspace。

## AIC GMM 子系统：`block_mmad_preload_async_fixpipe_quant.hpp`

AIC 侧 GMM1/GMM2 都走 `BlockMmad<MmadAtlasA5PreloadAsyncFixpipe<...>>`。它负责把 GM 中的 A/B/scale 分块送入 L1/L0，并通过 PTO matmul / fixpipe 输出到 GM workspace。

### MMAD 分层流水

```text
GM A/B/Scale
  -> TLOAD GM -> L1(Mat)
  -> TMOV L1(Mat) -> L0A(Left) / L0B(Right)
  -> TMATMUL / TMATMUL_ACC
  -> TSTORE_FP L0C/Scale -> GM C
```

对应文件分工：

| 文件 | 职责 |
| ---- | ---- |
| `block_mmad_preload_async_fixpipe_quant.hpp` | 管理 L1/L0A/L0B/L0C 多 stage buffer、k-loop preload、CrossCoreSetFlag finalize |
| `pto_mmad_ops.hpp` | 封装具体 PTO primitive：`PtoLoadNdGmToNzL1` / `PtoLoadNzGmToNzL1` 做 GM→L1，`PtoMoveL1ToL0A` / `PtoMoveL1ToL0B` 做 L1→L0A/L0B，`PtoTileMmad` 做 matmul，`PtoStoreAccTileToGm` / `PtoStoreAccToGm` 做 fixpipe store |
| `dispatch_policy_custom.hpp` | 定义 `MmadAtlasA5PreloadAsyncFixpipe` 和 epilogue policy tag |
| `moe_pto_utils.hpp` | 定义 arch resource、layout helper、tile copy traits、公共 shape/coord 工具 |

### GMM1 与 GMM2 的差异

- `GMM1` 输入是 `RunDispatchGatherImpl()` 生成的 `gmA`，B 是 `weight1`，输出到 `gmC`，scale 是 `scale1`。
- `GMM2` 输入是 `RunSwigluImpl()` 生成的 `gmPermutedToken`，B 是 `weight2`，输出到 `gmC2`，scale 是 `scale2`。
- 两者都按 local expert `groupIdx` 遍历，`BlockScheduler` 把每个 expert 的 `[M, N, K]` 问题切成 block/tile 分给 AIC core。

### AIC/AIV interlock

- `GMM1` 在每个 expert group 前等待 AIV dispatch-gather 的 `CrossCoreSetFlag`。
- `GMM1` 在 `epilogueGranularity` 边界 `Finalize(..., SYNCFLAGC2V)`，通知 AIV 可以开始前半段 SwiGLU。
- `RunGmmInterlockImpl()` 等待 AIV 的 `SYNCFLAGV2C`，避免 GMM2 早于 SwiGLU/quant 消费数据。
- `GMM2` 在分段边界继续等待 `SYNCFLAGV2C`，最后在 CombineV1 场景下发出 combine 可消费的 flag。

## Epilogue 子系统：三个 `block_epilogue_*` 文件的分工

Epilogue 层是当前代码里最容易混淆的部分。三个文件不是重复实现，而是服务不同阶段和不同 combine 粒度。

### `block_epilogue_pertoken_swiglu.hpp`：GMM1 后处理

使用场景：`RunSwigluImpl()`。

输入输出：

```text
gmC: GMM1 输出 half [routed_rows, N]
gmPerTokenScale1: routing quant 产生的 per-token scale
  -> dequant to fp32
  -> split N into N/2 + N/2
  -> x0 * sigmoid(x0) * gate
  -> dynamic quant to int8
gmPermutedToken: GMM2 输入 int8 [routed_rows, N/2]
gmPerTokenScale2: GMM2 per-token scale
```

关键实现点：

- 按 row 处理 GMM1 输出，每个 row 的 `N` 被拆成 `ChunkTileLen = N/2`。
- 先把 `half` C cast 到 FP32，再乘 `gmPerTokenScale1[row]` 完成 per-token dequant。
- SwiGLU 由 PTO vector primitive 组合完成：`TMULS`、`TEXP`、`TADDS`、`TDIV`、`TMUL`。
- 再对 SwiGLU 输出做 abs max / reduce max，得到 `gmPerTokenScale2[row]`，并把结果 cast/quant 成 int8 写入 `gmPermutedToken`。
- `RunSwigluImpl()` 会按 `stageDequantSum1` / `stageDequantSum2` 两段调用它，对齐 GMM1 的分段 finalize。

### `block_epilogue_pertoken_row.hpp`：CombineV1 row 级后处理

使用场景：`RunCombineImpl()` 选择 `CombineV1` 时。

输入输出：

```text
gmC2: GMM2 输出 half [routed_rows, K]
gmPerTokenScale2: SwiGLU quant 产生的 per-token scale
  -> dequant to fp32
  -> cast to output dtype
  -> local store or remote TPUT to peer offsetD
```

关键实现点：

- 以 row 为单位处理 GMM2 输出，每次处理一行 `K`。
- 对每行执行：GM→UB load、half→float cast、乘 per-token scale、float→输出 dtype cast。
- 如果目标 rank 是本 rank，直接 `TSTORE` 到 `offsetD` 对应位置。
- 如果目标 rank 是 peer rank，先写本地 remote-window scratch，再用 `pto::comm::TPUT` 推到 peer 的 `offsetD`。
- 适合 row 级直接回写路径，逻辑简单，但跨 rank 写回粒度较细。

### `block_epilogue_pertoken_v2.hpp`：CombineV2 tile 级后处理

使用场景：`RunCombineImpl()` 选择 `CombineV2` 时。当前 `initBuffer()` 中的选择逻辑是：

```text
isCombineV1 = true
if M * topK <= 4096:
  isCombineV1 = false   # 小规模默认走 CombineV2
```

输入输出：

```text
gmC2: GMM2 输出 half tile
blockCoord / actualBlockShape: BlockScheduler 生成的 GMM tile 坐标
preSrcExpertSum: 当前 expert 在 gmC2 中的起始行
preSumBeforeRank: 当前 source rank 在各 dst rank expert 段内的前缀
  -> tile 内 dequant
  -> 按 tokenPerExpert / preSumBeforeRank 切分给不同 dst rank
  -> local matrix rows store or remote per-row TPUT
```

关键实现点：

- 以 GMM tile/block 为单位处理 `gmC2`，而不是逐完整 row 顺序扫描。
- tile 内按 `m0 = 16` 行块拆给两个 AIV sub-core，形成更细的 AIV 并行粒度。
- 读取 `tokenPerExpert[dst_rank][src_rank][groupIdx]` 和 `preSumBeforeRank[dst_rank][groupIdx]`，判断当前 tile 的哪些 row 属于哪个目标 rank。
- 本地目标直接 `PtoStoreMatrixRows()`；远端目标先写 scratch，再逐 row `TPUT` 到 peer `offsetD`。
- 这个文件的职责是把 GMM tile 坐标、expert 内 row 区间、rank 内/跨 rank 写回三件事合并起来。

## Combine 与 restore 层

### CombineV1 / CombineV2 选择

`RunCombineImpl()` 会构造两个 epilogue：

- `BlockEpilogue2 = EpilogueAtlasA5PerTokenDequant`，对应 `CombineV1()`。
- `BlockEpilogue3 = EpilogueAtlasA5PerTokenDequantV2`，对应 `CombineV2()`。

两条路径的共同目标都是把 `gmC2` 中 expert-major 的 GMM2 输出写回各 source rank 的 remote-window `offsetD`，区别是粒度：

| 路径 | 粒度 | 主要文件 | 适用特点 |
| ---- | ---- | -------- | -------- |
| CombineV1 | expert group + row | `block_epilogue_pertoken_row.hpp` | 逐行逻辑直观，按 group 等待 AIC flag 后写回 |
| CombineV2 | GMM block/tile + sub-core rows | `block_epilogue_pertoken_v2.hpp` | 利用 BlockScheduler 的 tile 坐标，按 rank 区间切 tile，AIV sub-core 并行更细 |

### Restore / unpermute：`token_reorder/unpermute`

`RunRestoreImpl()` 是 AIV 主链最后一段：

```text
SyncAll
ResetTokenPerExpert
remoteWindow.CrossRankSync()
MoeTokenUnpermuteTiling(M * topK, K, topK, ...)
KernelMoeTokenUnpermute<ElementD2, int32_t, float, true>
  input: remoteWindow + offsetD
  index: expandedRowIdx
  prob: probs
  output: out
```

`token_reorder/unpermute/moe_token_unpermute.h` 的职责是把 combine 后仍处于 expanded/top-k 顺序的 token 输出累加回原 token 顺序：

1. 每个 AIV core 负责一段 output token。
2. 每个 output token 内部遍历 `topK` 个 expanded row。
3. 从 `expandedRowIdx` 读取原 token 映射。
4. 从 `remoteWindow + offsetD` 读取 expert 输出切片。
5. 如果 `PROBS=true`，读取 `probs` 并做概率加权。
6. 对 hidden 维按 chunk 处理，避免 UB 超限。
7. 将累加结果写到最终 `out[M, K]`。

`moe_token_unpermute_tiling.h` 只负责把 `M*topK`、`K`、`topK` 和 core 数切成 token/core 维度与 hidden chunk 维度。

## Remote window 布局

只有跨 rank 可见的数据和信号放在 HCCL RDMA window 中；普通输入、输出、workspace 和 tiling 仍由 `aclrtMalloc` 分配。

每个 rank 的 remote-window payload 布局如下：

```text
offsetA                  = 0
offsetPeerPerTokenScale  = AlignUp(windowBytes / 3, 512)
offsetD                  = offsetPeerPerTokenScale + 1 MiB
offsetPeerTokenPerExpert = windowBytes - 2 MiB
signalBase               = windowBytes - 1 MiB
```

| 区域 | 生产者 | 消费者 | 用途 |
| ---- | ------ | ------ | ---- |
| `offsetA` | routing / dynamic quant | dispatch-gather | expanded int8 token payload |
| `offsetPeerPerTokenScale` | routing / dispatch-gather / combine scratch | GMM1 epilogue / TGET/TPUT scratch | per-token scale 与临时 scratch |
| `offsetD` | combine | restore/unpermute | GMM2 dequant 后、按 source rank 回传的 expanded output |
| `offsetPeerTokenPerExpert` | routing count exchange | dispatch-gather / combine | `tokenPerExpert[dst][src][expert]` |
| signal region | `PtoRemoteWindow` | `PtoRemoteWindow` | barrier、token-ready、notify/wait signal |

## 计算通信 overlap

当前流水的核心不是把通信和 GMM 完全串行化，而是在 expert 维度形成错位执行：

```text
Dispatch phase:  Comm(i + 1)  ||  GMM(i)
Combine phase:   GMM(i)       ||  Comm(i - 1)
```

这里的 `Comm` 不是单独调用一个 op 级 AlltoAllV，而是由 remote-window 上的 `TGET/TPUT`、token-ready signal、rank 内 cross-core flag 组合出来。理解 overlap 要抓三条依赖链：

1. GMM1 只依赖“当前 expert group 的输入行已经 dispatch-gather 到 `gmA`”，不需要等所有 expert 都 gather 完。
2. SwiGLU 只依赖“GMM1 的前一段输出已经 finalize”，不需要等所有 GMM1 group 都结束。
3. Combine 只依赖“GMM2 对应 group/tile 已经可见”，不需要等 restore 阶段开始。

分阶段看：

```text
时间  ───────────────────────────────────────────────────────────────>

AIV : routing/count/quant -> token-count exchange -> cumsumMM
AIV : | dispatch gather expert0 | dispatch gather expert1 | dispatch gather expert2 | ...
AIC :                         | GMM1 expert0            | GMM1 expert1            | ...
AIV :                                      | dequant + SwiGLU + quant segment0 |
AIC :                                                         | GMM2 expert0 | GMM2 expert1 | ...
AIV :                                                  | combine tile/row expert-1 | combine expert0 | ...
AIV : restore / unpermute / probability accumulation
```

### Producer / consumer / signal 对照

| 生产者 | 消费者 | 数据 | 同步抓手 | 代码位置 |
| ------ | ------ | ---- | -------- | -------- |
| routing count exchange | dispatch-gather / GMM planning | `tokenPerExpert`, `preSumBeforeRank`, `cumsumMM` | `remoteWindow.NotifyRemoteTokenReady()` / `WaitTokenReady()` + `PtoSyncAll` | `CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2()`、`GetCumsumForMMAIV()` |
| AIV dispatch-gather | AIC GMM1 | `gmA`, `gmPerTokenScale1` | `CrossCoreSetFlag<0x2, PIPE_MTE3>`，AIC 侧 `CrossCoreWaitFlag` | `RunDispatchGatherImpl()`、`GMM1()` |
| AIC GMM1 | AIV SwiGLU | `gmC` | `blockMmad.Finalize(..., SYNCFLAGC2V)`，AIV 侧 `CrossCoreWaitFlag(SYNCFLAGC2V)` | `GMM1()`、`RunSwigluImpl()` |
| AIV SwiGLU | AIC GMM2 | `gmPermutedToken`, `gmPerTokenScale2` | `CrossCoreSetFlag(SYNCFLAGV2C)`，AIC 侧 `RunGmmInterlockImpl()` / `GMM2()` wait | `RunSwigluImpl()`、`RunGmmInterlockImpl()`、`GMM2()` |
| AIC GMM2 | AIV combine | `gmC2` | AIC finalize flag；CombineV1/CombineV2 按 group/tile wait | `GMM2()`、`CombineV1()`、`CombineV2()` |
| AIV combine | AIV restore | remote `offsetD` | `ResetTokenPerExpert()` + `remoteWindow.CrossRankSync()` | `RunCombineImpl()`、`RunRestoreImpl()` |

### `epilogueGranularity` 如何形成两段 SwiGLU/GMM2 overlap

`epilogueGranularity` 把 local experts 分成前后两段：

```text
front groups: [0, epilogueGranularity)
back groups : [epilogueGranularity, expertPerRank)
```

- `RunDispatchGatherImpl()` 统计 `stageDequantSum1` / `stageDequantSum2`，分别表示前后两段需要做 SwiGLU 的 routed rows。
- `GMM1()` 当 `groupIdx + 1 == epilogueGranularity` 时先 `Finalize(..., SYNCFLAGC2V)`，释放 AIV 处理前半段 `gmC`。
- `RunSwigluImpl()` 处理完前半段后发 `SYNCFLAGV2C`，AIC 的 `GMM2()` 可以先消费前半段 `gmPermutedToken`。
- `GMM1()` 完成剩余 group 后再次 `Finalize(..., SYNCFLAGC2V)`，AIV 再处理后半段，并再次发 `SYNCFLAGV2C`。

这就是 README 图里 `GMM1 -> SwiGLU -> GMM2` 不是全局大 barrier 串行，而是按 expert 段拆开的底层逻辑。

## PTO 原语分层

当前代码没有直接把所有逻辑压到裸 AscendC API，而是在几个文件里形成了 PTO bridge 分层：

| 层级 | 文件 | 典型原语 | 说明 |
| ---- | ---- | -------- | ---- |
| Vector bridge | `pto_vector_ops.hpp` | `TLOAD`, `TSTORE`, `TCVT`, `TMOV`, `TEXPANDS`, `TMULS`, `TADD`, `TDIV`, `TREDUCE` | 面向 UB/Vec tile 的连续向量搬运和算术封装 |
| MMAD bridge | `pto_mmad_ops.hpp` | `TLOAD`, `TMOV`, `TMATMUL`, `TMATMUL_ACC`, `TSTORE_FP` | 面向 AIC 的 GM→L1、L1→L0A/L0B、L0C→GM 封装 |
| Comm bridge | `hccl_window.hpp` + `pto::comm` | `TGET`, `TPUT`, notify/wait | remote-window 上的跨 rank 数据搬运和信号同步 |
| Sync wrapper | `moe_pto_utils.hpp` / routing helpers | `SetFlag`, `WaitFlag`, `SyncAll`, `PipeBarrier` wrappers | 当前仍有 AscendC/CCE 同步封装，用于补齐 PTO primitive 间的 pipeline 依赖 |

常见数据搬运语义：

```text
GM -> UB(Vec):       pto::TLOAD(Vec tile, GlobalTensor)
UB(Vec) -> GM:       pto::TSTORE(GlobalTensor, Vec tile)
GM -> L1(Mat):       pto::TLOAD(Mat tile, GlobalTensor)
L1(Mat) -> L0A:      pto::TMOV(TileLeft, Mat tile)
L1(Mat) -> L0B:      pto::TMOV(TileRight, Mat tile)
L0A/L0B -> L0C:      pto::TMATMUL / pto::TMATMUL_ACC
L0C/FIXPIPE -> GM:   pto::TSTORE_FP(...)
Rank A -> Rank B:    pto::comm::TPUT / pto::comm::TGET over remote window
```

## 默认参数与 Tiling

`run.sh` 默认生成一个小形状 smoke case：

| 参数 | 默认值 |
| ---- | ------ |
| `world_size` | 2 |
| `M` | 16 |
| `K` | 128 |
| `N` | 128 |
| `topK` | 2 |
| `expert_per_rank` | 2 |
| `max_output_size` | 32 |
| `seed` | 20260515 |
| `warmup_iters` | 3 |
| `measure_iters` | 5 |

Host tiling 中固定或派生的关键参数：

| 参数 | 值 / 来源 |
| ---- | --------- |
| `m0` | 128 |
| `k0` | 256 |
| `n0` | 256 |
| `swizzleDirect` | 1 |
| `swizzleOffset` | 7 |
| `ubMoveNum` | 16 KiB |
| `commNpuSplit` | `world_size` |
| `commDataSplit` | 1 |
| `lenPerLoop` | `m0 * n0 / 2` |
| `totalUbSize` | 196352 |
| system workspace | 16 MiB |
| `aivNum` | `PlatformAscendCManager::GetCoreNumAiv()` |
| `block_dim` | `CalcTschBlockDim(aivNum, aicNum, aivNum)` |

## 构建与运行

### 一键 smoke 运行

在 A5-capable 环境执行：

```bash
bash kernels/manual/a5/dispatch_combine_moe/run.sh \
  --soc Ascend950PR_958b \
  --world-size 2 \
  --m 16 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 32
```

如果目标环境的 CANN platform config 使用其他 SoC 名称，可把 `--soc` / `--soc-version` 换成对应值。若不传 `--soc`，host tiling 使用默认 `PlatformAscendCManager::GetInstance()`。

### `run.sh` 参数

| 参数 | 说明 | 默认值 |
| ---- | ---- | ------ |
| `--soc`, `--soc-version` | 传给 host tiling 的 SoC version | 空 |
| `--world-size` | MPI rank 数 | 2 |
| `--m` | token 数 | 16 |
| `--k` | 输入 hidden size / 输出 hidden size | 128 |
| `--n` | GMM1 输出维度，SwiGLU 前维度 | 128 |
| `--topk` | 每 token 路由 expert 数 | 2 |
| `--experts` | 每 rank expert 数 | 2 |
| `--max-output-size` | 每 rank routed token workspace 上限 | 32 |
| `--seed` | 数据生成随机种子 | 20260515 |
| `--atol` | FP16 compare 绝对误差 | 1e-3 |
| `--rtol` | FP16 compare 相对误差 | 1e-3 |
| `--warmup-iters` | warmup 次数 | 3 |
| `--measure-iters` | measure 次数 | 5 |

### 环境变量

| 环境变量 | 用途 | 默认行为 |
| -------- | ---- | -------- |
| `ASCEND_CANN_PATH` | CANN install 目录或 `set_env.sh` 路径 | 自动查找 `/usr/local/Ascend/cann-*/set_env.sh` 的最新版 |
| `ASCEND_HOME_PATH` | CMake 查找 CANN include/lib/platform 根目录 | 通常由 `set_env.sh` 设置；未设置时 CMake 报错 |
| `ASCEND_DRIVER_PATH` | driver kernel include 根目录 | 默认 `/usr/local/Ascend/driver` |
| `MPI_ENV_BIN` | 指定包含 `mpirun` 的目录 | 未设置时按 `MPI_SEARCH_DIRS` 查找 |
| `MPI_ENV_LIB` | 指定 MPI lib 目录 | 设置后补到 `LD_LIBRARY_PATH` 并推导 `MPI_LIB_PATH` |
| `MPI_SEARCH_DIRS` | MPI 搜索目录列表 | 包含 conda `ltr_pto`、常见 MPICH 路径 |
| `MPI_LIB_PATH` | `libmpi.so` 绝对路径 | 由 `run.sh` 根据 MPI 路径推导 |
| `MPI_RUNNER` | 可选 MPI runner 覆盖 | 未设置时使用 `mpirun` |
| `DISPATCH_COMBINE_MOE_BUILD_DIR` | CMake build 目录 | `/tmp/dispatch_combine_moe_a5_run_build` |
| `DISPATCH_COMBINE_MOE_CASE_DIR` | host 读取 case 的目录 | `run.sh` 固定导出为本目录 `out/`；直接运行 host 时默认 `../out` |
| `DISPATCH_COMBINE_MOE_SOC_VERSION` | host tiling 使用的 SoC version | 由 `--soc` / `--soc-version` 设置 |
| `DISPATCH_COMBINE_MOE_WARMUP_ITERS` | host warmup 次数 | 3 |
| `DISPATCH_COMBINE_MOE_MEASURE_ITERS` | host measure 次数 | 5 |

## 数据文件

`gen_data.py` 会生成以下文件：

| 文件 | 内容 |
| ---- | ---- |
| `case.json` | shape、topK、expert 数、compare 容差、逻辑 workload 统计 |
| `rank{r}_x.bin` | `M×K` BF16 bits |
| `rank{r}_weight1.bin` | 本 rank experts 的 W1，int8 Zn packed |
| `rank{r}_weight2.bin` | 本 rank experts 的 W2，int8 Zn packed |
| `rank{r}_expert_idx.bin` | `M×topK` global expert id |
| `rank{r}_scale1.bin` | W1 dequant scale，FP32 packed into int64 view |
| `rank{r}_scale2.bin` | W2 dequant scale，FP32 packed into int64 view |
| `rank{r}_probs.bin` | `M×topK` top-k 概率 |
| `rank{r}_x_active_mask.bin` | active token mask |
| `rank{r}_expected_out.bin` | CPU golden FP16 output |
| `output_rank{r}.bin` | host verify 阶段写出的 device output |

CPU golden 的计算顺序与 device 语义对齐：按 dst rank / local expert / src rank / token / topK 遍历 routed token，超过 `max_output_size` 的 routed token 会被截断。

## 常见问题

| 问题 | 原因与解决 |
| ---- | ---------- |
| `Cannot find CANN set_env.sh` | 设置 `ASCEND_CANN_PATH` 到 CANN install 目录或 `set_env.sh`。 |
| `Cannot find ASCEND_HOME_PATH` | 先 source CANN `set_env.sh`，或显式导出 `ASCEND_HOME_PATH=<cann-install>`。 |
| `Unsupported A5 SOC_VERSION` | `run.sh` 要求 `--soc` / `--soc-version` 以 `Ascend` 开头；检查平台配置中的 SoC 名称。 |
| `per-token-scale region is too small` | `max_output_size` 太大或 window 太小，导致 scale 区域超过 `offsetD`。 |
| `dispatch output region is too small` | `max_output_size * K * sizeof(int16_t)` 超过 dispatch output 区域。 |
| `token-count region overlaps signal region` | token-count 区域侵入最后 1 MiB signal region，需要更大的 window 或更小的 `world_size/expert_per_rank`。 |
| 运行时卡在 HCCL/MPI barrier | 检查所有 rank 是否都启动、设备编号是否与 rank 匹配、上次 HCCL 残留是否清理。`run.sh` 会尝试清理 `/dev/shm/sem.hccl*` 和 `ipcrm -a`。 |
| `FAIL rank=<id>` 或 mismatch | 先确认 shape、`N` 偶数、`max_output_size` 足够，再查看 `output_rank*.bin` 与 `expected_out` 的首个 bad index。 |

## 构建系统

- 编译器：`bisheng`
- C++ 标准：C++17
- A5 编译宏：`PTO_NPU_ARCH_A5`
- Kernel target：`dispatch_combine_moe_kernel`
- Host target：`dispatch_combine_moe`
- Kernel 编译目标：`--cce-aicore-arch=dav-c310`
- Kernel 数据类型宏：`DTYPE_W1=int8_t`，`DTYPE_OUT=half`
- Kernel source：`op_kernel/dispatch_combine_moe.cpp`
- Host sources：`main.cpp`、`op_host/runtime_context.cpp`、`op_host/tiling_builder.cpp`、`op_host/data_utils.cpp`
- 关键链接库：`runtime`、`ascendcl`、`hcomm`、`tiling_api`、`nnopbase`
- PTO include 路径放在 CANN include 前面，确保使用仓内 PTO 头文件。

## 当前验证状态

当前机器可编译 A5 `dav-c310` 代码，但不作为 A5 runtime validation target。端到端 PASS/性能需要在 A5-capable 环境重新验证。

README 本身的正确性检查应至少包含：

```bash
git diff --check -- kernels/manual/a5/dispatch_combine_moe/README_zh.md
python3 - <<'PY'
from pathlib import Path
s = Path('kernels/manual/a5/dispatch_combine_moe/README_zh.md').read_text(encoding='utf-8')
required = [
    '从 README 到代码的阅读路线',
    '核心变量字典',
    'Producer / consumer / signal 对照',
    '`epilogueGranularity` 如何形成两段 SwiGLU/GMM2 overlap',
    'token_reorder/routing',
    'token_reorder/unpermute',
    'block_epilogue_pertoken_swiglu.hpp',
    'block_epilogue_pertoken_row.hpp',
    'block_epilogue_pertoken_v2.hpp',
    'block_mmad_preload_async_fixpipe_quant.hpp',
    'RunDispatchGatherImpl',
    'RunRestoreImpl',
]
missing = [marker for marker in required if marker not in s]
raise SystemExit(f'missing README markers: {missing}' if missing else 0)
PY
```

两条命令都应无输出。
