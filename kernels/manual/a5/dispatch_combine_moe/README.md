# Dispatch FFN Combine V3 A5 融合算子示例

## 概览

本示例演示一个面向 A5 / Ascend950 形态的 MoE `dispatch -> FFN -> combine` 融合 kernel。它把跨 rank token dispatch、两段 int8 GMM、SwiGLU、概率加权 combine 和最终 restore 放进一个混合 AIC/AIV kernel 中，通过 PTO 通信原语直接访问 HCCL RDMA window。

## 支持的 AI 处理器

- A5 / Ascend950 系列（kernel 编译目标：`dav-c310`）
- 当前仓库所在机器按项目约定不是 A5 runtime 验证环境，本目录默认只做 A5 编译验证；端到端运行请在 A5-capable 环境执行。

## 目录结构

```text
kernels/manual/a5/dispatch_combine_moe/
├── CMakeLists.txt                       # A5 构建配置：kernel shared lib + host exe
├── run.sh                               # 数据生成、构建、MPI 运行一键脚本
├── main.cpp                             # Host 入口：MPI/ACL/HCCL 初始化、launch、计时、精度校验
├── runtime_context.{hpp,cpp}            # standalone HCCL/ACL runtime 与 remote-window context 解析
├── tiling_builder.{hpp,cpp}             # Host tiling 构造、block_dim 计算、HCCL window 容量校验
├── data_utils.{hpp,cpp}                 # case.json 解析、rank 文件路径、FP16 精度比较
├── kernel_launch.hpp                    # Host launcher 参数结构
├── comm_mpi.h                           # dlopen/dlsym 方式加载 MPI，避免硬链接 MPI
├── scripts/gen_data.py                  # CPU golden 数据生成器
├── op_kernel/
│   ├── dispatch_ffn_combine.cpp         # device kernel 入口与 host launch stub
│   ├── dispatch_ffn_combine.h           # op Init/Process、A5 policy 与 kernel params 装配
│   ├── dispatch_ffn_combine_kernel.hpp  # 主 device orchestrator 与各 stage 实现
│   ├── dispatch_ffn_combine_tiling.h    # tiling/runtime/launch config 结构
│   ├── stages/                          # routing/gather/GMM/SwiGLU/combine/restore stage facade
│   ├── utils/                           # PTO bridge、HCCL window、layout、MMAD/epilogue policy
│   ├── moe_init_routing_quant_v2/        # routing/sort/gather/quant 子流程
│   └── unpermute/                       # token restore/unpermute 子流程
├── DESIGN.md                            # 设计背景文档
├── IMPLEMENTATION_PLAN.md               # V4 staged implementation plan
├── api_interface.md                     # 非 PTO 依赖清单
└── out/                                 # 生成的 case.json、rank 输入、expected/output 文件
```

## 算子说明

### 计算功能

每个 rank 持有本 rank 的输入 token、专家权重和 scale。`expert_idx` 决定 token 的 top-k 目标专家，专家按 rank 分片：

```text
global_expert = dst_rank * expert_per_rank + local_expert
```

对每个 active token 和每个 top-k expert，kernel 完成：

$$
Y_i = \sum_{j=0}^{topK-1} prob_{i,j} \cdot FFN_{expert_{i,j}}(X_i)
$$

其中 FFN 当前路径为：

```text
BF16 X
  -> per-token quantize to int8
  -> GMM1: int8 X @ int8 W1
  -> per-token/per-channel dequant
  -> SwiGLU: split N into N/2 + N/2
  -> per-token quantize to int8
  -> GMM2: int8 hidden @ int8 W2
  -> per-token/per-channel dequant
  -> top-k probability weighted combine
  -> restore to original token order
```

### 规格

| 项目 | 值 |
| ---- | ---- |
| OpType | `Dispatch + FFN + Combine` |
| Kernel 名称 | `dispatch_ffn_combine` |
| Kernel 类型 | `KERNEL_TYPE_MIX_AIC_1_2` |
| Tiling key | `1000010` |
| 输入 `x` | `M×K`, BF16 bits (`uint16_t` 文件表示) |
| 输入 `weight1` | `expert_per_rank×K×N`, `int8`, Zn packed |
| 输入 `weight2` | `expert_per_rank×(N/2)×K`, `int8`, Zn packed |
| 输入 `expert_idx` | `M×topK`, `int32`，全局 expert id |
| 输入 `scale1/scale2` | FP32 scale 以 `int64` packing 形式存放 |
| 输入 `probs` | `M×topK`, `float32` |
| 输入 `x_active_mask` | `M`, `uint8` |
| 输出 `out` | `M×K`, `float16` |
| 输出 `expert_token_nums` | `expert_per_rank`, `int32` |

### 形状约束

- `N` 必须是偶数，因为 SwiGLU 会把 GMM1 输出最后一维拆成 `N/2 + N/2`。
- `expert_idx` 使用全局 expert id，合法范围是 `[0, world_size * expert_per_rank)`。
- `max_output_size` 限制单 rank 聚合后的 routed token workspace 容量；过小会截断 CPU golden 和 device 路径中的 routed token。
- `PTO_HCCL_MAX_RANKS` 当前为 64，A5 HCCL context 解析按该上限保存 `windowIn/windowOut`。

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
| `totalUbSize` | 196352 |
| `SYSTEM_NEED_WORKSPACE` | 16 MiB |
| `aivNum` | 来自 `PlatformAscendCManager::GetCoreNumAiv()` |
| `block_dim` | `CalcTschBlockDim(aivNum, aicNum, aivNum)` |

## 整体架构

```text
                         Host / MPI / HCCL
┌─────────────────────────────────────────────────────────────────────────┐
│ gen_data.py -> case.json + rank*.bin                                    │
│ main.cpp: MPI init -> ACL set device -> HCCL root info broadcast         │
│ runtime_context.cpp: HcclAllocComResourceByTiling -> remote window ctx   │
│ tiling_builder.cpp: BuildDispatchFFNCombineTiling -> launch args         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ launchDispatchFFNCombine
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ dispatch_ffn_combine mixed AIC/AIV kernel                               │
│                                                                         │
│ AIC path:                                                               │
│   GMM1 -> CrossCoreWaitFlag interlock -> GMM2                            │
│                                                                         │
│ AIV path:                                                               │
│   routing -> dispatch_gather -> swiglu -> combine -> restore             │
│                                                                         │
│ Cross-rank data path:                                                   │
│   PTO TNOTIFY/TWAIT + HCCL RDMA window payload/signal regions            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Host 流程

1. `run.sh` 生成 `out/case.json` 与每个 rank 的输入/golden 文件。
2. CMake 构建 `dispatch_combine_moe_kernel` 和 host 可执行文件 `dispatch_combine_moe`。
3. Host 进程由 MPI 启动，每个 rank 使用同编号 NPU 设备。
4. Rank 0 生成 `HcclRootInfo` 并广播给所有 rank。
5. `InitStandaloneRankRuntime()` 创建 compute stream、HCCL stream 和 HCCL comm，并通过 `HcclAllocComResourceByTiling()` 分配通信资源。
6. Runtime 优先按 A5 direct context 解析 `HcclDeviceContextA5`，失败时回退到 ring 参数解析。
7. `BuildDispatchFFNCombineTiling()` 校验 HCCL window 容量，填充 tiling、workspace bytes 与 block dim。
8. warmup 和 measure 每轮都会清零 HCCL windows、`out`、`expert_token_nums` 和 workspace。
9. measure 使用 ACL event 统计 kernel 时间，并在 rank 0 汇总每轮 max-rank 样本。
10. verify 再 launch 一轮，D2H 拷回 `out`，写出 `output_rank*.bin`，与 `expected_out` 做 FP16 比较。

成功时每个 rank 输出 `PASS rank=<id>`；rank 0 额外输出 `[PROFILE] dispatch_combine_moe` 性能摘要。

## Kernel 流程

### 计算链路总览

这个 kernel 不是把 `dispatch`、`GMM`、`combine` 简单串起来，而是按 MegaMoE 风格把 MoE FFN 的数据组织、跨 rank 通信和两段 GMM 放进同一个混合 AIC/AIV kernel 中：

```text
x / expert_idx / probs
  -> routing / expand / quant / count
  -> token count exchange + cumsum address planning
  -> dispatch-gather directly into expert-major GMM input
  -> GMM1
  -> per-token dequant + SwiGLU + quant
  -> GMM2
  -> combine / remote result exchange
  -> unpermute + top-k probability accumulation
  -> out[M, K]
```

关键点是：GMM 需要的不是原始 token 顺序，而是每个 local expert 的 token 连续排列。原始 token 位置和 top-k 权重通过 `expandedRowIdx`、`expert_idx`、`probs` 等 bookkeeping 信息保留到 restore/combine 阶段。

### MegaMoE 式布局优化

传统 `AlltoAllV` 路径常见形态是：通信前本地重排，通信后接收端得到 source-rank-major buffer，再根据 count 把数据密排成 expert-major，最后交给 GMM。这个“通信后密排”会引入额外 GM 搬运和同步边界。

MegaMoE 风格的核心改法不是取消 routing，而是把重排位置前移和地址化：

1. 本 rank 先根据 gating 输出执行 routing / expand / quant / count。
2. 通过跨 rank token count 同步得到 `tokenPerExpert[dst_rank][src_rank][local_expert]` 视图。
3. 对 count 做 cumsum，提前算出每个 source rank 的 token 在目标 expert 连续矩阵中的落点。
4. dispatch 通信时直接写入或读取到 expert-major 的最终位置。
5. 某个 expert 或 expert block 的连续输入一旦 ready，就具备启动对应 GMM tile 的条件。

因此这里的优化本质是：

```text
通信后再密排
  -> 通信地址计算时直接落到 GMM 可消费的 expert-major 布局
```

在原 MC2 设计文档中，这条主线的抓手是 `tokenPerExpert`、`cumsumMM`、`CopyGMToGMPerToken`；在当前 A5 PTO 版本中，对应落在 routing/count、remote-window layout、`dispatch_gather` stage 和 GMM input workspace 的协作上。

### AIC / AIV 分工

`dispatch_ffn_combine` 是 `KERNEL_TYPE_MIX_AIC_1_2` 混合 kernel。主 kernel 构造 `MatmulKernel` 后调用 `kernel(params)`，再由 `g_coreType` 分发到 AIC 或 AIV 特化：

```text
DispatchFFNCombine::Process()
  -> MatmulKernel kernel(params)
  -> kernel(params)
       AIC: operator()<AscendC::AIC>() -> RunAicMain()
       AIV: operator()<AscendC::AIV>() -> RunAivMain()
```

AIC 负责重计算链路，AIV 负责数据组织、通信搬运、量化/反量化 epilogue、combine 和 restore。

### AIC 路径

AIC 侧执行两段 grouped matmul 主链：

```text
RunGmm1Stage()
  -> RunGmmInterlockStage()
  -> RunGmm2Stage()
```

`GMM1` 消费 dispatch-gather 后按 local expert 连续排布的 int8 token 矩阵和本 rank expert 的 `weight1`。`GMM2` 消费 AIV 侧 SwiGLU 后重新量化得到的 hidden token 和 `weight2`。两段 GMM 中间通过 interlock 与 AIV 的 epilogue/activation 阶段对齐，而不是完全串行地等所有 GMM1 完成后再启动后处理。

当前 A5 policy 使用：

- `ArchTag = pto_ext::Arch::AtlasA5`
- `MmadAtlasA5PreloadAsyncFixpipe`
- `EpilogueAtlasA5PerTokenDequantSwigluQuant`
- `EpilogueAtlasA5PerTokenDequant`
- `EpilogueAtlasA5PerTokenDequantV2`

### AIV 路径

AIV 侧贯穿整个融合流水，不只是前处理和后处理：

```text
RunRoutingStage()
  -> RunDispatchGatherStage()
  -> RunSwigluStage()
  -> RunCombineStage()
  -> RunRestoreStage()
```

各 stage 的职责可以按数据流理解：

1. `routing`：读取 `expert_idx`、`x_active_mask` 和 `x`，按 top-k 展开 token，按 expert 分组，生成 `expandedRowIdx`、本地 expert token count，并在量化路径中生成 per-token scale。
2. `dispatch_gather`：拉通本地 token count 与 peer rank 的 count/cumsum 信息，把跨 rank token payload 和 scale 放到 GMM 可消费的位置，减少通信后的二次密排。
3. `swiglu`：等待 GMM1 的分段结果，对 `N` 维结果做 per-token/per-channel dequant，拆成 `N/2 + N/2`，执行 SwiGLU，再重新量化为 GMM2 输入。
4. `combine`：消费 GMM2 输出，根据 token 所属原 rank / original row / top-k slot，把 expert 输出写回本地或 peer window 中的 combine 区域。
5. `restore`：按 `expandedRowIdx` 和 `probs` 把 top-k expert 输出累加回原 token 顺序，生成最终 `out[M, K]`。

`dispatch_gather` 和 combine 相关路径会通过 `PtoRemoteWindow` 访问本 rank 与 peer rank 的 HCCL window，并使用 PTO `TNOTIFY/TWAIT` 做跨 rank ready / wait 同步。

### 计算通信 overlap

MegaMoE 背景里的 overlap 可以拆成两个方向：

- `Dispatch -> GMM`：通信先产出 expert 输入，GMM 消费输入。Dispatch 阶段可以让第 `i` 个 expert 的 `GMM` 与第 `i+1` 个 expert 的跨 rank dispatch/all-to-all 数据搬运并行。
- `GMM -> Combine`：GMM 先产出 expert 输出，通信或 combine 消费输出。Combine 阶段可以让第 `i` 个 expert 的 `GMM` 与第 `i-1` 个 expert 的结果回传/all-to-all 搬运并行。

整体流水可以按下图理解：本地 routing 和最终 unpermute 仍保留，变化是把中间通信输入/输出放到 remote window，并在专家维度上做计算通信错位执行。

```text
时间  ───────────────────────────────────────────────────────────────>

AIV : InitRouting / local reorder / HBM->window
      | dispatch expert0 | dispatch expert1 | dispatch expert2 | ...
AIC :                    |   GMM1 expert0   |   GMM1 expert1   | ...
AIV :                                      | dequant + SwiGLU + quant |
AIC :                                                    | GMM2 expert0 | GMM2 expert1 | ...
AIV :                                         | combine expert-1 | combine expert0 | combine expert1 |
      window->HBM / Unpermute / local inverse reorder
```

更抽象地说，稳态时希望形成：

```text
Dispatch phase:  Comm(i + 1)  ||  GMM(i)
Combine phase:   GMM(i)       ||  Comm(i - 1)
```

这里的 `Comm` 在当前 A5 PTO 版本里不是普通 op 级 `AlltoAllV` 调用，而是通过 `PtoRemoteWindow`、HCCL window payload 和 PTO `TNOTIFY/TWAIT` 显式完成的跨 rank ready/wait 与数据搬运。再叠加通信发起方式，就有远端读和远端写两类取舍。当前 A5 PTO 版本更关注 remote-window 上的显式地址计算和 ready/wait，同步尽量压缩到 rank 内 AIC/AIV 协同和 window signal，而不是把每个小块都做成跨 rank 往返握手。

当前实现中的 overlap 逻辑可以按下面几个实现点展开：

1. **routing / quant / count 前置**：`RunRoutingStage()` 调用 `moe_init_routing_quant_v2`，根据 `expert_idx` 展开 token、生成 `expandedRowIdx`、统计本 rank 的 `localTokenPerExpert`，并把 dispatch payload 与 per-token scale 写入 remote window 的固定区域。后续阶段不再把 gating 输出当作随机索引逐 token 处理，而是消费这批已经按 expert 组织过的数据和计数。
2. **count exchange 与 cumsum 地址规划**：`CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2()` 先把本 rank 的 `localTokenPerExpert` 发布到 peer window，再等待 peer token-ready；随后读取 `tokenPerExpert[dst_rank][src_rank][local_expert]`，计算 `preSumBeforeRank`。`GetCumsumForMMAIV()` 生成 `cumsumMM`，让每个 local expert 的输入行区间在 GMM workspace 中有确定落点。
3. **dispatch-gather 直接落到 GMM 输入布局**：`RunDispatchGatherImpl()` 按 `groupIdx` 遍历 local expert，根据 `cumsumMM` 算 `rowStart`，根据 `preSumBeforeRank` 算 peer 源行 `rowSrc`，再用 `CopyGMToGMPerToken()` 把 peer window 的 token 行拷到 `gmA[rowStart, :]`，同时写 `gmPerTokenScale1[rowStart]`。这一步把“通信后密排”折进了通信地址计算。
4. **GMM1 与 SwiGLU 分段 interlock**：dispatch-gather 每完成一个 expert group 后通过 `CrossCoreSetFlag` 通知 AIC；`GMM1` 按 `groupIdx` 等待这些 flag 后执行对应 expert 的 grouped matmul。当 `groupIdx + 1 == epilogueGranularity` 时，`GMM1` finalize 并发出 `SYNCFLAGC2V`，AIV 的 `RunSwigluImpl()` 先处理 `stageDequantSum1` 对应的前半段，再等待第二次 `SYNCFLAGC2V` 处理 `stageDequantSum2`。SwiGLU 完成每段后用 `SYNCFLAGV2C` 通知 `GMM2`。
5. **GMM2 与 combine 分段消费**：`GMM2` 同样按 expert group 遍历，等待 AIV 的 `SYNCFLAGV2C` 后消费 `gmPermutedToken`。后续 `CombineV1` 按 expert group 等待 AIC flag 后写回 peer window；`CombineV2` 则用 `BlockScheduler` 生成 GMM block/tile 坐标，再把每个 tile 按 `m0 = 16` 行块拆给两个 AIV sub-core 处理。
6. **restore / unpermute 保留语义顺序**：GMM 和 combine 阶段始终以 expert-major 连续布局优先，最终由 restore 根据 `expandedRowIdx`、`probs` 和 top-k slot 把 expert 输出加权累加回原 token 行。这样 GMM 不需要为了原始 token 顺序牺牲连续专家矩阵布局。
7. **soft-progress helper**：原始项目和当前 A5 PTO 目录都保留了 `ptrSoftFlagBase`、`InitArithProgress()`、`UpdateAicFlags()` 这类 helper。它们的逻辑是清零进度区、读取各 EP/AIC flag 并取最小完成进度，再写回 `aicFinishPtr`。当前主链路的推进仍以 `SyncAll`、`CrossCoreSetFlag/WaitFlag`、remote-window ready/wait 为主；这些 helper 记录了可用于汇总进度的实现形态。

因此当前 A5 PTO 目录继承的是原始 `dispatch_ffn_combine` 的专家级流水和 block/tile combine 逻辑：dispatch 阶段按 `groupIdx` 形成 `Comm(i+1) || GMM(i)` 的错位执行条件，combine 阶段按 AIC flag 和 tile 坐标形成 `GMM(i) || Comm(i-1)` 的消费条件。

## HCCL Remote Window 布局

只有跨 rank 可见的数据和信号放在 HCCL RDMA window 中；普通输入、输出、workspace 和 tiling 仍由 `aclrtMalloc` 分配。

每个 rank 的 remote window payload 布局如下：

```text
offsetA                   = 0
offsetPeerPerTokenScale   = AlignUp(windowBytes / 3, 512)
offsetD                   = offsetPeerPerTokenScale + 1 MiB
offsetPeerTokenPerExpert  = windowBytes - 2 MiB
signalBase                = windowBytes - 1 MiB
```

| 区域 | 用途 |
| ---- | ---- |
| `offsetA` 起始区域 | dispatch/gather token payload |
| `offsetPeerPerTokenScale` | peer per-token scale |
| `offsetD` | dispatch output / intermediate payload |
| `offsetPeerTokenPerExpert` | peer token-per-expert 计数 |
| 最后 1 MiB signal region | barrier counter、token-ready counter、PTO notify/wait signal |

Host 侧 `ValidateRemoteWindowCapacity()` 会检查：

- `windowBytes > 3 MiB`
- per-token-scale 区域能容纳 `max_output_size * sizeof(float)`
- dispatch output 区域能容纳 `max_output_size * K * sizeof(int16_t)`
- token-count 区域不会覆盖最后 1 MiB signal region

## 构建与运行

### 一键 smoke 运行

在 A5-capable 环境执行：

```bash
bash kernels/manual/a5/dispatch_combine_moe/run.sh \
  --soc-version Ascend910_950 \
  --world-size 2 \
  --m 16 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 32
```

如果目标环境的 CANN platform config 使用更具体 SoC 名称，可把 `--soc-version` 换成对应值，例如 `Ascend950PR_958b`。如果不传 `--soc-version`，host tiling 使用默认 `PlatformAscendCManager::GetInstance()`。


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
| `MPI_ENV_BIN` | 指定包含 `mpirun` 的目录 | 未设置时按 `MPI_SEARCH_DIRS` 查找 |
| `MPI_ENV_LIB` | 指定 MPI lib 目录 | 设置后补到 `LD_LIBRARY_PATH` 并推导 `MPI_LIB_PATH` |
| `MPI_SEARCH_DIRS` | MPI 搜索目录列表 | 包含 conda `ltr_pto`、常见 MPICH 路径 |
| `MPI_LIB_PATH` | `libmpi.so` 绝对路径 | 由 `run.sh` 根据 MPI 路径推导 |
| `DISPATCH_COMBINE_MOE_BUILD_DIR` | CMake build 目录 | `/tmp/dispatch_combine_moe_a5_run_build` |
| `DISPATCH_COMBINE_MOE_CASE_DIR` | host 读取 case 的目录 | `run.sh` 设置为本目录 `out/`；host 默认 `../out` |
| `DISPATCH_COMBINE_MOE_SOC_VERSION` | host tiling 使用的 SoC version | 由 `--soc-version` 设置 |
| `DISPATCH_COMBINE_MOE_WARMUP_ITERS` | host warmup 次数 | 3 |
| `DISPATCH_COMBINE_MOE_MEASURE_ITERS` | host measure 次数 | 5 |

## 数据文件

`gen_data.py` 会生成以下文件：

| 文件 | 内容 |
| ---- | ---- |
| `case.json` | shape、topk、expert 数、compare 容差、逻辑 workload 统计 |
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

## 常见问题

| 问题 | 原因与解决 |
| ---- | ---------- |
| `Cannot find CANN set_env.sh` | 设置 `ASCEND_CANN_PATH` 到 CANN install 目录或 `set_env.sh`。 |
| `Unsupported A5 SOC_VERSION` | `run.sh` 要求 `--soc-version` 以 `Ascend` 开头；检查是否传入了平台配置中的 SoC 名称。 |
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
- 关键链接库：`runtime`、`ascendcl`、`hcomm`、`tiling_api`、`nnopbase`
- PTO include 路径放在 CANN include 前面，确保使用仓内 PTO 头文件。

## 当前验证状态

当前机器可编译 A5 `dav-c310` 代码，但不作为 A5 runtime validation target。本文档中的运行路径来自代码解析；端到端 PASS/性能需要在 A5-capable 环境重新验证。

最近一次已记录的 compile-only 证据为：

```bash
source /home/ntlab/liulei/can/cann-9.0.0-beta.1/set_env.sh
export PATH=/home/ntlab/miniconda3/envs/ltr_pto/bin:$PATH
export LD_LIBRARY_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib:${LD_LIBRARY_PATH:-}
export MPI_LIB_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib/libmpi.so
cmake -S kernels/manual/a5/dispatch_combine_moe -B /tmp/dispatch_combine_moe_a5_readme_verify -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/dispatch_combine_moe_a5_readme_verify --target dispatch_combine_moe -j1
```

Observed result:

```text
[100%] Built target dispatch_combine_moe
```
