# dispatch_ffn_combine_v3 A5 算子示例

## 概览

本示例演示 A5 / Ascend950PR 平台上的 `dispatch_ffn_combine_v3` 独立直调版本，用于验证 int8 MoE `dispatch + FFN + combine` 融合链路。

该目录是从 A2/A3 `dispatch_ffn_combine_v3` 迁移得到的 A5 独立平台版本，保留 standalone ACL + HCCL + MPI 直调执行方式，并在设备侧展示 PTO 数据搬运、PTO matmul、PTO vector helper 和 PTO 跨 rank 通信 primitive 的组合使用。

当前验证路径包括：

- standalone ACL + HCCL + MPI direct launch
- deterministic non-zero `cpu-golden` 数据生成
- fp16 输出精度对比
- warmup / measure 计时统计
- 基于逻辑 workload 的等效吞吐、TFLOPS、GB/s 指标

不覆盖：

- `bf16`
- `w4_a8`
- 硬件计数器级别的真实带宽或 FLOPS 统计

## 支持的 AI 处理器

- Ascend950PR
- 默认 SoC：`Ascend950PR_958b`
- 备用 SoC：`Ascend910_9599`（需要本机 CANN / OPP / platform_config 支持）

## 目录结构

```text
kernels/manual/a5/dispatch_ffn_combine_v3/
├── CMakeLists.txt              # A5 构建配置：device kernel so + host exe
├── run.sh                      # 一键生成数据、构建、MPI 多 rank 运行
├── README.md                   # 当前说明文档
├── perf.md                     # 历史性能数据和 checkpoint 记录
├── DESIGN.md                   # V4 PTO showcase 设计说明
├── api_interface.md            # 非 PTO 依赖归类台账
├── main.cpp                    # Host 入口：ACL/HCCL/MPI 初始化、数据加载、kernel launch、验证和计时
├── kernel_launch.hpp           # Host 侧 kernel launcher 声明与封装
├── runtime_context.hpp/.cpp    # ACL/HCCL runtime、remote window context、设备资源管理
├── tiling_builder.hpp/.cpp     # Host 侧 tiling、workspace、blockDim、remote window 容量校验
├── data_utils.hpp/.cpp         # case 数据加载、golden 对比、profile 统计辅助
├── comm_mpi.h                  # MPI 动态加载包装
├── scripts/gen_data.py         # 生成 deterministic case 输入和 CPU golden
└── op_kernel/
    ├── dispatch_ffn_combine.cpp          # device kernel 编译入口
    ├── dispatch_ffn_combine.h            # host/device launch ABI 和 policy 绑定
    ├── dispatch_ffn_combine_kernel.hpp   # 主 kernel 编排和业务链路实现
    ├── dispatch_ffn_combine_tiling.h     # tiling 数据结构
    ├── stages/stage_sequence.hpp         # AIC/AIV stage facade 顺序
    ├── utils/                            # PTO helper、HCCL window、matmul/fixpipe substrate
    ├── moe_init_routing_quant_v2/        # routing / dispatch gather / quant routing 子链路
    └── unpermute/                        # restore / unpermute 子链路
```

## 算子说明

### 计算功能

本示例实现多 rank MoE FFN 的 `dispatch + GMM1 + SwiGLU + GMM2 + combine + restore` 融合链路。每个 rank 输入本地 token，根据上游 gating 产生的 `expertIdx[token, topK]` 和 `probs[token, topK]` 把 token 展开到目标 expert，跨 rank 交换 token payload，在 expert 所在 rank 上执行两段 grouped matmul，最后把结果回传到 owner rank 并按原 token 顺序加权恢复。

需要区分两层语义：

- **gating**：决定每个 token 要去哪些 expert，即生成 `expertIdx` 和 `probs`。
- **routing**：根据已经给定的 `expertIdx` / `probs`，把 token 展开、分组、计数、量化，并组织成 dispatch / GMM 可消费的布局。

因此本算子的输入已经包含 gating 结果；kernel 内部负责的是 MegaMoE 风格的 routing、跨 rank dispatch、专家 FFN 计算和 combine/restore。

### MegaMoE 计算图

传统 routed-expert 路径通常拆成多个 op：本地 routing、AlltoAllV、通信后重排、GMM1、activation、GMM2、combine、unpermute。MegaMoE 类融合方向的核心是把这些阶段收敛进一个大算子，并让 AIC 计算与 AIV 数据组织 / 通信尽量形成流水。

本 A5 standalone 版本对应的计算图如下：

```text
输入：
  x[M, K]
  expertIdx[M, topK]
  probs[M, topK]
  w1[expert, K, N]
  w2[expert, N/2, K]
  scale1 / scale2

MegaMoE fused chain:

  ┌────────────────────────────────────────────────────────────────────┐
  │ AIV: routing / dispatch / combine / restore                         │
  │                                                                    │
  │  ApplyXActiveMask                                                   │
  │    -> moe_init_routing_quant_v2                                     │
  │       - token 按 topK 展开                                           │
  │       - 按 expert 分组                                               │
  │       - 生成 expandedRowIdx                                          │
  │       - 统计 localTokenPerExpert                                     │
  │       - 生成 per-token quant scale                                   │
  │    -> CrossRank token count sync                                     │
  │    -> cumsum / pre-rank sum                                          │
  │    -> TGET remote token payload                                      │
  │    -> 直接落到本地 expert-major gmA                                  │
  │                                                                    │
  │                          ┌──────────────────────────────────────┐  │
  │                          │ AIC: expert grouped matmul            │  │
  │                          │                                      │  │
  │                          │  GMM1: x_expert @ w1                 │  │
  │                          │    -> TMATMUL / TMATMUL_ACC          │  │
  │                          └──────────────────────────────────────┘  │
  │                                                                    │
  │    -> dequant + SwiGLU + quant                                      │
  │       - PerTokenDequantSwigluQuant                                  │
  │       - PTO vector arithmetic                                       │
  │                                                                    │
  │                          ┌──────────────────────────────────────┐  │
  │                          │ AIC: expert grouped matmul            │  │
  │                          │                                      │  │
  │                          │  GMM2: swiglu_out @ w2               │  │
  │                          │    -> TMATMUL / TMATMUL_ACC          │  │
  │                          │    -> TSTORE_FP                      │  │
  │                          └──────────────────────────────────────┘  │
  │                                                                    │
  │    -> CombineV1 / CombineV2                                         │
  │       - local writeback or TPUT to owner rank                        │
  │    -> MoeTokenUnpermute                                             │
  │       - 根据 expandedRowIdx 和 probs 加权恢复                         │
  │                                                                    │
  └────────────────────────────────────────────────────────────────────┘

输出：
  out[M, K]
  expertTokenNums[rank/expert]
```

### 关键数据布局

MegaMoE 链路的关键不是“完全不重排”，而是把通信后的二次密排变成通信落点计算的一部分。

传统 AlltoAllV 接收端可能先得到 source-rank-major 布局：

```text
来自 rank0: expertC_tokens, expertD_tokens
来自 rank1: expertC_tokens, expertD_tokens

recv buffer: C0, D0, C1, D1
```

GMM 真正想消费的是 expert-major 连续布局：

```text
expert-major gmA: C0, C1, D0, D1
```

当前实现的抓手是：

1. `moe_init_routing_quant_v2(...)` 在本地完成 token 展开、expert 分组、`expandedRowIdx` 和本地 token count。
2. `CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(...)` 交换各 rank 的 `localTokenPerExpert`。
3. `GetCumsumForMMAIV(...)` 生成每个 local expert 的全局连续落点，也就是 GMM 使用的 `cumsumMM`。
4. dispatch gather 阶段通过 PTO `TGET` 从远端读取 token，并按 cumsum 直接写入本地 `gmA` 的 expert-major 位置。

所以这里的优化本质是：

```text
通信后再密排
  -> count/cumsum 先行
  -> 通信地址直接按 expert-major 计算
  -> remote gather 结果直接成为 GMM 输入
```

### AIC / AIV 流水关系

A5 版本仍采用 mixed AIC/AIV kernel：

| 角色 | Stage | 职责 |
| --- | --- | --- |
| AIV | `RunRoutingStage` | active mask、routing、expand、quant、local token count |
| AIV | `RunDispatchGatherStage` | 跨 rank count 同步、pre-rank sum、remote gather、expert-major 落位 |
| AIC | `RunGmm1Stage` | 按 local expert 执行第一段 grouped matmul |
| AIV | `RunSwigluStage` | GMM1 epilogue、dequant、SwiGLU、quant |
| AIC | `RunGmm2Stage` | 执行第二段 grouped matmul 和 fixpipe store |
| AIV | `RunCombineStage` | 本地 combine 或 PTO `TPUT` 回 owner rank |
| AIV | `RunRestoreStage` | unpermute，按 `expandedRowIdx` / `probs` 恢复原 token 输出 |

`GMM1 -> SwiGLU -> GMM2` 之间不是完全串行的大 barrier。当前实现通过 `epilogueGranularity`、`SYNCFLAGC2V`、`SYNCFLAGV2C` 和 cross-core flag 把 expert 维度切成分段流水：AIC 完成一段 GMM1 后，AIV 可以先做该段 SwiGLU / quant；GMM2 再消费已经 ready 的中间结果。

### 与 MegaMoE 背景方案的对应和边界

当前 A5 `dispatch_ffn_combine_v3` 已经体现的 MegaMoE 关键点：

- **通算融合**：routing、dispatch、GMM1、SwiGLU、GMM2、combine、restore 在一个 fused kernel 主链路内完成。
- **AIC/AIV 分工**：AIC 负责 grouped matmul，AIV 负责 routing、通信搬运、activation/quant epilogue、combine 和 restore。
- **routing quant 前置**：`moe_init_routing_quant_v2` 在 dispatch 前完成 expand、count、per-token scale。
- **count/cumsum 直接落位**：`tokenPerExpert`、`preSumBeforeRank`、`cumsumMM` 决定 remote payload 的目标位置。
- **跨 rank window 通信**：dispatch gather 使用 `TGET`，combine return 使用 `TPUT`，ready / barrier 使用 `TNOTIFY` / `TWAIT` / `TTEST`。

当前仍保留的工程边界：

- 主同步仍以 `SyncAll`、`CrossCoreSetFlag/CrossCoreWaitFlag` 和 remote-window signal 为主，没有把所有依赖都演化成完整软件计分板主控。
- `DataAsFlag` 是背景算法里的通信优化思想；当前代码没有暴露同名模块，不应把它写成当前实现的独立 API。
- W4A8 / BF16 不在本 A5 standalone 主路径内；本目录当前验证的是 int8 + NZ / Zn 权重语义路径。
- `TPipe/TQue/TBuf`、`LocalTensor/GlobalTensor`、`LoadData/Fixpipe` 等仍是 AscendC / CANN substrate，PTO 不替代这层生命周期管理。

### 规格

| 项目 | 值 |
| --- | --- |
| OpType | `dispatch + FFN + combine` |
| 平台目录 | `kernels/manual/a5/dispatch_ffn_combine_v3` |
| 默认 SoC | `Ascend950PR_958b` |
| Kernel 类型 | mixed AIC/AIV kernel，`KERNEL_TYPE_MIX_AIC_1_2` |
| Device arch | `dav-c310` |
| 输入 token | `M × K`, `int16_t` / fp16 语义 |
| 权重 | int8，默认 NZ / Zn 路径 |
| 输出 | `M × K`, `half` |
| 默认小用例 | `M=16, K=128, N=128, topK=2, experts/rank=2, world_size=2` |
| 默认大用例 | `M=4097, K=128, N=128, topK=2, experts/rank=2, world_size=2` |
| 精度阈值 | `atol=1e-3`, `rtol=1e-3` |

## 优化说明

本示例的目标是 A5 迁移和 PTO showcase，而不是重新设计 MoE 算法或做最终性能调优。当前实现保留生产 v3 主链路，并把可展示的 PTO seam 集中化：

- **PTO stage facade**：`op_kernel/stages/stage_sequence.hpp` 集中 AIC/AIV stage 顺序，让 routing → dispatch gather → GMM1 → SwiGLU → GMM2 → combine → restore 的链路在入口上可见。
- **PTO GM view / vector helper**：`pto_global_view.hpp`、`pto_vector_ops.hpp` 统一 GM view、UB tile load/store、vector compute、fill、cast、atomic store 等表达。
- **PTO matmul seam**：`block_mmad_preload_async_fixpipe_quant.hpp` 使用 `TMATMUL / TMATMUL_ACC / TSTORE_FP` 展示 L0/L1/FIX 计算边界。
- **PTO remote communication**：remote gather / combine return 使用 `TGET / TPUT`，rank 间 signal 使用 `TNOTIFY / TWAIT / TTEST`。
- **AscendC substrate 明确归类**：`TPipe/TQue/TBuf`、`LocalTensor/GlobalTensor`、HardEvent、CrossCore、cache coherence、`LoadData/Fixpipe` 等保留为 runtime / coordination / matmul-fixpipe substrate，不做机械替换。

## Tiling 参数

Host 侧 tiling 在 `tiling_builder.cpp` 中生成，核心默认值如下：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `m0` | 128 | CoC / GMM tile M 维 |
| `k0` | 256 | CoC / GMM tile K 维 |
| `n0` | 256 | CoC / GMM tile N 维 |
| `ubMoveNum` | 16384 | UB 搬运元素粒度 |
| `swizzleDirect` | 1 | block swizzle 方向参数 |
| `swizzleOffset` | 7 | block swizzle 偏移参数 |
| `totalUbSize` | 196352 | routing tiling 使用的保守 UB budget |
| `SYSTEM_NEED_WORKSPACE` | 16 MiB | HCCL / system workspace 预留 |
| `blockDim` | platform API 计算 | `PlatformAscendC::CalcTschBlockDim(aivNum, aicNum, aivNum)` |

`max-output-size` 是每个目标 rank 的 routed-token 容量，会影响 workspace sizing、remote window payload 区域以及截断行为。

## 整体架构

```text
┌────────────────────────────────────────────────────────────────────────────────────┐
│ Host                                                                               │
│  gen_data.py -> case.json / golden                                                 │
│  main.cpp -> ACL runtime -> MPI/HCCL init -> tiling_builder -> kernel launch       │
│                                                                                    │
│ Device mixed kernel                                                                │
│                                                                                    │
│  AIV control/data path                                                             │
│  ┌────────────────────┐   ┌─────────────────────┐   ┌──────────────────────────┐ │
│  │ routing/expand     │──▶│ count/cumsum sync   │──▶│ TGET remote gather       │ │
│  │ quant/per-token    │   │ pre-rank sum        │   │ direct expert-major gmA  │ │
│  └────────────────────┘   └─────────────────────┘   └────────────┬─────────────┘ │
│                                                                   │               │
│                                                                   ▼               │
│  AIC compute path                                      ┌───────────────────────┐  │
│                                                       │ GMM1 grouped matmul   │  │
│                                                       │ TMATMUL/TMATMUL_ACC   │  │
│                                                       └───────────┬───────────┘  │
│                                                                   │               │
│                                                                   ▼               │
│  AIV epilogue path                                    ┌───────────────────────┐  │
│                                                       │ dequant + SwiGLU      │  │
│                                                       │ quant/per-token scale │  │
│                                                       └───────────┬───────────┘  │
│                                                                   │               │
│                                                                   ▼               │
│  AIC compute path                                      ┌───────────────────────┐  │
│                                                       │ GMM2 grouped matmul   │  │
│                                                       │ TMATMUL + TSTORE_FP   │  │
│                                                       └───────────┬───────────┘  │
│                                                                   │               │
│                                                                   ▼               │
│  AIV return/restore path                              ┌───────────────────────┐  │
│                                                       │ combine local/TPUT    │  │
│                                                       │ unpermute + probs     │  │
│                                                       └───────────────────────┘  │
│                                                                                    │
│ HCCL remote window                                                                 │
│  payload region: per-token scale / dispatch output / token count                   │
│  signal region: token-ready / barrier counters                                     │
└────────────────────────────────────────────────────────────────────────────────────┘
```

## 计算内核详解

### AIC 路径

AIC 侧由 `RunAicMain()` 串起：

1. `RunGmm1Stage()`：按 expert 分组执行第一段 GMM，主计算 seam 使用 `TMATMUL / TMATMUL_ACC`。
2. `RunGmmInterlockStage()`：处理 GMM1 与 SwiGLU / GMM2 之间的本地同步。
3. `RunGmm2Stage()`：执行第二段 GMM，并通过 fixpipe / `TSTORE_FP` 写回 accumulator 结果。

AIC 路径保留 L1/L0/FIX 执行骨架中的 AscendC substrate。PTO 的展示重点在 tile 形状、matmul primitive 和 fixpipe store seam，而不是替换底层 allocator / queue / HardEvent 生命周期。

### AIV 路径

AIV 侧由 `RunAivMain()` 串起：

1. `RunRoutingStage()`：生成 active mask、expert token count、prefix / cumsum。
2. `RunDispatchGatherStage()`：完成跨 rank token count 同步和 remote gather，核心 remote payload 搬运使用 `TGET`。
3. `RunSwigluStage()`：执行 GMM1 epilogue、SwiGLU、dequant / quant seam。
4. `RunCombineStage()`：将 GMM2 输出本地写回或通过 `TPUT` 返回 owner rank。
5. `RunRestoreStage()`：按原 token 顺序 restore / unpermute，输出最终结果。

## HCCL 窗口与通信

A5 版本 host runtime 会从 HCCL device context 重建 `PtoRemoteWindowContext`，再传入 kernel。窗口信息包括：

- `rankId`
- `rankNum`
- `winSize`
- `windowsIn[rank]`
- `windowsOut[rank]`
- `workSpace / workSpaceSize`

设备侧通过 `op_kernel/utils/hccl_window.hpp` 访问 remote window。

窗口布局由 kernel 侧 `PtoRemoteWindow` 解释，host 侧 `ValidateRemoteWindowCapacity()` 会在 tiling 前检查：

| 区域 | 用途 |
| --- | --- |
| `offsetPeerPerTokenScale` | peer per-token scale payload |
| `offsetD` | dispatch / combine 中间输出 payload |
| `offsetPeerTokenPerExpert` | peer token-per-expert / count payload |
| signal base | token-ready / barrier signal counters |

跨 rank 数据与同步使用 PTO comm primitive：

- `TGET`：remote gather source payload
- `TPUT`：combine return / peer payload 写入
- `TNOTIFY`：发布 remote signal
- `TWAIT`：等待 remote signal
- `TTEST`：非阻塞检查 signal

## 构建与运行

1. 配置 CANN 环境。`ASCEND_CANN_PATH` 可以指向 CANN 安装目录，也可以直接指向 `set_env.sh`：

```bash
export ASCEND_CANN_PATH=/home/ntlab/liulei/can/cann-9.0.0-beta.1
```

2. 运行 small case（2 rank）：

```bash
bash kernels/manual/a5/dispatch_ffn_combine_v3/run.sh \
  --world-size 2 \
  --m 16 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 32
```

3. 运行 large case（2 rank）：

```bash
bash kernels/manual/a5/dispatch_ffn_combine_v3/run.sh \
  --world-size 2 \
  --m 4097 \
  --k 128 \
  --n 128 \
  --topk 2 \
  --experts 2 \
  --max-output-size 8194
```

成功时每个 rank 输出：

```text
PASS rank=0
PASS rank=1
```

### 环境变量说明

| 环境变量 | 用途 | 默认行为 |
| --- | --- | --- |
| `ASCEND_CANN_PATH` | CANN 安装目录或 `set_env.sh` 路径 | 自动 glob `/usr/local/Ascend/cann-*/set_env.sh` 取最新版 |
| `MPI_ENV_BIN` | MPI `bin/` 目录 | 若设置且存在 `mpirun`，优先加入 `PATH` |
| `MPI_ENV_LIB` | MPI `lib/` 目录 | 若设置则加入 `LD_LIBRARY_PATH` 并默认推导 `MPI_LIB_PATH` |
| `MPI_SEARCH_DIRS` | MPI 搜索目录，空格分隔 | 搜索 ltr_pto、`/usr/local/mpich/bin`、`/home/mpich/bin` 等 |
| `MPI_RUNNER` | MPI runner 命令 | 默认 `mpirun` |
| `MPI_LIB_PATH` | `libmpi.so` 路径 | 由 MPI 环境自动推导 |
| `ASCEND_DRIVER_PATH` | Ascend driver 路径 | CMake 默认 `/usr/local/Ascend/driver` |

## 运行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--soc` / `--soc-version` | 自动探测 | 可选覆盖 A5 SoC 版本；默认由 runtime `aclrtGetSocName()` 获取 |
| `--world-size` | `2` | MPI rank 数 |
| `--m` | `16` | token 数 / M 维 |
| `--k` | `128` | 输入 hidden size |
| `--n` | `128` | FFN 中间维度参数 |
| `--topk` | `2` | 每 token 选择 expert 数 |
| `--experts` | `2` | 每 rank expert 数 |
| `--max-output-size` | `32` | 每目标 rank routed-token 容量 |
| `--seed` | `20260515` | 输入数据随机种子 |
| `--atol` | `1e-3` | 绝对误差阈值 |
| `--rtol` | `1e-3` | 相对误差阈值 |
| `--warmup-iters` | `3` | warmup 迭代数 |
| `--measure-iters` | `5` | 计时迭代数 |

## 生成文件

`run.sh` 会在当前目录下生成：

| 路径 | 说明 |
| --- | --- |
| `build/dispatch_ffn_combine_v3` | host 可执行文件 |
| `build/lib/libdispatch_ffn_combine_v3_kernel.so` | device kernel so |
| `out/case.json` | case 元信息和逻辑 workload 指标 |
| `out/rank{rank}_expected_out.bin` | CPU golden 输出 |
| `out/output_rank{rank}.bin` | NPU 输出 |

## Runtime 输出

每个 rank 输出结构化精度摘要：

- `mismatch`
- `nan_or_inf`
- `max_abs_err`
- `max_rel_err`
- `mean_abs_err`
- `rmse`
- `PASS` / `FAIL`

rank0 额外输出 profile 摘要：

- kernel time `avg/min/max/std`
- e2e time `avg/min/max/std`
- input / routed tokens per second
- equivalent TFLOPS
- equivalent GB/s

这些性能指标基于 `case.json` 中的逻辑 workload 字段计算：

- `input_tokens_all_ranks`
- `routed_tokens_all_ranks`
- `remote_routed_tokens_all_ranks`
- `compute_flops_all_ranks`
- `comm_bytes_all_ranks`

它们适合做同 shape、同环境下的趋势比较，不等价于硬件 counter 统计。

## 性能记录

性能数据、历史 checkpoint 和 v2/v3 对比表已从 README 拆分到 `perf.md`。

查看：

```bash
less kernels/manual/a5/dispatch_ffn_combine_v3/perf.md
```

## 常见问题

| 问题 | 原因与解决 |
| --- | --- |
| `Cannot find CANN set_env.sh` | 设置 `ASCEND_CANN_PATH` 为 CANN 安装目录或 `set_env.sh` 完整路径 |
| `Unsupported A5 SOC_VERSION` | 仅在显式传 `--soc` / `--soc-version` 时校验，SoC 字符串需以 `Ascend` 开头，例如 `Ascend950PR_958b` |
| CMake 混用旧 CANN 编译器 | 删除旧 `build/` 或换新 build 目录，避免缓存 CANN8.5 的 `bisheng` 路径后再 source CANN9 头文件 |
| `CheckLogLevel` undefined | CANN9 `tiling_api` 依赖 `unified_dlog`；当前 CMake 已显式链接 `unified_dlog` |
| HCCL 初始化后挂死 | 清理 `/dev/shm/sem.hccl*` 和 System V IPC；`run.sh` 已在运行前执行清理 |
| remote window too small | `tiling_builder.cpp` 会在 host 侧校验窗口容量，检查 `max-output-size`、`K` 和 HCCL window size 是否匹配 |
| 精度对比失败 | 先用 small case 复现，再检查 routing count、remote gather、GMM scale、combine restore 顺序 |
| `--allow-run-as-root` 失败 | 本项目使用 MPICH 风格 runner，不依赖 OpenMPI 专用参数 |

## 构建系统

- **编译器**：bisheng（CANN 内置 clang 15.0.5）
- **Device kernel target**：`dispatch_ffn_combine_v3_kernel`
- **Host executable target**：`dispatch_ffn_combine_v3`
- **Device arch**：`--cce-aicore-arch=dav-c310`
- **关键宏**：`PTO_NPU_ARCH_A5`、`MEMORY_BASE`、`DTYPE_W1=int8_t`、`DTYPE_OUT=half`
- **Host 链接库**：`runtime`、`ascendcl`、`hcomm`、`tiling_api`、`unified_dlog`、`register`、`platform`、`c_sec`、`nnopbase`、`pthread`
- **PTO include**：`${PTO_ROOT}/include` 放在 include 列表前部，优先使用仓内 PTO ISA 头文件

## 变更记录

| 日期 | 变更 |
| --- | --- |
| 2026-05-19 | 新增 A5 / Ascend950PR 迁移版本，保留 standalone ACL + HCCL + MPI 直调链路 |
| 2026-05-19 | 修复 A5 迁移适配问题：A5 buffer limit、mixed role dispatch、HCCL context、remote window 校验、chunked fallback |
| 2026-05-19 | 修复 CANN9 `CheckLogLevel` 链接依赖，显式链接 `unified_dlog` |
| 2026-05-19 | 同步 A3 第一批纯瘦身结构，集中 stage facade 到 `stage_sequence.hpp` |
| 2026-05-19 | 将 README 中历史性能数据拆分到 `perf.md`，README 刷新为中文项目入口文档 |
