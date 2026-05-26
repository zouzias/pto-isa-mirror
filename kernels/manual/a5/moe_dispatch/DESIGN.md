# moe_dispatch A5 设计文档

## 1. 项目定位

`moe_dispatch` 是面向 A5 / Ascend950 的独立 MoE dispatch-only PTO 项目。它作为
`kernels/manual/a2a3/moe_combine` 的前置阶段雏形存在，只覆盖 token routing 和跨 rank dispatch
写入，不覆盖 FFN/GMM，也不覆盖 combine/restore。

目标数据流：

```text
inputA[M, K] + expertIdx[M, topK]
  -> tokenPerExpert / prefix / expandedRowIdx
  -> packedA 写入 owner rank remote window
  -> dispatch ready signal
```

第一版以 deterministic host fixture 和 CPU golden 校验 dispatch 结果；后续可以和 A5 combine 或
`dispatch_combine_moe` 的融合主线对接。

## 2. 设计依据

本设计只使用仓内现有代码事实作为依据：

- A2/A3 `moe_combine`：`kernels/manual/a2a3/moe_combine`，提供 shape、host fixture、CPU golden、
  HCCL window bootstrap 和 PTO combine stage 的可拆分骨架。
- A5 `dispatch_combine_moe`：`kernels/manual/a5/dispatch_combine_moe`，提供 A5 `PtoRemoteWindowContext`、
  `PtoRemoteWindow`、尾部 signal/control region、A5 resource config 和 compile-only 验收口径。当前新代码的
  fusion layout 中 `offsetA = 0`，`offsetPeerPerTokenScale = AlignUp(windowBytes / 3, 512)`，
  `offsetD = offsetPeerPerTokenScale + 1 MiB`，`offsetPeerTokenPerExpert = windowBytes - 2 MiB`，
  `signalBase = windowBytes - 1 MiB`。
- A2/A3 与 A5 `gemm_ar`：`kernels/manual/a2a3/gemm_ar`、`kernels/manual/a5/gemm_ar`，提供
  A5 HCCL window head guard、ready queue cache-line 隔离和 stronger ordering 的迁移依据。
- `kernels/a3vsa5.md`：记录 A5 HCCL context ABI、remote window layout、queue 可见性、DDR fence
  等平台差异。

## 3. 范围与非目标

保留范围：

- host 侧 MPI/HCCL 初始化、rank/device 绑定、A5 remote-window context 解析。
- deterministic 输入生成：`inputA`、`expertIdx`、可选 `probs`。
- dispatch metadata：`tokenPerExpert`、`prefix`、`expandedRowIdx`。
- dispatch payload：把 `inputA[token, :]` 按 `expertIdx[token, slot]` 写入目标 expert owner rank 的
  `packedA`。
- 跨 rank ready signal：发布 dispatch 完成状态，供后续 combine 或独立校验使用。
- CPU golden：校验 `packedA`、`tokenPerExpert`、`expandedRowIdx`。

非目标：

- 不实现 FFN/GMM。
- 不实现 combine return 或 restore。
- 不引入 HCCL collective AllToAllV。
- 不在 A3 机器上声明 A5 runtime pass。
- 不把所有 count/index/prefix metadata 强行 Tile 化；只有批量 payload 和 SIMD 数据处理走 PTO Tile。

## 4. A5 HCCL Window 设计约束

A5 不能照搬 A2/A3 `moe_combine` 的 `peerWindowOffset = 0` 连续小 layout。原因有两类。

第一，`gemm_ar` 的 A5 版本已经显式保留 window 头部 guard。这里的结论不是
“HCCL API 自动保留 padding”，而是“项目自己的 A5 window layout 必须主动保留 padding 区域”：

```text
WINDOW_GUARD_BYTES = 4096
window + 0                -> reduced_output_head_pad
window + 4096             -> reduced_output
after reduced_output      -> signal_matrix
```

A2/A3 `gemm_ar` 则直接从 window 起始处分配 `reduced_output`。这说明 A5 live payload 不应从
remote window 0 偏移开始，设计上要为 window 起始区预留 guard/padding。该 guard 区域只做占位和清零，
不承载 `packedA`、count、prefix、signal 等业务数据，也不参与 golden 校验。

第二，新版 A5 `dispatch_combine_moe` 已经把 window 尾部作为控制/信号保留区，但它的融合 payload
入口是 `offsetA = 0`：

```text
offsetA                  = 0
offsetPeerPerTokenScale  = AlignUp(windowBytes / 3, 512)
offsetD                  = offsetPeerPerTokenScale + 1 MiB
offsetPeerTokenPerExpert = windowBytes - 2 MiB
signalBase               = windowBytes - 1 MiB
```

这和 A5 `gemm_ar` 的 head guard 经验并不完全一致。`moe_dispatch` 是新建 dispatch-only 项目，不要求和
fusion kernel 的 `offsetA=0` 字节兼容；第一版默认采用 guarded layout，把 A5 头部 padding 显式保留下来：

```text
window + 0
  [headGuard]                 4 KiB, 不承载 live payload
  [packedA]                   half[expandedRows, K], 512B 对齐
  [expandedRowIdx]            int32[M * topK], 可选放 window，512B 对齐
  [dispatchScratch/reserved]  保留给后续 per-token scale 或临时 remote row
  [tokenPerExpert]            从 windowBytes - 2 MiB 开始
  [signal]                    从 windowBytes - 1 MiB 开始
```

host 必须做容量校验：

```text
headGuard + packedABytes + expandedRowIdxBytes <= tokenPerExpertOffset
tokenPerExpertOffset + tokenPerExpertBytes <= signalBase
signalBase + signalBytes <= windowBytes
```

所有 rank 使用相同 layout 和 offset。device 侧只通过语义 accessor 或 `PtoRemoteWindow` 获取子区地址，
不在 stage 主流程散落裸 offset。

如果后续要把 `moe_dispatch` 直接接入现有 `dispatch_combine_moe` 的 fusion consumer，需要新增
`layoutMode=fusion-compatible`，让 `packedA` 对齐到 `offsetA=0`。这个模式必须单独命名，不能偷偷改变
guarded 默认布局。

### Padding 回答

对“hccl window 保留 padding 区域了吗”的设计回答：

- HCCL runtime 返回的是 window base/size，不假设 runtime 会替业务 layout 自动跳过头部 padding。
- A5 `moe_dispatch` 必须在自己的 layout 中显式保留 `WINDOW_HEAD_GUARD_BYTES = 4096`。
- live payload 的第一个可写业务 offset 是 `AlignUp(WINDOW_HEAD_GUARD_BYTES, 512)`，不是 0。
- 尾部 `windowBytes - 2 MiB` 到 `windowBytes - 1 MiB` 保留给 count/control，`windowBytes - 1 MiB`
  到 window 末尾保留给 signal。
- host 初始化时清零 head guard 和 tail control/signal 区；kernel 不读写 head guard。

## 5. AIV 参数化设计

`aivBlocks` 不能写死在 kernel 或 host 默认值里。第一版采用“命令行显式值优先，项目资源表兜底”的设计，
风格对齐新版 `dispatch_combine_moe/op_host/pto_resource_config.hpp`：

```text
run.sh / argv --aiv-blocks N
  -> MoeDispatchArgs.shape.aivBlocks
  -> ValidateArgs / resource clamp
  -> launchConfig.blockDim
  -> kernel tiling/runtime config
  -> device 使用 PTO logical AIV helper 获取 block id/count
```

参数规则：

- `--aiv-blocks N` 或 `-aivBlocks N`：显式指定 dispatch AIV block 数，`N > 0`。
- 未指定或传 0：调用 `ChooseDefaultAivBlocks(socVersion, shape)`。
- 默认值来自本项目资源表，不从 `PlatformAscendCManager` 查询，也不硬编码在 kernel 内。
- `blockDim` 对 AIV-only kernel 等于 `aivBlocks`；后续若改 mixed AIC/AIV，再拆为 `aicBlocks/aivRatio/blockDim`。
- `aivBlocks` 必须进入 tiling/runtime config，kernel 不再靠编译期常量猜默认 block 数。

建议第一版资源表：

```text
Ascend950 / Ascend950DT_* / Ascend950PR_*:
  defaultAicBlocks = CONFIG_MOE_DISPATCH_MIX_AIC_BLOCKS      // default 20
  defaultAivRatio  = CONFIG_MOE_DISPATCH_MIX_AIV_RATIO       // default 2
  defaultAivBlocks = CONFIG_MOE_DISPATCH_AIV_NUM             // default 40
  defaultBlockDim  = CONFIG_MOE_DISPATCH_BLOCK_DIM           // AIV-only default 40
```

A5 默认 40 的依据是当前 `dispatch_combine_moe` 资源表的 mixed 1:2 baseline：
`aicBlocks=20, aivRatio=2, aivNum=40, blockDim=60`。`moe_dispatch` 是 AIV-only，所以 launch
`blockDim` 默认先等于 `aivBlocks`；如果未来改成 mixed kernel，再切到 `blockDim=aicBlocks*(1+aivRatio)`。
性能最优值仍需要在 A5 机器上 profile，因此必须保留命令行覆盖。

当前 A2/A3 `moe_combine` README 写的是 `aivBlocks 0->24`，但代码里的 `ChooseDefaultAivBlocks()` 仍返回
`8`。如果本轮实现继续覆盖 A2/A3 默认值，应把该不一致作为单独补丁修掉；A5 `moe_dispatch` 不复用这个
A2/A3 默认函数。

## 6. 工程结构

第一版目录结构：

```text
kernels/manual/a5/moe_dispatch/
  DESIGN.md
  README.md
  CMakeLists.txt
  run.sh
  common.h
  args.h
  layout.h
  golden.h
  comm_mpi.h
  kernel_launchers.h
  main.cpp
  op_host/
    runtime_context.hpp
    runtime_context.cpp
  op_kernel/
    moe_dispatch_kernel.cpp
    utils/
      const_args.hpp
      hccl_context.hpp
      hccl_window.hpp
```

其中 `op_host/runtime_context.*` 优先从 A5 `dispatch_combine_moe` 精简移植，不从 A2/A3
`moe_combine/hccl_context.h` 直接复制。

## 7. Host Flow

host 流程：

```text
ParseArgs -> ValidateArgs
  -> ComputeWorkspaceLayout / ComputeRemoteWindowLayout
  -> InitRankInfo(MPI)
  -> PrepareHostData(inputA, expertIdx, CPU golden)
  -> BindDevice / CreateStreams
  -> InitA5HcclRuntime(PtoRemoteWindowContext)
  -> ValidateRemoteWindowCapacity
  -> AllocateLocalBuffers
  -> CopyInputsToDevice
  -> ClearWindowAndWorkspace
  -> LaunchMoeDispatch
  -> CopyDispatchOutputs
  -> VerifyAndDump
  -> Cleanup
```

`run.sh` 负责：

- 加载 A5 CANN 9.0 环境。
- 默认 `SOC_VERSION` 使用 Ascend950 系列命名。
- 默认 `aivBlocks` 走资源表，允许 `--aiv-blocks` 显式覆盖，并把值传到 host binary。
- 根据 `packedA`、metadata、head guard、尾部 2 MiB control/signal region 计算 `HCCL_BUFFSIZE`。

## 8. Kernel Stage

第一版 device kernel 是 AIV-only dispatch kernel，kernel entry 使用 A5 vec arch 编译，逻辑 block 数由
host launch 的 `aivBlocks` 决定：

```text
MoeDispatchKernel
  -> InitViews
  -> BuildRouteCounts
  -> PrefixAndExpandedRows
  -> PackRowsToOwner
  -> PublishDispatchReady
```

### BuildRouteCounts

按 token shard 遍历 `expertIdx[token, slot]`，计算目标 expert 和 owner rank。计数 metadata 是控制面数据，
第一版不强行 Tile 化。为了避免多个 AIV 直接抢写同一 expert count，采用两段式设计：

```text
perBlockTokenPerExpert[aivBlocks, expertNum]  // workspace
  -> kernel 每个 AIV 写自己的 per-block count
  -> host 或 second-stage kernel 做 reduce/prefix
```

若第一轮只做 compile-only 和 deterministic 小 shape，可以先用 host-assisted prefix 验证 payload layout；
第二轮再补纯 device reduce/scan。

### PrefixAndExpandedRows

生成每个 route 的 packed row index，写 `expandedRowIdx[token, slot]`。这是 combine/restore 后续消费的
关键 metadata。第一版采用两阶段方案：device 生成 local block counts，host fixture 或 second-stage kernel
完成全局 prefix。若要求纯 device E2E，再补并行 scan。

### PackRowsToOwner

payload 数据必须 PTO 化：

- 本 rank owner：`TLOAD(inputA row)` + `TSTORE(local packedA row)`。
- remote owner：构造 remote `GlobalTensor` view，用 `TPUT` 写到 owner rank 的 `packedA`。
- UB 使用 ping/pong tile，tile width 默认按 A5 Vec 容量和 `K` 对齐选择。

### PublishDispatchReady

A5 signal 发布前必须保证 payload/count 可见：

```text
pipe_barrier(PIPE_ALL)
SYNCALL_SOFT_DCCI / DDR fence wrapper
TNOTIFY(remote signal)
```

消费者通过 `TWAIT(signal >= epoch)` 或 `PtoRemoteWindow::WaitTokenReady()` 等接口等待。

## 9. 数据与 Layout

核心 shape：

```cpp
struct MoeDispatchShape {
    uint32_t ep;
    uint32_t m;
    uint32_t k;
    uint32_t topK;
    uint32_t expertPerRank;
    uint32_t expertNum;
    uint32_t maxOutputSize;
    uint32_t aivBlocks;
    uint32_t tileCols;
    uint32_t metadataPad;
    uint32_t signalValue;
};
```

资源配置：

```cpp
struct MoeDispatchResourceConfig {
    uint32_t defaultAicBlocks;
    uint32_t defaultAivRatio;
    uint32_t defaultAivBlocks;
    uint32_t maxAivBlocks;
    uint32_t blockDim;
};
```

remote window layout：

```cpp
struct MoeDispatchWindowLayout {
    uint64_t headGuard;
    uint64_t packedA;
    uint64_t expandedRowIdx;
    uint64_t reservedScratch;
    uint64_t tokenPerExpert;
    uint64_t signal;
    uint64_t totalVisibleBytes;
};
```

`headGuard` 固定 4096 bytes；`tokenPerExpert` 和 `signal` 默认跟随 A5 tail-region 规则。layout 计算
必须同时提供 host 版本和 device 版本，或把 host 计算结果通过 tiling/runtime config 显式传给 kernel。
信号槽按 16 个 `int32` 为 stride，即 64B cache-line 隔离。

layout 结构需要带 mode，避免 guarded layout 和 fusion-compatible layout 混用：

```cpp
enum class MoeDispatchWindowLayoutMode : uint32_t {
    GuardedDispatchOnly = 0,
    FusionCompatible = 1,
};
```

## 10. A3/A5 差异检查表

实现时必须逐项检查：

- A5 HCCL context 是否使用 A5 ABI 或 `PtoRemoteWindowContext`，不要复用 A3 小 struct 假设。
- token/control metadata 是否避开 signal region。
- 如果使用 guarded 默认布局，payload 是否从 4 KiB head guard 之后开始。
- 如果使用 fusion-compatible 布局，是否明确标注 `offsetA=0` 且只用于对接现有 fusion consumer。
- signal 发布前是否有 A5 ordering fence。
- queue/count/ready metadata 是否 cache-line 隔离，至少信号槽按 64B stride 放置。
- `windowsIn[]` / `windowsOut[]` 是否都解析并校验非 0。
- `aivBlocks` 是否从参数或资源表进入 launch config 和 tiling，而不是固定在源码默认值。
- A5 编译宏是否是 `PTO_NPU_ARCH_A5`，kernel arch 是否是 A5 对应目标。
- 当前 A3 机器只允许 compile-only，不跑 A5 runtime。

## 11. 验收标准

静态验收：

- `moe_dispatch` 下不存在 `peerWindowOffset = 0` 作为 live payload 起点。
- guarded 默认布局下，`WINDOW_GUARD_BYTES` 或等价 head guard 明确存在，live payload 不从 0 开始。
- 若新增 fusion-compatible 布局，必须在 args/config/log 中显式输出 layout mode，不能伪装成 guarded。
- remote window 尾部 2 MiB control region 和 1 MiB signal region 有 host overlap 校验。
- `--aiv-blocks` 能覆盖默认资源表，默认值按 `socVersion` 选择。
- device stage 主流程能看到 `TLOAD`、`TSTORE`、`TPUT`、`TNOTIFY/TWAIT`。
- payload 使用 `GlobalTensor` + `Tile` 表达；纯 scalar metadata 不强行 Tile 化。

编译验收：

```bash
source /home/ntlab/liulei/can/cann-9.0.0-beta.1/set_env.sh
cmake -S kernels/manual/a5/moe_dispatch -B /tmp/moe_dispatch_a5_build -DRUN_MODE=npu -DSOC_VERSION=Ascend950PR_958b
cmake --build /tmp/moe_dispatch_a5_build --target moe_dispatch -j8
```

runtime 验收只在 A5 环境执行：

```bash
bash kernels/manual/a5/moe_dispatch/run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 2
```

期望日志：

```text
rank=<r> dispatch_count_done
rank=<r> dispatch_pack_done
rank=<r> dispatch_ready_done
rank=<r> verify=PASS mismatch_count=0
```

## 12. 设计取舍

候选方案：

1. 直接复制 A2/A3 `moe_combine` window layout。优点是实现快；缺点是 A5 live payload 从 offset 0 开始，
   与 A5 `gemm_ar` 的 head guard 经验冲突，不采用。
2. 复用完整 A5 `dispatch_combine_moe` layout。优点是和融合项目贴近，且当前新代码已有 `offsetA=0`
   的文档和 kernel 实现；缺点是引入 FFN/GMM/per-token-scale 等非目标字段，并放弃 A5 `gemm_ar`
   的 head guard 经验，不作为默认方案。
3. 采用 A5 head guard + tail control/signal 的 dispatch-only layout。优点是保留 A5 关键差异，
   同时保持项目小而可验证，作为第一版默认推荐方案。
4. 增加双 layout mode。优点是默认保留 guarded 安全边界，同时能按需对接现有 fusion consumer；
   缺点是 host/kernel/golden 都要携带 mode。作为扩展点保留，第一轮只实现 guarded 默认模式。

## 13. 分期实现顺序

1. 建立目录、README、CMake、run.sh 和 shared ABI。
2. 实现 args/resource config：`--aiv-blocks` 显式覆盖，默认按 `socVersion` 和资源宏选择。
3. 移植 A5 `PtoRemoteWindowContext` / runtime context，并删掉融合项目无关字段。
4. 实现 guarded layout 计算和 window capacity validation，确保 4 KiB head guard 与尾部 control/signal 不重叠。
5. 实现 CPU golden 与 host fixture。
6. 实现最小 AIV dispatch kernel：先支持本地 owner，再支持 remote TPUT。
7. 补 signal 和 A5 ordering fence。
8. A5 compile-only gate。
9. 等 A5 硬件环境可用后补 runtime gate。
