# moe_combine A5 设计文档

## 目标

`kernels/manual/a5/moe_combine` 是 `kernels/manual/a2a3/moe_combine` 的 A5 / Ascend950 翻译版本。主体流程、
host fixture、CPU golden、kernel 入口和 combine return 行为都保持 A3 版本一致；只改 A5 必须不同的编译、
资源默认值、HCCL context 和 HCCL window padding。A5 workspace 只保留当前 kernel 实际使用的本地 AIV
soft sync 区。

## 主体保持一致

保留 A3 `moe_combine` 的这些结构：

- 目录内独立项目：`args.h`、`common.h`、`layout.h`、`golden.h`、`comm_mpi.h`、`hccl_context.h`、
  `kernel_launchers.h`、`main.cpp`、`moe_combine_kernel.cpp`、`run.sh`。
- host 主流程：
  `ParseArgs -> ComputeWorkspaceLayout/ComputePeerWindowLayout -> InitRankInfo -> PrepareHostData -> BindDevice -> CreateStreams -> InitHccl -> AllocateLocalBuffers -> CopyInputsToDevice -> PrepareCombineFixture -> RunCombine -> VerifyAndDump`。
- kernel 主体：`LaunchMoeCombineKernel()` 和 `MoeCombineKernel` 的参数顺序、workspace view、peer window view、
  `TPUT/TGET/TNOTIFY/TWAIT` 使用方式与 A3 对齐。
- `--aiv-blocks` 参数继续从 `run.sh/argv` 进入 `shape.aivBlocks`，最终作为 kernel launch block 数。

## A5 差异

### 编译差异

- `CMakeLists.txt` 使用 `PTO_NPU_ARCH_A5`。
- kernel arch 使用 `dav-c310-vec`。
- CANN include root 兼容 CANN 9.0 的 `x86_64-linux` / `aarch64-linux` 路径。
- 默认运行环境是 CANN 9.0 beta：
  `source /home/ntlab/liulei/can/cann-9.0.0-beta.1/set_env.sh`。
- 默认 SOC 是 `Ascend950PR_958b`。

### AIV 默认值

A3 版本脚本默认把 `AIV_BLOCKS=0` 落成 24。A5 版本保留显式传参能力，并把默认值改为资源表兜底：

```text
Ascend950 / Ascend950DT_* / Ascend950PR_*:
  defaultAicBlocks = 20
  defaultAivRatio  = 2
  defaultAivBlocks = 40
  maxAivBlocks     = 128
```

规则：

- 用户传 `--aiv-blocks N` 时直接使用 `N`。
- 用户不传或传 `0` 时，`args.h` 根据 A5 resource config 选择默认值。
- `run.sh` 不再把 0 强制改成 24；它只在估算 workspace/window 时使用同一套 A5 默认值。

### HCCL context 差异

A3 `HcclDeviceContext` 只包含 workspace/rank/window 表。A5 需要兼容 `HcclAllocComResourceByTiling()` 直接返回的
A5 context prefix，结构尾部增加 CCU register 字段：

```cpp
uint64_t xnAddr;
uint64_t ckeAddr;
uint64_t msAddr;
uint64_t msSize;
```

初始化策略：

1. 资源 tiling 仍沿用 A3/`allgather_gemm`/`dispatch_combine_moe` 的手填 `Mc2CommConfigV2` 路径，不引入
   `AscendC::Mc2CcTilingConfig` 作为必须依赖。
2. 先尝试 A5 direct decode：直接从 `ctxPtr` 拷贝 A5 `HcclDeviceContext` prefix。
3. direct decode 失败时，保留 A3 ring fallback：从 `HcclOpResParam.remoteRes` 提取 `windowsIn/windowsOut`，
   再把 host-side context 拷回 device。
4. mesh fallback 只作为兼容路径，不作为默认假设。

### HCCL window padding 差异

A3 的 `peerWindowOffset = 0`，live payload 从 HCCL window 起始地址开始。A5 版本不能这么做，按 A5 `gemm_ar`
经验显式保留头部 guard：

```text
window + 0
  [headGuard]        4096 bytes, only zeroed, no live payload
  [ptrD]
  [countReadySignal]
  [combineDoneSignal]
```

HCCL runtime 返回 window base/size，但业务 padding 由本项目 layout 保留，不假设 HCCL 自动替业务跳过 padding。
host 初始化和每轮 clear 都清零 `[0, peerWindowOffset + peerWindowBytes)`，kernel 接收到的 `peerWindow` 指针指向
`windowBase + peerWindowOffset`，因此 kernel 内 layout 主体仍与 A3 一致。

## 验收

- A5 编译：

```bash
source /home/ntlab/liulei/can/cann-9.0.0-beta.1/set_env.sh
cmake -S kernels/manual/a5/moe_combine -B /tmp/moe_combine_a5_build -DRUN_MODE=npu -DSOC_VERSION=Ascend950PR_958b
cmake --build /tmp/moe_combine_a5_build --target moe_combine -j8
```

- A5 执行入口：

```bash
bash kernels/manual/a5/moe_combine/run.sh -pes 2 -M 64 -K 7168 -topK 8 -expertPerPe 2 --aiv-blocks 24
```

- 当前非 A5 机器只要求编译通过；A5 runtime pass 需要在 Ascend950 环境验证。
