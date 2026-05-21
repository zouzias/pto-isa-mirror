# dispatch_ffn_combine_v2 → dispatch_ffn_combine_v3 PTO 转换点

## 文档范围

本文只记录从 MC2 `dispatch_ffn_combine_v2` 改造成 PTO `dispatch_ffn_combine_v3` 后的**最终转换点**，不记录阶段执行历史。

- 原始 MC2 工程：`/home/ntlab/zy/code/zhangyuan/vllm-ascend-zy/csrc/mc2/dispatch_ffn_combine_v2`
- 当前 PTO 工程：`kernels/manual/a5/dispatch_ffn_combine_v3`
- 当前目标形态：A5 / Ascend950，kernel 编译目标 `dav-c310`
- 当前验证边界：本机已完成 A5 compile-only；A5 runtime PASS 与性能数据需在 A5-capable 环境闭环。

## 1. 工程与构建系统转换

| MC2 v2 | PTO v3 最终落点 | 转换点 |
| ------ | ---------------- | ------ |
| `csrc/mc2/dispatch_ffn_combine_v2` | `kernels/manual/a5/dispatch_ffn_combine_v3` | 从 vLLM Ascend MC2 子工程迁到 PTO ISA 仓内手写 kernel 示例目录。 |
| `ascendc_library(dispatch_ffn_combine_v2_kernel ...)` | `add_library(dispatch_ffn_combine_v3_kernel SHARED ...)` | 从 AscendC/CATLASS 工程封装改成显式 CMake target。 |
| `CATLASS_ARCH=2201` / A2A3 默认 | `PTO_NPU_ARCH_A5` + `--cce-aicore-arch=dav-c310` | 编译目标切到 A5 / `dav-c310`。 |
| MC2 工程内 include 路径 | `${PTO_ROOT}/include` 前置 | PTO 头文件优先于 CANN 内置路径，确保使用仓内 PTO primitive。 |
| `dispatch_ffn_combine_v2` binary | `dispatch_ffn_combine_v3` binary | host runner、target 名称和环境变量统一切到 v3。 |

最终构建入口：

- `CMakeLists.txt`
- `run.sh`
- `kernel_launch.hpp`
- `main.cpp`

## 2. Host runtime 转换

| MC2 v2 | PTO v3 最终落点 | 转换点 |
| ------ | ---------------- | ------ |
| host 侧直接维护 HCCL window table | `StandaloneHcclContext::remote_window_ctx` | device 侧不再直接吃原始 symmetric window table，而是吃 PTO remote-window context。 |
| `HcclDeviceContext` / ring resource 解析 | `LoadA5RemoteWindowContext()` + ring fallback | A5 direct context 优先，无法解析时回退到 ring 参数解析。 |
| `window_table_dev` | `PtoRemoteWindowContext` device copy | host 把 `rank/rankSize/windowBytes/windowIn/windowOut` 规整成统一结构后拷到 device。 |
| 固定/隐式 SoC | `DISPATCH_FFN_COMBINE_V3_SOC_VERSION` | `run.sh --soc-version` 透传到 host tiling 的 `PlatformAscendCManager`。 |

最终 host flow 保持：MPI 初始化 → ACL 设卡 → HCCL root info 广播 → runtime 初始化 → tiling 构造 → warmup/measure/verify launch → D2H compare。

## 3. Kernel ABI 与 tiling runtime 转换

| MC2 v2 | PTO v3 最终落点 | 转换点 |
| ------ | ---------------- | ------ |
| kernel 入参多用 `GM_ADDR` | `__gm__ uint8_t *` typed ABI | A5 device 入口改成 typed GM pointer，host launch 统一 `static_cast<uint8_t *>`。 |
| `DispatchFFNCombineRuntimeInfo::symmetricPtr` | `DispatchFFNCombineRuntimeInfo::remoteWindowContext` | runtime 信息从 symmetric ptr 切到 PTO remote window context。 |
| `DispatchFFNCombineRuntimeInfo::segmentSize` | `PtoRemoteWindowContext::windowBytes` | window size 跟随 remote-window context，不再单独作为 tiling 字段暴露。 |
| MC2 tiling 结构包含 `Mc2InitTiling` / `Mc2CcTiling` | v3 tiling 保留 `DispatchFFNCombineInfo`、`CoCTiling`、`runtimeInfo`、`launchConfig` | 当前手写 PTO 示例只保留 v3 live path 需要的 tiling payload。 |
| tiling key | `1000010` | kernel launch 分支保持固定 tiling key。 |
| task type | `KERNEL_TYPE_MIX_AIC_1_2` | 保持混合 AIC/AIV kernel 形态。 |

关键文件：

- `op_kernel/dispatch_ffn_combine.cpp`
- `op_kernel/dispatch_ffn_combine.h`
- `op_kernel/dispatch_ffn_combine_tiling.h`
- `tiling_builder.cpp`

## 4. HCCL SHMEM → PTO Remote Window

| MC2 v2 | PTO v3 最终落点 | 转换点 |
| ------ | ---------------- | ------ |
| `op_kernel/utils/hccl_shmem.hpp` | `op_kernel/utils/hccl_window.hpp` | 从手写 `HcclShmem` 抽象改成 `PtoRemoteWindow`。 |
| `symmetricPtr + rank window table` | `PtoRemoteWindowContext::windowIn/windowOut` | remote window 地址由 host 规整为 context，再由 device helper 读取。 |
| GM store + dcci + polling wait | `pto::comm::TNOTIFY/TWAIT` | 跨 rank ready/wait 从手写 GM counter 改为 PTO comm signal。 |
| 手写 remote address 计算 | `PtoRemoteWindow::operator()(offset, rank)` | window base、rank、越界检查集中在 helper 内。 |
| 原 barrier counter 逻辑 | `PtoRemoteWindow::CrossRankSync()` | 仍保留 barrier epoch 语义，但通知/等待走 PTO comm primitive。 |

最终 device-visible context：

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

remote window payload 最终布局：

```text
offsetA                   = 0
offsetPeerPerTokenScale   = AlignUp(windowBytes / 3, 512)
offsetD                   = offsetPeerPerTokenScale + 1 MiB
offsetPeerTokenPerExpert  = windowBytes - 2 MiB
signalBase                = windowBytes - 1 MiB
```

## 5. AscendC direct API → PTO helper 转换

### 5.1 GM view 转换

| MC2 v2 | PTO v3 最终落点 | 转换点 |
| ------ | ---------------- | ------ |
| 文件内反复构造 `AscendC::GlobalTensor` | `op_kernel/utils/pto_global_view.hpp` | GM view 构造集中为 `PtoGlobalNd` 与 `Make*Global*` helper。 |
| 长期持有 GM tensor wrapper | raw `__gm__ *` 成员 + use-site PTO view | 长期状态保存 raw pointer，PTO `GlobalTensor` 只在 `TLOAD/TSTORE/TGET/TPUT` use site 临时构造。 |

关键 helper：

- `PtoGlobalNd<Element>`
- `MakeContiguousGlobalFromPtr(...)`
- `MakeContiguousGlobalFromAddr(...)`

### 5.2 GM/UB vector 与 scalar 转换

| 原 direct / 分散实现 | PTO v3 最终 helper | PTO primitive |
| ------------------- | ------------------ | ------------- |
| `DataCopy` / `DataCopyPad` 的连续或 row/tail copy | `PtoLoadVector` / `PtoStoreVector` | `TLOAD` / `TSTORE` |
| atomic count writeback | `PtoStoreAtomicAddVector` | `TSTORE<AtomicAdd>` |
| `Cast` | `PtoCastVector` | `TCVT` |
| `Duplicate` / fill | `PtoFillVector` | `TEXPANDS` |
| `Add` / `Adds` | `PtoAddVector` / `PtoAddScalarVector` | `TADD` / `TADDS` |
| `Mul` / `Muls` | `PtoMulElementwiseVector` / `PtoMulVector` | `TMUL` / `TMULS` |
| `Div` | `PtoDivVector` | `TDIV` |
| `Abs` | `PtoAbsVector` | `TABS` |
| `Exp` | `PtoExpVector` | `TEXP` |
| `ReduceMax` | `PtoReduceMaxVector` | `TROWMAX` / `TMAX` |
| UB scalar `GetValue/SetValue` seam | `PtoGetValue` / `PtoSetValue` | PTO tile assign + scalar access |

最终落点：`op_kernel/utils/pto_vector_ops.hpp`。

### 5.3 sort/gather 转换

| MC2 v2 | PTO v3 最终落点 | 转换点 |
| ------ | ---------------- | ------ |
| routing sort 中 AscendC copy/sort adapter | `moe_v2_pto_sort.h` | sort/gather 侧引入 PTO sort/gather primitive。 |
| merge sort / gather 临时路径 | `TSORT32` / `TMRGSORT` / `TGATHER` | 排序与 gather 逻辑保留原算法语义，但通过 PTO primitive 显性化。 |

## 6. Stage facade 与主 kernel 编排转换

| MC2 v2 | PTO v3 最终落点 | 转换点 |
| ------ | ---------------- | ------ |
| `dispatch_ffn_combine_kernel.hpp` 单体主流程 | `op_kernel/stages/*.hpp` facade | 把任务流拆成可读 stage facade，主实现仍 header-only。 |
| AIC 直接跑 `GMM1 -> interlock -> GMM2` | `RunAicMain()` | AIC 路径显式命名为 `RunGmm1Stage -> RunGmmInterlockStage -> RunGmm2Stage`。 |
| AIV 直接跑 dispatch/combine 合并逻辑 | `RunAivMain()` | AIV 路径显式命名为 `routing -> dispatch_gather -> swiglu -> combine -> restore`。 |

最终 stage 文件：

- `op_kernel/stages/kernel_context.hpp`
- `op_kernel/stages/routing_stage.hpp`
- `op_kernel/stages/dispatch_gather_stage.hpp`
- `op_kernel/stages/gmm_stage.hpp`
- `op_kernel/stages/swiglu_stage.hpp`
- `op_kernel/stages/combine_stage.hpp`
- `op_kernel/stages/restore_stage.hpp`

真实实现函数仍在 `op_kernel/dispatch_ffn_combine_kernel.hpp` 内，例如：

- `RunRoutingImpl`
- `RunDispatchGatherImpl`
- `RunSwigluImpl`
- `RunCombineImpl`
- `RunRestoreImpl`
- `GMM1`
- `GMM2`

## 7. Routing / quant / restore 最终转换点

| 范围 | PTO v3 最终转换点 |
| ---- | ----------------- |
| routing 输入搬运 | aligned/tail 路径统一走 `PtoLoadVector`。 |
| expanded token 写出 | aligned/tail 路径统一走 `PtoStoreVector`。 |
| dynamic quant | cast、abs、reduce max、scale、除法、int8 cast 走 `pto_vector_ops.hpp` helper。 |
| expert token count | 原 `SetAtomicAdd + DataCopyPad` 改成 `PtoStoreAtomicAddVector`。 |
| full-load quant | 多行 load/store 展开为 row-wise PTO vector load/store。 |
| unpermute / restore | indices/probs/token slice load、weighted accumulation、output store 改为 PTO helper。 |

当前 A5 活树中，`op_kernel/` 下 `DataCopyPad`、`DataCopyExtParams`、`SetAtomicAdd` 无业务残留；这些语义已经被 PTO helper 或更明确的 substrate 边界吸收。

## 8. MMAD / Fixpipe 最终边界

这一层的最终口径不是“完全 PTO 化”，而是：**主计算链路 PTO 化，硬件 substrate 边界显式保留**。

| MC2 v2 | PTO v3 最终落点 | 转换点 |
| ------ | ---------------- | ------ |
| CATLASS/AscendC MMAD helper | `MmadAtlasA5PreloadAsyncFixpipe` | MMAD policy 切到 PTO A5 policy。 |
| `DataCopy` GM→L1 | PTO Mat / explicit offset seam | A/B GM→L1 已从业务 `LocalTensor` slice 调用面收口为 explicit L1 offset + PTO load 语义。 |
| `LoadData` L1→L0 | `TMOV` / substrate bridge | L1→L0A/B 走 PTO move 语义；不能安全替换的底层 substrate 保留在 bridge body。 |
| `Fixpipe` / L0C→GM | `TSTORE` / `TSTORE_FP` + substrate bridge | half 路径走 PTO store，int8/scale/fixpipe 语义继续受 substrate bridge 管控。 |
| soft-flag GM↔L1 | PTO Mat `TLOAD/TSTORE` 或移除死分支 | 过时 GM↔L1 soft-flag bridge 已清理，live path 不再依赖该死分支。 |

仍明确保留的 substrate / coordination shell：

- `kernel_operator.h`
- `__aicore__` / `__gm__` / `GM_ADDR`
- `TPipe` / `TQue` / `TBuf`
- `LocalTensor` / `GlobalTensor` 的分配与生命周期层
- `GetBlockIdx` / `GetBlockNum` / `GetTaskRation`
- `PipeBarrier` / `SetFlag` / `WaitFlag`
- `CrossCoreSetFlag` / `CrossCoreWaitFlag`
- `DataCacheCleanAndInvalid`
- `LoadData` / `Fixpipe` 的不可替代底层语义

## 9. A5 专项转换点

| 范围 | 最终转换点 |
| ---- | ---------- |
| 架构 tag | `ArchTag = pto_ext::Arch::AtlasA5`。 |
| MMAD policy | `pto_ext::Gemm::MmadAtlasA5PreloadAsyncFixpipe`。 |
| epilogue policy | `EpilogueAtlasA5PerTokenDequantSwigluQuant`、`EpilogueAtlasA5PerTokenDequant`、`EpilogueAtlasA5PerTokenDequantV2`。 |
| compile arch | `--cce-aicore-arch=dav-c310`。 |
| vector sync | A5 不支持路径中的 `pipe_barrier(PIPE_V)` / `TSYNC<TROWMAX/TMAX>` 已替换为 A5 可编译形式。 |
| cast | A5 `TCVT` 只在 `__DAV_VEC__` 下发射；`int32_t -> half` 走 `int32_t -> float -> half` 两步路径。 |
| HCCL window layout | per-token-scale、dispatch-output、token-count、signal region 使用 host/device 一致的 padded layout。 |
| namespace | A5 PTO headers 引入同名符号后，`TPipe`、`GlobalTensor` 等歧义引用显式限定。 |

## 10. 最终保留边界

PTO 改造完成后，当前 v3 不是纯 PTO kernel。最终边界如下：

1. **PTO 主链路**：搬运、vector math、sort/gather、remote signal、remote get/put、MMAD 主 primitive 均通过 PTO helper 或 PTO primitive 显性化。
2. **PTO bridge**：AscendC tensor/raw GM/UB offset 到 PTO tile/global view 的转换集中到 `pto_global_view.hpp`、`pto_vector_ops.hpp` 和局部 substrate bridge。
3. **AscendC substrate**：kernel ABI、queue/buffer 生命周期、本地 pipe event、cross-core、cache coherence、部分 fixpipe/load-data 语义继续保留。
4. **host/runtime**：ACL/HCCL/MPI bootstrap 不是 PTO 替换范围，继续保留在 host 层。

## 11. 验证状态

当前已验证：

```text
[100%] Built target dispatch_ffn_combine_v3
```

本机验证命令使用 A5 CANN beta 环境完成 compile-only：

```bash
source /home/ntlab/liulei/can/cann-9.0.0-beta.1/set_env.sh
export PATH=/home/ntlab/miniconda3/envs/ltr_pto/bin:$PATH
export LD_LIBRARY_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib:${LD_LIBRARY_PATH:-}
export MPI_LIB_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib/libmpi.so
cmake -S kernels/manual/a5/dispatch_ffn_combine_v3 -B /tmp/dispatch_ffn_combine_v3_a5_readme_verify -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/dispatch_ffn_combine_v3_a5_readme_verify --target dispatch_ffn_combine_v3 -j1
```

未声明完成：

- 不声明 A5 runtime 已 PASS。
- 不声明 A5 性能数据已闭环。
- A5 端到端验证需要在 A5-capable 环境运行 `run.sh` 并确认所有 rank compare PASS。
