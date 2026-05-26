# combine_tile 设计文档

## 1. 项目定位

`combine_tile` 是从 `dispatch_combine_tile` 拆出来的独立 combine 项目，目标只覆盖 MoE combine
返回和 restore 链路：

```text
expertOutput[local experts, source-rank-major]
  + dispatch 产出的 workspace/window metadata
  + probs[local M, topK]
    -> PTO TPUT 写回 owner rank peerWindow.ptrD
    -> PTO TWAIT 等待所有 peer combine done
    -> expandedRowIdx + probs 加权 restore
    -> outputC[local M, K]
```

当前目录已从 `kernels/manual/a2a3/dispatch_combine_tile` 初始化拷贝，保留源码、脚本、设计/计划文档和
`out/` 校验 dump，未拷贝 `build/` 产物。后续清理只删除 dispatch kernel 和 dispatch-only host 路径。

## 2. 拆分边界

### 保留内容

- HCCL/MPI 多进程初始化、rank/device 绑定、HCCL window 分配和 remote pointer helper。
- `DispatchCombineTileShape`、workspace/window layout 中 combine 必需字段。
- CPU golden 中 combine restore 所需的数据生成、debug dump 和 `CompareOutputs`。
- host 侧 combine e2e 计时、`actual_ptrD_head` dump、`outputC` dump、verify 日志。
- `out/` 下已有二进制校验材料，作为拆分前基线保留。

### 删除内容

- device 侧 `DispatchCombineTileDispatch` kernel 入口和 launcher。
- dispatch-only helper：本地 routing count、pack、expandedRowIdx rebuild、count row publish/wait、
  dispatch gather 等只服务 dispatch kernel 的函数。
- host 侧 `RunDispatch`、`PrepareExpertOutputIdentity` 中作为 dispatch 后处理的强绑定流程。
- `--dispatch-only`、`--dispatch-metadata-only` 等 dispatch 项目专属 gate。
- CMake 中的 combined 项目名、库名和可执行文件名。

## 3. 输入与前置状态

combine 不能凭空运行，必须拿到 dispatch 阶段已经生成的状态。独立工程第一版支持两种来源：

1. `--gen-data 1`：host 用 CPU golden 在本 rank 初始化 combine fixture，填充 workspace/window metadata，
   并生成 identity `expertOutput`。这是单项目自校验路径。
2. `--gen-data 0 --data-dir <dispatch_tile/out>`：从 dispatch 项目产物读取输入和 metadata。第一版先复用当前
   rank 文件命名，后续再补 manifest。

combine kernel 运行前必须满足：

- `workspace.dispatchOffset` 和 `workspace.prevSumBeforeRank` 已有效；
- `peerWindow.peerTokenPerExpert` 和 `peerWindow.expandedRowIdx` 已有效；
- `expertOutput` 已按 `localExpert -> sourceRank -> rows` 布局；
- `probs` 与 `expandedRowIdx` 使用同一份 routing；
- `peerWindow.ptrD` 和 `combineDoneSignal` 已清零；
- 所有 rank 使用一致的 HCCL window layout 和 signal value。

## 4. 工程结构

目标文件保持和原项目相近，降低拆分风险：

```text
combine_tile/
  CMakeLists.txt
  run.sh
  main.cpp
  dispatch_combine_tile_kernel.cpp   # 下一步改名为 combine_tile_kernel.cpp
  kernel_launchers.h
  common.h
  args.h
  layout.h
  golden.h
  hccl_context.h
  comm_mpi.h
  out/
```

第一轮清理后命名调整为：

| 当前符号 | combine 项目目标 |
| --- | --- |
| `project(pto_dispatch_combine_tile)` | `project(pto_combine_tile)` |
| `dispatch_combine_tile_kernel` | `combine_tile_kernel` |
| `dispatch_combine_tile` executable | `combine_tile` |
| `DispatchCombineTileCombine` | `CombineTileKernel` 或保留旧名到二轮统一改名 |
| `LaunchDispatchCombineTileCombine` | `LaunchCombineTile` |

命名重构分两步做：先删掉不需要的 dispatch 代码保证可编译，再统一 namespace/文件名，避免同时大改导致定位困难。

## 5. 日志与校验

combine 项目必须保留独立日志，不再依赖 combined 项目的 total 语义。

必须保留的 stdout 标记：

```text
rank=<r> stage=combine begin
rank=<r> combine_return_done
rank=<r> combine_wait_done peers=<ep>
rank=<r> restore_done
rank=<r> stage=verify begin
rank=<r> verify=PASS mismatch_count=0
combine_e2e min/avg/max
total_e2e min/avg/max
```

必须保留或生成的 `out/` 文件：

```text
rank_<r>_actual_ptrD_head.bin
rank_<r>_outputC.bin
rank_<r>_golden_outputC.bin
rank_<r>_ptrD_head.bin
rank_<r>_probs.bin
rank_<r>_expandedRowIdx.bin
```

`total_e2e` 在本项目中表示 combine 项目 e2e，包括 fixture 准备、combine kernel、verify；`combine_e2e`
只统计 combine kernel launch 到 stream sync，并包含 rank 间 barrier。

## 6. 实施顺序

1. 重命名 CMake target 和 run binary：`combine_tile` / `combine_tile_kernel`。
2. 将 `kernel_launchers.h` 缩减为单一 combine launcher。
3. 在 kernel 文件中删除 dispatch global kernel 和 dispatch-only helper；保留 combine 所需的 remote pointer、
   TPUT、TWAIT、restore helper。
4. 在 host 中改造流程为 `PrepareCombineFixture -> RunCombine -> VerifyAndDump`。
5. `PrepareCombineFixture` 由 CPU golden 填充设备侧 combine 前置状态：
   - `workspace.cumsumPerExpert`
   - `workspace.dispatchOffset`
   - `workspace.prevSumBeforeRank`
   - `workspace.dispatchedA`
   - `peerWindow.peerTokenPerExpert`
   - `peerWindow.expandedRowIdx`
   - `expertOutput`
6. 删除 dispatch-only 参数、run.sh gate 和 `RunDispatch` 调用。
7. 用静态检查确认 combine 目录中不再有 dispatch kernel/launcher/host gate，再用
   `bash run.sh --skip-run 1` 做编译检查。硬件空闲时再用 A3 单任务方式跑最小 2-rank 校验。

## 6.1 删除 dispatch 的设计

### Kernel 删除范围

从 `combine_tile_kernel.cpp` 删除：

- `DispatchCombineTileDispatch` global kernel；
- `LaunchDispatchCombineTileDispatch` launcher；
- dispatch pack/count/prefix/gather helper：
  `ClearDispatchState`、`InitPackCursors`、`PackedExpertOffset`、`PackLocalRowsToWindow`、
  `CountLocalRoutes`、`RebuildExpandedRowIdx`、`BuildBlockPrefixAndLocalCounts`、
  `BuildPackedExpertOffset`、`PublishCountRows`、`WaitCountRows`、`BuildPrefixMetadata`、
  `TGetRowsHalf`、`GatherLocalExpertPayload`。

保留：

- `RemotePtr` / `MakeRemotePeerWindowView`；
- `TPutRowsHalf`；
- `WaitCombinePhase`；
- `ReturnExpertRowsToOwners`；
- `RestoreOutputRows`。

### Host 删除范围

从 `main.cpp` 删除：

- `RunDispatch`；
- `PrepareExpertOutputIdentity`；
- dispatch metadata dump / payload verify / dispatch segment print；
- main loop 中 dispatch-only 早退分支。

新增 `PrepareCombineFixture`，它只做 deterministic fixture 初始化，不承担 dispatch kernel 行为：

```text
ClearDeviceState
PrepareCombineFixture
RunCombine
VerifyAndDump
```

### 参数删除范围

从 `common.h`、`args.h`、`run.sh` 删除：

- `dispatchMetadataOnly`
- `dispatchOnly`
- `--dispatch-metadata-only`
- `--dispatch-only`

`--combine-return-only` 保留，用于只校验 return path 和 `ptrD`。

## 7. 验收标准

- 项目可独立构建出 `combine_tile` 和 `libcombine_tile_kernel.so`。
- kernel 文件中没有 dispatch kernel 入口和 launcher。
- `run.sh` 默认 `--data-dir` 指向本目录 `out/`。
- `--debug 2 --verify 1` 能输出 combine return、restore、verify 和 e2e 日志。
- `out/` 中 combine 校验文件独立保留，不覆盖 `dispacth_tile/out`。
