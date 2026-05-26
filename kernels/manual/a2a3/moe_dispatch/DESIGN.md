# moe_dispatch 设计文档

## 1. 项目定位

`moe_dispatch` 是从 `dispacth_tile`/`dispatch_combine_tile` 拆出的独立 dispatch 算子项目。第一阶段目标是先保留现有可运行数据路径，把 combine kernel、combine launcher 和 host combine 流程移出本项目；后续再按 `moe_combine/PHASE2_DESIGN.md` 的 PTO 风格继续整改。

当前算子只覆盖 MoE dispatch：

```text
inputA[local M, K] + expertIdx[local M, topK]
    -> local peerWindow.packedA / peerTokenPerExpert / expandedRowIdx
    -> TPUT count rows + TWAIT count ready
    -> prefix metadata
    -> TGET owner local-expert payload
    -> workspace.dispatchedA[local experts, source-rank-major]
```

## 2. 拆分边界

保留：

- HCCL/MPI 多进程初始化、rank/device 绑定、HCCL window 分配和 remote pointer helper。
- dispatch 需要的 workspace/window metadata：local count、peer count、prefix、dispatch offset、expanded row index、packed payload、ready signal。
- CPU golden 中 routing、metadata、packed payload、dispatched payload 的计算和 dump。
- host 侧 dispatch kernel timing、metadata 校验、payload 校验和 debug segment 日志。

移除：

- device 侧 combine kernel 和 launcher。
- host 侧 expertOutput identity fixture、combine return、restore、outputC verify。
- `--combine-return-only` gate。

Peer window layout 暂时保留 `ptrD`、`combineDoneSignal` 字段，保持与下游 `moe_combine` fixture/window ABI 对齐；这不是本算子的执行路径。

## 3. 输出契约

dispatch kernel 完成后产出：

- `peerWindow.peerTokenPerExpert`：所有 source rank 到所有 global expert 的 count matrix；
- `peerWindow.expandedRowIdx`：本 rank token route 到 packed row 的反查表；
- `peerWindow.packedA`：本 rank token 按 global expert 打包后的 payload；
- `workspace.dispatchOffset`：本 rank local expert 在 `dispatchedA` 中的起始 row；
- `workspace.prevSumBeforeRank`：按 source rank 的 local expert prefix；
- `workspace.dispatchedA`：属于本 rank local experts 的 dispatch 后 payload。

## 4. 工程结构

```text
moe_dispatch/
  CMakeLists.txt
  run.sh
  main.cpp
  moe_dispatch_kernel.cpp
  kernel_launchers.h
  common.h
  args.h
  layout.h
  golden.h
  hccl_context.h
  comm_mpi.h
```

## 5. 验收标准

- 项目可独立构建出 `moe_dispatch` 和 `libmoe_dispatch_kernel.so`。
- kernel 文件中只有 `MoeDispatchKernel` 一个 public device kernel。
- host 默认流程是 `ClearDeviceState -> RunDispatch -> dispatch metadata/payload verify`。
- 默认 `run.sh --skip-run 1` 可完成 build；硬件可用时默认运行输出 `dispatch_metadata_mismatches=0` 和 `dispatch_payload_mismatches=0`。
