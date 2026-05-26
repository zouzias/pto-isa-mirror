# combine_tile

`combine_tile` 是 A2/A3 手写 PTO MoE combine 项目，覆盖 expert 输出已经生成之后的 combine 计算链路：

```text
expertOutput[local experts, source-rank-major]
  + dispatch metadata
  + probs[local M, topK]
    -> variable-size return 到 token owner rank 的 peerWindow.ptrD
    -> 等待所有 peer return 完成
    -> expandedRowIdx + probs 加权还原
    -> outputC[local M, K]
```

这里的 return 阶段对标 MoE combine 中的 all2allv 语义，但当前实现不是调用 HCCL `AllToAllV`
collective，而是基于 HCCL window remote pointer 和 PTO `TPUT/TWAIT` 手写变长跨 rank 写回。

## 计算范畴

当前项目覆盖 MoE combine 的三个核心过程：

1. **变长 return**
   每个 rank 持有本 rank local experts 的 `expertOutput`。kernel 根据 `peerTokenPerExpert`、
   `dispatchOffset`、`prevSumBeforeRank` 和 `cumsumPerExpert` 计算每个目标 rank、每个 expert 的变长行段，
   将对应行写回目标 rank 的 `peerWindow.ptrD`。

2. **跨 rank 完成同步**
   return 写回完成后，每个 rank 向目标 peer 的 `combineDoneSignal[myRank]` 发布完成信号。
   本 rank 通过 `TWAIT` 等待所有 peer 的 combine done 信号，确保本地 `ptrD` 已完整可读。

3. **按权重 restore**
   本 rank 根据 `expandedRowIdx[token, slot]` 找到每个 token/topK route 对应的 `ptrD` 行，
   再用 `probs[token, slot]` 做加权累加：

   ```text
   outputC[token, :] = sum(slot in topK) probs[token, slot] * ptrD[expandedRowIdx[token, slot], :]
   ```

当前项目不包含 dispatch pack/gather，也不包含真实 expert FFN/GMM 计算。kernel 入口假设
`expertOutput` 和 dispatch metadata 已经准备好。

## 关键输入

主要 shape 参数定义在 `common.h` 的 `DispatchCombineTileShape`：

- `ep`：rank 数。
- `m`：本 rank token 数。
- `k`：hidden size。
- `topK`：每个 token 的 expert route 数。
- `expertPerRank`：每个 rank 拥有的 local expert 数。
- `expertNum`：全局 expert 数，等于 `ep * expertPerRank`。
- `maxOutputSize`：每个 rank 的 expert output 最大行容量。
- `aivBlocks`：AIV block 数。
- `tileCols`：restore/拷贝时的列方向 tile 宽度。
- `rowChunk`：return 阶段按行分块的 chunk 大小，0 表示默认值。
- `metadataPad`：expert metadata 行的 padding 粒度。
- `signalValue`：当前迭代使用的 done signal epoch。

运行前必须满足：

- `workspace.dispatchOffset` 有效。
- `workspace.prevSumBeforeRank` 有效。
- `workspace.cumsumPerExpert` 有效。
- `peerWindow.peerTokenPerExpert` 有效。
- `peerWindow.expandedRowIdx` 有效。
- `expertOutput` 按 local expert、source rank、row 的顺序排列。
- `probs` 与 `expandedRowIdx` 来自同一份 routing。
- `peerWindow.ptrD` 已清零。
- 所有 rank 使用一致的 HCCL window layout 和 signal epoch。

## Host 流程

host 侧入口在 `main.cpp`，主流程如下：

```text
ParseArgs
  -> ComputeWorkspaceLayout / ComputePeerWindowLayout
  -> InitRankInfo
  -> PrepareHostData
  -> BindDeviceContinuous
  -> CreateStreams
  -> InitHccl
  -> AllocateLocalBuffers
  -> CopyInputsToDevice
  -> loop(warmup + iters):
       ClearDeviceState
       PrepareCombineFixture
       RunCombine
       VerifyAndDump
  -> PrintProfileSummary
  -> Cleanup
```

`PrepareHostData` 负责生成或加载 deterministic 输入，并计算 CPU golden。`PrepareCombineFixture`
负责把 combine kernel 需要的前置状态写入 device：

- `workspace.cumsumPerExpert`
- `workspace.dispatchOffset`
- `workspace.prevSumBeforeRank`
- `workspace.dispatchedA`
- `peerWindow.peerTokenPerExpert`
- `peerWindow.expandedRowIdx`
- `expertOutput`

当前自校验路径中，`expertOutput` 使用 identity fixture，即 `expertOutput = dispatchedA`。
这样可以单独验证 combine return 和 restore 的正确性。

## Device 流程

device 侧入口在 `combine_tile_kernel.cpp`：

```text
DispatchCombineTileCombine
  -> ReturnExpertRowsToOwners
  -> WaitCombinePhase
  -> RestoreOutputRows
```

`ReturnExpertRowsToOwners`：

- 按 `src rank * expertPerRank + localExpert` 遍历所有 return segment。
- 用 `peerTokenPerExpert[src, globalExpert]` 得到当前 segment 行数。
- 用 `dispatchOffset[localExpert] + prevSumBeforeRank[src, localExpert]` 得到 `expertOutput` 源行。
- 用 `cumsumPerExpert[src, globalExpert - 1]` 得到目标 rank `ptrD` 目标行。
- `src == myRank` 时本地拷贝。
- `src != myRank` 时通过 remote peer window 执行 `TPUT`。
- 所有 return chunk 完成后，对 peer 发布 `combineDoneSignal[myRank]`。

`WaitCombinePhase`：

- 本 rank 等待 `combineDoneSignal[peer] >= signalValue`。
- signal 使用每轮递增 epoch，避免多轮迭代复用旧信号造成误判。

`RestoreOutputRows`：

- 每个 AIV block 负责一段 token。
- 对每个 token 先将 `outputC[token, :]` 清零。
- 遍历 topK route，读取 `expandedRowIdx` 和 `probs`。
- 使用 PTO vector tile 执行 `output += prob * ptrD`。

## 内存布局

layout 由 `layout.h` 计算，host 和 kernel 侧保持字段顺序一致。

`WorkspaceLayout` 包含：

- `localTokenPerExpert`
- `blockTokenPerExpert`
- `blockPrefixPerExpert`
- `cumsumPerExpert`
- `dispatchOffset`
- `prevSumBeforeRank`
- `localSync`
- `floatScratch`
- `dispatchedA`
- `ptrDLocal`

`PeerWindowLayout` 包含：

- `peerTokenPerExpert`
- `expandedRowIdx`
- `packedA`
- `ptrD`
- `countReadySignal`
- `combineDoneSignal`

其中 combine 主路径直接依赖 `cumsumPerExpert`、`dispatchOffset`、`prevSumBeforeRank`、
`peerTokenPerExpert`、`expandedRowIdx`、`ptrD` 和 `combineDoneSignal`。

## 校验数据

`golden.h` 提供 deterministic 输入、CPU golden 和二进制 dump：

- `inputA`：本 rank token 输入。
- `expertIdx`：本 rank token/topK route 到 global expert。
- `probs`：本 rank token/topK 权重。
- `dispatchedA`：CPU golden 计算出的 expert 输入排列。
- `expertOutput`：当前 fixture 下等于 `dispatchedA`。
- `ptrD`：combine return 之后本 rank 应收到的按 route packed 数据。
- `outputC`：按 `probs` restore 之后的最终输出。

常用 dump 文件：

- `rank_<r>_probs.bin`
- `rank_<r>_expandedRowIdx.bin`
- `rank_<r>_ptrD_head.bin`
- `rank_<r>_actual_ptrD_head.bin`
- `rank_<r>_golden_outputC.bin`
- `rank_<r>_outputC.bin`

## 构建与运行

默认使用 A2/A3 CANN 环境，`run.sh` 会加载 `ASCEND_CANN_PATH` 指向的 `set_env.sh`。

只编译不运行：

```bash
bash run.sh --skip-run 1
```

host golden 自校验：

```bash
bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 2 --host-golden-only 1 --skip-build 1
```

最小 combine return 校验：

```bash
timeout 60s bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 2 --combine-return-only 1
```

最小 combine restore 校验：

```bash
timeout 60s bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1 --debug 2 --verify 1
```

当前机器是 A3 时可以运行 A3 用例。卡不支持多任务并行，硬件运行前建议先查看设备占用：

```bash
npu-smi info
```

## 运行日志

`--debug 1` 或 `--debug 2` 下应重点观察：

```text
rank=<r> stage=combine begin
rank=<r> combine_return_done
rank=<r> combine_wait_done peers=<ep>
rank=<r> restore_done
rank=<r> stage=verify begin
rank=<r> verify=PASS mismatch_count=0
```

profile 汇总包含：

```text
prepare_host
combine_e2e
total_e2e
```

在当前项目中，`combine_e2e` 统计 combine kernel launch 到 stream sync，并包含 rank 间 barrier；
`total_e2e` 当前记录为 combine e2e。

## 当前边界

- 当前只覆盖 MoE combine 阶段，不覆盖 dispatch 和 expert 计算。
- 当前跨 rank return 是 HCCL window 上的 PTO `TPUT/TWAIT` 协议，不是 HCCL collective API。
- 当前 fixture 使用 `expertOutput = dispatchedA`，用于隔离验证 combine return/restore。
- 当前默认数据来源是 deterministic host 生成；`--gen-data 0 --data-dir <dir>` 可复用已有 rank 文件。
- 当前命名仍保留 `DispatchCombineTileShape`、`DispatchCombineTileCombine` 等 ABI 名称，后续如要统一命名，应保持 host/kernel ABI 一次性同步。

## 后续任务输入

后续任务可以按以下顺序推进：

1. 固化 README 中的 combine-only 范围，避免把 dispatch 或 expert 计算混入本项目验收。
2. 补充 `--gen-data 0` 对外部数据目录的 manifest 或完整文件检查。
3. 将 `expertOutput` 输入从 identity fixture 扩展为可加载真实 expert 输出。
4. 清理当前未参与 combine 主路径的 layout 字段，或明确保留原因。
5. 统一项目内命名，将公开 launcher/kernel 名称改为 `CombineTile*`，同时保证 ABI 和 run.sh 同步。
6. 做性能侧任务：rowChunk、tileCols、aivBlocks 的 sweep，分别统计 return 和 restore 的计时。
