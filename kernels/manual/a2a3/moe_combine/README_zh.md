# moe_combine — A2/A3 PTO MoE Combine Kernel

基于 PTO 手写的 MoE **combine 阶段** device kernel，目标硬件为 Ascend A2/A3（Atlas 910B）。
覆盖 expert 计算完成之后的完整 combine 数据通路：

```
expertOutput[本地 experts × 来源 rank 行, K]
  ─── 变长 return (TPUT) ──→  token 归属 rank 的 peerWindow.ptrD
  ─── 跨 rank 完成信号 (TNOTIFY/TWAIT) ──→  同步屏障
  ─── 加权还原 (TAXPY) ──→  outputC[M, K]
```

## 架构总览

```
┌─────────────────────────────── Device Kernel ────────────────────────────────┐
│                                                                              │
│  MoeCombineKernel(shape, myRank, expertOutput, probs, outputC,     │
│                              peerWindow, hcclCtx, workspace)                │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │ 阶段 1: ReturnExpertRowsToOwners                                     │    │
│  │   • 按 segment 遍历: src_rank × expertPerRank                        │    │
│  │   • 从 peerTokenPerExpert 读取变长行数                                │    │
│  │   • 本地 return: TPUT（MTE2→MTE3，UB ping/pong 双缓冲）              │    │
│  │   • 远端 return: TPUT（经 UB ping/pong 写入 peer window）            │    │
│  │   • 按 chunk 在 AIV block 之间轮转分片                                │    │
│  │   • 完成后: TNOTIFY 各 peer 的 combineDoneSignal[myRank]             │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │ 阶段 2: WaitCombinePhase                                             │    │
│  │   • TWAIT combineDoneSignal[peer] >= signalValue  ∀ peer             │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │ 阶段 3: RestoreOutputRows                                            │    │
│  │   • Token 按 block 范围分片                                           │    │
│  │   • 每个 token: 先清零 outputC，再                                    │    │
│  │     for slot in topK:                                                │    │
│  │       outputC[token,:] += probs[token,slot] * ptrD[expandedRowIdx]   │    │
│  │   • 使用 TEXPANDS(清零) / TLOAD / TAXPY / TSTORE                    │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 覆盖范围


| 包含                        | 不包含                  |
| ------------------------- | -------------------- |
| 基于 HCCL window 的变长 return | Dispatch pack/gather |
| 跨 rank 完成信号同步             | Expert FFN/GMM 计算    |
| probs 加权还原                | HCCL AllToAllV 集合通信  |
| 多 AIV block 并行            | 动态路由/门控              |


## 核心参数

定义在 `common.h` → `MoeCombineShape`：


| 字段              | 含义                                |
| --------------- | --------------------------------- |
| `ep`            | rank 数（endpoint 数）                |
| `m`             | 本 rank token 数                    |
| `k`             | hidden size                       |
| `topK`          | 每 token 的 expert 路由数              |
| `expertPerRank` | 每 rank 的本地 expert 数               |
| `expertNum`     | 全局 expert 数（`ep × expertPerRank`） |
| `maxOutputSize` | 每 rank expert output 最大行容量        |
| `aivBlocks`     | AIV block 并行度（0→24）               |
| `tileCols`      | 向量操作列 tile 宽度（内部参数，固定 1024）       |
| `rowChunk`      | return 阶段行 chunk 大小（0→8）          |
| `metadataPad`   | expert metadata 对齐粒度              |
| `signalValue`   | 当前迭代信号 epoch                      |


## 计算链路与数据 Shape 流

```
输入:
  expertOutput [maxOutputSize, K]  ← 本 rank 所有 expert 处理后的结果（行按 expert×src 排列）
  probs        [M, topK]           ← 路由概率
  peerTokenPerExpert [ep, expertNumPadded]  ← 每个 (src, expert) 对的 token 行数
  expandedRowIdx     [M × topK]             ← 每个 (token, slot) 在 ptrD 中的目标行号
  cumsumPerExpert    [ep, expertNumPadded]  ← 累积行数，确定 ptrD 写入位置
  dispatchOffset     [expertPerRank]         ← 每个本地 expert 在 expertOutput 中的起始行
  prevSumBeforeRank  [ep, expertPerRank]     ← 本 rank 之前的累积行数

阶段 1: ReturnExpertRowsToOwners
  ┌─────────────────────────────────────────────────────────────────────┐
  │ for each (src, localExpert):                                        │
  │   rows = peerTokenPerExpert[src, globalExpert]                      │
  │   srcStart = dispatchOffset[localExpert]                            │
  │            + prevSumBeforeRank[src, localExpert]                    │
  │   dstStart = cumsumPerExpert[src, globalExpert - 1]  (或 0)         │
  │                                                                     │
  │   expertOutput[srcStart : srcStart+rows, 0:K]                       │
  │        ──TPUT──→  peer[src].ptrD[dstStart : dstStart+rows, 0:K]    │
  │                                                                     │
  │   shape: [rows, K] per chunk                                        │
  └─────────────────────────────────────────────────────────────────────┘
  数据流: expertOutput[maxOutputSize, K] → ptrD[M×topK, K] (跨 rank 写入)

阶段 2: WaitCombinePhase
  ┌─────────────────────────────────────────────────────────────────────┐
  │ TWAIT(combineDoneSignal[peer] >= signalValue)  ∀ peer               │
  │ 确保所有 rank 的 Return 完成，ptrD 数据可见                           │
  └─────────────────────────────────────────────────────────────────────┘

阶段 3: RestoreOutputRows
  ┌─────────────────────────────────────────────────────────────────────┐
  │ for each token in [tokenBegin, tokenEnd):                           │
  │   outputC[token, 0:K] = 0                                          │
  │   for slot in [0, topK):                                            │
  │     row = expandedRowIdx[token × topK + slot]                       │
  │     if row < 0: skip                                                │
  │     prob = probs[token × topK + slot]                               │
  │     outputC[token, 0:K] += prob × ptrD[row, 0:K]   ← TAXPY        │
  └─────────────────────────────────────────────────────────────────────┘
  数据流: ptrD[M×topK, K] × probs[M, topK] → outputC[M, K]

输出:
  outputC [M, K]  ← 每个 token 的加权 expert 结果之和
```

### Shape 总结

| 数据 | Shape | dtype | 说明 |
|------|-------|-------|------|
| expertOutput | [maxOutputSize, K] | half | 本 rank expert 输出，按 (expert, src) 分段排列 |
| probs | [M, topK] | float | 路由概率 |
| outputC | [M, K] | half | 最终输出 |
| ptrD | [M × topK, K] | half | combine return 缓冲区（peerWindow 内） |
| expandedRowIdx | [M × topK] | int32 | (token, slot) → ptrD 行号映射 |
| peerTokenPerExpert | [ep, expertNumPadded] | int32 | 路由行数表 |

## Kernel 参数语义

```cpp
MoeCombineKernel(shape, myRank, expertOutput, probs, outputC, peerWindow, hcclCtx, workspace)
```

| 参数 | 方向 | 类型 | Shape | 说明 |
|------|------|------|-------|------|
| `shape` | 入参 | 值传递 | — | 形状描述结构体，见"核心参数"表 |
| `myRank` | 入参 | uint32 | — | 当前 rank ID（0..ep-1） |
| `expertOutput` | 入参 | half* | [maxOutputSize, K] | 本 rank 所有 expert 处理完的输出，由 dispatch+FFN 阶段产出 |
| `probs` | 入参 | float* | [M, topK] | 每个 token 对各 expert 的路由概率 |
| `outputC` | **出参** | half* | [M, K] | 最终 combine 结果：各 expert 加权和 |
| `peerWindow` | 入参/中间态 | uint8* | 见下方 layout | HCCL 分配的跨 rank 共享内存窗口 |
| `hcclCtx` | 入参 | uint8* | — | HCCL 设备上下文，含各 rank 的 window 基地址映射 |
| `workspace` | 入参/中间态 | uint8* | 见下方 layout | rank 本地工作空间 |

**数据生命周期**：`peerWindow` 中的路由元数据（peerTokenPerExpert、expandedRowIdx）由 dispatch 阶段写入，combine 阶段只读。`ptrD` 区域在 combine 阶段由各 rank 互相写入（Return），然后本地读取（Restore）。

## 内存布局

### Workspace（rank 本地）

由 `MakeWorkspaceLayout(shape)` 计算偏移，所有字段 64 字节对齐，顺序排列。

| 字段 | 类型 | 元素数 | 说明 |
|------|------|--------|------|
| `localTokenPerExpert` | int32 | expertNumPadded | 本 rank 路由到各 expert 的 token 数 |
| `blockTokenPerExpert` | int32 | aivBlocks × expertNumPadded | 每 AIV block 分配的 token 数 |
| `blockPrefixPerExpert` | int32 | aivBlocks × expertNumPadded | 每 block 的前缀和（dispatch 用） |
| `cumsumPerExpert` | int32 | ep × expertNumPadded | 各 src rank 发给各 expert 的累积 token 数，用于计算 ptrD 写入偏移 |
| `dispatchOffset` | int32 | expertPerRank | 每个本地 expert 在 expertOutput 中的起始行偏移 |
| `prevSumBeforeRank` | int32 | ep × expertPerRank | 对于每个 (src, localExpert) 对，本 rank 之前所有 rank 的累积行数 |
| `localSync` | int32 | max(64, aivBlocks × (8 + expertNumPadded)) | SoftSyncAiv 的 GM polling 工作区 |
| `floatScratch` | float | aivBlocks × tileCols | 临时 float 缓冲（预留） |
| `dispatchedA` | half | maxOutputSize × K | dispatch 后的 expert 输入（测试 fixture 用） |
| `ptrDLocal` | half | M × topK × K | 本地 ptrD 镜像（预留） |

### Peer Window（HCCL 共享，每 rank 一份）

由 `MakePeerWindowLayout(shape)` 计算偏移。每个 rank 拥有一份，其他 rank 通过 `hcclCtx->windowsIn[rank]` 获取远端地址。

| 字段 | 类型 | 元素数 | 写入方 | 读取方 | 说明 |
|------|------|--------|--------|--------|------|
| `peerTokenPerExpert` | int32 | ep × expertNumPadded | dispatch 阶段 | combine Return | 各 src rank 发给各 expert 的 token 数（路由表） |
| `expandedRowIdx` | int32 | M × topK | dispatch 阶段 | combine Restore | 每个 (token, slot) 在 ptrD 中的行号（-1 表示无效路由） |
| `packedA` | half | M × topK × K | dispatch 阶段 | — | dispatch 打包数据（combine 不使用） |
| `ptrD` | half | M × topK × K | combine Return | combine Restore | 各 expert 处理后的结果写回区，按 expandedRowIdx 索引 |
| `countReadySignal` | int32 | ep | dispatch 阶段 | dispatch 阶段 | dispatch 就绪信号（combine 不使用） |
| `combineDoneSignal` | int32 | ep | combine Return | combine Wait | 跨 rank 完成信号：rank i 写 peer[j].combineDoneSignal[i]，rank j 等待所有 signal |

所有字段 64 字节对齐。host（`layout.h`）和 device 使用相同计算逻辑。

## UB 存储规划


| 区域     | 偏移     | 大小             | 用途              |
| ------ | ------ | -------------- | --------------- |
| Ping   | 0x0000 | 4 KiB          | 双缓冲 tile A      |
| Pong   | 0x1000 | 4 KiB          | 双缓冲 tile B      |
| Meta   | 0x2000 | 4 KiB          | 元数据 TLOAD tile  |
| Sync   | 0x3000 | 256 B          | SoftSyncAiv 工作区 |
| **总计** |        | **~12.25 KiB** | ≤ 192 KiB 预算    |


## 文件结构

```
moe_combine/
├── moe_combine_kernel.cpp  — PTO device kernel（return + wait + restore）
├── main.cpp                 — Host 编排（MPI、HCCL、验证）
├── common.h                 — 共享 ABI 结构体（Shape、Layout、Context）
├── kernel_launchers.h       — Kernel launch 声明
├── layout.h                 — Host 侧 layout 计算器
├── args.h                   — 命令行解析与校验
├── golden.h                 — CPU golden 参考实现与二进制 I/O
├── hccl_context.h           — HCCL window 初始化（MESH/RING）
├── comm_mpi.h               — dlopen MPI 封装
├── CMakeLists.txt           — Bisheng CCE + host 构建配置
├── run.sh                   — 构建/运行封装脚本（mpirun）
├── DESIGN.md                — 从 dispatch 拆分的设计说明
├── PHASE2_DESIGN.md         — PTO 风格重构计划
└── IMPLEMENTATION_PLAN.md   — 原始任务追踪器
```

## Host 流程

```
ParseArgs → ValidateArgs → ComputeLayouts
  → InitRankInfo（MPI）
  → PrepareHostData（确定性生成 + CPU golden）
  → BindDeviceContinuous → CreateStreams → InitHccl
  → AllocateLocalBuffers → CopyInputsToDevice
  → loop(warmup + 计时迭代):
       ClearDeviceState
       PrepareCombineFixture    ← 写入 metadata + expertOutput 到 device
       RunCombine               ← kernel launch + stream sync
       VerifyAndDump            ← 对比 outputC 与 CPU golden
  → PrintProfileSummary → Cleanup
```

## 构建与运行

### 前置条件

- Ascend CANN 8.5+ 含 Bisheng 编译器
- MPI 库（`MPI_LIB_PATH`）
- PTO 头文件库位于 `../../../../include`
- 硬件：Atlas 910B1（A3）或兼容

### 仅编译

```bash
bash run.sh --skip-build 0 --clean-build 1
```

### 快速验证（小 shape）

```bash
bash run.sh -pes 2 -M 8 -K 64 -topK 2 -expertPerPe 1
```

### 生产规模运行

```bash
bash run.sh -pes 2 -M 64 -K 7168 -topK 8 -expertPerPe 2 --aiv-blocks 24
```

### 主要命令行选项


| 选项             | 默认值    | 说明                 |
| -------------- | ------ | ------------------ |
| `-pes`         | 2      | rank 数             |
| `-M`           | 64     | 每 rank token 数     |
| `-K`           | 7168   | hidden size        |
| `-topK`        | 8      | 每 token expert 路由数 |
| `-expertPerPe` | 2      | 每 rank expert 数    |
| `--aiv-blocks` | 0（→24） | AIV block 并行度      |
| `--row-chunk`  | 0（→8）  | return chunk 大小    |
| `--debug`      | 0      | 0=静默, 1=摘要, 2=详细   |
| `--iters`      | 1      | 计时迭代次数             |
| `--warmup`     | 1      | warmup 迭代次数        |


验证默认始终开启，kernel 输出与 CPU golden 对比，容差 `rtol=1e-2, atol=1e-2`。

### 检查设备可用性

```bash
npu-smi info
```

## 验证方案

测试框架使用 **identity fixture**（`expertOutput = dispatchedA`）来隔离验证 combine 正确性：

1. **CPU Golden**：`golden.h` 在 host 计算完整 dispatch→combine 流水线
2. **完整校验**：device `outputC` 与 golden 对比，容差 `rtol=1e-2, atol=1e-2`
3. **二进制 dump**：保存到 `--data-dir` 供离线分析

预期通过输出：

```
verify=PASS
```

## 实现细节

### 跨 Rank 通信

- **本地 return**（`src == myRank`）：同步 `TPUT`，经 UB ping/pong tile
- **远端 return**（`src != myRank`）：同步 `TPUT`，经 UB ping/pong tile 写入 peer window
- **完成通知**：`TNOTIFY` + `AtomicAdd` 写 peer 的 `combineDoneSignal`
- **等待屏障**：`TWAIT` + `GE` 比较，阈值为 epoch `signalValue`

### 元数据访问

使用 `LoadMetadataScalar`：通过 MTE2 将 256 元素 int32 tile 从 GM TLOAD 到 UB（绕过标量 D-cache），再从 UB 中提取目标索引值。

### Block 分片策略

- Return 阶段：chunk 轮转分片（`chunkBase % blockNum == blockId`）
- Restore 阶段：token 范围分片（`TokenShardBegin/End`）
- 同步：阶段间使用 `SoftSyncAiv`（SYNCALL Soft 模式）

## 性能优化与 Overlap 分析

### 当前流水线结构

```
Time ──────────────────────────────────────────────────────────────────────────→

Block 0: ┃ Return chunk0 ┃ Return chunk2 ┃ ... ┃ Wait ┃ Restore tok0..N/B ┃
Block 1: ┃ Return chunk1 ┃ Return chunk3 ┃ ... ┃ Wait ┃ Restore tokN/B..  ┃
          ├─── 阶段 1: Return ────────────────┤sync├── 阶段 3: Restore ──┤
                                               ↑
                                          SoftSyncAiv
```

### 已实现的 Overlap 机制


| 机制                   | 位置                                             | 重叠效果                                     |
| -------------------- | ---------------------------------------------- | ---------------------------------------- |
| **Ping/Pong 双缓冲**    | Return 阶段: `TPUT(dst, src, ping, pong)`        | MTE2 加载下一 chunk 与 MTE3 写出当前 chunk 重叠     |
| **Event 驱动 TAXPY 链** | Restore 阶段: `TLOAD → TAXPY → TSTORE` 带 `Event` | MTE2 加载、Vector 计算、MTE3 写回形成 3 级流水        |
| **多 Block 并行**       | 所有阶段                                           | 多个 AIV block 并行执行，通过 chunk/token 分片共享工作量 |


### 关键调优参数


| 参数               | 影响                           | 建议                                    |
| ---------------- | ---------------------------- | ------------------------------------- |
| `aivBlocks`      | 核级并行度；更多 block = 更多并行工作量     | 24（默认）利用 910B 全部可用 AIV block          |
| `rowChunk`       | Return 粒度；控制 overlap 粒度和负载均衡 | 较大 = 更少元数据读取、更低开销；较小 = 更好的 block 负载均衡 |
| `K`（hidden size） | 决定每行数据量；主导带宽开销               | K=7168 fp16 = 14 KiB/行 → 带宽受限         |


### 瓶颈分析

#### 阶段 1: Return（通信受限）

```
┌───── AIV 标量路径 ─────────┐    ┌───── MTE2/MTE3 (ping/pong) ────────┐
│ LoadMetadata(peerToken)    │    │                                      │
│ LoadMetadata(dispatchOff)  │    │  ┌─ TPUT chunk N（经 UB）────────┐  │
│ LoadMetadata(prevSum)      │    │  │  MTE2 加载 → MTE3 写出         │  │
│ LoadMetadata(cumsum)       │    │  └──────────────────────────────────┘│
│ ... 下一 segment ...       │    │  ┌─ TPUT chunk N+1 ─────────────┐  │
│                            │    │  │  MTE2 加载 → MTE3 写出         │  │
└────────────────────────────┘    │  └──────────────────────────────────┘│
                                  └──────────────────────────────────────┘
```

- **主要开销**：远端写入的 MTE 传输延迟（`rowChunk × K × 2B` 每 chunk）
- **次要开销**：元数据标量读取（每 segment 4 次 `LoadMetadataScalar` → 4 次 MTE2 TLOAD）
- **优化杠杆**：增大 `rowChunk` 分摊元数据开销；`rowChunk=8, K=7168` 时每次传输 112 KiB

#### 阶段 2: Wait（延迟受限）

- 纯同步等待；开销 = max(各 peer return 延迟)
- `TWAIT` 信号轮询 — 硬件等待，几乎零 AIV 周期消耗
- **优化杠杆**：与 Stage 3 部分重叠（见下方 Future Optimizations）

#### 阶段 3: Restore（计算 + 带宽受限）

```
每 token、每 slot、每 tile：
  TLOAD(ptrTile, ptrD[row])     ← MTE2: 读取 ptrD 行 tile
  TLOAD(outTile, outputC[tok])  ← MTE2: 读取当前累加器
  TAXPY(outTile, ptrTile, prob) ← VEC:  融合乘加
  TSTORE(outputGlobal, outTile) ← MTE3: 写回累加器

每 tile 流水深度 = 3 级（MTE2 → VEC → MTE3）
每 token 总迭代数 = topK × ceil(K / tileCols)
```

- **主要开销**：MTE2 带宽，每 token-tile 需 `2 × topK` 次加载（ptrD + outputC 回读）
- **次要开销**：每 slot 后 MTE3 写回累加器
- **优化杠杆**：增大 `tileCols` 减少循环开销；多 slot 合并后再写回

### 未来优化方向


| 优化项                          | 预期收益                                                | 复杂度                                             |
| ---------------------------- | --------------------------------------------------- | ----------------------------------------------- |
| **Return ↔ Restore 部分重叠**    | 本地 return 完成的行可提前 restore，减少 Stage 2 空闲等待           | 高 — 需逐 peer 就绪跟踪和行级 restore 门控                  |
| **多 Slot 累加**                | 在 UB 中累加多个 topK slot 后再 TSTORE，MTE3 写次数减少 topK 倍    | 中 — 需额外 UB 累加器 buffer；UB 预算允许约 4 个额外 tile       |
| **元数据预取**                    | Return 开始时批量加载所有 segment 元数据，消除逐 segment 的 TLOAD 停顿 | 低 — 分配专用 UB 区域存放完整元数据数组                         |
| **自适应 rowChunk**             | 根据实际行数 vs block 数自动调优 chunk 大小                      | 低 — 在 kernel prologue 添加运行时启发式                  |
| **TPUT_ASYNCSDMA 远端 return** | 远端写入与 MTE 流水解耦；多笔传输同时在途                             | 中 — 需 CANN 9.0+ 及 `aclnnShmemSdmaStarsQuery` 支持 |
| **Restore 行预取**              | 计算当前 ptrD 行时预取下一行到 ping（当前正在用 pong）                 | 中 — 将 return 阶段的双缓冲扩展到 restore 阶段               |
| **K 维度 Cube 切片**             | 大 K 时使用 Cube（MTE1→Cube→MTE3）路径替代 VEC TAXPY          | 高 — 需要 Cube tile reshape 和不同的 UB 预算分配           |


### Overlap 理想流水（未来目标）

```
Time ──────────────────────────────────────────────────────────────────────────→

MTE2 (加载):  ║ meta_seg0    ║ meta_seg1    ║ ... ║ ptrD_tok0 ║ out_tok0 ║ ...║
VEC (计算):   ║              ║              ║     ║           ║ TAXPY_0  ║ ...║
MTE3 (写回):  ║ chunk0→peer  ║ chunk1→peer  ║ ... ║           ║          ║ ST ║
AIV (标量):   ║ addr_calc    ║ addr_calc    ║ ... ║ idx_read  ║ ...      ║   ║

目标：让 MTE2、VEC、MTE3 同时保持繁忙，
      在 return→restore 边界处尽可能重叠。
```

## 性能参考

```
[PROFILE] CombineTile
  M=64 K=7168 ranks=2 topK=8 expertPerPe=2 warmup=1 measured=1 samples=1
  logical work: input tokens(all ranks)=128 routed tokens(all ranks)=1024
  combine_e2e: avg=963.7 us max=963.7 us
  verify=PASS
```

运行环境：Atlas 910B1，2 ranks，AIV blocks=24。

## 当前边界

- 仅覆盖 MoE combine 阶段，不包含 dispatch 和 expert 计算
- 使用 HCCL window + PTO TPUT/TWAIT 协议，非 HCCL collective API
- Identity fixture（`expertOutput = dispatchedA`）用于正确性隔离验证
- 远端 return 使用同步 TPUT（SDMA 异步需 CANN 9.0+）
- 数据张量仅支持 half（fp16）精度

## 后续方向

1. 扩展 `expertOutput` 输入源，支持加载真实 expert 输出
2. 升级到 CANN 9.0 后启用 `TPUT_ASYNC<SDMA>` 提升远端 return 性能
3. 性能 sweep：`rowChunk`、`aivBlocks` 参数扫描
4. 补充 `--gen-data 0` 模式的外部数据目录校验
5. 清理未参与 combine 主路径的 layout 字段

