# CCU GEMM AllReduce 融合算子示例

## 概览

本示例演示如何在 A5 上使用 PTO + **CCU** 实现多卡 GEMM + AllReduce 融合算子：Compute Stream 跑 AIC GEMM，AIV 只做就绪与 CKE 调度，数据面由 **Persistent CCU** 完成 owner 作用域的 Pull Reduce + Push Broadcast（逻辑上等价 RS + AG）。Host 侧通过 `HcommCcuKernelRegister*` / `HcommCcuKernelLaunch` 注册与下发微码。

相对同目录兄弟示例 `gemm_ar`（AIV `TPUT` / AtomicAdd 数据面），本示例通信量同为约 `2*(P-1)/P * D`，但通信搬运改走 CCU。

## 支持的 AI 处理器

- Ascend950PR

## 目录结构

```text
kernels/manual/a5/ccu_gemm_ar/
├── CMakeLists.txt                      # 构建配置（3 个 target：cube / vec / host）
├── run.sh                              # 一键构建+运行（自动抬高 HCCL_BUFFSIZE、发现 MPI）
├── config.h                            # 全局参数（矩阵维度、tile、group、signal 布局）
├── host_comm.hpp                       # Host 通信栈（HCCL window + CCU channel/CKE/Register/Launch）
├── main.cpp                            # Demo 编排：MPI、bench、校验、数据 IO
├── compute_kernel.cpp                  # Device 计算（AIC Cube，`dav-c310-cube`）
├── progress_kernel.cpp                 # Device 通信控制（AIV progress / seq peer-sync / gate / unpack）
├── ccu_reduce_broadcast_kernel.hpp     # CCU 微码合成（host 写、CCU 跑）
├── kernel_launchers.h                  # Device kernel launcher 声明
├── comm_context.h                      # CommDeviceContext（窗口地址等）
└── comm_mpi.h                          # MPI 动态加载包装
```

共享头文件：`include/pto/comm/async/ccu/ccu_gate_registry.hpp`、`ccu_loopgroup_utils.hpp`。

## 算子说明

### 计算功能

本示例实现多卡 GEMM + AllReduce：

$$
C_{final} = \sum_{i=0}^{nranks-1} A_i \times B
$$

其中：

- `A_i` 为 `M×K`（每 rank 独立）
- `B` 为 `K×N`（所有 rank 共享）
- `C_i` 为 `M×N`（每 rank 本地 GEMM 结果）
- `C_final` 为 `M×N`（AllReduce 归约后的最终输出）

`config.h` 中默认参考配置为 `M=5416, K=6144, N=1408`，与 `gemm_ar` 对齐。

### 规格

| 项目 | 值 |
| --- | --- |
| OpType | `GEMM + AllReduce`（CCU 数据面） |
| 输入 | `A_i`: `M×K`, `float16`, `ND`（每 rank 独立）; `B`: `K×N`, `float16`, `DN`（共享） |
| 输出 | `C_final`: `M×N`, `float16`, `ND`（AllReduce 归约结果） |
| 计算 Kernel | `CcuGemmArComputeKernel`（Cube，`dav-c310-cube`） |
| Progress Kernel | `ccu_gemm_ar_progress_kernel`（Vector，`dav-c310-vec`） |
| CCU 数据面 | fused Pull Reduce + Push Broadcast（`ccu_reduce_broadcast_kernel.hpp`） |

## 优化说明

本示例以 Ascend950PR 作为验证平台。Cube（AIC）与 Vector（AIV）物理分立，可配合双流做计算与通信重叠；通信 payload 由 CCU 完成，AIV 不承担 RS/AG 搬运。

> **核数以 CANN** `platform_config` **为准（推荐），以** `950PR_958b` **为例**：
>
> - `cube_core_cnt=32`（Cube / AIC 侧并行度）
> - `vector_core_cnt=64`（Vector / AIV 侧并行度）

- **双流重叠**：Compute Stream 跑 AIC GEMM；AIV Stream 跑 progress/gate；CCU Stream 跑 fused RS+AG。tile/group 就绪后经 CKE 触发 CCU，与后续计算重叠。
- **逻辑 RS + AG，执行上 fused**：每个 owner group 一次 CCU mission——先 Pull Reduce 到 owner，再 Push Broadcast；通信量约 `2*(P-1)/P * D`。
- **owner 作用域 packed 布局**：`owner = tile % nranks`，packed 按 owner 连续，便于 CCU 按 group 发固定长度 WQE；残余 group 对 owner shard 按 `comm-group-tiles` 对齐补零。
- **Progress 同步**：AIC 对本地 window `groupDone[flat]` 做 `AtomicAdd`；peer AIV poll 齐后 `TNOTIFY(owner groupReady)`；owner 等 `>= P-1` 再 `TriggerProgressCke`（**depth-1**，避免丢边沿）。
- **CKE 生命周期**：seq one-shot 与 pipelined 在**同一次** `RegisterStart/End` 中注册，Translate 后分别 Publish 到 `seqGate` / `gate`（+ progress），保证物理 CKE 不复用。
- **AG 错峰**：CCU Broadcast 的 peer 写序按 `(rankId + group) % (P-1)` 轮转。
- **Block Swizzle + L1/L0 双缓冲**：计算侧与 `gemm_ar` 同类（zigzag tile、`stepK=4`、L0 ping/pong）。
- **每 tile store fence**：`TSTORE` 后 `pipe_barrier(PIPE_FIX) + dsb`，再发 `groupDone`。
- **Stable 约束**：`CCU_MISSION_PARALLEL=1`；每 rank 有独立的 pipelined `gate` 与 Sequential `seqGate`。

## Tiling 参数

| 参数 | 值 |
| --- | --- |
| `M`（原始） | 5416 |
| `K` | 6144 |
| `N`（原始） | 1408 |
| `M`（对齐后） | 5504 |
| `N`（对齐后） | 1536 |
| `baseM` | 128 |
| `baseK` | 64 |
| `baseN` | 256 |
| `stepKa` / `stepKb` | 4 |
| `commSubM` | 128（`== baseM`，当前路径要求 subtile=1） |
| `commGroupTiles` | 默认 16（4 卡建议 8） |
| `tile 数` | 258（43×6） |
| `COMPUTE_BLOCK_NUM` | 24（可用 `--compute-blocks` 覆盖） |
| `COMM_BLOCK_NUM` | 24 |
| `CCU_MISSION_PARALLEL` | 1 |

## 整体架构

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│  Compute Stream (AIC)     AIV Stream (progress)        CCU Stream            │
│                                                                              │
│  CcuGemmArComputeKernel:  ccu_gemm_ar_progress_kernel:  fused RS+AG:         │
│  ┌────────────────────┐   ┌─────────────────────────┐  ┌──────────────────┐  │
│  │ for each tile:     │   │ poll groupDone          │  │ WaitEvent(gate)  │  │
│  │   K-loop → TSTORE  │──►│ TNOTIFY owner groupReady│  │ per group:       │  │
│  │   PIPE_FIX + dsb   │   │ owner: wait >= P-1      │──►│   Pull Reduce    │  │
│  │   AtomicAdd        │   │ TriggerProgressCke      │  │   Push Broadcast │  │
│  │     groupDone[flat]│   │ (depth-1 backpressure)  │  │ itemsDone++      │  │
│  └────────────────────┘   └─────────────────────────┘  └──────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 计算内核详解

每个 AIC 负责一组 tile（按 `block_idx` 分配），对每个 tile：

1. **Block Swizzle**：zigzag 遍历，奇数行反向，提升 B 的 L1 复用。
2. **K-loop**：`stepKa=4` 批量 TLOAD 到 L1，TEXTRACT 到 L0，TMATMUL / TMATMUL_ACC。
3. **TSTORE**：L0C FP32 经 FixPipe cast 为 FP16，写入 **owner-packed** `gemm_output`。
4. **`pipe_barrier(PIPE_FIX) + dsb(DSB_DDR)`**：保证 store 可见后发信号。
5. **Progress 信号**：`AtomicAdd(+1)` 到本 rank window 的 `groupDone[flat(owner,g)]`。

## 通信路径详解

### AIV Progress

1. 各 rank AIV 轮询本地 `groupDone[flat]`，达到 `CcuOwnerGroupTilesInGroup` 后，对 **owner** 的 `groupReady[flat]` 做 `TNOTIFY(+1)`。
2. Owner AIV 对每个 flat group：`TTEST(groupReady >= P-1)` 通过后，在 depth-1 背压允许时 `TriggerProgressCke`（ping-pong slot A/B）。
3. Host 侧 `PrepareCcu`：Launch persistent CCU → poke gate CKE；CCU 在 `WaitEvent(gate)` 后进入 `CCU_WHILE`，按 progress CKE 消费 group。

### CCU fused RS + AG

对每个就绪 group（owner 作用域连续 tile）：

1. **Pull Reduce**：从各 peer 的 packed `gemm_output` `Read` 到 MS，`LocalReduce`，结果落在 owner `reduced_output` 对应 shard。
2. **Push Broadcast**：按错峰 peer 序 `Write` 到各 rank 的 `reduced_output` 同偏移。
3. `itemsDone` 递增，供 AIV 背压判断下一 group 是否可开火。

### Sequential baseline

与 Pipelined **分路径**：数据面仍是 one-shot（只开 **seqGate** CKE，无 progress CKE / 无按 group 开火），但对端就绪复用同一套 ready-counter `TNOTIFY`。`seqGate` 与 pipelined `gate` 是不同物理 CKE，避免 `PrepareCcu` 的 gate poke 误放行 Sequential。

1. Host `Launch` one-shot（CCU 停在 `WaitEvent(seqGate)`，**计时外**；Launch 可早于 GEMM，与 Pipelined 一样）。
2. 整段 GEMM → sync（必须先算完再通信；此时 CCU 仍应卡在 `WaitEvent`）。
3. AIV `seq_peer_sync_gate`：走与 Pipelined 共用的 peer-ready walk（**仅在整段 GEMM 之后**），然后 **单次** Trigger `seqGate`（禁止 follow-up poke，否则会 latch 下一轮 WaitEvent）。
4. CCU **一次** fused RS→AG（全 owner 真实 footprint）。

计时：`seq` 为连续墙钟 `GEMM → peer sync → CCU sync`；`seq_comm` 含 peer sync + one-shot 通信。可用 `PTO_CCU_GEMM_AR_COMM_DIAG=1` 查看 `aiv_wall` / `ccu_wall`（`ccu_wall≈0` 表示 gate 未挡住）。

## 内存布局与 HCCL 窗口

CCU payload（`gemm_output` / `reduced_output`）与最终 `row_output` 在高带宽 HBM block（带 token，供 CCU 访问）；跨 rank 同步计数器在 HCCL window 的 `signal_matrix`。

| 缓冲区 | 大小 | 位置 | 原因 |
| --- | --- | --- | --- |
| `gemm_output`（packed） | owner-padded × tile bytes | **HBM + token** | CCU Pull 源 |
| `reduced_output`（packed） | 同上 | **HBM + token** | CCU Reduce 目的 / Broadcast 源与目的 |
| `row_output` | `M × N × 2B` | **HBM block** | unpack 后校验用 |
| `signal_matrix` | `G_SIGNAL_MATRIX_SLOTS × 4B` | **HCCL 窗口** | `groupDone` / `groupReady` |
| `src0_dev`, `src1_dev` | 输入矩阵 | **aclrtMalloc** | 仅本地读写 |

`run.sh` 按 pad 后体积估算并抬高 `HCCL_BUFFSIZE`（约 `3 × packed + 128MB` margin）。

## 实测性能（参考）

以下数据在 2 卡 Ascend950PR 上测得，参数 `M=5416, K=6144, N=1408`（padded `5504×1536`），`258 tiles (43×6)`，`compute_blocks=32`，`ccu_group_tiles=16`。每 rank 计算完整 GEMM `C_i = A_i × B`，AllReduce 对 2 个 `C_i` 求和；`comm_data=0.016 GB/rank`。

| 指标 | 值 |
| --- | --- |
| Compute-only | `313.2 us`（`299162 GFLOPS`） |
| Sequential | `735.6 us`（compute `314.7 us` + one-shot fused RS→AG `420.9 us @ 37.4 GB/s`） |
| Pipelined | **`574.6 us`**（compute done `302.8 us`，comm done `574.1 us @ 27.4 GB/s`） |
| Speedup | `1.280x` |
| Time saved | `161.0 us`（`21.9%`） |
| Overlap eff | `52.7%` |
| Throughput | `326138 GFLOPS`（total） |

### 这些数字意味着什么

- **Compute-only**：纯 GEMM 时间（无通信）。当前 `313.2 us`，对应 `299162 GFLOPS`。
- **Sequential**：整段 GEMM 后再 peer-sync + one-shot CCU，无算通重叠。Launch 在计时外；`seq` 为连续墙钟。当前 `735.6 us`，其中 compute `314.7 us`、comm `420.9 us`。
- **Pipelined**：`PrepareCcu` 在计时外；AIC / AIV / CCU 重叠端到端。当前 `574.6 us`，相对 Sequential 加速 `1.280x`；`compute done = 302.8 us`。
- **Speedup**：Sequential / Pipelined。
- **Time saved**：相对串行路径节省的总时长。当前节省 `161.0 us`，约占 `21.9%`。
- **Overlap eff**：重叠带来的时间节省占较短阶段时间的百分比。

## 构建与运行

1. 配置 Ascend CANN 环境（需 **cann-9.2**）：

```bash
export ASCEND_CANN_PATH=/usr/local/Ascend/cann-<version>/set_env.sh
source "${ASCEND_CANN_PATH}"
```

2. 运行示例（2 卡）：

```bash
cd ${git_clone_path}/kernels/manual/a5/ccu_gemm_ar
./run.sh -r npu -v Ascend950PR_958b -n 2 -d 2 --compute-blocks 32
```

3. 指定起始设备编号：

```bash
FIRST_DEVICE=0 ./run.sh -r npu -v Ascend950PR_958b -n 2 -d 2 --compute-blocks 32
```

4. 4 卡（推荐 `comm-group-tiles=8`）：

```bash
FIRST_DEVICE=0 ./run.sh -r npu -v Ascend950PR_958b -n 4 -d 4 \
  --compute-blocks 32 --comm-group-tiles 8
```

成功时输出：

```text
CCU GEMM AllReduce demo completed successfully.
```

### 环境变量说明

| 环境变量 | 用途 | 默认行为 |
| --- | --- | --- |
| `ASCEND_HOME_PATH` | CANN 根路径（cmake 必需） | 由 `set_env.sh` 设置 |
| `ASCEND_CANN_PATH` | CANN `set_env.sh` 完整路径 | 按本机 CANN 安装填写 |
| `MPI_SEARCH_DIRS` | MPI `bin/` 搜索路径 | 常见 mpich 路径 |
| `MPI_LIB_PATH` | `libmpi.so` | `run.sh` 自动设置 |
| `HCCL_BUFFSIZE` | HCCL 窗口（MB） | `run.sh` 按 M/N 自动抬高 |
| `HCCL_CCU_CUSTOM_OP_MODE` | 自定义 CCU kernel | `run.sh` 置为 `1` |
| `FIRST_DEVICE` | 起始 NPU 编号 | 默认 `0` |
| `CCU_MISSION_PARALLEL` | CCU mission 并行度 | 必须为 `1` |
| `HCOMM_PKG_INC` | 内部 hcomm pkg_inc（可选） | cmake 自动探测 |
| `PTO_CCU_GEMM_AR_VERBOSE` | 打印 seq/pipe gate VA 等 | 默认关闭 |
| `PTO_CCU_GEMM_AR_COMM_DIAG` | 打印 seq/pipe `aiv_wall`/`ccu_wall` | 默认关闭 |

## 修改矩阵维度

修改 `config.h` 中的 `CONFIG_G_M` / `CONFIG_G_K` / `CONFIG_G_N`，或通过环境变量传入：

```bash
G_M=8192 G_K=8192 G_N=2048 ./run.sh -r npu -v Ascend950PR_958b -n 2 -d 2
```

约束：

- `K` 须能被 `G_BASE_K × stepKa`（默认 64×4=256）整除
- `M`/`N` 自动 pad 到 `baseM`/`baseN`
- `CONFIG_COMM_SUB_M == G_BASE_M`
- `CCU_MISSION_PARALLEL=1`


## 构建系统

- **编译器**：bisheng（CANN 内置）
- **Cube kernel**：`--cce-aicore-arch=dav-c310-cube`
- **Vec kernel**：`--cce-aicore-arch=dav-c310-vec`
- **Host**：`-xc++`，链接 `runtime`、`ascendcl`、`hccl`、`hcomm`、`tiling_api` 等
- **ABI**：Host `-D_GLIBCXX_USE_CXX11_ABI=0`（与 hcomm/hccl 一致）
- pto-isa `include/` 须优先于 CANN 自带头文件



