# TGET / TGET_ASYNC 带宽对比示例

## 概览

本示例对比 **TGET**（同步远程读）与 **TGET_ASYNC**（异步 SDMA 远程读）的点对点通信带宽，覆盖 4 KB ~ 4 MB 的传输规模，同时测量 host 侧带宽（GB/s）和 device 侧平均执行 cycle 数。

- **TGET** 通过 UB（Unified Buffer）暂存进行远程读：`Remote GM → UB → Local GM`，受 UB 带宽限制，大传输量时饱和在约 4 GB/s。
- **TGET_ASYNC** 通过 SDMA 引擎直传：`Remote GM → SDMA → Local GM`，绕过 UB 瓶颈，在 4 MB 时可达约 13–14 GB/s。

## 支持的 AI 处理器

- A2/A3

## 目录结构

```
kernels/manual/a2a3/tget_bandwidth/
├── scripts/
│   └── plot_bw_compare.py           # 绘制带宽对比图
├── CMakeLists.txt                   # 构建配置
├── tget_bandwidth_kernel.cpp        # Kernel 实现（AICORE + Host 编排）
├── tget_bandwidth_kernel.h          # Kernel 头文件
├── main.cpp                         # Host 侧入口（MPI 初始化）
├── run.sh                           # 便捷脚本
├── README_zh.md                     # 本文件
└── README.md                        # 英文版
```

## 算子说明

### 数据流

**TGET（同步）**：
```
Peer NPU GM ──TGET──▶ Local UB ──TSTORE──▶ Local GM
```

**TGET_ASYNC（异步）**：
```
Peer NPU GM ──SDMA──▶ Local GM   (直传，无 UB 中转)
```

### 测试流程

1. 每个 rank 在 HCCL shared memory 中准备发送数据（`PrepareSendBufferKernel`）
2. root rank 对每种传输规模分别执行 TGET 和 TGET_ASYNC
3. Host 侧计时测量带宽，device 侧通过 `SYS_CNT` 测量 cycle 数
4. 验证接收数据正确性

### 规格

| 项目        | 值 |
| ----------- | ----- |
| 数据类型    | `float` |
| NPU 数量    | 2（点对点） |
| 传输规模    | 4 KB, 16 KB, 64 KB, 256 KB, 1 MB, 4 MB |
| 测量指标    | host 带宽（GB/s）、device 平均 cycle 数 |

## 实测性能（参考）

以下数据在 Ascend A2/A3 上测得（float 类型，2 卡点对点）。

| 传输大小 | TGET 带宽 (GB/s) | TGET_ASYNC 带宽 (GB/s) | TGET device 平均 cycles | TGET_ASYNC device 平均 cycles |
| -------- | ----------------: | ----------------------: | ----------------------: | ----------------------------: |
| 4 KB     | 0.21              | 0.19                    | 50.85                   | 118.18                        |
| 16 KB    | 0.72              | 0.75                    | 202.05                  | 166.42                        |
| 64 KB    | 1.75              | 2.55                    | 780.73                  | 338.10                        |
| 256 KB   | 3.01              | 6.08                    | 3347.12                 | 1094.37                       |
| 1 MB     | 3.75              | 10.48                   | 12703.39                | 3791.18                       |
| 4 MB     | 3.99              | 12.95                   | 52878.12                | 14834.47                      |

### 分析

- **TGET** 随传输规模增大，带宽逐步上升但在约 **4 GB/s** 处饱和——这是 UB 暂存路径的单核带宽上限。
- **TGET_ASYNC** 在大传输量（≥256 KB）时显著超越 TGET，4 MB 时达到约 **13 GB/s**，接近 SDMA 引擎的理论带宽。
- 在极小传输量（4 KB）下，TGET_ASYNC 由于 SDMA 启动开销反而略慢于 TGET。

### 带宽对比图

运行绘图脚本生成对比图：

```bash
python3 scripts/plot_bw_compare.py
```

## 构建与运行

### 前置条件

- CANN Toolkit >= 8.5.0（TGET 同步指令）；>= 9.0.0（TGET_ASYNC 异步指令）
- MPI >= 3.2.1（如 OpenMPI）
- 2 张及以上 Ascend NPU

### 步骤

1. 配置 Ascend CANN 环境：

```bash
source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
```

2. 运行示例（默认 2 卡）：

```bash
cd ${git_clone_path}/kernels/manual/a2a3/tget_bandwidth
bash run.sh -r npu -v Ascend910B1
```

可通过 `-n` 参数指定 rank 数（默认为 2）：

```bash
bash run.sh -r npu -v Ascend910B1 -n 2
```

成功时输出：

```text
================ TGET/TGET_ASYNC Bandwidth Sweep ================
peer_rank=1 dtype=float tile_elems=1024
[BW] instr=TGET bytes=4096 iters=1000 ...
[BW] instr=TGET_ASYNC bytes=4096 iters=1000 ...
...
test success
```

## 变更记录

| 日期       | 变更 |
| ---------- | ------ |
| 2026-04-02 | 从 ST 测试迁移为独立性能示例 |
