# Allgather 同步通信 Demo

本示例展示如何使用 PTO 的同步 `TPUT`（远程写）和 `TGET`（远程读）指令在多个 NPU 设备之间实现 allgather 集合通信操作。

与 `allgather_async` demo 不同，本 demo 使用的是**同步**通信指令：每次 `TPUT`/`TGET` 调用都通过 UB staging tile 中转数据，在调用返回前即完成传输，无需异步会话和事件等待。

## 前置条件

- 已安装 CANN Toolkit（9.0.0 及以上版本），并通过 `set_env.sh` 设置 `ASCEND_HOME_PATH`
- 已安装 CANN Ops 包（9.0.0 及以上版本）
- 已安装 MPICH
- 机器上至少有 2 个 NPU 设备

## 快速开始

```bash
source /path/to/set_env.sh
./run.sh                      # 8 ranks，默认 SoC
./run.sh 4                    # 4 ranks
./run.sh 2 Ascend910_9599     # A5 设备
```

## 功能说明

每个 rank 贡献 256 个 `int32_t` 数据。allgather 操作完成后，每个 rank 都持有所有 rank 的完整数据。

1. **TPUT Allgather（同步远程写）**：每个 rank 通过 `pto::comm::TPUT` 将自身数据同步写入所有其他 rank 的接收缓冲区对应位置。数据经由 UB staging tile 中转（本地 GM → UB → 远端 GM）。

2. **TGET Allgather（同步远程读）**：每个 rank 通过 `pto::comm::TGET` 从所有其他 rank 同步拉取数据到本地接收缓冲区。数据经由 UB staging tile 中转（远端 GM → UB → 本地 GM）。

### 关键 PTO API

- `pto::comm::TPUT`：同步远程写（本地 GM → UB staging tile → 远端 GM）
- `pto::comm::TGET`：同步远程读（远端 GM → UB staging tile → 本地 GM）
- `GlobalTensor`：基于 Shape/Stride 构造的 5 维全局内存张量视图
- `Tile` + `TASSIGN`：UB 缓冲区 Tile 分配
- `TLOAD` / `TSTORE`：GM 与 UB 之间的数据搬运（用于本地拷贝）
- `HcclRemotePtr`：将本地共享内存地址转换为远端可访问地址
- `set_flag` / `wait_flag`：硬件流水线间的事件同步

### 与异步版本的对比

| 特性 | 同步版本（本 demo） | 异步版本（allgather_async） |
| --- | --- | --- |
| 通信指令 | `TPUT` / `TGET` | `TPUT_ASYNC` / `TGET_ASYNC` |
| 数据路径 | GM → UB → GM（经 UB staging tile 中转） | GM → GM（SDMA 直传） |
| 完成方式 | 每次调用阻塞完成 | 返回 `AsyncEvent`，需 `Wait(session)` |
| 额外依赖 | 无 | 需要 `AsyncSession` + `SdmaWorkspaceManager` |
| 适用场景 | 数据量较小、逻辑简单 | 大数据量、需要通信与计算重叠 |

## 目录结构

```
allgather_sync/
├── CMakeLists.txt                       # 构建配置（bisheng 编译器 + CCE）
├── csrc/
│   ├── kernel/
│   │   ├── allgather_kernel.cpp         # AICORE kernel 实现 + Host 侧启动函数
│   │   └── allgather_kernel.h           # Host 侧函数声明
│   └── host/
│       └── main.cpp                     # 入口（MPI 初始化、运行 Demo、输出结果）
├── run.sh                               # 一键构建并运行
└── README_zh.md                         # 本文档
```

## 手动构建与运行

```bash
# 构建
mkdir -p build && cd build
cmake .. -DSOC_VERSION=ascend910b1
make -j$(nproc)
cd ..

# 运行（N 个 MPI 进程，对应 N 个 NPU 设备）
mpirun -n 2 ./build/bin/allgather_sync_demo
```

## 预期输出（2 ranks）

```
========================================
 PTO Allgather Sync Demo
 Ranks: 2
========================================

--- Demo 1: Allgather via TPUT (Sync) ---
[TPUT_SYNC PASS] Rank 0: slot[0]=[0,1,2,...] slot[1]=[1000,1001,1002,...]
[TPUT_SYNC PASS] Rank 1: slot[0]=[0,1,2,...] slot[1]=[1000,1001,1002,...]

--- Demo 2: Allgather via TGET (Sync) ---
[TGET_SYNC PASS] Rank 0: slot[0]=[0,1,2,...] slot[1]=[1000,1001,1002,...]
[TGET_SYNC PASS] Rank 1: slot[0]=[0,1,2,...] slot[1]=[1000,1001,1002,...]

========================================
 All demos PASSED
========================================
```
