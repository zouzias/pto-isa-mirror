# Ring Allgather Demo

本示例展示如何使用 PTO 的 `TPUT_ASYNC`（异步远程写）SDMA 指令，基于 **Ring 算法** 实现 allgather 集合通信操作。

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
./run.sh 2 Ascend910_9599     # 2 ranks，A5 设备
```

## 功能说明

每个 rank 贡献 256 个 `int32_t` 数据。allgather 操作完成后，每个 rank 都持有所有 rank 的完整数据。

### Ring 算法原理

Ring allgather 共执行 N-1 轮（N 为 rank 总数），每轮中每个 rank 向其环形邻居（下一个 rank）推送一个 chunk：

- **第 0 轮**：rank i 先将自身 sendBuf 拷贝到 recvBuf[i]（本地拷贝），然后通过 `TPUT_ASYNC` 将 sendBuf 推送到 rank (i+1) 的 recvBuf[i]。
- **第 r 轮（r >= 1）**：rank i 将上一轮接收到的 chunk 继续转发给 rank (i+1)，即把 recvBuf[(i-r+N)%N] 推送到 rank (i+1) 的 recvBuf[(i-r+N)%N]。

每轮通过独立的 kernel launch 执行，host 端在轮次间执行 `aclrtSynchronizeStream` + `HcclHostBarrier` 保证所有 SDMA 写入完成后再开始下一轮。

### 已知限制

当 N > 2 时，第 1 轮及之后的轮次中，rank 需要读取上一轮由远端 SDMA 写入本地 HBM 的数据。由于当前硬件上 SDMA 远程写入不会主动使本地 AICORE 的 L2 缓存失效，可能导致读到过期数据。此 demo 为算法参考实现，已知此限制。

### 关键 PTO API

- `pto::comm::AsyncSession` / `BuildAsyncSession`：SDMA 异步会话初始化
- `pto::comm::TPUT_ASYNC`：异步远程写（通过 SDMA 引擎）
- `pto::comm::AsyncEvent` / `Wait`：异步事件等待与同步
- `SdmaWorkspaceManager`：Host 侧 SDMA 工作空间管理
- `HcclRemotePtr`：将本地共享内存地址转换为远端可访问地址

## 目录结构

```
allgather_ring/
├── CMakeLists.txt                            # 构建配置（bisheng 编译器 + CCE）
├── csrc/
│   ├── kernel/
│   │   ├── allgather_ring_kernel.cpp         # AICORE kernel 实现 + Host 侧启动函数
│   │   └── allgather_ring_kernel.h           # Host 侧函数声明
│   └── host/
│       └── main.cpp                          # 入口（MPI 初始化、运行 Demo、输出结果）
├── run.sh                                    # 一键构建并运行
├── README.md                                 # 英文说明
└── README_zh.md                              # 本文档
```

## 手动构建与运行

```bash
# 构建
mkdir -p build && cd build
cmake .. -DSOC_VERSION=ascend910b1
make -j$(nproc)
cd ..

# 运行（N 个 MPI 进程，对应 N 个 NPU 设备）
mpirun -n 2 ./build/bin/allgather_ring_demo
```

## 预期输出（2 ranks）

```
========================================
 PTO Ring Allgather Demo (TPUT_ASYNC)
 Ranks: 2
 Rounds: 1
========================================

--- Ring Allgather via TPUT_ASYNC ---
[RING_TPUT_ASYNC PASS] Rank 0: slot[0]=[0,1,2,...] slot[1]=[1000,1001,1002,...]
[RING_TPUT_ASYNC PASS] Rank 1: slot[0]=[0,1,2,...] slot[1]=[1000,1001,1002,...]

========================================
 Demo PASSED
========================================
```
