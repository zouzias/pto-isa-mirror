# 点对点通信 Demo

本示例展示如何使用 PTO 的 `TPUT`（远程写）和 `TGET`（远程读）指令在两个 NPU 设备之间进行点对点通信。

## 前置条件

- 已安装 CANN Toolkit，并通过 `set_env.sh` 设置 `ASCEND_HOME_PATH`
- MPI 库可用（如 mpich）
- 机器上至少有 2 个 NPU 设备

## 快速开始

```bash
source /path/to/set_env.sh   # 设置 ASCEND_HOME_PATH
./run.sh                      # 构建并运行（默认 SoC: ascend910b1）
./run.sh Ascend910_9599       # A5 设备
```

## 功能说明

本 Demo 包含两个示例，分别演示 PTO 的两种点对点通信方式：

1. **TPUT 示例（远程写）**：Rank 0 将 256 个 `int32_t` 数据通过 `pto::comm::TPUT` 写入 Rank 1 的共享内存窗口，Rank 1 读取并校验接收到的数据。

2. **TGET 示例（远程读）**：Rank 0 将数据准备在自身的共享内存窗口中，Rank 1 通过 `pto::comm::TGET` 从 Rank 0 的共享内存中拉取数据并校验。

### 关键 PTO API

- `GlobalTensor`：基于 Shape/Stride 构造的 5 维全局内存张量视图
- `Tile` + `TASSIGN`：UB 缓冲区 Tile 分配
- `TLOAD` / `TSTORE`：GM 与 UB 之间的数据搬运
- `TPUT`：远程写（将本地数据写入远端 NPU 的 GM）
- `TGET`：远程读（从远端 NPU 的 GM 拉取数据到本地）
- `HcclRemotePtr`：将本地共享内存地址转换为远端可访问地址
- `set_flag` / `wait_flag`：硬件流水线间的事件同步

## 目录结构

```
p2p_comm/
├── CMakeLists.txt                 # 构建配置（bisheng 编译器 + CCE）
├── csrc/
│   ├── kernel/
│   │   ├── p2p_kernel.cpp         # AICORE kernel 实现 + Host 侧启动函数
│   │   └── p2p_kernel.h           # Host 侧函数声明
│   └── host/
│       └── main.cpp               # 入口（MPI 初始化、运行 Demo、输出结果）
├── run.sh                         # 一键构建并运行
├── README.md                      # 英文说明
└── README_zh.md                   # 本文档
```

## 手动构建与运行

```bash
# 构建
mkdir -p build && cd build
cmake .. -DSOC_VERSION=ascend910b1
make -j$(nproc)
cd ..

# 运行（需要 2 个 MPI 进程，对应 2 个 NPU 设备）
mpirun -n 2 ./build/bin/p2p_demo
```

可通过 `SOC_VERSION` 参数指定目标芯片，例如 A5 设备使用 `Ascend910_9599`。

## 预期输出

```
========================================
 PTO P2P Communication Demo
 Ranks: 2
========================================

--- Demo 1: TPUT (Remote Write) ---
[TPUT PASS] Rank 1 received Rank 0's data. First 5: [0, 1, 2, 3, 4, ...]

--- Demo 2: TGET (Remote Read) ---
[TGET PASS] Rank 1 pulled Rank 0's data. First 5: [1000, 1001, 1002, 1003, 1004, ...]

========================================
 All demos PASSED
========================================
```
