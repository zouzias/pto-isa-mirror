# TPut_sdma 测试用例

## 概述

本测试用例验证 `TPUT_SDMA` 指令的正确性。`TPUT_SDMA` 是一个异步远程写入操作，使用 SDMA（System DMA）引擎直接将数据从本地 GM（Global Memory）传输到远程 NPU 的 GM，无需 UB（Unified Buffer）中转。

## 与 TPUT 的区别

| 特性 | TPUT | TPUT_SDMA |
|------|------|-----------|
| 执行方式 | 同步 | 异步 |
| 数据路径 | GM → UB → GM | GM → GM（直接传输）|
| 需要 UB Tile | 是 | 否 |
| 计算通信重叠 | 否 | 是 |
| 返回值 | void | SdmaEvent |
| 同步方式 | 隐式（指令完成即同步）| 显式（需调用 `SDMA::wait()`）|

## 测试模式

采用**环形通信模式（Ring Communication）**：每个 rank 将数据写入前一个 rank 的接收缓冲区。

```
Rank 0 → Rank N-1
Rank 1 → Rank 0
Rank 2 → Rank 1
...
Rank N-1 → Rank N-2
```

## 实现逻辑

### 内核执行流程

```
1. 初始化阶段
   ├── 创建 GlobalTensor（srcG, dstG, sendG, recvG）
   └── 计算目标 rank（prev_rank = (my_rank + nranks - 1) % nranks）

2. 本地数据准备
   ├── TPUT_SDMA(sendG, srcG)  // 将源数据复制到发送缓冲区
   └── SDMA::wait()             // 等待本地复制完成

3. 全局同步
   └── ShmemDeviceBarrierAll()  // 确保所有 rank 的发送缓冲区已准备好

4. 远程写入
   ├── ShmemPtr() 获取远程 recv 缓冲区地址
   ├── TPUT_SDMA(remoteRecvG, sendG)  // 异步写入远程 rank
   └── SDMA::wait()                    // 等待远程传输完成

5. 同步与完成
   ├── ShmemDeviceQuiet()       // 确保本 PE 的所有远程操作完成
   └── ShmemDeviceBarrierAll()  // 全局同步

6. 结果输出
   ├── TPUT_SDMA(dstG, recvG)  // 将接收到的数据复制到输出
   └── SDMA::wait()
```

### 数据流图

```
Rank i:
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  srcG ──SDMA──> sendG ──SDMA──> [Remote Rank (i-1)] recvG  │
│                                                             │
│  [From Rank (i+1)] sendG ──SDMA──> recvG ──SDMA──> dstG    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Host 端执行流程

```
1. 进程创建
   └── fork() 创建 n_ranks 个子进程

2. 每个子进程独立执行
   ├── ShmemSetConfStoreTls()    // 初始化 TLS 配置
   ├── aclInit / aclrtSetDevice  // 初始化 ACL 运行时
   ├── ShmemInitFromEnv()        // 初始化 shmem 对称堆
   ├── 分配设备内存和主机内存
   ├── 初始化输入数据（input[i] = i + rank_id * 10000）
   ├── ShmemMalloc()             // 分配对称堆内存
   ├── ShmemBarrierAll()         // 全局同步
   ├── 启动内核执行
   ├── ShmemBarrierAll()         // 内核执行后同步
   └── 验证结果（期望值 = i + next_rank * 10000）

3. 父进程等待所有子进程完成
   └── waitpid() 收集退出状态
```

## 测试用例列表

### 1D 向量测试

| 测试名称 | 数据类型 | 元素数量 | Rank 数 | 设备数 |
|----------|----------|----------|---------|--------|
| Vec_FloatSmall | float | 256 | 4 | 4 |
| Vec_Int32Large | int32_t | 4096 | 2 | 2 |
| Vec_Uint8Small | uint8_t | 512 | 8 | 8 |

### 2D 形状测试

| 测试名称 | 数据类型 | 形状 (rows × cols) | Rank 数 | 设备数 |
|----------|----------|-------------------|---------|--------|
| Shape2D_Float16x16 | float | 16 × 16 | 2 | 2 |
| Shape2D_Float8x32 | float | 8 × 32 | 2 | 2 |
| Shape2D_Int32_4x64 | int32_t | 4 × 64 | 2 | 2 |

## 验证逻辑

每个 rank 初始化数据为 `input[i] = i + rank_id * 10000`，执行环形 PUT 后：
- Rank 0 应收到 Rank 1 的数据：`output[i] = i + 1 * 10000`
- Rank 1 应收到 Rank 2 的数据：`output[i] = i + 2 * 10000`
- ...
- Rank N-1 应收到 Rank 0 的数据：`output[i] = i + 0 * 10000`

## 文件说明

| 文件 | 说明 |
|------|------|
| `main.cpp` | GTest 测试入口，定义测试用例 |
| `tput_sdma_kernel.cpp` | 内核实现，包含 1D 和 2D 测试的设备代码和主机代码 |
| `CMakeLists.txt` | 构建配置 |
