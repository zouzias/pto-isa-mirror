# TGet_sdma 测试用例

## 概述

本测试用例验证 `TGET_SDMA` 指令的正确性。`TGET_SDMA` 是一个异步远程读取操作，使用 SDMA（System DMA）引擎直接将数据从远程 NPU 的 GM（Global Memory）传输到本地 GM，无需 UB（Unified Buffer）中转。

## 与 TGET 的区别

| 特性 | TGET | TGET_SDMA |
|------|------|-----------|
| 执行方式 | 同步 | 异步 |
| 数据路径 | GM → UB → GM | GM → GM（直接传输）|
| 需要 UB Tile | 是 | 否 |
| 计算通信重叠 | 否 | 是 |
| 返回值 | void | SdmaEvent |
| 同步方式 | 隐式（指令完成即同步）| 显式（需调用 `SDMA::wait()`）|

## 与 TPUT_SDMA 的对比

| 操作 | 数据方向 | 说明 |
|------|----------|------|
| TPUT_SDMA | 本地 → 远程 | 主动推送数据到远程 PE |
| TGET_SDMA | 远程 → 本地 | 主动从远程 PE 拉取数据 |

## 测试模式

采用**环形通信模式（Ring Communication）**：每个 rank 从下一个 rank 读取数据。

```
Rank 0 ← Rank 1
Rank 1 ← Rank 2
Rank 2 ← Rank 3
...
Rank N-1 ← Rank 0
```

## 实现逻辑

### 内核执行流程

```
1. 初始化阶段
   ├── 创建 GlobalTensor（srcG, dstG, sendG, recvG）
   └── 计算源 rank（next_rank = (my_rank + 1) % nranks）

2. 本地数据准备
   ├── TPUT_SDMA(sendG, srcG)  // 将源数据复制到发送缓冲区
   └── SDMA::wait()             // 等待本地复制完成

3. 全局同步
   └── ShmemDeviceBarrierAll()  // 确保所有 rank 的发送缓冲区已准备好

4. 远程读取
   ├── ShmemPtr() 获取远程 send 缓冲区地址
   ├── TGET_SDMA(recvG, remoteSendG)  // 异步从远程 rank 读取
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
│  srcG ──SDMA──> sendG  (供其他 rank 读取)                   │
│                                                             │
│  [Remote Rank (i+1)] sendG ──SDMA──> recvG ──SDMA──> dstG  │
│                          ↑                                  │
│                     TGET_SDMA 读取                          │
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
| Vec_FloatSmall | float | 256 | 2 | 2 |
| Vec_Int32Large | int32_t | 4096 | 2 | 2 |
| Vec_Uint8Small | uint8_t | 512 | 2 | 2 |

### 2D 形状测试

| 测试名称 | 数据类型 | 形状 (rows × cols) | Rank 数 | 设备数 |
|----------|----------|-------------------|---------|--------|
| Shape2D_Float16x16 | float | 16 × 16 | 2 | 2 |
| Shape2D_Float8x32 | float | 8 × 32 | 2 | 2 |
| Shape2D_Int32_4x64 | int32_t | 4 × 64 | 2 | 2 |

## 验证逻辑

每个 rank 初始化数据为 `input[i] = i + rank_id * 10000`，执行环形 GET 后：
- Rank 0 从 Rank 1 读取：`output[i] = i + 1 * 10000`
- Rank 1 从 Rank 2 读取：`output[i] = i + 2 * 10000`
- ...
- Rank N-1 从 Rank 0 读取：`output[i] = i + 0 * 10000`

## 文件说明

| 文件 | 说明 |
|------|------|
| `main.cpp` | GTest 测试入口，定义测试用例 |
| `tget_sdma_kernel.cpp` | 内核实现，包含 1D 和 2D 测试的设备代码和主机代码 |
| `CMakeLists.txt` | 构建配置 |

## 使用场景

`TGET_SDMA` 适用于以下场景：

1. **数据预取（Prefetch）**：在计算当前数据时，异步预取下一批需要的远程数据
2. **Pull 模式通信**：由数据消费方主动拉取数据，而非由生产方推送
3. **计算通信重叠**：利用异步特性，在等待数据传输时执行其他计算任务

```cpp
// 示例：计算与通信重叠
auto get_event = TGET_SDMA(localBuf, remoteSrc);

// 在等待远程数据时执行本地计算
TLOAD(localTile, localData);
TADD(resultTile, localTile, otherTile);
TSTORE(outputData, resultTile);

// 等待远程数据到达后再使用
SDMA::wait(get_event);
// 现在可以使用 localBuf 中的数据
```
