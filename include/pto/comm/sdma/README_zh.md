# SDMA 类使用指南

## 概述

`SDMA` 类为 PTO 通信中的 SDMA（System DMA）操作提供了高层接口。它封装了初始化和数据传输操作，使得使用 SDMA 进行异步 GM 到 GM 传输变得简单。

## 文件结构

```
include/pto/comm/sdma/
├── sdma.hpp              # SDMA 主类定义及公共接口
├── sdma_impl.hpp         # SDMA 类方法实现（init、wait、test）
├── sdma_host_init.h      # Host 侧初始化函数声明和数据结构
├── sdma_host_init.cpp    # Host 侧初始化函数实现
├── sdma_device_impl.hpp  # Device 侧 SDMA put/get 实现
├── sdma_types.hpp        # SDMA 数据类型定义
└── README.md             # 英文文档
```

## 架构设计

### 调用流程

```
用户代码
  ↓
TPUT_SDMA(dstGlobal, srcGlobal)
  ↓
TPUT_SDMA_IMPL() [TPut_sdma.hpp]
  ↓
sdma::SDMA::put() [sdma.hpp]
  ↓
detail::put() [sdma_device_impl.hpp]
  ↓
sdma_post_send() [sdma_device_impl.hpp]
  ↓
硬件执行 SDMA 传输
```

### 初始化流程

```
Host 侧代码
  ↓
sdma::SDMA::init(attributes) [sdma_impl.hpp]
  ↓
pto_sdma_init(attributes) [sdma_host_init.cpp]
  ↓
1. 创建 AICPU stream
  ↓
2. create_sdma_streams(): 创建 40 个 SDMA streams
  ↓
3. 分配 16KB 共享 workspace 内存
  ↓
4. 将资源信息复制到 device 内存 (H2D)
  ↓
5. run_aicpu_kernel(): 运行 AICPU kernel 初始化 device 侧映射
  ↓
更新全局状态供 device 侧访问
```

## 初始化

在使用 SDMA 操作之前，必须初始化 SDMA 引擎：

```cpp
#include "pto/comm/sdma/sdma.hpp"
#include "pto/comm/sdma/sdma_impl.hpp"

// 初始化 SDMA（通常在程序启动时调用一次）
pto_comm_init_attr_t init_attr = {};
init_attr.my_pe = 0;           // 本地 PE 编号
init_attr.n_pes = 8;           // PE 总数
init_attr.local_mem_size = 1024 * 1024 * 1024;  // 1GB 本地内存

bool success = pto::comm::sdma::SDMA::init(&init_attr);
if (!success) {
    // 处理初始化错误
}
```

`init()` 函数内部调用 `sdma_host_init.cpp` 中的 `pto_tput_sdma_init()` 来设置 SDMA 资源，包括：
- 40 个用于数据传输的 SDMA streams
- 16KB 共享 workspace 内存，用于 AICPU 和 AIV 通信
- 通过 AICPU kernel 进行 device 侧资源映射

## PUT 操作

PUT 执行异步远程写操作（本地 GM → 远程 GM）。

### 使用 GlobalTensor 形状

```cpp
using namespace pto;
using GTensor = GlobalTensor<float, Shape<1,1,1,64,256>, ...>;

GTensor localSrcG(local_data);   // 本地 PE 上的源数据
GTensor remoteDstG(remote_data); // 远程 PE 上的目标

// 异步传输：本地 -> 远程
auto event = comm::sdma::SDMA::put(remoteDstG, localSrcG);

// 在传输进行时执行其他计算...

// 等待传输完成
comm::sdma::SDMA::wait(event);
```

### 使用显式大小

```cpp
// 传输指定字节数
uint64_t transfer_size = numRows * numCols * sizeof(float);
auto event = comm::sdma::SDMA::put(remoteDstG, localSrcG, transfer_size);
comm::sdma::SDMA::wait(event);
```

## GET 操作

GET 执行异步远程读操作（远程 GM → 本地 GM）。

### 使用 GlobalTensor 形状

```cpp
GTensor localDstG(local_data);   // 本地 PE 上的目标
GTensor remoteSrcG(remote_data); // 远程 PE 上的源数据

// 异步传输：远程 -> 本地
auto event = comm::sdma::SDMA::get(localDstG, remoteSrcG);

// 在传输进行时执行其他计算...

// 等待传输完成
comm::sdma::SDMA::wait(event);
```

### 使用显式大小

```cpp
// 传输指定字节数
uint64_t transfer_size = numRows * numCols * sizeof(float);
auto event = comm::sdma::SDMA::get(localDstG, remoteSrcG, transfer_size);
comm::sdma::SDMA::wait(event);
```

## 同步

### 等待完成

```cpp
auto event = comm::sdma::SDMA::put(remoteDstG, localSrcG);
comm::sdma::SDMA::wait(event);  // 阻塞直到传输完成
```

### 测试完成（非阻塞）

```cpp
auto event = comm::sdma::SDMA::put(remoteDstG, localSrcG);
while (!comm::sdma::SDMA::test(event)) {
    // 等待时执行其他工作
    // ...
}
```

## 与 TPUT_SDMA/TGET_SDMA 指令的集成

`TPUT_SDMA` 和 `TGET_SDMA` 指令委托给 `SDMA` 类：

```cpp
// 在 TPut_sdma.hpp 中
template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL SdmaEvent TPUT_SDMA_IMPL(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal)
{
    return sdma::SDMA::put(dstGlobal, srcGlobal);
}

// 在 TGet_sdma.hpp 中
template <typename GlobalDstData, typename GlobalSrcData>
PTO_INTERNAL SdmaEvent TGET_SDMA_IMPL(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal)
{
    return sdma::SDMA::get(dstGlobal, srcGlobal);
}
```

这允许用户使用以下任一方式：
- 直接使用高层 `SDMA` 类 API（`SDMA::put`、`SDMA::get`）
- 使用指令接口（`TPUT_SDMA`、`TGET_SDMA`）

## 约束条件

- 源和目标的元素类型必须匹配
- 源和目标的布局必须匹配
- 元素大小必须是 1、2、4 或 8 字节
- 地址应 32 字节对齐以获得最佳性能
- 使用前必须初始化 SDMA

## 错误处理

- 如果 SDMA 资源不可用，操作返回无效的 `SdmaEvent`（event_id == 0）
- 可以通过测试 `event.event_id == 0` 来检查无效事件

## 数据结构

### sdma_types.hpp

| 结构体 | 描述 |
|--------|------|
| `sdma_config_t` | SDMA 配置参数 |
| `workspace_layout_t` | Workspace 内存布局 |
| `batch_write_flag_info_t` | Flag 同步信息 |
| `batch_write_channel_info_t` | 通道信息 |
| `batch_write_item_t` | SQE（提交队列条目）结构 |

### sdma_host_init.h

| 结构体/函数 | 描述 |
|-------------|------|
| `pto_sdma_op_res_info_t` | SDMA 操作资源信息 |
| `pto_host_stream_info_t` | Host 流信息 |
| `pto_sdma_init()` | Host 侧初始化函数 |
| `pto_sdma_finalize()` | 释放 SDMA 资源 |

## Device 侧实现

`sdma_device_impl.hpp` 实现了完整的 SDMA put 功能：

| 函数 | 描述 |
|------|------|
| `sdma_post_send()` | 主函数，协调整个 SDMA 传输流程 |
| `add_one_memcpy_sqe()` | 构建 SQE（提交队列条目） |
| `init_sdma_config()` | 初始化配置参数 |
| `prepare_workspace()` | 准备 workspace 内存布局 |
| `submit_data_transfer_sqes()` | 提交数据传输 SQE |
| `submit_flag_transfer_sqes()` | 提交 flag 同步 SQE |
| `flush_cache_and_ring_doorbell()` | 刷新缓存并敲 Doorbell |
| `poll_for_completion()` | 轮询等待传输完成 |
| `put()` | 公共 put 接口（在 detail 命名空间下） |
| `get()` | 公共 get 接口（在 detail 命名空间下） |

## 重要注意事项

1. **Host/Device 分离**：
   - `init()` 方法应在 host 侧调用
   - `put()`/`get()` 方法在 device 侧调用

2. **初始化顺序**：
   - 必须先调用 `SDMA::init()` 初始化 SDMA
   - 然后才能使用 `SDMA::put()`/`SDMA::get()` 进行传输

3. **命名空间**：
   - Device 侧实现位于 `pto::comm::sdma::detail` 命名空间
   - SDMA 类位于 `pto::comm::sdma` 命名空间

## 编译说明

`sdma_host_init.cpp` 需要链接以下库：

- `libruntime.so`：用于 `rtsStreamCreate` 和 `rtGetDeviceInfo`
- `libopapi.so`：用于 `aclnnSdmaMap` AICPU kernel
- ACL 运行时库：用于 stream 和内存管理

编译时需要添加以下链接选项：

```bash
-ldl -lascendcl
```

## 实现说明

- SDMA 类使用静态标志跟踪初始化状态
- 通道选择使用基于地址的简单轮询算法
- 实际的 SDMA API 调用标有 TODO 注释，需要根据 Ascend SDK 实现
