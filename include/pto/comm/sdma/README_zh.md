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

---

## `sdma_post_send` 函数实现详解

本节详细说明 `sdma_post_send` 函数及其调用的子函数实现逻辑。

### 调用层次结构

```
TPUT_SDMA (TPut_sdma.hpp)
    └── sdma::SDMA::put (sdma.hpp)
            └── detail::put (sdma_device_impl.hpp)
                    └── sdma_write
                            └── sdma_post_send  ← 核心实现
                                    ├── init_sdma_config
                                    ├── prepare_workspace
                                    ├── init_sq_tail_array
                                    ├── submit_data_transfer_sqes
                                    │       └── add_one_memcpy_sqe
                                    ├── submit_flag_transfer_sqes
                                    │       └── add_one_memcpy_sqe
                                    ├── flush_cache_and_ring_doorbell
                                    └── poll_for_completion
```

### 核心数据结构

#### 1. 全局状态结构 `pto_comm_global_state_t`

```cpp
struct pto_comm_global_state_t {
    uint64_t sdma_workspace_addr;    // SDMA 工作空间地址
    uint64_t sdma_flag_addr;         // SDMA 同步标志地址
    uint64_t sdma_op_res_info_addr;  // SDMA 操作资源信息地址
};
```

#### 2. SDMA 配置结构 `sdma_config_t`

```cpp
struct sdma_config_t {
    uint32_t queue_num;           // 每个 core 使用的队列数（默认 1）
    uint64_t block_bytes;         // 每个 SQE 传输的数据块大小（1MB）
    uint64_t per_core_bytes;      // 当前 core 需传输的总字节数
    uint64_t comm_block_offset;   // 当前 core 在数据中的起始偏移
    uint32_t iter_num;            // 需要提交的 SQE 数量
};
```

#### 3. 工作空间布局 `workspace_layout_t`

```cpp
struct workspace_layout_t {
    __gm__ uint8_t* send_workspace;        // 本地发送标志区
    __gm__ uint8_t* recv_workspace;        // 本地接收标志区
    __gm__ uint8_t* remote_recv_workspace; // 远程接收标志区
};
```

#### 4. 通道信息 `batch_write_channel_info_t`

```cpp
struct batch_write_channel_info_t {
    uint32_t sq_head;      // SQ 头指针
    uint32_t sq_tail;      // SQ 尾指针
    uint32_t sq_depth;     // SQ 深度
    uint64_t sq_base;      // SQ 基地址
    uint64_t sq_reg_base;  // SQ 寄存器基地址（门铃）
    uint32_t stream_id;    // 流 ID
    // ...
};
```

### `sdma_post_send` 主函数流程

```cpp
void sdma_post_send(__gm__ uint8_t* recv_buffer,
                    __gm__ uint8_t* send_buffer,
                    uint64_t opcode,
                    uint64_t message_len)
```

#### 执行流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          sdma_post_send                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Step 1: 获取全局状态                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  device_state = pto_comm_get_state()                            │   │
│  │  context_gm   = device_state->sdma_workspace_addr               │   │
│  │  flag_addr    = device_state->sdma_flag_addr                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               ↓                                         │
│  Step 2: 初始化 UB 临时缓冲区                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  tmp_buf.InitBuffer(UB_ALIGN_SIZE * 2)                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               ↓                                         │
│  Step 3: 获取当前 core 信息                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  block_idx = GetBlockIdx()                                      │   │
│  │  comm_block_dim = GetBlockNum() * GetSubBlockNum()              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               ↓                                         │
│  Step 4: 初始化 SDMA 配置 → init_sdma_config()                          │
│                               ↓                                         │
│  Step 5: 准备工作空间 → prepare_workspace()                             │
│                               ↓                                         │
│  Step 6: 初始化 SQ 尾指针数组 → init_sq_tail_array()                    │
│                               ↓                                         │
│  Step 7: 提交数据传输 SQE → submit_data_transfer_sqes()                 │
│                               ↓                                         │
│  Step 8: 提交标志传输 SQE → submit_flag_transfer_sqes()                 │
│                               ↓                                         │
│  Step 9: 刷缓存并敲门铃 → flush_cache_and_ring_doorbell()               │
│                               ↓                                         │
│  Step 10: 轮询等待完成 → poll_for_completion()                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 各子函数详解

#### 1. `init_sdma_config` - 初始化 SDMA 配置

**功能**：根据消息长度和 core 数量，计算每个 core 的传输任务分配。

```cpp
bool init_sdma_config(__gm__ uint8_t* context_gm,
                      uint64_t message_len,
                      uint32_t block_idx,
                      uint32_t comm_block_dim,
                      sdma_config_t& config,
                      TBuf& tmp_buf)
```

**计算逻辑**：

```
输入: message_len = 10MB, comm_block_dim = 4 (4个core)

1. 基本分配:
   per_core_bytes = 10MB / 4 = 2.5MB

2. 余数处理 (如果 message_len % comm_block_dim != 0):
   前 extra_bytes 个 core 各多传 1 字节

3. SQE 数量计算:
   block_bytes = 1MB (每个 SQE 最大传输量)
   iter_num = ceil(per_core_bytes / block_bytes) = ceil(2.5) = 3

4. 偏移计算:
   block_idx=0: offset = 0
   block_idx=1: offset = 2.5MB
   block_idx=2: offset = 5MB
   block_idx=3: offset = 7.5MB
```

**数据分片示意图**：

```
message_len = 10MB, 4 cores

Core 0: [0MB ─────── 2.5MB]     iter_num=3: [0-1MB][1-2MB][2-2.5MB]
Core 1: [2.5MB ───── 5MB]       iter_num=3: [2.5-3.5MB][3.5-4.5MB][4.5-5MB]
Core 2: [5MB ─────── 7.5MB]     iter_num=3: [5-6MB][6-7MB][7-7.5MB]
Core 3: [7.5MB ───── 10MB]      iter_num=3: [7.5-8.5MB][8.5-9.5MB][9.5-10MB]
```

---

#### 2. `prepare_workspace` - 准备工作空间

**功能**：为每个 core 分配同步标志的发送/接收区域。

```cpp
void prepare_workspace(__gm__ uint8_t* workspace,
                       __gm__ uint8_t* flag_addr,
                       const sdma_config_t& config,
                       workspace_layout_t& layout,
                       uint32_t block_idx,
                       uint32_t my_pe,
                       TBuf& tmp_buf)
```

**工作空间布局**：

```
workspace (context_gm + header_size):
┌──────────────────────────────────────────────────────────────┐
│  send_workspace (8B)  │  Core 0 recv │  Core 1 recv │  ...  │
│    (公共发送标志)      │   (8B)       │   (8B)       │       │
└──────────────────────────────────────────────────────────────┘
                        ↑
                        └── 每个 core 的接收标志区

flag_addr (远程标志区):
┌──────────────────────────────────────────────────────────────┐
│  PE 0 flags  │  PE 1 flags  │  PE 2 flags  │  ...           │
│  (40*8B)     │  (40*8B)     │  (40*8B)     │                 │
└──────────────────────────────────────────────────────────────┘
```

**初始化操作**：
- 设置 `send_workspace` 值为 `queue_num`（作为完成标志）

---

#### 3. `init_sq_tail_array` - 初始化 SQ 尾指针

**功能**：从通道信息中读取当前的 SQ tail 值。

```cpp
void init_sq_tail_array(__gm__ batch_write_channel_info_t* channel_info,
                        uint32_t queue_num,
                        uint32_t* sq_tail,
                        TBuf& tmp_buf)
```

**操作**：
- 遍历每个队列，读取 `channel_info->sq_tail` 到本地数组
- 后续 SQE 提交时基于此值递增

---

#### 4. `add_one_memcpy_sqe` - 构建单个 SDMA SQE

**功能**：在 SQ 中构建一个 SDMA 内存拷贝描述符。

```cpp
void add_one_memcpy_sqe(__gm__ batch_write_channel_info_t* channel_info,
                        __gm__ uint8_t* src,
                        __gm__ uint8_t* dst,
                        uint64_t opcode,
                        uint32_t length,
                        uint32_t sq_tail,
                        uint32_t task_id)
```

**SQE 结构填充**：

```
batch_write_item_t (SDMA SQE):
┌────────────────────────────────────────────────────────────┐
│  type        = RT_STARS_SQE_TYPE_SDMA                      │
│  blockDim    = 0                                           │
│  rtStreamId  = channel_info->stream_id                     │
│  taskId      = task_id (用于跟踪)                          │
│  opcode      = 0 (memcpy)                                  │
│  length      = 传输字节数                                   │
│  srcAddrLow  = src 地址低 32 位                             │
│  srcAddrHigh = src 地址高 32 位                             │
│  dstAddrLow  = dst 地址低 32 位                             │
│  dstAddrHigh = dst 地址高 32 位                             │
│  sssv/dssv   = 1 (地址有效)                                 │
│  sns/dns     = 1 (非安全)                                   │
│  qos         = 6 (服务质量)                                 │
│  linkType    = 255 (无链接)                                 │
└────────────────────────────────────────────────────────────┘
```

**SQ 环形缓冲区**：

```
SQ (Submission Queue):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ SQE │ SQE │ SQE │     │     │     │     │     │
│  0  │  1  │  2  │     │     │     │     │     │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  ↑                       ↑
  head                   tail (新 SQE 写入位置)

位置计算: sqe_idx = sq_tail % sq_depth
```

---

#### 5. `submit_data_transfer_sqes` - 提交数据传输 SQE

**功能**：为当前 core 的数据传输任务提交所有 SQE。

```cpp
void submit_data_transfer_sqes(
    __gm__ batch_write_channel_info_t* channel_info,
    __gm__ uint8_t* send_buffer,
    __gm__ uint8_t* recv_buffer,
    uint32_t opcode,
    const sdma_config_t& config,
    uint32_t* sq_tail,
    TBuf& tmp_buf)
```

**执行逻辑**：

```
for idx in 0..iter_num:
    queue_idx = idx % queue_num  // 轮询使用多个队列
    
    // 计算本次传输大小
    if idx == iter_num - 1:
        transfer_bytes = per_core_bytes - idx * block_bytes  // 最后一块可能不足 1MB
    else:
        transfer_bytes = block_bytes  // 1MB
    
    // 计算源/目标地址
    src_addr = send_buffer + comm_block_offset + idx * block_bytes
    dst_addr = recv_buffer + comm_block_offset + idx * block_bytes
    
    // 提交 SQE
    add_one_memcpy_sqe(channel_info, src_addr, dst_addr, ...)
    sq_tail[queue_idx]++
```

**示意图**：

```
Core 0 提交 3 个 SQE:

SQE 0: send[0MB-1MB]    → recv[0MB-1MB]
SQE 1: send[1MB-2MB]    → recv[1MB-2MB]
SQE 2: send[2MB-2.5MB]  → recv[2MB-2.5MB]  (最后一块)
```

---

#### 6. `submit_flag_transfer_sqes` - 提交标志传输 SQE

**功能**：提交同步标志的传输 SQE，用于通知远端数据传输完成。

```cpp
void submit_flag_transfer_sqes(
    __gm__ batch_write_channel_info_t* channel_info,
    const workspace_layout_t& layout,
    const sdma_config_t& config,
    uint32_t* sq_tail,
    TBuf& tmp_buf)
```

**执行逻辑**：
- 将本地 `send_workspace` 的标志值传输到远端 `remote_recv_workspace`
- 标志值非零表示传输完成

```
send_workspace (本地)  ──SDMA──>  remote_recv_workspace (远端)
     [queue_num]                      [queue_num]
```

---

#### 7. `flush_cache_and_ring_doorbell` - 刷缓存并敲门铃

**功能**：确保 SQE 写入 HBM，然后通知硬件开始处理。

```cpp
void flush_cache_and_ring_doorbell(
    __gm__ batch_write_channel_info_t* channel_info,
    const sdma_config_t& config,
    uint32_t* sq_tail,
    TBuf& tmp_buf)
```

**执行步骤**：

```
1. 刷新数据缓存 (DCCI)
   ┌─────────────────────────────────────────────┐
   │  DataCacheCleanAndInvalid(sq_base, size)   │
   │  确保所有 SQE 从 cache 写入 HBM             │
   └─────────────────────────────────────────────┘
                        ↓
2. 敲门铃 (Ring Doorbell)
   ┌─────────────────────────────────────────────┐
   │  写入 sq_reg_base + 8 = sq_tail            │
   │  通知硬件有新的 SQE 待处理                  │
   └─────────────────────────────────────────────┘
```

**门铃寄存器布局**：

```
sq_reg_base:
┌──────────┬──────────┬──────────┬──────────┐
│  reg[0]  │  reg[1]  │  reg[2]  │  ...     │
│  (4B)    │  (4B)    │  tail    │          │
└──────────┴──────────┴──────────┴──────────┘
                       ↑
                       offset = 8 (第三个 uint32)
```

---

#### 8. `poll_for_completion` - 轮询等待完成

**功能**：等待远端发送的完成标志。

```cpp
bool poll_for_completion(
    __gm__ batch_write_channel_info_t* channel_info,
    const workspace_layout_t& layout,
    const sdma_config_t& config,
    uint32_t* sq_tail,
    TBuf& tmp_buf)
```

**执行逻辑**：

```
for each queue:
    times = 0
    while send_value == 0 && times < max_times:
        // 从远端拷贝标志到本地
        copy_gm_to_gm(local_recv, remote_recv, 1)
        send_value = get_value(local_recv)
        times++
    
    // 清理标志区
    set_value(remote_recv, 0)
    set_value(local_recv, 0)
    
    // 更新通道 tail 值
    set_value(channel_info->sq_tail, sq_tail[queue])
```

**完成检测机制**：

```
发送端                              接收端
────────                           ────────
1. 提交数据 SQE                    
2. 提交标志 SQE ─────────────────> 标志区被写入非零值
3. 敲门铃                          
4. 轮询等待... <───────────────── 检测到标志非零，传输完成
```

---

### 完整数据流时序图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             SDMA 传输时序                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  AI Core                    SDMA Engine                     Remote PE       │
│  ────────                   ───────────                     ─────────       │
│     │                           │                               │           │
│     │  1. 初始化配置            │                               │           │
│     │─────────────────>         │                               │           │
│     │                           │                               │           │
│     │  2. 写入数据 SQE          │                               │           │
│     │─────────────────>         │                               │           │
│     │                           │                               │           │
│     │  3. 写入标志 SQE          │                               │           │
│     │─────────────────>         │                               │           │
│     │                           │                               │           │
│     │  4. DCCI 刷缓存           │                               │           │
│     │─────────────────>         │                               │           │
│     │                           │                               │           │
│     │  5. 敲门铃                │                               │           │
│     │─────────────────>         │                               │           │
│     │                           │  6. 执行数据 DMA              │           │
│     │                           │──────────────────────────────>│           │
│     │                           │                               │           │
│     │                           │  7. 执行标志 DMA              │           │
│     │                           │──────────────────────────────>│           │
│     │                           │                               │           │
│     │  8. 轮询标志区            │                               │           │
│     │<─ ─ ─ ─ ─ ─ ─ ─ ─         │                               │           │
│     │     (等待非零)            │                               │           │
│     │                           │                               │           │
│     │  9. 检测到完成            │                               │           │
│     │<──────────────────────────│───────────────────────────────│           │
│     │                           │                               │           │
│     │  10. 清理标志，更新 tail  │                               │           │
│     │─────────────────>         │                               │           │
│     │                           │                               │           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 辅助函数说明

| 函数 | 功能 |
|------|------|
| `dcci_cacheline(addr)` | 无效化单个 cache line |
| `copy_gm_to_gm<T>(dst, src, size, buf)` | GM 到 GM 拷贝（经 UB 中转） |
| `set_value<T>(addr, buf, value)` | 写入单个值到 GM |
| `get_value<T>(addr, buf)` | 从 GM 读取单个值 |
| `pto_comm_get_state()` | 获取全局通信状态指针 |
| `pto_comm_select_sdma_channel(idx)` | 选择 SDMA 通道（轮询） |

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
