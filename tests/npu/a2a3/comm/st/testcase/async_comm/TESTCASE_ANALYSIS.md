# async_comm 测试用例详细解析

> 文件路径：`tests/npu/a2a3/comm/st/testcase/async_comm/`
> 原始 kernel：`D:\cann\async_comm-pto.cpp`（由 PTOAS 编译器自动生成，**不得修改**）

---

## 目录

1. [整体架构](#1-整体架构)
2. [被测 Kernel 解析](#2-被测-kernel-解析)
3. [Test 1 — SmokeTest_Float128](#3-test-1--smoketest_float128)
4. [Test 2 — ZeroData_Float128](#4-test-2--zerodata_float128)
5. [Test 3 — NegativeData_Float128](#5-test-3--negativedata_float128)
6. [Test 4 — MultiLaunch_Float128](#6-test-4--multilaunch_float128)
7. [Test 5 — CrossRank_Float128](#7-test-5--crossrank_float128)
8. [Test 6 — RootPut_4Ranks_Float128](#8-test-6--rootput_4ranks_float128)
9. [已发现的 Kernel Bug](#9-已发现的-kernel-bug)
10. [运行命令](#10-运行命令)

---

## 1. 整体架构

```
main.cpp                    ← GTest 入口，定义 5 个 TEST 宏
async_comm_kernel.cpp       ← 被测 kernel + 5 个 host runner
async_comm_kernel.h         ← 5 个 runner 的函数声明
```

测试框架由 `run_st.py` 驱动，每次运行固定跑三轮 `mpirun`：

| 轮次 | 进程数 | GTest filter | 命中本套用例 |
|---|---|---|---|
| 1 | 2 | `*-*4Ranks*:*8Ranks*` | **全部 5 个**（用例名不含后缀） |
| 2 | 4 | `*4Ranks*` | 无 |
| 3 | 8 | `*8Ranks*` | 无 |

Test 1-4 只在 MPI rank 0 上真正执行（rank 1 直接 `SUCCEED()`）；  
Test 5 两个 rank 都完整参与。

---

## 2. 被测 Kernel 解析

**文件**：`async_comm-pto.cpp`（PTOAS 自动生成，原样复制进测试）

```cpp
__global__ AICORE void async_comm_kernel(
    __gm__ float  *v1,   // 目标缓冲区，128 个 float
    __gm__ float  *v2,   // 源缓冲区，  128 个 float
    __gm__ int8_t *v3    // SDMA workspace（主机预分配）
)
```

### Kernel 内部流程

```
① 常量准备
   v4 = UINT32_MAX      → 传输量上限（不限）
   v5 = 0u              → SDMA channel 编号
   v6 = 0               → 用于清零 scratch buffer
   v7 = 256, v8 = 1     → ⚠️ 声明但未使用（疑似 PTOAS 遗漏，见 Bug 分析）

② 申请 AICore 片上 scratch buffer（UB 内存）
   Tile v9(256 bytes)
   TASSIGN(v9, 0)       → 清零，用于 BuildAsyncSession 的初始化数据

③ 建立 SDMA 异步会话
   AsyncSession v11
   SdmaBaseConfig = {block_bytes=32768, offset=0, channel_count=1}
   BuildAsyncSession(v9, v10=v3, v11, channel=0, config, UINT32_MAX)
   → 在 workspace 里初始化 SQ 指针和 8 个 completion flag 槽

④ 把裸指针包装成带形状描述的 GlobalTensor
   v15 = GlobalTensor(v1, Shape<1,1,1,1,128>, Stride<128,128,128,128,1>)
   v18 = GlobalTensor(v2, ...)
   v22 = GlobalTensor(v2, ...)  ← v2 的第二份描述（TGET 用）
   v25 = GlobalTensor(v1, ...)  ← v1 的第二份描述

⑤ 发起第一次异步 DMA：TPUT_ASYNC
   v19 = TPUT_ASYNC(dst=v15→v1, src=v18→v2, session=v11)
   → 把 DMA 命令写入 SDMA Send Queue，立即返回（AICore 不等待）

⑥ 发起第二次异步 DMA：TGET_ASYNC
   v26 = TGET_ASYNC(src=v22→v2, dst=v25→v1, session=v11)
   → 同上，再写一条 DMA 命令到 SQ

⑦ return  ← ⚠️ 没有 Wait()，见 Bug 分析
```

### SDMA 工作原理简图

```
AICore                          SDMA 硬件
  │                                │
  │── BuildAsyncSession ─────────→ │  初始化 SQ/completion 槽
  │── TPUT_ASYNC ─────────────────→│  写 SQE 到 Send Queue
  │── TGET_ASYNC ─────────────────→│  写 SQE 到 Send Queue
  │── return ──────────────────    │
  │（AICore 退出）                 │── 独立执行 DMA: v2 → v1
  │                                │── 独立执行 DMA: v2 → v1
```

---

## 3. Test 1 — SmokeTest_Float128

**目的**：最基础的冒烟测试，验证 kernel 能跑通，SDMA 能完成数据搬运。

### GTest 入口（main.cpp）

```cpp
TEST(AsyncComm, SmokeTest_Float128)
{
    if (CommMpiRank() != 0) { SUCCEED(); return; }  // rank 1 直接跳过
    int device_id = CommMpiRank() % 1;               // device_id = 0
    ASSERT_TRUE(RunAsyncCommTest(1, 1, 0, device_id));
}
```

### Host Runner 流程（RunAsyncCommTest）

```
1. aclInit(nullptr)               → 初始化 CANN ACL 运行时
2. aclrtSetDevice(0)              → 绑定 device 0
3. rtStreamCreate(&stream)        → 创建 AICore 执行流
4. aclrtMalloc(v1_dev, 512B)      → 在 device 上分配目标缓冲区
5. aclrtMalloc(v2_dev, 512B)      → 在 device 上分配源缓冲区
6. Host → Device memcpy:
     v1_host = [0, 1, 2, ..., 127]      → v1_dev
     v2_host = [1000, 1001, ..., 1127]  → v2_dev
7. SdmaWorkspaceManager::Init()   → 分配 SDMA workspace（SQ + completion 槽）
8. async_comm_kernel<<<1, stream>>>(v1_dev, v2_dev, ws)
                                  → 启动 kernel（1 个 AICore block）
9. aclrtSynchronizeStream(stream) → 等待 AICore kernel 完成
10. Device → Host memcpy: v1_dev → result[]
11. 验证: result[i] == v2_host[i] == i + 1000
12. 清理: Finalize, Free, Destroy, Reset, aclFinalize
```

### 输入/输出数据

| 缓冲区 | 初始值 | 预期结果 |
|---|---|---|
| `v1` | `[0, 1, 2, ..., 127]` | `[1000, 1001, ..., 1127]` |
| `v2` | `[1000, 1001, ..., 1127]` | 不变（只作为源） |

### 验证逻辑

两次 DMA（TPUT + TGET）都把 v2 写入 v1，所以 v1 最终应与 v2 完全相同。用 `result[i] != v2_host[i]` 逐元素检查，发现第一个不匹配就报错退出。

---

## 4. Test 2 — ZeroData_Float128

**目的**：验证 SDMA 能正确把全零数据写入非零目标，排除"目标从未被写入"类 bug。

### GTest 入口

```cpp
TEST(AsyncComm, ZeroData_Float128)
{
    if (CommMpiRank() != 0) { SUCCEED(); return; }
    int device_id = CommMpiRank() % 1;
    ASSERT_TRUE(RunZeroDataTest(device_id));
}
```

### Host Runner 流程（RunZeroDataTest）

使用公共帮助结构 `AsyncCommCtx` 封装 ACL 初始化：

```
v1_host = [1, 2, 3, ..., 128]    ← 非零初始值（刻意与全零不同）
v2_host = [0, 0, 0, ..., 0]      ← 源数据全零

AsyncCommCtxInit()
  └─ aclInit / aclrtSetDevice / rtStreamCreate
  └─ aclrtMalloc v1_dev, v2_dev
  └─ memcpy v1_host→v1_dev, v2_host→v2_dev
  └─ SdmaWorkspaceManager::Init

async_comm_kernel<<<1, stream>>>(v1_dev, v2_dev, ws_dev)
aclrtSynchronizeStream(stream)

验证: result[i] == 0.0f  对所有 i

AsyncCommCtxDestroy()
  └─ Finalize, Free, Destroy, Reset, aclFinalize
```

### 测试意图

如果 SDMA 有"跳过写入全零数据"的优化 bug（某些 DMA 引擎会省略全零传输），或者目标内存根本没被写入，result 将仍是 `[1, 2, ..., 128]` 而非 `[0, 0, ..., 0]`，测试即失败。

---

## 5. Test 3 — NegativeData_Float128

**目的**：验证 SDMA 正确传输负数浮点值，排除符号位截断或无符号传输 bug。

### GTest 入口

```cpp
TEST(AsyncComm, NegativeData_Float128)
{
    if (CommMpiRank() != 0) { SUCCEED(); return; }
    int device_id = CommMpiRank() % 1;
    ASSERT_TRUE(RunNegativeDataTest(device_id));
}
```

### Host Runner 流程（RunNegativeDataTest）

```
v1_host = [0, 1, 2, ..., 127]         ← 正数初始值
v2_host = [-1, -2, -3, ..., -128]     ← 负数源数据

AsyncCommCtxInit()  →  memcpy  →  kernel  →  sync

验证: result[i] == v2_host[i] == -(i+1)
```

### 浮点负数的内存表示

IEEE 754 单精度浮点数最高位是符号位。`-1.0f` 的二进制是 `0xBF800000`，`-128.0f` 是 `0xC3000000`。如果 DMA 把数据当无符号整数处理、丢弃高位、或只传低 24 位，符号位会丢失，result 变成正数，测试失败。

---

## 6. Test 4 — MultiLaunch_Float128

**目的**：验证同一设备上连续多次启动 kernel，SDMA workspace（completion flag 槽、sq_tail）在每次启动之间正确重置，不发生状态泄漏。

### GTest 入口

```cpp
TEST(AsyncComm, MultiLaunch_Float128)
{
    if (CommMpiRank() != 0) { SUCCEED(); return; }
    int device_id = CommMpiRank() % 1;
    ASSERT_TRUE(RunMultiLaunchTest(device_id));
}
```

### Host Runner 流程（RunMultiLaunchTest）

```
aclInit / aclrtSetDevice / rtStreamCreate
aclrtMalloc v1_dev, v2_dev   ← 复用同一块 device 内存，共 3 轮

for round in [0, 1, 2]:
    ① SdmaWorkspaceManager sdmaMgr   ← 每轮新建，关键！
       sdmaMgr.Init()                 → 分配新 workspace，completion 槽全为 0
       ws_dev = sdmaMgr.GetWorkspaceAddr()

    ② v2_host[i] = i + 5000 + round*1000
       第 0 轮: [5000, 5001, ..., 5127]
       第 1 轮: [6000, 6001, ..., 6127]
       第 2 轮: [7000, 7001, ..., 7127]

    ③ memcpy v2_host → v2_dev

    ④ async_comm_kernel<<<1, stream>>>(v1_dev, v2_dev, ws_dev)
    ⑤ aclrtSynchronizeStream(stream)

    ⑥ memcpy v1_dev → result[]
    ⑦ 验证 result[i] == v2_host[i]

    ⑧ sdmaMgr.Finalize()   ← 释放本轮 workspace

aclrtFree / rtStreamDestroy / aclrtResetDevice / aclFinalize
```

### 为什么每轮必须重建 SdmaWorkspaceManager？

SDMA workspace 内存布局：

```
┌─────────────────────────────────────────────────┐
│ BatchWriteFlagInfo (64B)                        │
├─────────────────────────────────────────────────┤
│ BatchWriteChannelInfo × 40 (每个 64B)           │
│   ├─ sq_head    ← SDMA 读指针                   │
│   └─ sq_tail    ← AICore 写指针（记录进度）     │
├─────────────────────────────────────────────────┤
│ SdmaEventRecord × 8 (completion flag 槽)        │
│   ├─ flag       ← 0=pending, 非0=完成           │
│   ├─ sq_tail    ← 完成时要提交的队列尾           │
│   └─ channel_info                               │
└─────────────────────────────────────────────────┘
```

**第 0 轮结束后的状态**：
- `sq_tail = 2`（提交了 2 条 DMA 命令）
- `slots[0].flag = 非零`（DMA 0 已完成）
- `slots[1].flag = 非零`（DMA 1 已完成）

**如果第 1 轮复用同一 workspace**：
- `BuildAsyncSession` 读到 `sq_tail=2`，新 DMA 从位置 2 开始
- `next_event_id` 从 0 重新计数，分配到 slot 0
- `slot 0.flag` 已是非零 → `Wait()` 立即返回（误判为完成）
- SDMA 收到新命令但 AICore 已认为完成，数据未更新
- `v1` 仍是 `[5000..]`，期望 `[6000..]` → **测试失败**

**每轮重建 SdmaWorkspaceManager 的效果**：
- `sdmaMgr.Init()` 分配全新的设备内存并清零
- `sq_tail=0`，所有 `flag=0`，状态完全干净
- 下一轮 DMA 正确提交并执行

---

## 7. Test 5 — CrossRank_Float128

**目的**：不修改原始 kernel，通过传入另一个 rank 的 RDMA 远端地址作为 `v1`，验证 SDMA 能跨设备（device 6 → device 7）搬运数据。

### GTest 入口

```cpp
TEST(AsyncComm, CrossRank_Float128)
{
    // n_ranks=2, n_devices=2, first_rank_id=0, first_device_id=6
    // MPI rank 0 → device 6（sender）
    // MPI rank 1 → device 7（receiver）
    ASSERT_TRUE(RunCrossRankTest(2, 2, 0, 6));
}
```

### 设备分配

```
RunCrossRankTest(n_ranks=2, n_devices=2, first_rank_id=0, first_device_id=6)
  └─ ForkAndRunWithHcclRootInfo(2, 0, 6, ...)
       ├─ MPI rank 0: rankId = 0+0 = 0, deviceId = 0%2+6 = 6
       └─ MPI rank 1: rankId = 0+1 = 1, deviceId = 1%2+6 = 7
```

### Host Runner 流程（RunCrossRankKernel）

两个 MPI 进程分别执行相同的 `RunCrossRankKernel`，根据 `myRank` 走不同分支：

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rank 0 (device 6)                  Rank 1 (device 7)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TestContext::Init()                TestContext::Init()
  └─ HcclCommInitRootInfo(2,0)       └─ HcclCommInitRootInfo(2,1)
  └─ HcclAllocComResourceByTiling    └─ HcclAllocComResourceByTiling
  └─ hostCtx.windowsIn[0] = dev6 window base
     hostCtx.windowsIn[1] = dev7 window base（RDMA 地址）

sendBuf = windowsIn[0] + 0        sendBuf = windowsIn[1] + 0
recvBuf = windowsIn[0] + kBytes   recvBuf = windowsIn[1] + kBytes
remoteRecvBuf = windowsIn[1]+kBytes  (rank0 计算的 rank1 recvBuf 地址)

v1_host = [0, 1, ..., 127]        v1_host = [1000, ..., 1127]
v2_host = [0, 0, ..., 0]          v2_host = [0, 0, ..., 0]
memcpy sendBuf, recvBuf            memcpy sendBuf, recvBuf

┄┄┄┄┄┄ HcclHostBarrier (Barrier 1) ┄┄┄┄┄┄ 两个 rank 都初始化完毕

SdmaWorkspaceManager::Init()      （等待 Barrier 2）
async_comm_kernel<<<1, stream>>>(
  v1 = remoteRecvBuf,  ← dev7 的 recvBuf RDMA 地址
  v2 = sendBuf,        ← dev6 本地 sendBuf
  v3 = workspace
)
  └─ TPUT_ASYNC(dst=dev7_recvBuf, src=dev6_sendBuf)
     ← SDMA 通过 NVLink/PCIe 把 [0..127] 搬到 device 7

aclrtSynchronizeStream(stream)
sdmaMgr.Finalize()

┄┄┄┄┄┄ HcclHostBarrier (Barrier 2) ┄┄┄┄┄┄ rank0 DMA 完成，rank1 可以读

（等待）                           memcpy recvBuf → result[]
                                   验证 result[i] == i (rank0 的数据)
                                   [PASS] 打印前 5 个元素

ctx.Finalize()                     ctx.Finalize()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### RDMA 地址计算原理

HCCL 为每个 rank 分配一段**对等可访问的 window 内存**。所有 rank 在各自 window 内的分配偏移相同，因此：

```
rank 0 的 recvBuf 地址 = windowsIn[0] + kBytes
rank 1 的 recvBuf 地址 = windowsIn[1] + kBytes   ← 传给 kernel 的 v1
```

rank 0 把 `windowsIn[1] + kBytes` 作为 `v1` 传给 kernel。SDMA 硬件识别出这是远端地址，自动通过设备间互联（NVLink 或 PCIe）完成 RDMA Write。**kernel 本身无需感知跨 rank 通信**。

### Barrier 2 的时序保证

```
rank 0: SDMA DMA 提交 → aclrtSynchronizeStream → HcclBarrier(信号发出)
rank 1:                                          HcclBarrier(收到信号) → 读 recvBuf
```

`HcclBarrier` 是网络集合操作，延迟远大于 512 字节的 SDMA 传输时间（微秒级）。因此 rank 1 在 Barrier 2 后读取 recvBuf 时，SDMA DMA 必然已经完成。

### 防死锁设计

rank 0 内部所有错误路径（`SdmaWorkspaceManager::Init` 失败、`aclrtSynchronizeStream` 失败）只设 `is_ok=false`，**不提前 return**，确保两个 rank 都能到达 Barrier 2，避免 rank 1 永久阻塞。

---

## 8. Test 6 — RootPut_4Ranks_Float128

**目的**：在 4 个 rank 之间测试多目标广播模式：rank 0（root）对 rank 1、2、3 各发起一次 `async_comm_kernel`，每次调用内部**同时执行 TPUT_ASYNC 和 TGET_ASYNC**（原始 kernel 固有行为，两者 src/dst 相同），验证 SDMA RDMA 对多目标的顺序传输正确性。

### GTest 入口

```cpp
TEST(AsyncComm, RootPut_4Ranks_Float128)
{
    // MPI rank 0 → device 4, rank 1 → device 5, rank 2 → device 6, rank 3 → device 7
    ASSERT_TRUE(RunRootPut4RanksTest(4, 4, 0, 4));
}
```

> **用例名含 `4Ranks`** → `run_st.py` 在 `mpirun -n 4` 层运行，filter `*4Ranks*` 命中。

### 设备分配

```
RunRootPut4RanksTest(n_ranks=4, n_devices=4, first_rank_id=0, first_device_id=4)
  └─ ForkAndRunWithHcclRootInfo(4, 0, 4, ...)
       ├─ MPI rank 0: rankId=0, deviceId=0%4+4=4  (root,   sender)
       ├─ MPI rank 1: rankId=1, deviceId=1%4+4=5  (target 1)
       ├─ MPI rank 2: rankId=2, deviceId=2%4+4=6  (target 2)
       └─ MPI rank 3: rankId=3, deviceId=3%4+4=7  (target 3)
```

### Host Runner 流程（RunRootPut4RanksKernel）

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Rank 0 (root, dev4)           Rank 1/2/3 (dev5/6/7)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TestContext::Init()           TestContext::Init()
  HcclCommInitRootInfo(4,0)    HcclCommInitRootInfo(4,1/2/3)
  获取所有 rank 的 windowsIn   获取所有 rank 的 windowsIn

sendBuf = windowsIn[0] + 0    sendBuf = windowsIn[rank] + 0
recvBuf = windowsIn[0]+kBytes recvBuf = windowsIn[rank]+kBytes

sendHost = [0..127]           sendHost = [rank*1000..rank*1000+127]
recvHost = [0, 0, ..., 0]     recvHost = [0, 0, ..., 0]
memcpy → device               memcpy → device

┄┄┄┄┄┄┄┄ HcclHostBarrier (Barrier 1) ┄┄┄┄┄┄┄┄

for target in [1, 2, 3]:     （等待 Barrier 2）
  remoteRecvBuf = windowsIn[target] + kBytes
  sdmaMgr.Init()              ← 每次发送独立初始化 workspace
  async_comm_kernel(
    v1=remoteRecvBuf,          ← 目标 rank 的 recvBuf（RDMA 地址）
    v2=sendBuf,                ← root 的本地 sendBuf
    v3=workspace
  )
  ↳ kernel 内部：
      TPUT_ASYNC(v1=remoteRecvBuf, v2=sendBuf) → root 数据推送到 target
      TGET_ASYNC(v1=remoteRecvBuf, v2=sendBuf) → 同 src/dst，与 TPUT 方向相同
  aclrtSynchronizeStream
  sdmaMgr.Finalize()
  打印: "[Root dev4] → rankX OK"

┄┄┄┄┄┄┄┄ HcclHostBarrier (Barrier 2) ┄┄┄┄┄┄┄┄

（等待）                       memcpy recvBuf → result[]
                               验证 result[i] == i (root 的数据)
                               [PASS] 打印前 5 个元素

ctx.Finalize()                ctx.Finalize()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 输入/输出数据

| rank | sendBuf 初始值 | recvBuf 初始值 | recvBuf 预期结果 |
|---|---|---|---|
| 0 (root) | `[0, 1, ..., 127]` | `[0, 0, ..., 0]` | 不验证（只发送） |
| 1 | `[1000, ..., 1127]` | `[0, 0, ..., 0]` | `[0, 1, ..., 127]`（root 的数据）|
| 2 | `[2000, ..., 2127]` | `[0, 0, ..., 0]` | `[0, 1, ..., 127]` |
| 3 | `[3000, ..., 3127]` | `[0, 0, ..., 0]` | `[0, 1, ..., 127]` |

### 为什么每次发送必须用新的 SdmaWorkspaceManager？

这是 Test 4 发现的 Bug 的直接延伸。第一次 kernel 执行结束后：
- `completion_slot[0].flag = 非零`（TPUT 已完成）
- `completion_slot[1].flag = 非零`（TGET 已完成）

如果第二次发送复用同一 workspace，新的 `BuildAsyncSession` 看到的 `next_event_id=0`，拿到的是已完成的 slot，`Wait()` 立即返回，DMA 实际上没有执行。target rank 2 的 recvBuf 不会被写入。

每次创建新的 `SdmaWorkspaceManager` 可保证 completion 槽全为 0，DMA 被正确提交和执行。

### 顺序发送的时序分析

```
root:  kernel→rank1  ──sync──  kernel→rank2  ──sync──  kernel→rank3  ──sync──  Barrier2
rank1: ←─────────── DMA 512B ──────────────────────────────────────────────────────┤读recvBuf
rank2:              ←─────────────────── DMA 512B ─────────────────────────────────┤读recvBuf
rank3:                                              ←──────────── DMA 512B ─────────┤读recvBuf
```

rank 1 的 DMA 是三次里最早发出的，等到 Barrier 2 完成时（需等 rank2、rank3 的 kernel + sync + 网络 barrier），已有充裕时间完成。

### 防死锁设计

root 内循环的所有错误路径（`sdmaMgr.Init()` 失败、`aclrtSynchronizeStream` 失败）均设 `is_ok=false` 并 `break`，**不提前 return**，保证 root 一定到达 Barrier 2，避免其余 rank 永久阻塞。

---

## 9. 已发现的 Kernel Bug

### Bug 1（严重）：缺少 `Wait()` 调用

```cpp
pto::comm::AsyncEvent v19 = TPUT_ASYNC(...);
pto::comm::AsyncEvent v26 = TGET_ASYNC(...);
return;   // ← 没有 v19.Wait(v11) 和 v26.Wait(v11)
```

**影响**：
- DMA 可能在 kernel 返回后仍在执行
- `aclrtSynchronizeStream` 只等 AICore，不等 SDMA
- 消费方读取 v1 时可能看到旧数据
- Completion flag 槽留在 pending 状态，workspace 复用时 DMA 被静默跳过（Test 4 验证了此问题）

**当前测试为何能通过**：单次运行时 512 字节 SDMA 非常快，主机侧代码的执行延迟掩盖了这个问题。Test 4 通过每轮重建 workspace 规避了 slot 污染，而不是修复 kernel。

### Bug 2（中等）：TPUT 和 TGET 底层实现完全相同

```cpp
// sdma_async_intrin.hpp
uint64_t __sdma_put_async(dst, src, ...) { return SdmaWrite(dst, src, ...); }
uint64_t __sdma_get_async(dst, src, ...) { return SdmaWrite(dst, src, ...); }
```

两个函数调用同一个 `SdmaWrite`，opcode 都是 0。同时，两次 DMA 的 src/dst 完全相同（都是 v2→v1），存在并发写竞争，写入顺序不确定。

### Bug 3（轻微）：未使用的变量 `v7`、`v8`

```cpp
int32_t v7 = 256;   // 未被任何操作使用
int32_t v8 = 1;     // 未被任何操作使用
```

这两个值通常用于 Tile 操作的 repeat/stride 参数。它们的存在暗示 PTOAS 可能遗漏了某些中间操作。

---

## 10. 运行命令

### 运行全部 6 个用例

```bash
python3 tests/script/run_st.py -r npu -v a3 -t comm/async_comm
```

`run_st.py` 会依次执行三轮：

| 轮次 | mpirun -n | GTest filter | 匹配用例 |
|---|---|---|---|
| 1 | 2 | `*-*4Ranks*:*4ranks*:*8Ranks*:*8ranks*` | Test 1–5 |
| 2 | 4 | `*4Ranks*:*4ranks*` | Test 6 (`RootPut_4Ranks_Float128`) |
| 3 | 8 | `*8Ranks*:*8ranks*` | （当前无 8-rank 用例）|

### 单独运行某个用例

```bash
# Test 1–5（2-rank 层）
python3 tests/script/run_st.py -r npu -v a3 -t comm/async_comm \
    -g AsyncComm.SmokeTest_Float128

# Test 6（4-rank 层，run_st.py 自动路由到 mpirun -n 4）
python3 tests/script/run_st.py -r npu -v a3 -t comm/async_comm \
    -g AsyncComm.RootPut_4Ranks_Float128
```

### 用例与 rank 的对应关系

| 用例名 | mpirun -n | 使用设备 | 验证内容 |
|---|---|---|---|
| `SmokeTest_Float128` | 2 | device 0 | v2→v1 基础搬运 |
| `ZeroData_Float128` | 2 | device 0 | 全零覆盖非零目标 |
| `NegativeData_Float128` | 2 | device 0 | 负数浮点传输 |
| `MultiLaunch_Float128` | 2 | device 0 | 3 轮启动 workspace 重置 |
| `CrossRank_Float128` | 2 | device 6 + 7 | 2 rank 跨设备 RDMA TPUT |
| `RootPut_4Ranks_Float128` | 4 | device 4/5/6/7 | root 顺序 PUT 到 3 个目标 |
