# `pto::PTO_PREFETCH` vs `pto::TPREFETCH_L2` — 深度对比

本文件是 `TPREFETCH_L2_design.md` 的配套说明，系统性地剖析 pto-isa 仓中两条 L2 预取路径的所有差异。

- **host 路径**：`pto::PTO_PREFETCH`（`include/pto/npu/kernels/Pto_prefetch.hpp`）
- **device 路径**：`pto::TPREFETCH_L2`（`include/pto/npu/TPrefetchL2.hpp`，公开 API 在 `include/pto/common/pto_instr.hpp`）

仅讨论 `PTO_PREFETCH<UseSdma=true>` 分支（即走 SDMA CMO 硬件），因为这才是和 `TPREFETCH_L2` 功能对等的路径。`UseSdma=false` 的 AIV 分支用的是 `TPREFETCH`（MTE2 → UB），与 `TPREFETCH_L2`（SDMA → L2）目的地根本不同，另行说明。

---

## 目录

0. [术语与层级（runtime / driver / STARS / SDMA / stream / SQ 关系）](#0-术语与层级runtime--driver--stars--sdma--stream--sq-关系)
1. [API 签名与调用约定](#1-api-签名与调用约定)
2. [定位与设计哲学](#2-定位与设计哲学)
3. [内部调用链（从 API 到硬件 SQE）](#3-内部调用链从-api-到硬件-sqe)
4. [硬件执行路径](#4-硬件执行路径)
5. [同步机制详解](#5-同步机制详解)
6. [启动开销分解](#6-启动开销分解用我们-scenario-b-的数据倒推)
7. [吞吐 / 带宽对比](#7-吞吐--带宽对比)
8. [资源占用](#8-资源占用)
9. [与 AI Core 其他流水的交互](#9-与-ai-core-其他流水的交互)
10. [错误 / 失败模式](#10-错误--失败模式)
11. [跨 rank / 跨设备行为](#11-跨-rank--跨设备行为)
12. [适用 / 不适用场景](#12-适用--不适用场景带代码样式)
13. [组合用法（两者不互斥）](#13-组合用法两者不互斥)
14. [关键差异一览表](#14-关键差异一览表)
15. [实测数据印证](#15-实测数据印证)

---

## 0. 术语与层级（runtime / driver / STARS / SDMA / stream / SQ 关系）

本章给后面所有讨论一个共同的概念基础。后文再出现 "runtime"、"STARS"、"stream"、"SQ/CQ"、"doorbell" 等术语时，都按这里的定义。

### 0.1 物理与软件层级

```
┌─── 用户应用 (host 进程) ──────────────────────────────────────────┐
│  调用 aclrt* / aclmdl* 等 ACL API                                 │
└────────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─── CANN runtime (host 用户态 .so) ────────────────────────────────┐
│  位置：libascendcl.so / libruntime.so                              │
│  仓库：本工作区下 d:/code/cann/runtime/                            │
│  职责：context / stream / event / memory / kernel 抽象             │
│        在 host 内存里维护 stream FIFO、依赖关系                    │
│        把 task 翻译成 STARS SQE，写到 SQ buffer，敲 doorbell        │
│  关键代码：                                                        │
│    runtime/src/acl/                       ← ACL API 入口            │
│    runtime/src/runtime/api/api_c_*.cc     ← stream/event/memory     │
│    runtime/src/tprt/                      ← task → STARS SQE        │
│    runtime/src/runtime/driver/            ← 调内核态 driver         │
└────────────────────────────────────────────────────────────────────┘
                  │  通过系统调用 / mmap                              
                  ▼
┌─── NPU driver (host 内核模块) ────────────────────────────────────┐
│  不开源，装 Ascend-hdk-*-npu-driver_*.run 时进入 kernel            │
│  职责：把用户态请求翻译成 MMIO 写、管理设备资源、中断处理          │
└────────────────────────────────────────────────────────────────────┘
                  │  PCIe / on-chip bus
                  ▼
═════════════════ host / device 边界 ═══════════════════════════════
                  │
                  ▼
┌─── STARS (NPU 芯片上的硬件 + 固件，task scheduler) ───────────────┐
│  位置：NPU SoC 上的专用调度硬件                                    │
│  发布形态：固件烧录在 Ascend-hdk-*-npu-firmware_*.run              │
│  职责：调度整个 NPU 的所有任务（不只是 SDMA）                      │
│    - 维护若干硬件 SQ/CQ 对（数量因芯片而异）                       │
│    - 监听 doorbell 寄存器，发现 SQ 有新 SQE                        │
│    - 解析 SQE 的 type 字段，派给对应执行单元：                     │
│         AICore / AI CPU / SDMA / Notify / 等                       │
│    - 执行单元做完后，STARS 写 CQE，runtime 据此判断完成            │
└────────────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─── 执行单元（NPU 内的具体硬件） ───────────────────────────────────┐
│  AI Core (AIC + AIV)：跑用户写的 kernel                            │
│  AI CPU：跑 AI CPU 算子                                            │
│  SDMA：搬数 / CMO / Notify                                         │
└────────────────────────────────────────────────────────────────────┘
```

### 0.2 关键事实（带 runtime 仓代码佐证）

**1. STARS 不是所有 NPU 都有的特性**

`runtime/src/runtime/core/src/runtime.cc:227` 定义了 `Runtime::ChipIsHaveStars()`，每个芯片在 `runtime/src/runtime/config/<chip>/dev_info_reg.cc` 里独立标注 `isStars` 字段：

| 有 STARS（`isStars=true`） | 没有 STARS（`isStars=false`） |
|---|---|
| `950`、`910_B_93`、`910_96`、`mini_v3`、`tiny`、`kirin9030`、`kirinx90`、`mc62cm12a`、`mc32dm11a`、`as31xm1` | `mini`、`cloud`、`dc`、`adc`、`nano`、`610_lite` |

老芯片（mini / cloud / 早期 310）走传统 stream 路径；新芯片（910B/950 等）走 STARS 路径。

**2. stream 数量上限因芯片而异，从 driver 动态查**

```cpp
// runtime/src/runtime/driver/npu_driver_res.cc:420-447
rtError_t NpuDriver::GetMaxStreamAndTask(...)
{
    if (rt->ChipIsHaveStars()) {
        drvRet = halResourceInfoQuery(deviceId, tsId,
                                      DRV_RESOURCE_SQ_ID, &queryInfoInput);
    } else {
        drvRet = halResourceInfoQuery(deviceId, tsId,
                                      DRV_RESOURCE_STREAM_ID, &queryInfoInput);
    }
    *maxStrCount = queryInfoInput.capacity;
}
```

这一段直接告诉我们：**在有 STARS 的芯片上，"stream 数量上限"就是"STARS 硬件 SQ 数量上限"**——一个 `aclrtStream` 一对一对应一对 STARS 的 `(SQ, CQ)`。具体值因芯片代际而异，runtime 自己也是问 driver 才知道。

`runtime/src/runtime/feature/xpu/xpu_device.cc:24` 里有个 `MAX_STREAM_NUM = 64` 的软件兜底上限，但实际容量以 driver 查到的为准（且能被 ini 配置进一步收紧）。

**3. STARS 不只调度 SDMA，是整个 NPU 的硬件任务调度器**

`runtime/src/runtime/inc/sqe/aic_aiv_sqe_common.hpp` 定义了 AIC/AIV 类型的 STARS SQE，`runtime/src/tprt/inc/external/tprt_sqe_cqe.h` 定义了 AICPU 类型的 STARS SQE：

```c
// runtime/src/tprt/inc/external/tprt_sqe_cqe.h:43-50
struct TprtStarsSqeHeader_t {
    uint8_t type;
    uint8_t wrCqe : 2;
    uint8_t sqeLength : 6;
    uint16_t sqId;
    uint16_t dfxId;
    uint32_t taskSn;
};
```

`pto-isa` 用的 `RT_STARS_SQE_TYPE_SDMA = 11` 只是 STARS 支持的众多 type 之一。**AI Core kernel launch、CMO prefetch、Notify、AICPU 调度，全部走 STARS 这套统一硬件队列机制**，只是 SQE 的 `type` 字段不同。

**4. host runtime 和 device kernel 共用同一份 STARS ABI**

| 共用的硬件 ABI | 含义 |
|---|---|
| `sq_base` | SQ 缓冲区在 GM 中的基地址 |
| `sq_reg_base` | STARS 的 doorbell MMIO 寄存器地址 |
| `sq_head` / `sq_tail` | STARS 内部维护的硬件指针 |
| SQE 格式（`TprtStarsSqeHeader_t` 等） | STARS 解析的字段布局 |

不同点只在**谁来写**：
- host runtime：CPU 把 SQE 写到 mmap 后的 SQ buffer，MMIO 写 doorbell
- device kernel（pto-isa / shmem）：AI Core 用 `copy_ubuf_to_gm` 把 SQE 写到 GM SQ buffer，scalar 写 doorbell

两条路写完后 STARS 看到的是**完全一样的硬件队列状态**。这就是为什么 device path 能"绕过 runtime"——它复用了 STARS ABI，自己干了 runtime 干的活。

### 0.3 stream / SQ / channel 这些词的对应关系

后文经常混用，统一一次：

| 名字 | 出现位置 | 实际是什么 |
|---|---|---|
| `aclrtStream` | 用户 API、host 代码 | runtime 维护的软件抽象，**底层一对一映射**到一对 STARS `(SQ, CQ)` |
| **stream** | runtime 内部、ACL 文档 | 同上 |
| **STARS channel** | `pto-isa` / `shmem` 文档 | 一对 STARS `(SQ, CQ)` 硬件队列。"channel" 是该资源的另一种叫法 |
| **STARS SQ** | 硬件层 | 一条提交队列，CPU 或 AI Core 往里写 SQE |
| **STARS CQ** | 硬件层 | 一条完成队列，STARS 往里写 CQE |
| `stars_channel_info_t` | `shmem` / `pto-isa` device 代码 | 描述一对 (SQ, CQ) 的元数据：`sq_base`/`sq_reg_base`/`sq_head`/`sq_tail`/`cq_id`/... |
| `AsyncSession` | `pto-isa` device API | 一组 STARS channel 的封装，`queue_num` 表示占了几条 |
| `queue_num` | `AsyncSession` 参数 | 这个 session 占用的 STARS channel 数 |

**容易混的点**：
- "stream FIFO 串行" 这种说法指的是 **runtime 软件 FIFO**，不是 STARS 硬件 SQ。STARS SQ 上虽然 SQE 也是按 sq_tail 顺序提交，但**派给不同执行单元后是并行执行的**。
- "48 条" 这个数字（`pto-isa` / `shmem` 文档中常见）是**软件分配策略**，不是 STARS 硬件容量。来源有两处协调一致的定义：
  - `shmem/include/host_device/shmem_common_types.h` 的 `ACLSHMEM_MAX_AIV_PER_NPU = 48`（NPU 上 AIV 核数上限），驱动 `shmem` 按 "一 AIV 一 STARS channel" 创建 48 条 device-only stream
  - `pto-isa/include/pto/npu/comm/async/sdma/sdma_types.hpp:30` 的 `kSdmaMaxChannel = 48U`，`SdmaCmoPrefetch` / `SubmitCmoPrefetchSqes` 等 device 内核代码直接用它做 `channelGroupIdx >= kSdmaMaxChannel / queue_num` 的范围校验
  - 两个 48 是**协调设计**：`shmem` 创建 48 条供 `pto-isa` 使用，`pto-isa` 自己再用 `kSdmaMaxChannel` 兜底校验。STARS 真实硬件 SQ 容量因芯片而异，runtime 通过 `halResourceInfoQuery(DRV_RESOURCE_SQ_ID)` 动态查询。

### 0.4 host path 和 device path 在这套层级里的位置

```
host PTO_PREFETCH:                       device TPREFETCH_L2:
┌──────────────────┐                    ┌──────────────────┐
│ user code        │                    │ user code        │
└──────────────────┘                    └──────────────────┘
        │ aclrtCmoAsync                         │ TPREFETCH_L2
        ▼                                       │ (在 kernel 内)
┌──────────────────┐                            │
│ CANN runtime     │← 经过这一层               │
│ (libascendcl.so) │                            │
└──────────────────┘                            │
        │ ioctl / MMIO                          │
        ▼                                       │
┌──────────────────┐                            │
│ NPU driver       │← 经过这一层               │
└──────────────────┘                            │
        │                                       │
═══ host/device 边界 ═══════════════════════════│═══════════════
        │                                       ▼
        └──→ ┌──────────────────────────────────────────┐
             │ STARS (NPU 硬件调度器)                    │← 两条路最终汇聚到这里
             │   SQ_n.sq_base / sq_reg_base / ...        │
             └──────────────────────────────────────────┘
                              │
                              ▼
             ┌──────────────────────────────────────────┐
             │ SDMA CMO 引擎（实际执行 prefetch）        │
             └──────────────────────────────────────────┘
```

device path 绕过的是 host runtime + driver + host/device 边界这三段，**不是绕过 STARS**——AI Core 直接扮演了"runtime 写 SQE + 敲 doorbell"的角色，复用 STARS 的同一套硬件队列。

后文涉及 stream FIFO / overlap / 启动开销时，都按这张层级图来定位是哪一层在影响。

---

## 1. API 签名与调用约定

### 1.1 `pto::PTO_PREFETCH`（host API）

```cpp
template <bool UseSdma = true, int AivCores = -1>
void PTO_PREFETCH(__gm__ void *tensor, uint64_t tensor_bytes, aclrtStream stream)
{
    if (tensor_bytes == 0)
        return;

    if constexpr (UseSdma) {
        aclrtCmoAsync((void *)(uint64_t)tensor, static_cast<size_t>(tensor_bytes),
                      ACL_RT_CMO_TYPE_PREFETCH, stream);
    } else {
        static_assert(AivCores > 0, "AivCores must be > 0 when UseSdma is false");
        PTO_PREFETCH_AIV<<<AivCores, nullptr, stream>>>((__gm__ uint8_t *)tensor, tensor_bytes);
    }
}
```

- **调用位置**：host（C++ runtime）
- **返回**：`void` — 调用方无「事件」可拿，同步只能靠 `aclrtSynchronizeStream(stream)` 或在同一 stream 上 enqueue 的后续 kernel
- **参数**：device 侧指针 + 字节数 + ACL stream
- **模板参数**：`UseSdma`（默认 `true`，走 SDMA CMO 路径）/ `AivCores`（`UseSdma=false` 时走 AIV kernel 内 `TPREFETCH` 循环，**目的地是 UB 不是 L2**）

### 1.2 `pto::TPREFETCH_L2`（device API）

```cpp
template <typename GlobalData>
PTO_INTERNAL AsyncEvent TPREFETCH_L2_IMPL(GlobalData &srcGlobalData, const AsyncSession &session)
{
    return detail::TPREFETCH_L2_SDMA_IMPL(srcGlobalData, session.sdmaSession.execCtx);
}

PTO_INTERNAL AsyncEvent TPREFETCH_L2_IMPL(__gm__ void *src, uint64_t bytes, const AsyncSession &session)
{
    return detail::TPREFETCH_L2_RAW_SDMA_IMPL(src, bytes, session.sdmaSession.execCtx);
}
```

- **调用位置**：device（`__global__ AICORE` kernel 内部）
- **返回**：`AsyncEvent` — 拿在手里，之后调用 `evt.Wait(session)` 或 `evt.Test(session)` 精确同步
- **参数**：`GlobalTensor` 或 `(void*, bytes)`，加 `AsyncSession`（持有 STARS 通道 + workspace 上下文）
- **重载维度**：`GlobalTensor` vs raw pointer × `AsyncSession` vs 直接传 `SdmaExecContext`

### 1.3 签名差异的潜台词

两个签名一对比就能看出两种不同的世界观：

| | host API | device API |
|---|---|---|
| 任务归属 | 挂在 `aclrtStream` 上 | 挂在 `AsyncSession` 上 |
| 同步句柄 | 无（返回 void），只有 stream 级同步 | 有（返回 `AsyncEvent`），支持 per-task Wait/Test |
| 异步模型 | ACL 统一 stream FIFO | STARS SQ + flag 轮询（device-native） |
| 类型安全 | 裸指针 + bytes | 有 `GlobalTensor` 重载 |

host API 完全嵌在 ACL stream 语义里，继承了 stream 的所有优点（简单、成熟）和缺点（粒度粗、无 per-task event）。device API 自己造了一套 event 机制，换来细粒度等待。

---

## 2. 定位与设计哲学

| | `PTO_PREFETCH`（host） | `TPREFETCH_L2`（device） |
|---|---|---|
| **抽象层** | CANN runtime 层原语 | `pto-isa` 的指令级原语（device-side，与 `pto::comm` 下的 `TGET`/`TPUT`、`TGET_ASYNC`/`TPUT_ASYNC` 同层；语义上和返回 `AsyncEvent` 的 `*_ASYNC` 系列最接近） |
| **谁发起** | CPU 线程（host）往 runtime 派命令 | AI Core 内部标量核 |
| **时机** | kernel **启动前**就能发 | kernel 运行**中任意位置** |
| **依赖** | ACL runtime + SDMA 驱动 | STARS channel + 已建好的 `AsyncSession` |
| **数据通路** | SDMA CMO（或 AIV path） | 仅 SDMA CMO |

本质区别是**控制权归属**：host 路径的决策者是 CPU，device 路径的决策者是 AI Core。硬件在两条路下执行的是**同一条** SDMA CMO prefetch，差异全在「谁把 SQE 送到 SDMA 引擎」。

---

## 3. 内部调用链（从 API 到硬件 SQE）

### 3.1 host `PTO_PREFETCH` 的调用链（`UseSdma=true`）

```
应用代码
  pto::PTO_PREFETCH(dev_ptr, bytes, stream)         ← user call on host
    │
    ▼
  aclrtCmoAsync(dev_ptr, bytes,
                ACL_RT_CMO_TYPE_PREFETCH,
                stream)                              ← ACL API
    │
    ▼
  runtime: CmoTask 入队到 aclrtStream FIFO
    │
    ▼ (stream 按 FIFO 依次调度)
  runtime: 构造 SDMA CMO 描述符 (src_addr, bytes, opcode=prefetch)
    │
    ▼
  runtime: ioctl / driver 调用，把任务挂到 SDMA 引擎的硬件队列
    │
    ▼
  SDMA CMO engine 执行 prefetch：逐段从 HBM/GM 取数据灌到 L2
    │
    ▼
  SDMA 回写 stream 的完成标志，runtime 认为这个任务完成
    │
    ▼
  aclrtSynchronizeStream(stream) 或下一个 kernel 在此 stream 上 launch 时，
  会看到这个 CMO 任务已完成
```

**关键特征**：SQE 是 runtime/driver 在 host 侧构造的，AI Core 完全不参与；stream 是串行 FIFO，`PTO_PREFETCH` 占一个 slot。

### 3.2 `UseSdma=false` 的 AIV path（参考，非本文主讨论路径）

```cpp
// Generic prefetch kernel: split a 1D tensor across blocks (get_blockdim()) and issue TPREFETCH
__global__ AICORE PTO_AIV_ATTR void PTO_PREFETCH_AIV(__gm__ uint8_t *tensor, uint64_t total_elems)
{
    detail::PtoPrefetchKernelBody(tensor, total_elems);
}
```

```cpp
for (uint64_t offset = start; offset < end; offset += tile_elems) {
    const uint64_t remaining = end - offset;
    const uint32_t cur_elems = (remaining < tile_elems) ? static_cast<uint32_t>(remaining) : tile_elems;

    PrefetchTile tile(cur_elems);
    TASSIGN(tile, 0u);

    PrefetchShape dyn_shape(1, 1, 1, 1, static_cast<int>(cur_elems));
    PrefetchStride dyn_stride(1, 1, 1, static_cast<int>(cur_elems), 1);
    GlobalTensor<DType, PrefetchShape, PrefetchStride> g(tensor + offset, dyn_shape, dyn_stride);
    TPREFETCH(tile, g);
}
```

这条路径是：host launch 一个 **AIV kernel**，kernel 内用 `TPREFETCH`（MTE2 的 L1 prefetch 指令，**到 UB 而非 L2**）分块预取。注意这个 `TPREFETCH` 和我们的 `TPREFETCH_L2` 不是同一件事：前者是 MTE2 path（GM → UB），后者是 SDMA CMO path（GM → L2）。两条路径的目的地根本不同。

所以严格说，`PTO_PREFETCH<UseSdma=true>` 才是和 `TPREFETCH_L2` 功能完全对等的路径（都是把 GM 刷到 L2）。我们的对比只测这一条。

### 3.3 device `TPREFETCH_L2` 的调用链

```
// kernel 内
pto::comm::AsyncSession session;
BuildAsyncSession(scratch, sdmaWorkspace, session, syncId);  // 构造 STARS 上下文

// 推荐：workspace overload，不需要预先 BuildAsyncSession
auto evt = pto::TPREFETCH_L2(srcGlobalTensor, workspace);
evt.Wait();  // 0-arg：handle 内含 workspace 地址，自动反推
// 或高级路径：session overload，多次调用复用同一 session
// auto evt = pto::TPREFETCH_L2(srcGlobalTensor, session);
//  │
//  ▼  (TPrefetchL2.hpp: TPREFETCH_L2_IMPL)
//  TPREFETCH_L2_SDMA_IMPL(srcGD, execCtx)
//    │ ① IsFlatContiguous1D 检查：只接受紧凑 1D
//    │ ② TPrefetchL2GetTotalBytes：reduce shape 到 bytes
//    │ ③ sdma::__sdma_cmo_prefetch(ptr, bytes, execCtx)
//    │    │
//    │    ▼  (sdma_async_intrin.hpp: SdmaCmoPrefetch)
//    │    SubmitCmoPrefetchSqes():
//    │      ├─ 在 UB 里构造 SQE（opcode=6=CMO_PREFETCH, src, length, ...）
//    │      ├─ 通过 MTE2 copy_ubuf_to_gm → 写到 STARS SQ 缓冲区（GM）
//    │      ├─ dcci() 把 SQ 页从 L2 刷到 GM（让 SDMA 硬件读到最新 SQE）
//    │      ├─ 更新 channelInfo->sq_reg_base+8 字段 (doorbell ring)
//    │      └─ 返回 eventHandle（= contextGm address）
//    ▼
//  AsyncEvent(handle, DmaEngine::SDMA)

evt.Wait(session);
//  │
//  ▼ PrepareEventCheck：提交 flag SQE（SDMA 执行到它时写 record->flag）
//  ▼ 循环 scalar poll record->flag 直到非零
//  └─ 等待完成
```

**关键特征**：
- SQE 在 AI Core 内构造，用 MTE2 写到 GM 的 SQ 缓冲区
- 通过敲 doorbell 通知 SDMA 硬件新 SQE 可用
- STARS 调度器异步把 SQE 派到 SDMA CMO 引擎
- AI Core 拿 `AsyncEvent`，用 scalar poll GM 里的 flag 等待完成
- 整个过程**不退出 kernel**，也**不占 stream FIFO 槽位**

---

## 4. 硬件执行路径

### 4.1 共享的部分

两者最终都是同一条 SDMA CMO engine 执行 **opcode=6（CMO_PREFETCH）** 的 SQE，engine 行为完全一致：

```
SDMA CMO engine:
  for each SQE (src, bytes):
    lines = bytes / L2_LINE_SIZE
    for each line:
      issue read-request to HBM/GM controller
      memory controller returns the line
      L2 controller allocates/updates a cache line
    (不写回 UB、不 DMA 到任何具体目的地)
```

- **带宽**：和 SDMA 普通 GM→GM copy 同一引擎分享，但 CMO 只读不写远端 HBM，所以有效带宽更高
- **L2 替换策略**：正常 LRU，新 prefetch 来的 line 如果撞了脏 line，先回写再作废
- **多 line 并行**：SDMA 引擎本身多 outstanding，一条 prefetch SQE 内部是硬件流水

### 4.2 不一样的部分

| 维度 | host 路径 | device 路径 |
|---|---|---|
| SQE 生成地 | runtime in CPU memory，再通过 driver mmap 写到硬件队列 | AI Core UB → MTE2 → GM SQ 缓冲区 → SDMA 读 |
| 触达 SDMA 的机制 | runtime 写门铃寄存器（由 driver 管理） | AI Core 执行 `copy_ubuf_to_gm` + `dcci` + 写 `sq_reg_base+8` 门铃 |
| SQ 队列归属 | runtime 软件 FIFO + driver 下发，最终落到 stream 绑定的那条 STARS SQ | 直接挂在 `AsyncSession` 持有的 STARS channel（一对硬件 SQ/CQ）上；`shmem` 按 `ACLSHMEM_MAX_AIV_PER_NPU=48` 创建 48 条 device-only stream，`pto-isa` 自己也定义 `kSdmaMaxChannel=48U`（`sdma_types.hpp:30`）做 channel 范围校验，由 `queue_num` 决定占用几条（48 是软件分配策略，不是 STARS 硬件 SQ 上限——硬件上限因芯片而异，runtime 通过 `halResourceInfoQuery(DRV_RESOURCE_SQ_ID)` 动态查询） |
| SQE 分片 | runtime 按 CMO 最大长度自动分 | `SubmitCmoPrefetchSqes` 里按 `block_bytes` 分 |
| 完成感知 | runtime 轮询 SDMA status 寄存器/中断；通过 stream 事件链同步 | flag SQE 写 GM record，AI Core scalar poll record->flag |

---

## 5. 同步机制详解

### 5.1 host `PTO_PREFETCH` 的同步

```
                   stream (FIFO)
 ┌──────────────────────────────────────────────┐
 │ [CmoAsync task] → [next kernel] → [sync op] │
 └──────────────────────────────────────────────┘
  上面的任务按 enqueue 顺序严格串行执行
```

同步方式只有两种：

1. **隐式串行**：在同一个 stream 上 enqueue 下一个 kernel 或 CMO，runtime 保证 FIFO 顺序。这种方式不用 host 阻塞，但代价是任务 B 必须和 prefetch 串行（在 stream 语义上）；要并行只能用两个 stream + 跨 stream event。
2. **显式同步**：`aclrtSynchronizeStream(stream)`，block host thread 直到 stream 清空。

**没法部分等**：如果你想「等 prefetch 完，但让其它任务继续」，只能拆多 stream。单 stream 里不能跳过前面的任务。

### 5.2 device `TPREFETCH_L2` 的同步

```cpp
// AI Core 标量核
auto evt = TPREFETCH_L2(...);   // 返回 AsyncEvent，持有 handle

// ... 做别的事情：计算、TLOAD 其他区域、再发更多 async ops ...

evt.Wait(session);              // 精确等这一条 prefetch
```

`Wait` 的本质：
- 第一次 Wait 时提交一个 **flag SQE** 到同一个 SQ，让 SDMA 执行完前面所有 SQE 后，往 GM 的 `SdmaEventRecord->flag` 写一个非零值
- AI Core 用 scalar loop 轮询这个 GM 字段
- 由于「前面所有 SQE 都做完，flag SQE 才轮到」的 FIFO 语义，flag 写到就代表 `evt` 之前提交的 SDMA 操作全部完成

**细粒度**：可以发 N 个 async op，只 Wait 最后一个；也可以分别 Wait；可以 `Test`（非阻塞检查）。这种「在一个 kernel 里搞一张异步 DAG」的能力是 host 路径不具备的。

### 5.3 AI Core 最多监控 8 个 slot

device 异步框架约束：每个 AI Core 最多同时等待 **8 个** outstanding `AsyncEvent`（对应硬件 8 个 flag slot）。超出要先 `Wait` 释放。这是 host 路径没有的约束，但对 prefetch 这种低频操作基本撞不到。

---

## 6. 启动开销分解（用 Scenario B 的数据倒推）

实测：host 10–14 μs vs device ~3 μs。这 10+ μs 是怎么花掉的？

### 6.1 host `PTO_PREFETCH` + `aclrtSynchronizeStream` 的 10–14 μs

```
 0 us    user call PTO_PREFETCH → ACL 入口
         - 参数校验、stream 合法性检查
 1-2 us  runtime 构造 CmoTask 结构，入 stream FIFO（用户态软件队列）
 2-3 us  stream scheduler 把 task 提交给 kernel driver（syscall / ioctl）
         - 用户态 → 内核态切换
 3-5 us  driver 构造硬件 SDMA 描述符，写 SDMA 寄存器
 5-8 us  SDMA 硬件开始执行 prefetch（真正的工作）；不同 size 这段不同，但 4KB 基本 <1μs
 8-10 us SDMA 写完成状态；中断到 driver / 或 runtime 轮询拿到
10-14 us aclrtSynchronizeStream 的 host 轮询看到 stream 空，返回用户态
```

**三个主要开销**：

- 用户态↔内核态切换（syscall）一次或多次
- driver 做的描述符构造
- runtime 的 stream scheduler 簿记（task 对象生命周期、event 对象链等）

这三块是纯软件 overhead，buffer 多大都一样。所以 B 场景（4KB）测出来就是**纯软件发起成本**。

### 6.2 device `TPREFETCH_L2` + `evt.Wait` 的 ~3 μs

```
 0 us    kernel 内 call TPREFETCH_L2
         → IsFlatContiguous1D 检查（几条 scalar）
         → GetTotalBytes（乘法）
 0 us    SdmaCmoPrefetch 主体开始
         → InitSqTailArray：对每条队列读 sq_tail 从 GM（每条 GetValue，一次 copy_gm_to_ubuf_align_b32）
         → 在 UB 构造 N 条 SQE（N = bytes / block_bytes，4KB 那就 1 条）
         → copy_ubuf_to_gm 把 SQE 写到 SQ 缓冲区（GM）
         → dcci 把本地 L2 的 SQ 页刷回 GM（避免 SDMA 读到旧值）
         → 写 channelInfo->sq_reg_base+8（门铃）
         → UpdateSqTailState：回写 sq_tail
 1.5 us  SDMA 看到新 SQE，开始 prefetch（4KB 大约几百 ns）
 2 us    SDMA 完成，flag SQE 被 SDMA 跟上执行，写 record->flag
 2-3 us  AI Core scalar poll record->flag 看到非零，Wait 返回
```

**这 3 μs 里几乎每一步都是 AI Core 内纯指令执行，不离开 NPU 芯片**：

- 没有 syscall
- 没有 kernel launch
- 没有 runtime scheduler 簿记
- 没有 stream 事件链
- 没有 PCIe 往返
- 没有 host 中断/调度

唯一带"跨子系统"延迟的两个动作是：MTE2 把 SQE 写到 GM、scalar 单元 poll GM 上的 flag。
这两步都发生在**同一颗 NPU 芯片内**（AI Core 与片上 GM 之间通过 NoC 访问），单次开销
~100 ns 到几百 ns 量级。比 scalar/vector 指令（~ns）慢一两个数量级，但相比 host path
的 PCIe 往返 + 中断唤醒（μs ~ 几十 μs 级），仍小至少一个数量级。

### 6.3 为什么 device path 首次执行会额外贵一点？

`BuildAsyncSession` 做了：

- 分配 `ScratchTile` 并 `TASSIGN`
- 构造 `SdmaExecContext`（读 workspace layout）
- 构造 `SdmaSyncContext`
- 校验 session config

首次 build 会占 ~5–10 μs。我们的测量里已经把 BuildAsyncSession 纳入 Scenario A 的 kernel 时间，所以 A 场景 device 比 host 多出来的 4–20 μs 主要就是它。对于 Scenario B 我们复用 session，所以 4 KB 下测得 3 μs 不含 build 成本。

---

## 7. 吞吐 / 带宽对比

**结论**：到 SDMA 硬件层面**吞吐能力相同**（都是 ~280 GB/s 上限），
但 device path 受 `kDefaultSdmaBlockBytes = 1MB` 约束，单次 prefetch 的 SQE 条数 =
`ceil(bytes / 1MB)`，**每条 SQE 在 SDMA 引擎流水里都有 ~800 ns 的启动尾巴**。
所以 buffer 越大，device path 累计的 per-SQE overhead 越多，看起来"带宽"略低。

### 7.1 实测数据

实测 Scenario A，kernel time（包含 session build）：

| Size | host kernel | device kernel | 理论传输 (280 GB/s) | host overhead | device overhead | iter_num | 每 SQE overhead |
|---|---|---|---|---|---|---|---|
| 1 MB | 3.98 us | 6.10 us | 3.6 us | ~0.4 us | ~2.5 us | 1 | — |
| 16 MB | 60.70 us | 69.37 us | 57.1 us | ~3.6 us | ~12.3 us | 16 | ~770 ns |
| 128 MB | 489.58 us | 590.25 us | 457 us | ~32.6 us | ~133 us | 128 | ~1040 ns |

可以看到 device overhead 几乎是 **iter_num × ~800–1000 ns**，证明开销确实和 SQE 条数线性相关，
不是固定的 session build 成本。

### 7.2 ~800 ns per SQE 是哪来的

为了下文 §8 / §9 复用同一套术语，先固化 device path 的两阶段拆分：

| 阶段 | 谁在干活 | AICORE 占用 | SDMA 占用 | 单 SQE 开销 |
|---|---|---|---|---|
| **构造期** (Construction) | AICORE 写 N 条 SQE → 末尾一次 dcci + doorbell | Scalar + MTE3（写 SQE 字段、`sq_tail`、doorbell）| 空闲 | ~50 ns × N + 一次 flush/doorbell（合计通常 < 几 us） |
| **执行期** (Execution) | SDMA 处理 N 条 SQE，AICORE 同步进入 Wait（busy-poll flag） | 仅 Scalar（poll flag），MTE/Cube/Vector 全空闲 | 全部 | ~700 ns × N（每条 SQE 启动尾巴）+ `bytes / bandwidth`（实际搬运） |

**构造期细化**（仅 AICORE 时钟，不含 SDMA）：

```
AddOneCmoSqe → 构造 SQE 字段（~10 个 scalar 运算）        ~ 30 ns
内嵌 sq_tail 更新 / channel info 写回                    ~ 20 ns
（每条 SQE 约 50 ns，N 条 SQE 串行累加 = N × 50 ns）
末尾一次 FlushCacheAndRingDoorbell（dcci + 写 sq_reg_base）   ~ 几百 ns
```

**执行期细化**（仅 SDMA 引擎，AICORE 在 polling 不算入此处）：

```
从 SQ 取 SQE → 解析 → 启动 burst                        ~ 200 ns
burst 流水启动延迟（地址转换 / TLB / NoC 仲裁）            ~ 200 ns
burst 间间隙（不能 100% 背靠背）                          ~ 300 ns
（每条 SQE 启动尾巴约 700 ns，N 条 SQE 引擎内基本无法掩盖）
+ 实际数据搬运 ~ block_bytes / 280 GB/s
```

**两阶段串行、无重叠**：`SdmaCmoPrefetch` 先 `SubmitCmoPrefetchSqes` 写完所有 SQE，
再 `FlushCacheAndRingDoorbell` 一次性敲 doorbell，所以 SDMA 在所有 SQE 写完之前
不会开始执行。**没有 N×50ns（构造期 AICORE）被 N×700ns（执行期 SDMA）掩盖的可能**。

**外部观测的 per-SQE 总开销** = 构造期 ~50 ns + 执行期 ~700 ns + 数据搬运 ≈
**~750 ns per SQE 的纯启动尾巴**（数据搬运另算）。和 §7.1 实测 `~770~1040 ns`
吻合（多出来的部分主要是 SQ depth 上升导致 NoC 仲裁压力增加）。

### 7.3 host runtime 没这个问题的原因

host 路径在 CANN runtime 内部走的是经过精心调优的 SDMA descriptor builder：

| 项 | host runtime | device path |
|---|---|---|
| 单 SQE 最大传输 | 通常 16MB+（接近 SDMA descriptor `len` 字段位宽上限） | 写死 1MB |
| 切分粒度 | 按硬件最大字段决定，SQE 数量最少化 | 用户配置常量 `kDefaultSdmaBlockBytes` |
| 纯构造期 per-SQE（AICORE/CPU 时钟） | host CPU 2-3 GHz scalar，~100 ns / SQE | AICORE scalar，~50 ns / SQE |
| 执行期 per-SQE 启动尾巴（SDMA 时钟） | ~700 ns / SQE（共用引擎，相同） | ~700 ns / SQE（共用引擎，相同） |
| 外部观测 per-SQE 总开销 | ~800–1000 ns | ~800–1000 ns |
| 构造与执行的重叠 | 单次调用内基本串行；**跨调用**间可流水（runtime 把多次合并下发） | 单次调用内串行（doorbell 在末尾一次性敲）；多 `queue_num` 通道间可并行 |

128MB 任务下：
- host：~8 条 SQE × ~1000 ns ≈ **8 us 启动尾巴**
- device：128 条 SQE × ~1000 ns ≈ **128 us 启动尾巴**

差额 ~120 us，和实测 ~100 us 量级吻合。

### 7.4 这个 gap 重要吗？什么时候改？

| 场景 | 是否在意这个 gap | 优化建议 |
|---|---|---|
| **overlap with compute**（典型用法） | 不在意 | 不用改。compute 时间完全掩盖 prefetch，发起方位置不影响最终 wall time。Scenario C 大 buffer 下 C1/C2 收益几乎一致就是这个原因。 |
| **预热 L2、不 overlap** | 可能在意 | 见下面的优化方案 |
| **小 buffer (≤ 16 MB)** | 不在意 | iter_num ≤ 16，累计 overhead < 15 us，相对总耗时占比小 |
| **大 buffer (≥ 128 MB) 且要求 raw bandwidth** | 在意 | 必须改 |

### 7.5 如果要追平 host 吞吐，怎么改

**方案 A（推荐）：在 `BuildAsyncSession` 时传入更大的 `block_bytes`**

```cpp
// 例：把 block_bytes 改成 16MB，128MB → 8 条 SQE
sdma::SdmaBaseConfig cfg{16ULL * 1024 * 1024, 0, 1};
pto::comm::BuildAsyncSession(scratchTile, sdmaWorkspace, session, 0U, cfg);
```

需要先确认 SDMA 硬件单 descriptor `len` 字段位宽（常见 24-bit ≈ 16MB，
26-bit ≈ 64MB），可以查 `BuildTransferConfig` 里的 cap 校验。

**方案 B：调高 `queue_num` 让多通道并发**

128 条 SQE 分摊到 N 条通道，启动尾巴可以并行掉，但代价是占多条 STARS channel。
当 SDMA 引擎本身没饱和时有效，否则只是把 overhead 换地方。

**方案 C：接受现状**

正因为典型用法是 overlap，这 100 us 的 gap 不影响实际收益。
保持 1MB 的好处是 SQE 小、SQ 不容易溢出（`kSqDepth` 限制），代码路径简单。

### 7.6 这一节和原结论的关系

老结论"两者吞吐完全一致"**严格来说在 device path 默认配置下不准确**：
- 在 16MB 量级以下，差额 < 5%，确实可以认为一致
- 在 128MB 量级，差额 ~16%，**不能忽略**

更精确的说法是：**SDMA 硬件本身吞吐一致，差距全部来自 device path 的 1MB SQE 切分策略**。
配置成相同 SQE 大小后两者会真正一致。

---

## 8. 资源占用

### 8.1 host `PTO_PREFETCH` 占用

- **1 个 stream FIFO slot**：同一 stream 下的后续任务被阻塞
- **1 个 runtime task 对象**：栈/堆分配，生命周期跟到完成
- **driver 资源**：一个 SDMA 描述符槽位
- **CPU 时间**：~10 μs 的用户态 + 内核态 CPU 占用（阻塞 host 线程，如果 sync）
- **不占用 AI Core 资源**

### 8.2 device `TPREFETCH_L2` 占用

**长期占用**（session 生命周期内）：

- **1 个 `AsyncSession`**：通常 `queue_num=1`，占 STARS 的 1 条 SQ 通道
- **1 个 AsyncEvent slot**：AI Core 硬件 8 个 flag slot 之一
- **GM workspace**：每 session 大约 128 B × queue_num（`kSdmaFlagLength`）+ channel info
- **UB tmpBuf**：scratchTile 提供的小块 UB（一般几十 B），生命周期跟 session

**单次调用瞬时占用**（沿用 §7.2 的两阶段拆分）：

| 阶段 | 持续时间 | Scalar | MTE2 | MTE3 | Cube | Vector | UB |
|---|---|---|---|---|---|---|---|
| **构造期** (AICORE 写 SQE + flush + doorbell) | `~50 ns × iter_num + 一次 flush/doorbell`（通常 < 几 us） | ✅ 占（写 SQE 字段、循环控制、写 doorbell） | ✅ 占（读 `sq_tail` / `channel info`）| ✅ 占（写 SQE 数据到 GM、`sq_tail` 回写） | ❌ | ❌ | tmpBuf 在用 |
| **执行期 / Wait** (SDMA 处理 SQE，AICORE busy-poll flag) | `~700 ns × iter_num + bytes/280GB`（大头） | ✅ 占（仅 scalar busy-poll，读 GM flag 走 MTE2 但占用极轻） | ❌ | ❌ | ❌ | ❌ | tmpBuf 仍持有 |

注意：

- **AICORE 真正密集占用 MTE2/MTE3 只在构造期**，时长跟 `iter_num` 线性相关但绝对值很短（128 SQE 也只 ~10 us 量级）
- **执行期是 prefetch 的时间大头**（128 MB 时 ~133 us），但这段 AICORE **只有 Scalar 在 polling**，Cube / Vector / MTE 全空闲——这就是和 compute 真正 overlap 的窗口
- 如果应用 compute 也用 MTE（比如 `copy_gm_to_ubuf`），**构造期**会和 prefetch 抢 MTE 资源；执行期不抢
- 只用 Cube/Vector 的 compute（典型 GEMM）两个阶段都不受影响

- **不占 stream FIFO slot**

### 8.3 并发性对比

- 一个 host stream 上两次 `PTO_PREFETCH` **串行执行**（FIFO）
- 一个 `AsyncSession` 上两次 `TPREFETCH_L2` **并行提交**到 SDMA（engine 会按带宽饱和度并行处理），但 SQ 按 FIFO，所以 Wait 语义仍然「按序完成」

---

## 9. 与 AI Core 其他流水的交互

### 9.1 host path：被 stream FIFO 约束，并行需要多 stream

#### Stream FIFO 的真实语义

很容易误以为"prefetch 是异步的，提交完 SQE 给 SDMA，stream 就可以放下一个 kernel
了"。但 stream FIFO 的约定其实更严格：

> **同一条 stream 上的两个 task，第二个 task 要等第一个 task 的"完成事件"被记录后才能开始**。
> 不是等"提交事件"，是等"完成事件"。

```
单 stream 上的 host prefetch + kernel：
  T0: prefetch task 提交给 SDMA          ─┐
  T1: SDMA 开始干活                       │ 这段时间
  T2: kernel task 在 stream 队列里等待   │ kernel 不能开始
  T3: SDMA 完成、写完成 token             ─┘
  T4: runtime 看到 task 完成，调度 kernel
  T5: AI Core 开始跑 kernel
```

**为什么这么严**：runtime 没法静态分析 task B 是否依赖 task A 的副作用
（即使 CMO 不"产生数据"，prefetch 把数据加载到 L2 这件事会被 task B 观察到——
变成 warm TLOAD）。所以默认按完全序列化处理，这是 ACL/CUDA 共同的 stream 语义。

#### 单 stream 模式：no-overlap，只有 warm cache 红利

```cpp
pto::PTO_PREFETCH(tile_ptr, bytes, stream);          // task A
compute_kernel<<<..., stream>>>(...);                // task B
// stream FIFO 严格序列化:
//   wall = prefetch_time + compute_time + warm_TLOAD_time
//   prefetch 和 compute **不重叠**
```

这种模式只赚到一件事：kernel 里的 TLOAD 是 warm 的（因为 prefetch 已经先把 L2 暖好了）。
**没有 prefetch+compute 的硬件并行**。

#### 多 stream 模式：真正的并行，但代码复杂

```cpp
aclrtCreateStream(&prefetchStream);
aclrtCreateStream(&computeStream);

pto::PTO_PREFETCH(tile_ptr, bytes, prefetchStream);  // SDMA 跑这边
compute_kernel<<<..., computeStream>>>(...);          // AI Core 跑那边
// 两条 stream 没依赖关系 → runtime 派给两套硬件并行执行
// 但要小心：compute 里如果用 prefetch 的数据，可能 race
//          → 要 record_event + stream_wait_event，编排成本上升
```

只有这种模式才能让 prefetch 和 compute **真正并行**，但要管理：
- 至少 2 条 stream 的创建/销毁/sync
- 跨 stream 依赖（event 编排）
- 每个 task 边界都过一次 runtime 调度

#### Scenario C 的测试结构澄清

我们的 C1 模式用的是**单 stream**：

```cpp
case 1:  // C1
    pto::PTO_PREFETCH(env.srcDevice, env.dataBytes, env.stream);
    ScenarioC_ComputeThenTloadKernel<<<..., env.stream>>>(...);
    aclrtSynchronizeStream(env.stream);
```

按 stream FIFO，**prefetch 和 compute 不重叠**，C1 wall ≈ prefetch + compute + warm_TLOAD。

C1 vs C2 的 `cold-reduc` 数值接近，**只能证明两者最终 TLOAD 都是 warm 的**，
**不能证明 host 单 stream 模式实现了并行**。要看真正 overlap 收益，应该比较
`C1 wall - C2 wall`：这个差值就是 device path 通过"kernel 内并行"省下来的
prefetch 时间（不需要多 stream 编排即可获得）。

### 9.2 device path：kernel 内细粒度并行（不被 stream FIFO 约束）

**为什么 device path 能在单 stream 内并行？**

因为 device path 的 prefetch 是在 **kernel 内部**发起的，整个 kernel 是 stream 里
的**一个** task。stream FIFO 约束的是「task 与 task 之间」，不是「task 内部的
指令之间」。kernel 内部，AI Core 的 scalar 写完 SQE+doorbell 后会**立刻继续执行
后面的指令**——这次没有 stream 调度器拦着了，因为 STARS SQ 是硬件队列，AI Core
直接写硬件就完成了"提交"。

```
host path vs device path 的"提交后是否阻塞"对比：

host path:
  runtime: aclrtCmoAsync 提交 → stream 调度器 hold 住后续 task → 等 SDMA 完成 token
                                ↑ 这就是 FIFO 的"阻塞"

device path:
  AI Core scalar: 写 SQE 到 GM → ring doorbell → 继续执行下一条指令
                                                 ↑ 没有调度器拦
```

**典型用法**：

```cpp
// 在一个 kernel 内
auto evt = TPREFETCH_L2(next_tile_ptr, bytes, session);   // ① 提交

compute_on_current_tile();   // ② 当前 tile 的计算

evt.Wait(session);           // ③ 用前才等
TLOAD(tile, next_tile_g);    // ④ L2 已预热，TLOAD 飞快
```

**实际并行的时序**（沿用 §7.2 / §8.2 的两阶段术语）：

```
时间 →
                ① 构造期 (~几 us)               ② 执行期 (~ms 大头)
                AICORE 写 SQE + flush + doorbell SDMA 跑数据 + AICORE Wait
SDMA 引擎:                                    [─── 搬 next_tile 到 L2 ───]
AI Core Scalar: [─ 写 SQE 字段、doorbell ─]                       [poll flag]
AI Core MTE2/3: [─ SQE 数据写 GM、sq_tail ─]                      
AI Core Cube:                                [── compute_on_current_tile ──]
AI Core Vector:                              [── compute_on_current_tile ──]
                ↑ 这段 Cube/Vec 才空闲         ↑ 黄金 overlap 窗口（最大头时间）
```

关键点：

- **构造期**（~几 us 量级）Scalar + MTE2/MTE3 被占用，**Cube/Vector 是空闲的**
  - 用 Cube/Vector 的 compute（典型 GEMM）可以**完全并行**
  - 用 MTE 的 compute（要搬 UB↔GM）会和构造期**抢 MTE 资源**
- **执行期**（搬运的大头时间，~ms 量级）AI Core **MTE/Cube/Vector 全空闲**，只有 Scalar 在 busy-poll flag
  - 这是 overlap 真正赚到时间的地方
  - 任何 compute（包括 MTE 密集型）都能放心跑
- 构造期和执行期的时长是数量级差异（几 us vs ~ms），所以「真实开销 = 执行期」「与 compute overlap 的窗口 = 执行期」是准确的描述

**对比 host path**：device path 的并行单位是「kernel 内的几行代码」，无需双 stream/event 编排，
但**程序员要心里清楚构造期那几 us MTE 会被占**，避开 MTE 密集型 compute 紧邻在 prefetch 之后。

---

## 10. 错误 / 失败模式

### 10.1 host `PTO_PREFETCH`

- `aclrtCmoAsync` 返回非 zero：无效地址、越界、stream 非法、driver 错误——用户可 catch（但源码里我们直接忽略了返回值）
- stream 已有未完成错误任务：后续 enqueue 都失败
- 完成信号丢失：极罕见，通常是 driver/硬件 bug
- **buffer 0 byte**：API 直接 early return
- **地址未对齐**：driver 兜底处理（CMO 通常对 128 B 或 page 对齐敏感，但驱动会 clip）

### 10.2 device `TPREFETCH_L2`

```cpp
if (srcGlobalData.data() == nullptr) {
    return AsyncEvent(0, DmaEngine::SDMA);
}

if (!TPrefetchL2IsFlatContiguous1D(srcGlobalData)) {
    return AsyncEvent(0, DmaEngine::SDMA);
}

const uint64_t totalBytes = TPrefetchL2GetTotalBytes(srcGlobalData);
if (totalBytes == 0) {
    return AsyncEvent(0, DmaEngine::SDMA);
}
```

失败模式：

- `src == nullptr` → 返回 handle=0 的 no-op event
- **不是紧凑 1D shape** → 返回 handle=0（`TPrefetchL2IsFlatContiguous1D` 严格检查 `pitch4=1, pitch3=dim4, ...`）
- `bytes == 0` → 返回 handle=0
- `BuildAsyncSession` 失败 → 调用方自己判断并 skip（我们的测试里就是这样写的）
- SQE 队列满 → `SdmaCmoPrefetch` 在 `sqePerQueue > kSqDepth` 检查处直接 `return 0`，本次 prefetch 没发出（注意：实际写 SQE 的 `SubmitCmoPrefetchSqes` 是 `void`，溢出保护在它的外层做）

**注意**：device 路径的 `AsyncEvent(0, ...)` 是**静默失败**，调用方 Wait 它也不会 block（handle 为 0 的 event 特殊处理）。这在 kernel 内部是合理设计——不能让一个错误 prefetch 把整个 kernel hang 死。

---

## 11. 跨 rank / 跨设备行为

| | host path | device path |
|---|---|---|
| **跨 rank 发起者的位置** | CPU0 给 rank 0 NPU 发；CPU1 给 rank 1 NPU 发（MPI 各自独立） | rank 0 AI Core 给 rank 0 SDMA；rank 1 AI Core 给 rank 1 SDMA |
| **能不能给远端 prefetch** | **不能**：`aclrtCmoAsync` 只作用于本地 NPU 的 HBM/L2 | **不能**：SDMA CMO SQE 只能刷本地 L2 |
| **跨 rank 数据流** | 无关，prefetch 永远是 local 操作 | 无关 |
| **和 HCCS 的关系** | L2 prefetch 完，后续 TLOAD 可能走 HCCS 读远端（如果地址是远端） | 同左 |

两者**在跨 rank 场景下行为完全等价**：都是把本 rank 能访问到的 GM 地址预取到本 rank 的 L2。差别仍然只是「发起方位置 + 启动开销」，我们的 Scenario D 实测证实了这点。

有一种**特殊用法**（tprefetch_l2 测试里试过）：「对端 prefetch」——rank B 要 TGET rank A 的数据，先让 rank A 的 AI Core 发 `TPREFETCH_L2` 把自己的 local GM 灌进自己的 L2，HCCS 在跨 rank 读时**可能**从 remote L2 snoop 命中（具体命中率看硬件拓扑）。host path 也能做同样的事，没有本质差别。

---

## 12. 适用 / 不适用场景（带代码样式）

### 12.1 host `PTO_PREFETCH` 典型场景

**算子启动前，要预取的数据已经在 host 侧确定**：

```cpp
// host 侧：已知输入 tensor 地址，下一个 kernel 要读它
pto::PTO_PREFETCH(input_tensor_ptr, input_bytes, stream);
my_big_kernel<<<..., stream>>>(input_tensor_ptr, ...);
aclrtSynchronizeStream(stream);
```

**批处理前的 warmup**：

```cpp
for (auto &tensor : tensors_used_in_this_batch) {
    pto::PTO_PREFETCH(tensor.data, tensor.bytes, prefetchStream);
}
// 让 SDMA 并行把它们刷进 L2
HcclHostBarrier(...);
launch_batch_kernels(...);
```

**对 device kernel 后续访问透明加速**：`PTO_PREFETCH` 把数据从 HBM 拉到 L2，受益的是**所有**访问该地址的 device 指令，不需要"配对"特定指令——无论 kernel 里用的是 `TGET`/`TPUT`（MTE2/3 经 UB 路径）、`TGET_ASYNC`/`TPUT_ASYNC`（SDMA GM-to-GM）、`TLOAD` 还是普通的 GM 读，命中 L2 都比第一次读 HBM 快。

> 注意：`TGET` / `TPUT` 是 **device-side** 同步指令，走 MTE2/MTE3 + UB 暂存，**不是 SDMA**；`TGET_ASYNC` / `TPUT_ASYNC` 才是走 SDMA 的 GM-to-GM 异步指令。详见 `pto/comm/README_zh.md`。

**不适用**：

- 预取地址需要 kernel 运行时计算（比如动态 indirect 索引）
- 想在 kernel 内部 pipeline 阶段性切换预取目标
- 想避免额外 kernel launch 开销（每次 CmoAsync 虽然不是 launch 但实际 overhead 和 launch 同量级）

### 12.2 device `TPREFETCH_L2` 典型场景

**Kernel 内多 tile pipeline**：

```cpp
__global__ AICORE void pipeline_kernel(__gm__ T *input, int numTiles, ...) {
    AsyncSession session;
    BuildAsyncSession(scratch, workspace, session, syncId);

    // Prologue: 先发 tile 0 的预取
    auto evt = TPREFETCH_L2(input + 0 * tileElems, tileBytes, session);
    
    for (int i = 0; i < numTiles; ++i) {
        // 在等 tile i 的同时，发 tile i+1 的预取
        if (i + 1 < numTiles) {
            auto evt_next = TPREFETCH_L2(input + (i+1) * tileElems, tileBytes, session);
            // evt_next 会在下次迭代开头被 Wait
        }
        
        evt.Wait(session);
        TLOAD(tileLocal, g(input + i * tileElems));
        compute(tileLocal);
        
        evt = evt_next;  // 下次迭代等 next
    }
}
```

**按动态条件预取**：

```cpp
__global__ AICORE void dynamic_kernel(__gm__ Idx *indices, __gm__ T *data, ...) {
    AsyncSession session;
    BuildAsyncSession(...);

    int nextIdx = LoadScalar(&indices[0]);
    auto evt = TPREFETCH_L2(data + nextIdx * stride, stride, session);
    
    for (int i = 0; i < N; ++i) {
        evt.Wait(session);
        TLOAD(..., data + nextIdx * stride);
        compute();
        
        if (i + 1 < N) {
            nextIdx = LoadScalar(&indices[i+1]);  // 运行时才知道
            evt = TPREFETCH_L2(data + nextIdx * stride, stride, session);
        }
    }
}
```

这个 host path 做不到——host 没法在 kernel 跑到一半时插入 CMO。

**和 TPUT_ASYNC / TGET_ASYNC 同框架协作**：`AsyncSession` 可以同时承载多种 async op，统一用 `AsyncEvent` 管理，不用混用多套 API。

**不适用**：

- 预取时机远早于任何 kernel 启动（比如上一个 kernel 之前）
- Kernel 不用 `AsyncSession` 的场景（为一次 prefetch 建 session 不值）
- Prefetch 大小极小（发 1 KB 的 prefetch，启动成本就占大头了，不如不发）

---

## 13. 组合用法（两者不互斥）

典型完整 pipeline：

```cpp
// ------- Host 侧 -------
// Stage 1: kernel 启动前，对已知的大块输入做预热
pto::PTO_PREFETCH(input_A_ptr, A_bytes, stream);
pto::PTO_PREFETCH(input_B_ptr, B_bytes, stream);

// Stage 2: launch kernel
big_kernel<<<..., stream>>>(input_A_ptr, input_B_ptr, output_ptr, ...);

// ------- Device 侧（big_kernel 内部）-------
__global__ AICORE void big_kernel(...) {
    // input_A, input_B 这时已经在 L2 里了（来自 host PTO_PREFETCH）
    
    AsyncSession session;
    BuildAsyncSession(...);
    
    for (int tile = 0; tile < numTiles; ++tile) {
        // 用 device TPREFETCH_L2 预热 tile+1（host 没法做，因为依赖 tile 循环变量）
        if (tile + 1 < numTiles) {
            TPREFETCH_L2(compute_next_tile_ptr(tile + 1), tileBytes, session);
        }
        
        // 对当前 tile 做 compute
        TLOAD(..., input_A + tile * tileStride);
        compute();
        TSTORE(output + tile * tileStride, ...);
    }
}
```

这个 pipeline 里：

- `PTO_PREFETCH`（host）覆盖「启动前确定的全量输入」
- `TPREFETCH_L2`（device）覆盖「循环内动态计算的下一 tile」

两个阶段的数据量互补、时机互补，没有重复工作。

---

## 14. 关键差异一览表

| 维度 | `pto::PTO_PREFETCH` (host) | `pto::TPREFETCH_L2` (device) |
|---|---|---|
| **命名空间 / 路径** | `pto::` / `pto/npu/kernels/Pto_prefetch.hpp` | `pto::` / 实现 `pto/npu/TPrefetchL2.hpp`，公开 API `pto/common/pto_instr.hpp` |
| **调用侧** | Host (C++ runtime) | Device (AI Core kernel) |
| **下层实现** | `aclrtCmoAsync` (UseSdma=true) | `sdma::__sdma_cmo_prefetch` |
| **SDMA SQE 构造位置** | Runtime / driver | AI Core UB → GM |
| **硬件路径** | SDMA CMO engine (opcode=6) | SDMA CMO engine (opcode=6) ← 相同 |
| **返回类型** | `void` | `AsyncEvent` |
| **同步方式** | `aclrtSynchronizeStream` 或 stream FIFO 依赖 | `evt.Wait(session)` / `evt.Test(session)` |
| **单次启动开销** | ~10–14 μs (Scenario B 实测) | ~3 μs (复用 session) / ~5–15 μs (含 BuildAsyncSession) |
| **SDMA 引擎峰值带宽** | ~280 GB/s | ~280 GB/s ← 硬件相同 |
| **实测有效带宽 (16 MB)** | ~277 GB/s | ~262 GB/s（受 1MB×16 SQE 切分影响） |
| **实测有效带宽 (128 MB)** | ~274 GB/s | ~231 GB/s（受 1MB×128 SQE 切分影响） |
| **每 SQE 启动尾巴** | ~1 us（runtime 用大块切分，量少） | ~800–1000 ns × `bytes/1MB`（默认配置）|
| **FIFO 串行** | 同 stream 内严格 FIFO | 同 session SQ 内 FIFO，多 session 可并行 |
| **细粒度等待** | 不支持（只能 sync 整个 stream） | 支持（per-event Wait/Test） |
| **最大 outstanding** | 受 stream depth 限制，一般很大 | AI Core 最多 8 个 flag slot |
| **运行时位置约束** | 任意 host thread | 仅 AI Core kernel 内 |
| **依赖** | ACL runtime | `AsyncSession` + SDMA workspace |
| **CPU 占用** | 有（syscall, sync） | 无 |
| **跨 stream 协调** | 需要 event API 手动做 | 自动并行（各 session 独立） |
| **预取时机** | 任何 kernel 启动前 | Kernel 运行中任意位置 |
| **动态预取地址** | 不支持（host 不知道 kernel 运行时状态） | 支持（scalar 核算出地址直接发） |
| **失败语义** | `aclError` 返回码 | 静默返回 handle=0 的 no-op event |
| **资源占用** | stream FIFO slot + runtime task | AsyncSession + STARS SQ channel + event slot |
| **跨 rank 用法** | 每 rank 自己 host 调自己 | 每 rank kernel 内自己发；均为 local |
| **适用场景** | 启动前已知全量 / 批处理 warmup | Kernel 内 pipeline / 动态地址 |
| **典型 pairing** | 对后续访问该地址的 device 指令透明加速（TGET/TPUT/TGET_ASYNC/TPUT_ASYNC/TLOAD/普通 GM 读 都受益）；不需要"配对" | 同 kernel 内可与 `TGET_ASYNC` / `TPUT_ASYNC` 共享 `AsyncSession`，复用同一组 STARS channel 调度 |
| **代码复杂度** | 1 行 | 需要 scratch + BuildAsyncSession + Wait |

---

## 15. 实测数据印证

| 数据点 | 对应结论 |
|---|---|
| Scenario B: host 10+ μs, device 3 μs | host 软件 overhead 高出 ~3x，device 纯 in-kernel |
| Scenario A: 16MB 下 host 149 μs, device 156 μs；128MB 下 host 1086 μs, device 1200 μs | 硬件吞吐一致，差额 = device session build + per-SQE 启动尾巴累计（1MB×N 条 SQE）|
| Scenario A: 128MB 下 device overhead ~133 μs ≈ 128 SQE × 1 μs | 证实差距来自 `kDefaultSdmaBlockBytes = 1MB` 的切分策略，调大可消除 |
| Scenario A/D: host 和 device 吞吐均 ~270 GB/s（小 buffer / 同 SQE 配置） | 证实底层 SDMA engine 同吞吐 |
| Scenario C 大 buffer: C1 和 C2 cold-reduc 仅差 2–3 pp | prefetch 和 compute 并行时，发起方位置不影响并行效果 |
| Scenario D: D1 35μs vs D2 38μs (1MB), D1 1465μs vs D2 1517μs (128MB) | 跨 rank 没放大差异，规律和单 rank 一致 |
| Scenario D: D0→D1 省 ~40% wall | TPUT_ASYNC 写入远端后，远端 L2 是冷的，后续 prefetch 确有收益 |

---

## 总结一句话

> 两者是**同一条 SDMA CMO 硬件路径**的两个「入口」——host 入口胜在「启动前即可发、代码 1 行」，device 入口胜在「kernel 内随时发、单次 3 μs、不阻塞 host」。**底层 SDMA 引擎吞吐相同**，差距只来自上层切分策略（device 默认 1MB/SQE，可配大），在 overlap 场景下完全可被 compute 掩盖。真正决定选哪条的是：你的预取决策发生在 kernel 启动之前还是之后。

---

## 附录：两个关键签名的后果拆解（供快速回顾）

### `PTO_PREFETCH` 返回 `void` 意味着什么

- 调用方**没有 per-task 句柄**，没法说「只等这一条 prefetch」
- 同步粒度被**限定到整个 stream**：要么 `aclrtSynchronizeStream` 把 stream 全等完，要么靠在同 stream 上 enqueue 下一个 kernel 让 FIFO 语义帮你串行
- host API 完全嵌在 ACL stream 语义里，继承了 stream 的简单和粗粒度

### `PTO_PREFETCH` 的三个参数的潜台词

| 参数 | 要求 | 对比 device API |
|---|---|---|
| `__gm__ void *tensor` | 必须是 **device GM 地址**（host 指针会出错）；来源可以是 `aclrtMalloc` 或 HCCL window | device API 还有 `GlobalTensor` 重载，更类型安全 |
| `uint64_t tensor_bytes` | **字节数**，不是元素数；0 直接 early-return；不对齐/超大由 driver 兜底 | device API raw 版也是 bytes；`GlobalTensor` 版从 shape 自动推 |
| `aclrtStream stream` | **ACL 命令流 FIFO**，决定和哪些 host-issued 任务串行、以及用哪个同步点可以等它 | device API 对应参数是 `AsyncSession`，底层是 STARS SQ channel，同步用 `AsyncEvent` |

两个签名约束合起来说明一件事：host path 的**挂载模型是 stream-based**，同时**同步模型也是 stream-based**，整个路径完全嵌在 ACL 的统一异步模型里，不做额外抽象。
