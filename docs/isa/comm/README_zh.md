# PTO 通信 ISA 参考手册

本目录包含 PTO 通信 ISA 的逐指令参考文档。

- 权威来源（C++ 内建接口）：`include/pto/comm/pto_comm_inst.hpp`
- 类型定义：`include/pto/comm/comm_types.hpp`

## 点对点通信（同步）
- [**TPUT**](TPUT_zh.md)：远程写（GM → UB → GM）
- [**TGET**](TGET_zh.md)：远程读（GM → UB → GM）

## 点对点通信（异步）
- [**TPUT_ASYNC**](TPUT_ASYNC_zh.md)：异步远程写（GM → DMA 引擎 → GM）
- [**TGET_ASYNC**](TGET_ASYNC_zh.md)：异步远程读（GM → DMA 引擎 → GM）

## 基于信号的同步
- [**TNOTIFY**](TNOTIFY_zh.md)：向远端 NPU 发送通知
- [**TWAIT**](TWAIT_zh.md)：阻塞等待信号条件满足
- [**TTEST**](TTEST_zh.md)：非阻塞检测信号条件

## 集合通信

- [**TGATHER**](TGATHER_zh.md)：从所有 rank 收集数据
- [**TSCATTER**](TSCATTER_zh.md)：向所有 rank 分发数据
- [**TREDUCE**](TREDUCE_zh.md)：从所有 rank 归约数据到本地
- [**TBROADCAST**](TBROADCAST_zh.md)：从当前 NPU 广播数据到所有 rank

## 类型定义

### NotifyOp

`TNOTIFY` 的操作类型：

| 值 | 说明 |
|-------|-------------|
| `NotifyOp::AtomicAdd` | 原子加（`signal += value`）|
| `NotifyOp::Set` | 直接赋值（`signal = value`）|

### WaitCmp

`TWAIT` 和 `TTEST` 的比较运算符：

| 值 | 说明 |
|-------|-------------|
| `WaitCmp::EQ` | 等于（`==`）|
| `WaitCmp::NE` | 不等于（`!=`）|
| `WaitCmp::GT` | 大于（`>`）|
| `WaitCmp::GE` | 大于等于（`>=`）|
| `WaitCmp::LT` | 小于（`<`）|
| `WaitCmp::LE` | 小于等于（`<=`）|

```cpp
// 用法示例（统一运行时参数风格）：
comm::TNOTIFY(signal, 1, comm::NotifyOp::Set);
comm::TWAIT(signal, 1, comm::WaitCmp::EQ);
comm::TTEST(signal, 1, comm::WaitCmp::GE);
```

### ReduceOp

`TREDUCE` 的归约运算符：

| 值 | 说明 |
|-------|-------------|
| `ReduceOp::Sum` | 逐元素求和 |
| `ReduceOp::Max` | 逐元素取最大值 |
| `ReduceOp::Min` | 逐元素取最小值 |

### AtomicType

`TPUT` 的原子操作类型（定义于 `include/pto/common/constants.hpp`）：

| 值 | 说明 |
|-------|-------------|
| `AtomicType::AtomicNone` | 无原子操作（默认）|
| `AtomicType::AtomicAdd` | 原子加操作 |

### DmaEngine

`TPUT_ASYNC` 和 `TGET_ASYNC` 的 DMA 后端选择：

| 值 | 说明 |
|-------|-------------|
| `DmaEngine::SDMA` | SDMA 引擎。当前异步路径仅支持扁平连续的逻辑一维 tensor。|
| `DmaEngine::URMA` | URMA 引擎（User-level RDMA Memory Access）。仅支持扁平连续的逻辑一维 tensor；仅在 Ascend950（NPU_ARCH 3510）上可用，且要求 CANN Toolkit **>= 9.1.0**。|

### CollEngine

集合通信指令（`TGATHER`、`TSCATTER`、`TREDUCE`、`TBROADCAST`）的后端引擎选择：

| 值 | 说明 |
|-------|-------------|
| `CollEngine::AIV` | 默认。在 AI Vector 上通过 `TLOAD` + 计算 + `TSTORE` 的 tile 路径执行。|
| `CollEngine::CCU` | AIV 触发 CKE gate，由 CCU 硬件执行实际的集合通信。仅在 Ascend950（NPU_ARCH 3510）上可用。选用此后端时，调用方需要传入 `CcuTriggerContext` 作为第一个可变参数。|

### CcuTriggerContext

`CollEngine::CCU` 场景下由宿主传递给 AIV kernel 的不透明上下文。宿主在启动 kernel 前通过 `ccu::TryGet()` + `rtGetDevResAddress()` 填充。

```cpp
struct CcuTriggerContext {
    uint64_t ckeSlotVA;  // 来自 rtGetDevResAddress(dieId, ckeId)
    uint32_t mask;       // 16 位 CKE 触发掩码
};
```

### AsyncEvent

由 `TPUT_ASYNC` / `TGET_ASYNC` 返回：

```cpp
struct AsyncEvent {
    uint64_t handle;
    DmaEngine engine;

    bool valid() const;                           // handle != 0 时返回 true
    bool Wait(const AsyncSession &session) const; // quiet 语义等待（详见 TPUT_ASYNC）
    bool Test(const AsyncSession &session) const; // 非阻塞完成检测
};
```

### AsyncSession

用于异步 DMA 操作的引擎无关会话对象，构建一次后传递给所有异步调用：

```cpp
comm::AsyncSession session;
comm::BuildAsyncSession<comm::DmaEngine::SDMA>(scratchTile, workspace, session);
```

定义于 `include/pto/comm/async_common/async_types.hpp`。构建参数详见 [TPUT_ASYNC](TPUT_ASYNC_zh.md)。

### Signal / Signal2D / GlobalSignal

用于信号同步（`TNOTIFY` / `TWAIT` / `TTEST`）的 `GlobalTensor` 别名：

```cpp
using Signal = GlobalTensor<int32_t, Shape<1,1,1,1,1>, Stride<1,1,1,1,1>, Layout::ND>;

// 稠密：stride = Cols；带步幅构造函数可传入自定义 DIM_3 stride 表示大网格子区域。
template <int Rows, int Cols> struct Signal2D;

// 通用别名：自定义形状/步幅时使用。
template <typename E, typename S, typename St, Layout L = Layout::ND>
using GlobalSignal = GlobalTensor<E, S, St, L>;
```

### ParallelGroup

多 NPU 集合通信（`TGATHER` / `TSCATTER` / `TREDUCE` / `TBROADCAST`）的包装器：

```cpp
template <typename GlobalData>
struct ParallelGroup {
    GlobalData *tensors;   // 每个 rank 的 GlobalTensor 视图（本地或远端 GM）
    int nranks;
    int rootIdx;           // 组内所有 rank 必须传入相同的值

    static ParallelGroup Create(GlobalData *tensorArray, int size, int rootIdx);

    int  GetRootIdx() const;
    int  GetSize()    const;
    bool empty()      const;
    GlobalData&       operator[](int teamRank);
    const GlobalData& operator[](int teamRank) const;
};
```

