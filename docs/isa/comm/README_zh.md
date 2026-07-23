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
| `NotifyOp::Set` | 直接赋值（`signal = value`）|
| `NotifyOp::AtomicAdd` | 原子加（`signal += value`）|

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

`TPUT_ASYNC` 和 `TGET_ASYNC` 的 DMA 引擎选择：

| 值 | 说明 |
|-------|-------------|
| `DmaEngine::SDMA` | SDMA 引擎（支持一维传输，Ascend950 上仅支持TGET|
| `DmaEngine::URMA` | URMA 引擎（支持一维传输，仅 Ascend950 / NPU_ARCH 3510；要求 CANN >= 9.1.0）|
| `DmaEngine::RDMA` | RDMA 引擎（支持一维传输，仅 Ascend950 / NPU_ARCH 3510）。当前网卡后端为 HNS1825，且必须在配置阶段使能。 |

### AsyncEvent

由 `TPUT_ASYNC` / `TGET_ASYNC` 返回，用于同步传输完成状态：

```cpp
struct AsyncEvent {
    uint64_t handle;
    DmaEngine engine;

    bool valid() const;                        // handle != 0 时返回 true
    bool Wait(const AsyncSession &session) const; // 阻塞直到传输完成
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

### RDMA 后端与 Host 控制面

`DmaEngine::RDMA` 表示使用 RDMA 通信，`RdmaBackend` 表示编入二进制的具体网卡实现；一个二进制最多包含
一个 RDMA 后端。当前唯一支持的值是 `RdmaBackend::HNS_1825`。

Ascend950 / NPU_ARCH 3510 通信 ST 必须在首次 CMake 配置前选择后端：

```bash
export PTO_RDMA_BACKEND=HNS_1825
python3 tests/script/run_st.py -r npu -v a5 -t comm/tput_async_hns1825 -d -n 2
```

`PTO_RDMA_BACKEND` 是配置阶段输入。CMake 将其转换成 Host 与 Device 一致的编译定义，生成的二进制不会在
运行时读取该变量。未设置、空值或不支持的值都会生成不包含 RDMA 后端的产物。修改取值后必须重新配置构建
目录。

Host 代码包含 `pto/comm/async/rdma/rdma_workspace_manager.hpp`，并遵循以下生命周期：

```cpp
pto::comm::rdma::RdmaWorkspaceManager manager;
if (manager.Preflight() != pto::comm::rdma::WorkspaceInitResult::READY) {
    // RDMA 未使能，或所选后端不支持当前架构。
}

pto::comm::rdma::WorkspaceConfig config;
// 填写 rankId/rankCount、本地及各 peer 的 RDMA 网卡信息，
// 以及各 rank 注册通信缓冲区的地址。
if (manager.Init(config) != pto::comm::rdma::WorkspaceInitResult::READY) {
    // 初始化失败。
}
void *rdmaWorkspace = manager.GetWorkspaceAddr();

// 启动使用 rdmaWorkspace 构建 DmaEngine::RDMA 会话的 Kernel。

bool finalized = manager.Finalize();
```

应用必须在调用 `Init` 前，以一致方式在各 rank 间交换：

- rank id 和 rank 总数；
- 本地物理设备 id 与 RDMA 网卡 IPv4；
- 每个 rank 的物理设备 id 与 RDMA 网卡 IPv4；
- 每个 rank 注册通信缓冲区的 Device 虚拟地址；
- 公共 base port。

Manager 不负责 MPI/HCCL bootstrap。它负责创建 HCOMM endpoint、注册本地通信缓冲区、为每个 peer 创建一条
RoCE channel、等待建链完成，并发布包含 HNS1825 队列与 MR 元数据的 RDMA Device workspace。必须在释放
已注册缓冲区之前调用 `Finalize()`；该接口依次销毁 channel、注销内存、销毁 endpoint，并释放 Device
workspace。

当前 HNS1825 约束：

- HNS1825 后端仅支持 Ascend950 / NPU_ARCH 3510；在不支持的目标上，`Preflight()` 返回 `ERROR`。
- 本地和远端操作数的完整范围都必须位于 `Init` 时注册的通信缓冲区内。这里的“对称”表示各 rank 注册相同
  的逻辑区域；由于 peer base address 会显式交换，不要求各 rank 的 Device 虚拟地址相同。
- 单次传输最多 `0x7fffffff` 字节。
- RDMA `scratchTile` 必须提供至少 64 字节 UB，`syncId` 必须在 `[0, 7]` 范围内。
- 按当前端口映射，同一个 RDMA 网卡 IPv4 最多承载 16 个 rank。
- Host 控制面依赖 HCOMM。若 HNS1825 verbs provider 不在默认 provider 路径，部署环境可能还需要通过
  `IBV_EXTEND_DRIVERS` 指向 `libhrn5-rdmav34.so`。

`HCCL_RDMA_TC`（默认 `132`）和 `HCCL_RDMA_SL`（默认 `4`）分别配置 RoCE traffic class 与 service level。
`PTO_ROCE_VERBOSE=1` 可打开 Host 控制面过程日志。仓库 ST 使用的 root-info 与 MPI 便利变量见
[tests/README_zh.md](../../../tests/README_zh.md)；它们不是 PTO 库的运行时后端选择输入。

### ParallelGroup

用于多 NPU 集合通信的包装器：

```cpp
template <typename GlobalData>
struct ParallelGroup {
    // 指向 `GlobalData` 对象数组的指针（每个对象封装一个 GM 地址）。
    // 数组本身是本地元数据；封装的地址可以指向本地或远端 GM，
    // 具体取决于集合通信指令的语义。
    GlobalData *tensors;
    int nranks;   // rank 总数
    int rootIdx;  // 根 NPU 的 rank 索引

    // 工厂函数（推荐）：从已有 tensor 数组构建。
    static ParallelGroup Create(GlobalData *tensorArray, int size, int rank_id);
};
```
