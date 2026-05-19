# TREDUCE_SCATTER

## 简介

ReduceScatter 操作：所有 rank 集体对完整输入进行归约，然后每个 rank 获得归约结果的一个不同分片。

所有 rank 必须参与此操作。操作完成后，rank `r` 持有归约输出的第 `r` 个分片。

## 数学语义

给定 `N` 个 rank，总元素数 `M = N × sliceSize`：

$$\mathrm{output}^{(r)}_{i} = \bigoplus_{k=0}^{N-1} \mathrm{input}^{(k)}_{r \cdot S + i}, \quad i \in [0, S)$$

其中 $S$ 为每 rank 的分片大小，$r$ 为本地 rank 索引，$\oplus$ 为归约运算。

## 模板参数

- `engine`：
    - `CollEngine::CCU`（Ascend950，仅 NPU_ARCH 3510）

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
// 基础 ReduceScatter（累加 Tile + 接收 Tile）
template <CollEngine engine = CollEngine::CCU,
          typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TREDUCE_SCATTER(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                     TileData &accTileData, TileData &recvTileData, ReduceOp op, Args&... args);

// 乒乓 ReduceScatter（累加 Tile + ping/pong Tile）
template <CollEngine engine = CollEngine::CCU,
          typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TREDUCE_SCATTER(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                     TileData &accTileData, TileData &pingTileData, TileData &pongTileData,
                                     ReduceOp op, Args&... args);
```

当 `engine == CollEngine::CCU` 时，可变参数的第一个参数必须是 `CcuTriggerContext`。AIV kernel 触发 CKE gate，CCU 引擎执行实际的 ReduceScatter 数据路径（ReadNb 读取所有对端 + LocalReduceNb + LocalCopyNb）。

## 约束

- **引擎**：当前仅支持 `CollEngine::CCU`。
- **类型约束**：
    - `TileData::DType` 必须等于 `GlobalDstData::RawDType`。
- **内存约束**：
    - `dstGlobalData` 必须指向本地 HBM（当前 NPU）。
    - `accTileData`、`recvTileData` 必须为预分配的 UB Tile，且需初始化有效维度。
- **ParallelGroup 约束**：
    - `parallelGroup.tensors[r]` 必须指向 rank `r` 的输入缓冲区。
    - 所有 rank 必须通过宿主侧 API 注册并启动 CCU kernel。
- **CcuTriggerContext**：
    - `inputSource == AivStored`：AIV 将 `accTileData` TSTORE 到 `parallelGroup[selfIdx]` 后再触发 CKE。
    - `inputSource == HostManaged`：宿主已准备好输入 HBM，AIV 仅触发 CKE。

> **CCU 路径**：所有 rank 必须通过 `HcclCcuKernelRegister` / `HcclCcuKernelLaunch` 注册并启动 CCU kernel。完整示例参见 `tests/npu/a5/comm/st/testcase/treduce_scatter_ccu/`。

## 示例

### ReduceScatter 求和（AIV 融合路径）

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SLICE_SIZE, int NRANKS>
void reduce_scatter_sum(__gm__ T *inputVa, __gm__ T *outputVa, uint32_t selfIdx,
                        uint64_t ckeVA, uint32_t mask) {
    using TileT = Tile<TileType::Vec, T, 1, SLICE_SIZE, BLayout::RowMajor, -1, -1>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SLICE_SIZE>,
                                 Stride<SLICE_SIZE,SLICE_SIZE,SLICE_SIZE,SLICE_SIZE,1>, Layout::ND>;

    TileT accTile(1, SLICE_SIZE);
    TileT recvTile(1, SLICE_SIZE);
    GTensor inputGm(inputVa);
    GTensor outputGm(outputVa);

    GTensor rankTensors[NRANKS];
    for (int r = 0; r < NRANKS; ++r) rankTensors[r] = GTensor(inputVa);
    comm::ParallelGroup<GTensor> group(rankTensors, NRANKS, 0);

    // 用本 rank 的数据填充 accTile
    TEXPANDS(accTile, static_cast<T>(selfIdx + 1));

    comm::CcuTriggerContext ctx{ckeVA, mask, selfIdx, comm::CcuInputSource::AivStored};
    comm::TREDUCE_SCATTER<comm::CollEngine::CCU>(group, outputGm, accTile, recvTile, comm::ReduceOp::Sum, ctx);
}
```
