# TALL_TO_ALL

## 简介

AllToAll 操作（等长）：全数据交换，每个 rank 向其他所有 rank 发送各自不同的分片。

所有 rank 必须参与。每个 rank 的输入在逻辑上被划分为 `N` 个分片（每个 rank 一个）。操作完成后，rank `r` 的输出分片 `i` 包含 rank `i` 发送给 rank `r` 的数据。

## 数学语义

给定 `N` 个 rank，每个有 `N × S` 个元素的输入：

$$\mathrm{output}^{(r)}_{i \cdot S + j} = \mathrm{input}^{(i)}_{r \cdot S + j}, \quad i \in [0, N),\; j \in [0, S)$$

即：rank `r` 从 rank `i` 接收 rank `i` 指定发给 rank `r` 的分片。

## 模板参数

- `engine`：
    - `CollEngine::AIV`（默认）— 每个 rank 从各对端 TLOAD 其指定分片
    - `CollEngine::CCU`（Ascend950，仅 NPU_ARCH 3510）— AIV 触发 CKE gate，CCU 引擎处理数据路径

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
// AIV 路径（基于 ParallelGroup，A2A3）
template <CollEngine engine = CollEngine::AIV,
          typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TALL_TO_ALL(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                 TileData &stagingTileData, Args&... args);

// CCU 路径（GlobalSrc + GlobalDst，A5）
template <CollEngine engine = CollEngine::CCU,
          typename GlobalSrcData, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TALL_TO_ALL(GlobalSrcData &srcGlobalData, GlobalDstData &dstGlobalData,
                                 TileData &srcTileData, Args&... args);
```

当 `engine == CollEngine::AIV`（默认）时，AIV 引擎直接执行 all-to-all 交换：每个 rank 通过 `ParallelGroup` 从所有对端 TLOAD 其指定分片，并 TSTORE 到本地输出。

当 `engine == CollEngine::CCU` 时，可变参数的第一个参数必须是 `CcuTriggerContext`。AIV kernel 触发 CKE gate，CCU 引擎执行实际的 AllToAll 数据交换（WriteNb 从本地输入分片写到每个对端 output 的对应偏移 + 本地 LocalCopyNb）。

## 约束

- **类型约束**：
    - `TileData::DType` 必须等于 `GlobalSrcData::RawDType`。
- **AIV 路径**：
    - `ParallelGroup[r]` 指向 rank `r` 的完整输入缓冲区（N × sliceSize）。每个 rank 从各对端 TLOAD 第 selfIdx 个分片，并 TSTORE 到本地输出的相应偏移处。无需宿主侧 CCU kernel 注册。完整示例参见 `tests/npu/a2a3/comm/st/testcase/tall_to_all/`。
- **内存约束**：
    - `srcGlobalData` 指向本 rank 的输入缓冲区（本地 HBM，大小 = N × sliceSize）。
    - `dstGlobalData` 指向输出缓冲区（本地 HBM，大小 = N × sliceSize）。
    - `srcTileData` 必须为预分配的 UB Tile，且需初始化有效维度。
- **数据布局**：
    - 输入被划分为 N 个 `sliceSize` 字节的连续分片。
    - 本 rank 输入的分片 `i` 被发送到 rank `i`。
    - 操作完成后，本 rank 输出的分片 `i` 包含 rank `i` 发送给本 rank 的数据。
- **CcuTriggerContext**：
    - `inputSource == AivStored`：AIV 将 `srcTileData` TSTORE 到 `srcGlobalData` 后再触发 CKE。
    - `inputSource == HostManaged`：宿主已准备好输入 HBM，AIV 仅触发 CKE。

> **CCU 路径**：所有 rank 必须通过 `HcclCcuKernelRegister` / `HcclCcuKernelLaunch` 注册并启动 CCU kernel。完整示例参见 `tests/npu/a5/comm/st/testcase/tall_to_all_ccu/`。

## 示例

### AllToAll（AIV 路径，A2A3）

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SLICE_SIZE, int NRANKS>
void all_to_all(__gm__ T *remoteAddrs[NRANKS], __gm__ T *output, int myRank) {
    using TileT = Tile<TileType::Vec, T, 1, SLICE_SIZE>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,NRANKS,SLICE_SIZE>,
                                 Stride<NRANKS*SLICE_SIZE,NRANKS*SLICE_SIZE,NRANKS*SLICE_SIZE,SLICE_SIZE,1>, Layout::ND>;

    GTensor tensors[NRANKS];
    for (int r = 0; r < NRANKS; ++r) tensors[r] = GTensor(remoteAddrs[r]);
    comm::ParallelGroup<GTensor> group(tensors, NRANKS, myRank);

    GTensor dstG(output);
    TileT stagingTile;

    comm::TALL_TO_ALL(group, dstG, stagingTile);
}
```

### AllToAll（CCU 路径，A5）

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int TOTAL_SIZE>
void all_to_all(__gm__ T *inputVa, __gm__ T *outputVa, uint32_t selfIdx,
                uint64_t ckeVA, uint32_t mask) {
    using TileT = Tile<TileType::Vec, T, 1, TOTAL_SIZE, BLayout::RowMajor, -1, -1>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,TOTAL_SIZE>,
                                 Stride<TOTAL_SIZE,TOTAL_SIZE,TOTAL_SIZE,TOTAL_SIZE,1>, Layout::ND>;

    TileT srcTile(1, TOTAL_SIZE);
    GTensor inputGm(inputVa);
    GTensor outputGm(outputVa);

    // 填充输入：统一值 (selfIdx + 1) —— 实际场景中每个分片可能不同
    TEXPANDS(srcTile, static_cast<T>(selfIdx + 1));

    comm::CcuTriggerContext ctx{ckeVA, mask, selfIdx, comm::CcuInputSource::AivStored};
    comm::TALL_TO_ALL<comm::CollEngine::CCU>(inputGm, outputGm, srcTile, ctx);
}
```
