# TALL_GATHER

## 简介

AllGather 操作：每个 rank 贡献一块数据，操作完成后所有 rank 持有所有 rank 贡献的完整拼接结果。

所有 rank 必须参与。操作完成后，所有 rank 持有相同的完整输出缓冲区，按 rank 索引顺序包含所有数据块。

## 数学语义

给定 `N` 个 rank，每个贡献 `S` 个元素：

$$\mathrm{output}^{(\text{任意 rank})}_{r \cdot S + i} = \mathrm{input}^{(r)}_{i}, \quad r \in [0, N),\; i \in [0, S)$$

所有 rank 的输出相同，按 rank 顺序拼接包含所有 rank 的输入。

## 模板参数

- `engine`：
    - `CollEngine::AIV`（默认）— 每个 rank TLOAD 所有对端数据并组装完整输出
    - `CollEngine::CCU`（Ascend950，仅 NPU_ARCH 3510）— AIV 触发 CKE gate，CCU 引擎处理数据通路

## C++ 内建接口

声明于 `include/pto/comm/pto_comm_inst.hpp`：

```cpp
// AIV 路径（基于 ParallelGroup，A2A3）
template <CollEngine engine = CollEngine::AIV,
          typename ParallelGroupType, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TALL_GATHER(ParallelGroupType &parallelGroup, GlobalDstData &dstGlobalData,
                                 TileData &stagingTileData, Args&... args);

// CCU 路径（GlobalSrc + GlobalDst，A5）
template <CollEngine engine = CollEngine::CCU,
          typename GlobalSrcData, typename GlobalDstData, typename TileData, typename... Args>
PTO_INST RecordEvent TALL_GATHER(GlobalSrcData &srcGlobalData, GlobalDstData &dstGlobalData,
                                 TileData &srcTileData, Args&... args);
```

当 `engine == CollEngine::AIV` 时，每个 rank 遍历 ParallelGroup，TLOAD 每个对端的源缓冲区，然后 TSTORE 到本地输出的对应偏移位置。

当 `engine == CollEngine::CCU` 时，可变参数的第一个参数必须是 `CcuTriggerContext`。AIV kernel 触发 CKE gate，CCU 引擎执行实际的 AllGather（WriteNb 写入每个对端 output 的对应偏移 + 本地 LocalCopyNb）。

## 约束

- **类型约束**：
    - `TileData::DType` 必须等于 `GlobalSrcData::RawDType`。
- **内存约束**：
    - `srcGlobalData` 指向本 rank 的输入缓冲区（本地 HBM）。
    - `dstGlobalData` 指向输出缓冲区（本地 HBM，将被填充所有 rank 的数据）。
    - `srcTileData` 必须为预分配的 UB Tile，且需初始化有效维度。
- **AIV 路径**：ParallelGroup[r] 指向 rank r 的源缓冲区。每个 rank TLOAD 每个对端的数据并 TSTORE 到本地输出的对应偏移。无需宿主侧 CCU kernel 注册。完整示例参见 `tests/npu/a2a3/comm/st/testcase/tall_gather/`。
- **CcuTriggerContext**（仅 CCU 路径）：
    - `inputSource == AivStored`：AIV 将 `srcTileData` TSTORE 到 `srcGlobalData` 后再触发 CKE。
    - `inputSource == HostManaged`：宿主已准备好输入 HBM，AIV 仅触发 CKE。

> **CCU 路径**：所有 rank 必须通过 `HcclCcuKernelRegister` / `HcclCcuKernelLaunch` 注册并启动 CCU kernel。完整示例参见 `tests/npu/a5/comm/st/testcase/tall_gather_ccu/`。

## 示例

### AllGather（AIV 路径，A2A3）

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int COUNT, int NRANKS>
void all_gather(__gm__ T *remoteAddrs[NRANKS], __gm__ T *output, int myRank) {
    using TileT = Tile<TileType::Vec, T, 1, COUNT>;
    using SrcTensor = GlobalTensor<T, Shape<1,1,1,1,COUNT>,
                                   Stride<COUNT,COUNT,COUNT,COUNT,1>, Layout::ND>;
    using DstTensor = GlobalTensor<T, Shape<1,1,1,NRANKS,COUNT>,
                                   Stride<NRANKS*COUNT,NRANKS*COUNT,NRANKS*COUNT,COUNT,1>, Layout::ND>;

    SrcTensor tensors[NRANKS];
    for (int r = 0; r < NRANKS; ++r) tensors[r] = SrcTensor(remoteAddrs[r]);
    comm::ParallelGroup<SrcTensor> group(tensors, NRANKS, myRank);

    DstTensor dstG(output);
    TileT stagingTile;

    comm::TALL_GATHER(group, dstG, stagingTile);
}
```

### AllGather（CCU 路径，A5）

```cpp
#include <pto/comm/pto_comm_inst.hpp>

using namespace pto;

template <typename T, int SLICE_SIZE>
void all_gather(__gm__ T *inputVa, __gm__ T *outputVa, uint32_t selfIdx,
                uint64_t ckeVA, uint32_t mask) {
    using TileT = Tile<TileType::Vec, T, 1, SLICE_SIZE, BLayout::RowMajor, -1, -1>;
    using GTensor = GlobalTensor<T, Shape<1,1,1,1,SLICE_SIZE>,
                                 Stride<SLICE_SIZE,SLICE_SIZE,SLICE_SIZE,SLICE_SIZE,1>, Layout::ND>;

    TileT srcTile(1, SLICE_SIZE);
    GTensor inputGm(inputVa);
    GTensor outputGm(outputVa);

    TEXPANDS(srcTile, static_cast<T>(selfIdx + 1));

    comm::CcuTriggerContext ctx{ckeVA, mask, selfIdx, comm::CcuInputSource::AivStored};
    comm::TALL_GATHER<comm::CollEngine::CCU>(inputGm, outputGm, srcTile, ctx);
}
```
