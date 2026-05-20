# dispatch_ffn_combine_v3 AscendC Tensor PTO 化任务设计

## 目标

把 `op_kernel/utils/` 中仍暴露在业务函数里的 `AscendC::LocalTensor` / `AscendC::GlobalTensor` 逐步收口到 PTO 风格接口。

底层逻辑：

- 长期 GM 状态使用 raw `__gm__ *` 指针，不使用长期持有的 `pto::GlobalTensor` 成员。
- PTO `GlobalTensor` 只在 `TLOAD` / `TSTORE` / `TGET` / `TPUT` 的 use site 临时构造。
- 片上 buffer 从 `AscendC::LocalTensor` 逐步改为显式 UB/L1/L0/FP offset 或 PTO Tile wrapper。
- 不先替换无法证明等价的 matmul/fixpipe substrate；这类只先做 seam 收口。
- 每个小阶段都必须 build，并跑 A3 small/large。

## 验证命令

Build：

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
export PATH=/home/ntlab/miniconda3/envs/ltr_pto/bin:$PATH
export LD_LIBRARY_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib:$LD_LIBRARY_PATH
export MPI_LIB_PATH=/home/ntlab/miniconda3/envs/ltr_pto/lib/libmpi.so
cmake -S kernels/manual/a2a3/dispatch_ffn_combine_v3 \
  -B kernels/manual/a2a3/dispatch_ffn_combine_v3/build
cmake --build kernels/manual/a2a3/dispatch_ffn_combine_v3/build \
  --target dispatch_ffn_combine_v3 -j16
```

Small：

```bash
bash kernels/manual/a2a3/dispatch_ffn_combine_v3/run.sh \
  --world-size 2 \
  --m 16 --k 128 --n 128 \
  --topk 2 --experts 2 --max-output-size 32
```

Large：

```bash
bash kernels/manual/a2a3/dispatch_ffn_combine_v3/run.sh \
  --world-size 2 \
  --m 4097 --k 128 --n 128 \
  --topk 2 --experts 2 --max-output-size 8194
```

通过标准：build 成功，small/large 均 `PASS rank=0` 和 `PASS rank=1`。

---

## 修改顺序总览

| 顺序 | 范围 | 风险 | 目标 |
| --- | --- | --- | --- |
| 1 | `block_epilogue_pertoken_row.hpp` 的 GM 临时 wrapper | 低 | 去掉 operator 内 `AscendC::GlobalTensor` 临时变量，直接 raw pointer + PTO helper |
| 2 | `block_epilogue_pertoken_v2.hpp` 的 GM 临时 wrapper 与 token count wrapper | 低-中 | `gmC/gmPerTokenScale/gmRemotePeer/gmLocalScratch/tokenPerExpert` raw pointer 化 |
| 3 | `block_epilogue_pertoken_swiglu.hpp` 的 GM 临时 wrapper | 中 | `gmC/gmD/gmPerTokenScale*` raw pointer 化，并保留 UB LocalTensor 暂不动 |
| 4 | `pto_vector_ops.hpp` 增加 raw pointer / UB offset overload | 中 | 先新增接口，不替换全部调用点 |
| 5 | 三个 epilogue 的 UB `LocalTensor` 成员逐步改显式 UB offset | 中-高 | 用 `pto::Tile` 替代 UB LocalTensor 切片/计算 |
| 6 | `block_mmad_preload_async_fixpipe_quant.hpp` soft-flag/GM wrapper 去除 | 中 | raw pointer 输入 + use-site AscendC substrate 局部化 |
| 7 | `block_mmad_preload_async_fixpipe_quant.hpp` L1/L0/FP `LocalTensor` seam 设计 | 高 | 仅做 offset seam，不急于替换 MMAD substrate |
| 8 | `dispatch_policy_custom.hpp` substrate bridge 分层 | 高 | 保持 AscendC substrate 集中，不做机械 PTO drop-in |

### `dispatch_policy_custom.hpp` / `block_mmad_preload_async_fixpipe_quant.hpp` 剩余 LocalTensor 优先级

当前目标不是一次性清空 `dispatch_policy_custom.hpp` 的所有 `AscendC::LocalTensor`，而是把业务层依赖逐步压到 substrate seam 内部。优先级按“可证明等价 + 单批可验证”排序。

| 优先级 | 状态 | 范围 | 目标 | 验证 |
| --- | --- | --- | --- | --- |
| P0 | 已完成 | `LocalTensorBufferBase::GetBufferAddrByByte`、`PtoTileMmad`、soft-flag L1 load/store | 建立显式地址 seam；MMAD `TASSIGN` 改 explicit L0 offset；soft-flag 改 explicit L1 offset | build + A3 small/large 已通过 |
| P1 | 已完成 | int8 per-channel scale 链路：`l1STensor`、`fixpipeBuf`、`StagePerChannelScale`、`CopyL1ToFP` | S 的 GM/L1/FP 搬运接口已改成 raw pointer + explicit L1/FP offset；scale/fixpipe buffer 已收口，A/B matrix 主路径未扩散改动 | build + A3 small/large 已通过 |
| P2 | 已完成 | `CopyGmToL1` A/B `GM -> L1` | A/B GM→L1 已在 `block_mmad` detail 层改为 explicit L1 byte offset + PTO `TLOAD(Mat, GlobalTensor)`；旧 `CopyGmToL1` LocalTensor operator 与 DataCopy bridge 已删除 | build + A3 small/large 已通过 |
| P3 | 已完成 | `CopyL1ToL0A` / `CopyL1ToL0B` | 已替换为 PTO `TMOV(Mat -> Left/Right)`；`block_mmad` 不再向 policy 传 A/B 的 LocalTensor slice | build + A3 small/large 已通过 |
| P4 | 已完成 | `CopyL0CToGm` / `PtoFixpipeL0CToGm` | store use-site 已改为 explicit L0C offset；half 路径走 PTO `TSTORE(Acc -> GM)`，int8 路径走 `TSTORE_FP` 并保留 scaling/quant 语义 | build + A3 small/large 已通过 |
| P5 | 已完成 | `GetBufferByByte` 和 backing `AscendC::LocalTensor<uint8_t> tensor` | `GetBufferByByte` 已删除；resource buffer 不再长期保存 `AscendC::LocalTensor<uint8_t>`，只保存 `baseAddr`，构造期局部 tensor 仅用于取物理地址 | grep + build + A3 small/large 已通过 |

执行纪律：每完成一个优先级，只进入下一优先级前必须重新构建并跑 A3 small/large；若某一级无法证明 PTO 等价，保留 AscendC substrate 在 seam 内部，不把风险扩散到业务层。

#### P2 联动分析

`block_mmad_preload_async_fixpipe_quant.hpp` 里当前只剩两个业务层 `LocalTensor` 成员：

```cpp
AscendC::LocalTensor<ElementA> l1ATensorList[L1_STAGES];
uint64_t l1AOffsetList[L1_STAGES];
AscendC::LocalTensor<ElementB> l1BTensorList[L1_STAGES];
uint64_t l1BOffsetList[L1_STAGES];
```

它们只服务 `BlockMmad::operator()` 的 A/B GM→L1 preload：

```cpp
copyGmToL1A(l1ATensorList[l1ListId], gmTileA, MakeL1ALayout(), layoutTileA);
copyGmToL1B(l1BTensorList[l1ListId], gmTileB, MakeL1BLayout(), layoutTileB, AscendC::CacheMode::CACHE_MODE_DISABLE);
copyGmToL1B(l1BTensorList[l1ListId], gmTileB, MakeL1BLayout(), layoutTileB);
```

真正的 LocalTensor 依赖在 `dispatch_policy_custom.hpp` 的 `CopyGmToL1` cluster：

- `GemmType<Element, layout::ND> -> GemmType<Element, layout::Zn, A1>`：A 路 ND→Zn，用 `Nd2NzParams`，包含 stride 小于 `STRIDE_LIMIT` 的整块路径和逐行 fallback。
- `GemmType<Element, layout::Zn> -> GemmType<Element, layout::Zn, A1>`：B 路 Zn→Zn，用 `DataCopyParams`，包含 `stride(3)` 小于 `STRIDE_LIMIT` 的整块路径和按 C0 block fallback。
- `GemmType<Element, layout::VectorLayout> -> GemmType<Element, layout::VectorLayout, A1>`：当前主链路只剩 scale/flag 之外的旧泛化入口，先不和 A/B matrix P2 混改。

P2 的底层逻辑：A/B 的 L1 目标地址已经有 `l1AOffsetList/l1BOffsetList`，后续 L1→L0 已经按 explicit offset 走 `PtoMoveL1ToL0A/B`，所以删除 `l1ATensorList/l1BTensorList` 的抓手不是重新规划 L1，而是让 `CopyGmToL1` 直接消费 L1 byte offset。风险点集中在 GM→L1 搬运指令语义：当前 AscendC `DataCopy`/`Nd2NzParams` 对 ND→Zn 和 Zn→Zn 的布局转换、stride fallback、L2 cache hint 都已经正确；PTO `TLOAD` 是否能完全表达这些模式必须分路径证明。

#### P2 子任务设计

| 子任务 | 状态 | 范围 | 修改内容 | 验证 |
| --- | --- | --- | --- | --- |
| P2-0 | 待做 | live call site 分类 | 保留 `Resource::LocalTensorBufferBase` 的 backing tensor；只把本轮目标锁定到 `CopyGmToL1` A/B。`CopyL1ToL0A/B`、`CopyL1ToFP`、`CopyL0CToGm` 的旧 LocalTensor API 先标记为 policy 死/兼容 seam，不同批删除 | grep call site，无代码语义修改可不跑 |
| P2-1 | 待做 | `CopyGmToL1` offset overload seam | 在 `dispatch_policy_custom.hpp` 为 A 路/B 路新增 `operator()(uint64_t dstOffset, __gm__ Element *srcPtr, LayoutDst, LayoutSrc[, CacheMode])`，先内部用 `LocalTensorBufferBase::GetBufferByByte` 或同等临时 view 转回旧实现，保证业务层签名先切到 offset，算法语义不变 | build + A3 small/large |
| P2-2 | 待做 | `block_mmad` 调用面 | `InitL1` 不再填 `l1ATensorList/l1BTensorList`；`copyGmToL1A/B(...)` 改传 `l1AOffsetList[l1ListId]` / `l1BOffsetList[l1ListId]`；删除两个 `LocalTensor` 成员。此阶段允许 policy seam 内部仍临时构造 LocalTensor | build + A3 small/large；本文件 grep 不再有 `LocalTensor` |
| P2-3 | 待做 | A 路 ND→Zn PTO 化 | 只替换 `GemmType<Element, layout::ND> -> layout::Zn` 的 offset overload 内部实现。分别验证整块 `Nd2NzParams` 路径和逐行 fallback；若 PTO `TLOAD` 无法表达 stride/layout，保持 AscendC seam，不扩散回业务层 | build + A3 small/large |
| P2-4 | 待做 | B 路 Zn→Zn PTO 化 | 只替换 `GemmType<Element, layout::Zn> -> layout::Zn` 的 offset overload 内部实现。覆盖普通路径、`stride(3)` fallback、`CACHE_MODE_DISABLE` 分支；若 PTO `GlobalTensor` 没有等价 ZN layout 表达，保持 AscendC seam | build + A3 small/large |
| P2-5 | 待做 | policy 死 seam 清理 | P2-3/P2-4 通过后再清理未引用的 `PtoDataCopyGmToL1*`、`CopyL1ToL0A/B`、`CopyL1ToFP`、`CopyL0CToGm` LocalTensor 旧入口；清理前必须用 grep 证明没有 live call site | grep + build + A3 small/large |
| P2-6 | 待做 | `GetBufferByByte` / backing tensor 决策 | 如果所有 live policy 都已切 explicit offset，评估 `GetBufferByByte` 是否仍被其他文件使用；只有全局 grep 无业务引用时才删除或降级为内部兼容工具 | grep + build + A3 small/large |

P2 复盘：A/B GM→L1 已从 `l1ATensorList/l1BTensorList` + `AscendC::DataCopy` 改为 `l1AOffsetList/l1BOffsetList` + PTO `TLOAD(Mat, GlobalTensor)`。A 路用 ND global 写入 L1 Mat tile；B 路用 NZ global 写入 L1 Mat tile；旧 `PtoDataCopyGmToL1*` bridge 与 `CopyGmToL1` 的 LocalTensor operator 已删除。2026-05-20 已完成 grep、build、A3 small、A3 large，rank0/rank1 均 PASS。

P5 复盘：`Resource::LocalTensorBufferBase` 已从长期持有 `AscendC::LocalTensor<uint8_t> tensor` 改为只保存 `uint64_t baseAddr`；各 position buffer 构造函数只局部取 `buf.Get<uint8_t>()` 并保存物理地址。2026-05-20 已完成 grep、build、A3 small、A3 large，rank0/rank1 均 PASS。

P3 复盘：`CopyL1ToL0A` / `CopyL1ToL0B` 已从 `AscendC::LoadData` / `LoadDataWithTranspose` 调用面收口到 `PtoMoveL1ToL0A/B`，由显式 L1/L0 byte offset 绑定 `Mat` tile 与 `Left/Right` tile 后执行 `pto::TMOV`。2026-05-20 已完成 build、A3 small、A3 large，rank0/rank1 均 PASS。

P4 复盘：`StoreAccumulator` 已从 `AscendC::LocalTensor` L0C 输入改为 explicit L0C byte offset。half 路径通过 `pto::TSTORE(Acc -> GM)` 承接非量化输出，并按 `unitFlag` 映射 `pto::STPhase::Partial/Final`；int8 路径继续用 `pto::TSTORE_FP` 承接 per-channel scaling/quant。2026-05-20 已完成 build、A3 small、A3 large，rank0/rank1 均 PASS。

P5 复盘：已清理 `block_mmad_preload_async_fixpipe_quant.hpp` 中只用于初始化但实际已由 offset 替代的 `fixpipeBuf/l1STensor/l1FTensor/l0ATensorList/l0BTensorList/l0CTensorList`。当前该文件仅剩 `l1ATensorList/l1BTensorList`，用于 P2 的 GM→L1 substrate；`dispatch_policy_custom.hpp` 的 `GetBufferByByte` 和 backing `AscendC::LocalTensor<uint8_t> tensor` 仍是 L1 GM→L1 substrate 入口，不能在 P2-1/P2-2 前删除。2026-05-20 死成员清理已完成 build、A3 small、A3 large，rank0/rank1 均 PASS。

---

## moe_init_routing_quant_v2 LocalTensor 接口收口任务规划

详细规划已拆到独立文件：[`quant_task.md`](quant_task.md)。

当前建议下一步执行 `quant_task.md` 的 Batch A：先收口 `moe_v2_pto_sort.h` 的 PTO sort helper 签名，再处理 full-load/sort 调用链里的 slice alias。Batch A 完成后必须重新执行 build + A3 small + A3 large。


---

## 1. `block_epilogue_pertoken_row.hpp`

### 1.1 `BlockEpilogue::operator()`

现状：

- 入参已可改为 raw pointer：`__gm__ ElementC *gmCPtr`、`__gm__ ElementPerTokenScale *gmPerTokenScalePtr`。
- 函数内部仍构造：
  - `AscendC::GlobalTensor<ElementC> gmC`
  - `AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale`
  - `AscendC::GlobalTensor<ElementD> gmLocalScratch`
  - `AscendC::GlobalTensor<ElementD> gmTileD`
- 成员仍有：
  - `ubCList`
  - `ubDList`
  - `ubCFp32List`
  - `ubMulList`

PTO 化方案：

1. 保持 `operator()` raw pointer 入参。
2. 把 `gmC[loopIdx * blockN]` 改成 `gmCPtr + loopIdx * blockN`，调用 `row_detail::PtoLoadVector` 的 raw pointer overload。
3. 把 `gmPerTokenScale(loopIdx)` 改成 `gm_load(gmPerTokenScalePtr + loopIdx)`。
4. 本 rank store：把 `gmTileD.SetGlobalBuffer(dstRowBase)` + `PtoStoreVector(gmTileD[0], ...)` 改成 raw pointer store helper。
5. 远端 scratch：把 `gmLocalScratch[0]` 改成 raw `localScratch` store helper，再构造 `pto::GlobalTensor` 执行 `TPUT`。
6. 暂不动 `ubCList/ubDList/ubCFp32List/ubMulList`，等 `pto_vector_ops.hpp` 的 UB offset overload 稳定后再改。

修改顺序：

- Step 1：给 `row_detail::PtoLoadVector/PtoStoreVector` 增加 raw GM pointer overload。
- Step 2：替换 `gmC/gmPerTokenScale/gmLocalScratch/gmTileD` 临时 wrapper。
- Step 3：build + small + large。
- Step 4：再设计 UB LocalTensor 成员替换。

验收 grep：

```bash
rg -n "AscendC::GlobalTensor" kernels/manual/a2a3/dispatch_ffn_combine_v3/op_kernel/utils/block_epilogue_pertoken_row.hpp
```

允许残留：无。第一阶段目标是本文件内 `GlobalTensor` 清零。

---

## 2. `block_epilogue_pertoken_v2.hpp`

### 2.1 detail helper：`PtoLoadVector`

现状：

```cpp
PtoLoadVector(LocalTensor dst, GlobalTensor src, elemNum)
```

PTO 化方案：

- 新增 overload：`PtoLoadVector(LocalTensor dst, __gm__ Element *src, elemNum)`。
- 内部继续调用 `pto_bridge::PtoLoadVector` 的 raw pointer / global view helper。
- 第二阶段再新增 UB offset 版本。

### 2.2 detail helper：`PtoStoreVector`

现状：

```cpp
PtoStoreVector(GlobalTensor dst, LocalTensor src, elemNum)
```

PTO 化方案：

- 新增 overload：`PtoStoreVector(__gm__ Element *dst, LocalTensor src, elemNum)`。
- 后续把 `gmRemotePeer/gmLocalScratch` 临时 wrapper 删掉。

### 2.3 detail helper：`PtoLoadMatrixRows`

现状：

```cpp
PtoLoadMatrixRows(LocalTensor dst, GlobalTensor src, rowNum, colNum, dstStride, srcStride)
```

PTO 化方案：

- 新增 raw pointer 版本：`PtoLoadMatrixRows(LocalTensor dst, __gm__ Element *src, ...)`。
- 每行调用 raw pointer `PtoLoadVector(dst[row * dstStride], src + row * srcStride, colNum)`。

### 2.4 detail helper：`PtoStoreMatrixRows`

现状：

```cpp
PtoStoreMatrixRows(GlobalTensor dst, LocalTensor src, rowNum, colNum, dstStride, srcStride)
```

PTO 化方案：

- 新增 raw pointer 版本：`PtoStoreMatrixRows(__gm__ Element *dst, LocalTensor src, ...)`。
- 每行调用 raw pointer `PtoStoreVector(dst + row * dstStride, src[row * srcStride], colNum)`。

### 2.5 `BlockEpilogue::Params / constructor`

现状：

- `Params::ptrTokenPerExpert` 已是 raw pointer。
- constructor 仍设置成员 `AscendC::GlobalTensor<int32_t> tokenPerExpert`。

PTO 化方案：

- 成员 `tokenPerExpert` 改为 `__gm__ int32_t *tokenPerExpertPtr`。
- constructor 只赋值 raw pointer。
- `gm_load(tokenPerExpert.GetPhyAddr() + ...)` 改成 `gm_load(tokenPerExpertPtr + ...)`。

### 2.6 `BlockEpilogue::operator()`

现状：

- 入参已可 raw pointer：`gmCPtr` / `gmPerTokenScalePtr`。
- 内部仍构造 `AscendC::GlobalTensor<ElementC> gmC` / `gmPerTokenScale`。
- 内部仍构造 `gmRemotePeer`、`gmLocalScratch`。

PTO 化方案：

1. `gmTileC = gmC[gmCOffset]` 改成 `__gm__ ElementC *gmTileCPtr = gmCPtr + gmCOffset`。
2. `PtoLoadMatrixRows(ubC, gmTileC, ...)` 改成 raw pointer helper。
3. `PtoLoadVector(scaleUb, gmPerTokenScale[gmScaleOffset], actualM)` 改成 raw pointer helper。
4. `gmRemotePeer.SetGlobalBuffer(dstPeermemPtr)` 改成 `__gm__ ElementD *remotePeerPtr = reinterpret_cast<...>(dstPeermemPtr)`。
5. 本 rank `PtoStoreMatrixRows(gmTileD, ...)` 改成 raw pointer helper。
6. 远端 local scratch `PtoStoreVector(gmLocalScratch[0], ...)` 改成 raw pointer helper。
7. `TPUT` 仍使用 use-site `pto::GlobalTensor`。

修改顺序：

- Step 1：新增 raw pointer helper overload。
- Step 2：改 tokenPerExpert 成员。
- Step 3：改 `gmC/gmPerTokenScale/gmRemotePeer/gmLocalScratch` 临时 wrapper。
- Step 4：build + small + large。

验收 grep：

```bash
rg -n "AscendC::GlobalTensor" kernels/manual/a2a3/dispatch_ffn_combine_v3/op_kernel/utils/block_epilogue_pertoken_v2.hpp
```

第一阶段允许残留：detail helper 的旧 overload 可以先保留，但 `BlockEpilogue` 类体内不应再持有 GlobalTensor 成员。

---

## 3. `block_epilogue_pertoken_swiglu.hpp`

### 3.1 `BlockEpilogue::operator()`

现状：

- 入参已改 raw pointer。
- 函数开头仍构造：
  - `AscendC::GlobalTensor<ElementC> gmC`
  - `AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale1`
  - `AscendC::GlobalTensor<ElementD> gmD`
  - `AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale2`
- scalar scale 用 `gmPerTokenScale1(loopIdx)`。
- output scale store 用 `PtoStoreVector(gmPerTokenScale2[loopStartIdx], ...)`。
- 成员有多组 UB `LocalTensor`。

PTO 化方案：

1. `gmTileC = gmC[loopIdx * blockN]` 改成 `gmCPtr + loopIdx * blockN`。
2. `gmTileD = gmD[loopIdx * ChunkTileLen]` 改成 `gmDPtr + loopIdx * ChunkTileLen`。
3. `gmPerTokenScale1(loopIdx)` 改成 `gm_load(gmPerTokenScale1Ptr + loopIdx)`。
4. `PtoStoreVector(gmPerTokenScale2[loopStartIdx], ...)` 改成 raw pointer store helper。
5. 先保留 UB LocalTensor 成员，避免同时改 Swiglu 数学链路。

修改顺序：

- Step 1：依赖 `pto_vector_ops.hpp` raw pointer overload。
- Step 2：删 operator 内四个 `GlobalTensor` 临时 wrapper。
- Step 3：build + small + large。
- Step 4：再进入 UB offset 化。

验收 grep：

```bash
rg -n "AscendC::GlobalTensor" kernels/manual/a2a3/dispatch_ffn_combine_v3/op_kernel/utils/block_epilogue_pertoken_swiglu.hpp
```

允许残留：无。第一阶段目标是本文件内 `GlobalTensor` 清零。

### 3.2 剩余 UB `LocalTensor` offset 化任务设计

当前剩余 `LocalTensor` 都是 UB scratch / stage buffer，不再是 GM wrapper。处理原则：先建 explicit UB byte offset seam，再删成员；每批都必须 build + A3 small/large，不能把 scalar seam 和全量 vector 链路混在一批。

| 优先级 | 状态 | 范围 | 目标 | 验证 |
| --- | --- | --- | --- | --- |
| S0 | 已完成 | offset shadow：`ubCList/ubDList/ubCFp32*List/ubQuant*List/ubPerTokenScaleOutput` | 已在构造函数里为每个 UB buffer 同步记录 `uint64_t` offset，保持现有 `LocalTensor` 不删；所有原调用继续工作 | build + A3 small/large 已通过 |
| S1 | 已完成 | GM↔UB stage：`ubCList`、`ubDList` | 已将 `PtoLoadVector/PtoStoreVector` 调用从 `PtoUbBaseAddr(LocalTensor)` 改成 explicit offset；已删除 `ubCList/ubDList` 成员 | build + A3 small/large 已通过 |
| S2 | 已完成 | 主 swiglu vector scratch：`ubCFp32List`、`ubCFp32ChunkNList`、`ubCFp32ChunkNAbsList` | 已将 cast/mul/exp/add/div/abs/reduce/mul-output 等 helper 调用改成 offset-only；`ubCFp32ChunkNAbsList` 继续作为 `ubOutputTmp` 共享 scratch offset | build + A3 small/large 已通过 |
| S3 | 已完成 | quant scratch reinterpret：`ubQuantS32List`、`ubQuantF16List` | 已不再保存两个 dtype 的 `LocalTensor` view；用同一个 `ubQuantScratchOffset` 按阶段绑定 int32/half PTO tile | build + A3 small/large 已通过 |
| S4 | 已完成 | scalar seam：`ubReduceMax.GetValue(0)`、`ubPerTokenScaleOutput.SetValue(...)` | 已将 UB scalar read/write 收口到 PTO Tile `PtoGetValue/PtoSetValue`，不再依赖 `LocalTensor::GetValue/SetValue` | build + A3 small/large 已通过 |
| S5 | 已完成 | `ubPerTokenScaleOutput` 批量 store | 已将 output scale 暂存区改 explicit offset，并用 `PtoStoreVector(gmPerTokenScale2Ptr + loopStartIdx, offset, tasksForIdx)` 写回；已删除最后一个 UB `LocalTensor` 成员 | build + A3 small/large 已通过 |

#### S0 offset shadow 设计

需要新增成员：

```cpp
uint64_t ubCOffsetList[UB_STAGES];
uint64_t ubDOffsetList[UB_STAGES];
uint64_t ubCFp32OffsetList[UB_STAGES];
uint64_t ubCFp32ChunkNOffsetList[UB_STAGES];
uint64_t ubCFp32ChunkNAbsOffsetList[UB_STAGES];
uint64_t ubCFp32ChunkNMaxOffsetList[UB_STAGES];
uint64_t ubPerTokenScaleOutputOffset{0};
```

初始化规则：每次 `resource.ubBuf.GetBufferByByte<T>(ubOffset)` 的同一位置，同步记录 `resource.ubBuf.GetBufferAddrByByte(ubOffset)`。`ubQuantS32/ubQuantF16` 不单独分配 offset，复用 `ubCFp32ChunkNAbsOffsetList[i]`。

S0 复盘：`block_epilogue_pertoken_swiglu.hpp` 已为 `ubC/ubD/ubCFp32/ubCFp32ChunkN/ubCFp32ChunkNAbs/ubCFp32ChunkNMax/ubPerTokenScaleOutput` 增加 explicit UB byte offset shadow；`ubQuantS32/ubQuantF16` 继续复用 `ubCFp32ChunkNAbs` backing offset。当前不删 `LocalTensor`，只为 S1-S5 铺 seam。2026-05-20 已完成 build、A3 small、A3 large，rank0/rank1 均 PASS。

S1 复盘：`ubCList/ubDList` 已删除，GM→UB 的 C staging 改为 `PtoLoadVector<ElementC>(ubCOffset, gmTileC, blockN)`，UB→GM 的 D staging 改为 `PtoStoreVector<ElementD>(gmTileD, ubDOffset, ChunkTileLen)`；两端边界 cast 已同步使用 explicit UB offset，未改 swiglu 主数学链路。2026-05-20 已完成 build、A3 small、A3 large，rank0/rank1 均 PASS。

S2 复盘：`ubCFp32List/ubCFp32ChunkNList` 已删除，swiglu 主链路的 cast、scale mul、sigmoid exp/add/div、gated mul、abs、reduce max、quant 前 scale mul 均改为 explicit UB byte offset 调用。`ubCFp32ChunkNAbsList` 仍保留为 quant reinterpret backing，`ubCFp32ChunkNMaxList` 仍保留 scalar `GetValue` seam，未提前混入 S3/S4 风险。2026-05-20 已完成 build、A3 small、A3 large，rank0/rank1 均 PASS。

S3 复盘：`ubQuantS32List/ubQuantF16List` 已删除，quant scratch 不再保存 dtype-specific `LocalTensor` reinterpret view；int32 cast、half cast、最终 ElementD cast 都使用同一块 `ubCFp32ChunkNAbsOffsetList` backing offset 按阶段绑定 PTO tile。2026-05-20 已完成 build、A3 small、A3 large，rank0/rank1 均 PASS。

S4 复盘：新增 `pto_vector_ops.hpp` 的 `PtoGetValue/PtoSetValue`，用 PTO Vec tile `TASSIGN + GetValue/SetValue` 承接 UB scalar read/write；`ubReduceMax.GetValue(0)` 和 `ubPerTokenScaleOutput.SetValue(...)` 已替换为 explicit UB offset scalar seam，`ubCFp32ChunkNMaxList` 已删除。2026-05-20 已完成 build、A3 small、A3 large，rank0/rank1 均 PASS。

S5 复盘：`ubPerTokenScaleOutput` 已删除，output scale 暂存区只保留 `ubPerTokenScaleOutputOffset`；最终批量写回改为 `PtoStoreVector<ElementPerTokenScale>(gmPerTokenScale2Ptr + loopStartIdx, ubPerTokenScaleOutputOffset, tasksForIdx)`。2026-05-20 已完成 build、A3 small、A3 large，rank0/rank1 均 PASS。

#### S1/S2 vector helper 依赖

依赖 `pto_vector_ops.hpp` 已有或新增 offset overload：

```cpp
PtoLoadVector(uint64_t dstUbOffsetBytes, __gm__ Element *src, uint32_t elemNum);
PtoStoreVector(__gm__ Element *dst, uint64_t srcUbOffsetBytes, uint32_t elemNum);
PtoCastVector(uint64_t dstUbOffsetBytes, uint64_t srcUbOffsetBytes, uint32_t elemNum, pto::RoundMode mode);
PtoMulVector(uint64_t dstUbOffsetBytes, uint64_t srcUbOffsetBytes, uint32_t elemNum, float scalar);
PtoExpVector(uint64_t dstUbOffsetBytes, uint64_t srcUbOffsetBytes, uint32_t elemNum);
PtoReduceMaxVector(uint64_t dstUbOffsetBytes, uint64_t srcUbOffsetBytes, uint64_t tmpUbOffsetBytes, uint32_t elemNum);
```

若某个 helper 的 offset overload 不存在，先在 `pto_vector_ops.hpp` 补 seam 并单独 build + small/large，不在 swiglu 文件里临时回退到 `LocalTensor`。

#### S4 scalar seam 约束

这两处必须单独处理：

```cpp
ElementPerTokenScale GMubDequantScale = ubReduceMax.GetValue(0);
ubPerTokenScaleOutput.SetValue(ubPerTokenScaleOutputOffset, GMubDequantScale / 127.f);
```

`GetValue/SetValue` 是 scalar path，不能直接用 vector `TLOAD/TSTORE` 替换而不验证同步语义。允许短期保留极小 scalar seam，但必须把保留原因写在 P/S 任务表里，不能声称本文件 `LocalTensor` 清零。

验收 grep：

```bash
rg -n "AscendC::LocalTensor|GetValue\(|SetValue\(" \
  kernels/manual/a2a3/dispatch_ffn_combine_v3/op_kernel/utils/block_epilogue_pertoken_swiglu.hpp
```

最终目标：除非 S4 明确保留 scalar seam，否则本文件不再出现 `AscendC::LocalTensor`。

---

## 4. `pto_vector_ops.hpp`

这个文件是后续 PTO 化的抓手，要先扩接口再替换调用点。

### 4.1 `PtoLoadVector`

现状：

```cpp
PtoLoadVector(LocalTensor dst, GlobalTensor src, elemNum)
```

PTO 化方案：

- 新增：`PtoLoadVector(LocalTensor dst, __gm__ Element *src, elemNum)`。
- 新增：`PtoLoadVector(uint64_t dstUbOffsetBytes, __gm__ Element *src, elemNum)`。
- 当前调用点先用第一种，后续 UB 成员改 offset 时用第二种。

### 4.2 `PtoStoreVector`

现状：

```cpp
PtoStoreVector(GlobalTensor dst, LocalTensor src, elemNum)
```

PTO 化方案：

- 新增：`PtoStoreVector(__gm__ Element *dst, LocalTensor src, elemNum)`。
- 新增：`PtoStoreVector(__gm__ Element *dst, uint64_t srcUbOffsetBytes, elemNum)`。

### 4.3 `PtoStoreAtomicAddVector`

现状：

```cpp
PtoStoreAtomicAddVector(GlobalTensor dst, LocalTensor src, elemNum)
```

PTO 化方案：

- 新增 raw pointer dst overload。
- 当前没有必要先改所有 atomic 调用，先保证接口存在。

### 4.4 UB arithmetic helpers

涉及函数：

- `PtoCastVector`
- `PtoMoveVector`
- `PtoFillVector`
- `PtoMulVector`
- `PtoAddVector`
- `PtoAddScalarVector`
- `PtoMulElementwiseVector`
- `PtoDivVector`
- `PtoAbsVector`
- `PtoExpVector`
- `PtoReduceMaxVector`

PTO 化方案：

- 每个函数保留 LocalTensor 版本。
- 新增 UB offset 版本：参数从 `LocalTensor` 改为 `uint64_t ubOffsetBytes`。
- 对于多输入函数，参数形态统一为：
  - `dstUbOffsetBytes`
  - `src0UbOffsetBytes`
  - `src1UbOffsetBytes`
  - `elemNum`
- `PtoReduceMaxVector` 需要特别设计：当前依赖 `dst[0]` / `tmp[0]` 中间位置，offset 版要明确 `dstUbOffsetBytes`、`srcUbOffsetBytes`、`tmpUbOffsetBytes`，不能共用错。

修改顺序：

- Step 1：增加 GM raw pointer overload，服务 epilogue GlobalTensor 清理。
- Step 2：build + small + large。
- Step 3：增加 UB offset overload，不替换调用点。
- Step 4：逐个 epilogue 替换 UB 成员调用。

验收 grep：

```bash
rg -n "AscendC::GlobalTensor" kernels/manual/a2a3/dispatch_ffn_combine_v3/op_kernel/utils/pto_vector_ops.hpp
```

阶段目标：先允许旧 overload 残留；最终目标是业务文件不再依赖这些旧 overload。

---

## 5. `block_mmad_preload_async_fixpipe_quant.hpp`

这个文件风险最高，不能按“全部 LocalTensor 直接换 PTO Tile”硬推。

### 5.1 `PtoTileMmad`

现状：

```cpp
PtoTileMmad(LocalTensor l0C, LocalTensor l0A, LocalTensor l0B, ...)
```

性质：

- L0A/L0B/L0C substrate，直接参与 MMAD。
- 不能先改成普通 Vec Tile。

PTO 化方案：

- 第一阶段不改签名。
- 只把调用面归档为 MMAD substrate。
- 若后续要改，必须先确认 PTO 对 L0A/L0B/L0C tile assign 和 `TMMAD` 的完整等价路径。

### 5.2 `PtoLoadSoftFlagL1`

现状：

```cpp
PtoLoadSoftFlagL1(LocalTensor<int32_t> dst, GlobalTensor<int32_t> src, elemNum)
```

PTO 化方案：

- `src` 改 raw pointer：`__gm__ int32_t *src`。
- `dst` 暂保留 L1 LocalTensor，因为目标位置是 L1 flag matrix。
- 后续可改为 L1 offset：`uint64_t dstL1OffsetBytes`。

### 5.3 `PtoStoreSoftFlagL1`

现状：

```cpp
PtoStoreSoftFlagL1(GlobalTensor<int32_t> dst, LocalTensor<int32_t> src, elemNum)
```

PTO 化方案：

- `dst` 改 raw pointer：`__gm__ int32_t *dst`。
- `src` 暂保留 L1 LocalTensor。
- 删除 `Finalize()` 内的 `flagGlobal` 临时 wrapper。

### 5.4 `PtoStoreAccToGm`

现状：

```cpp
PtoStoreAccToGm(GlobalTensor<ElementDst> dst, LocalTensor<ElementAccumulator> src, LocalTensor<uint64_t> scale, ...)
```

性质：

- int8 fixpipe / accumulator store substrate。

PTO 化方案：

- 先把 `dst` 改 raw pointer，内部构造 PTO `GlobalTensor`。
- `src` / `scale` 暂保留 LocalTensor，因为对应 L0C / FP buffer。
- 后续若做 offset seam，单独设计 `srcL0COffsetBytes` / `scaleFpOffsetBytes`。

### 5.5 `StagePerChannelScale`

现状：

```cpp
StagePerChannelScale(copyGmToL1S, copyL1ToFP, LocalTensor l1STensor, LocalTensor fixpipeBuf, GlobalTensor gmBlockS, ...)
```

PTO 化方案：

- `gmBlockS` 改 raw pointer。
- `l1STensor/fixpipeBuf` 暂保留。
- 后续与 `CopyGmToL1S`、`CopyL1ToFP` substrate 一起收口。

### 5.6 `StoreAccumulator`

现状：

```cpp
StoreAccumulator(GlobalTensor<ElementC> dst, LocalTensor src, LocalTensor scale, ...)
```

PTO 化方案：

- `dst` 改 raw pointer。
- half 路径仍需要 `copyL0CToGm` substrate；先在内部构造最小 AscendC wrapper 或给 `copyL0CToGm` 加 raw pointer overload。
- int8 路径直接调用 raw pointer `PtoStoreAccToGm`。

### 5.7 `BlockMmad::operator()`

现状：

- 入参已 raw pointer。
- 函数内部仍构造 `gmBlockA/B/C/S` 临时 wrapper。
- `L1TileMmadParams` 仍保存 `AscendC::GlobalTensor<ElementC>`、`AscendC::GlobalTensor<uint64_t>`。

PTO 化方案：

1. `gmBlockA/B` wrapper 先保留，直到 `copyGmToL1A/B` 支持 raw pointer。
2. `L1TileMmadParams` 改存：
   - `__gm__ ElementC *gmBlockCPtr`
   - `__gm__ uint64_t *gmBlockSPtr`
3. `StagePerChannelScale` / `StoreAccumulator` 使用 raw pointer 参数。
4. `Finalize()` 内 `flagGlobal` 改 raw pointer。
5. `InitL1()` 内 `flagBase` 改 raw pointer。

修改顺序：

- Step 1：soft-flag load/store raw pointer 化。
- Step 2：`L1TileMmadParams` 的 C/S raw pointer 化。
- Step 3：`StagePerChannelScale` / `StoreAccumulator` raw pointer 化。
- Step 4：build + small + large。
- Step 5：评估 `copyGmToL1A/B` raw pointer overload。
- Step 6：仅在有等价 substrate 后再删除 A/B 临时 wrapper。

验收 grep：

```bash
rg -n "AscendC::GlobalTensor" kernels/manual/a2a3/dispatch_ffn_combine_v3/op_kernel/utils/block_mmad_preload_async_fixpipe_quant.hpp
```

第一阶段允许残留：A/B GM wrapper、half fixpipe substrate wrapper。目标是先清掉 soft-flag 和 C/S 存储链路的 wrapper。

---

## 6. `dispatch_policy_custom.hpp`

这个文件是 matmul/fixpipe substrate 的定义面，不应作为第一批“清零 LocalTensor”的目标。

### 6.1 `LocalTensorBufferBase` / `Resource`

现状：

- `Resource` 用 `AscendC::LocalTensor<uint8_t>` 表示 UB/L1/L0/FP backing buffer。
- `GetBufferByByte<T>` 返回 `AscendC::LocalTensor<T>`。

PTO 化方案：

- 不直接删除。
- 新增 parallel seam：
  - `GetOffsetByByte(offset)` 返回显式 byte offset。
  - 或 `PtoBufferView{basePosition, offset}`。
- 先让 epilogue 使用 offset seam；MMAD substrate 保留 LocalTensor seam。

### 6.2 `PtoDataCopyGmToL1Nd2Nz`

现状：

```cpp
PtoDataCopyGmToL1Nd2Nz(LocalTensor dst, GlobalTensor src, ...)
```

PTO 化方案：

- 这是 substrate bridge，暂不替换算法。
- 可增加 raw pointer `src` overload，内部构造 AscendC wrapper 或 PTO Mat view。
- 不在第一阶段做 L1 dst offset 化。

### 6.3 `PtoDataCopyGmToL1Nd2NzRow`

方案同 6.2。

### 6.4 `PtoDataCopyGmToL1`

方案同 6.2。

### 6.5 `PtoDataCopyL1ToFp`

现状：L1 -> FP substrate。

PTO 化方案：暂不替换；只保留在 substrate bridge 内。

### 6.6 `PtoLoadDataL1ToL0A`

现状：L1 -> L0A substrate。

PTO 化方案：暂不替换；等 PTO L0 tile/MMAD substrate 明确后再动。

### 6.7 `PtoLoadDataL1ToL0B`

方案同 6.6。

### 6.8 `PtoFixpipeL0CToGm`

现状：L0C -> GM fixpipe substrate。

PTO 化方案：

- 可以先加 raw pointer `dst` overload，内部最小构造 AscendC wrapper。
- 不替换 `Fixpipe` 本体。

### 6.9 `CopyGmToL1::*::operator()`

涉及三个 overload：

- ND/ND
- VectorLayout/VectorLayout
- Zn/Zn

PTO 化方案：

- 第一阶段增加 raw pointer `src` overload。
- 内部仍走 `PtoDataCopyGmToL1*` substrate。
- 让 `BlockMmad::operator()` 不再必须构造 `gmBlockA/gmBlockB` wrapper。

### 6.10 `CopyL1ToL0A/B`、`CopyL1ToFP`、`CopyL0CToGm`

PTO 化方案：

- 保留 LocalTensor substrate。
- `CopyL0CToGm` 可优先增加 raw pointer dst overload。
- 不做 L1/L0/FP offset 化，除非 `BlockMmad` 已经完成 offset seam。

修改顺序：

- Step 1：只加 raw pointer overload，不删旧接口。
- Step 2：让 `block_mmad_preload_async_fixpipe_quant.hpp` 调用 raw overload。
- Step 3：build + small + large。
- Step 4：再评估 LocalTensorBuffer offset seam。

验收标准：

- `dispatch_policy_custom.hpp` 可以继续保留 `AscendC::LocalTensor`，但只能集中在 substrate bridge。
- 不允许业务 kernel 重新直接调用 `DataCopy/LoadData/Fixpipe`。

---

## 7. 当前已完成基线

本轮已完成：

- `dispatch_ffn_combine_kernel.hpp` 主文件中的 `AscendC::GlobalTensor` 已清零。
- 主 kernel 成员已改为 raw pointer：
  - `gmAPtr`
  - `gmCPtr`
  - `gmPermutedTokenPtr`
  - `gmC2Ptr`
  - `gmPerTokenScale1Ptr`
  - `gmPerTokenScale2Ptr`
  - `cumsumMMPtr`
- `cumsumMM(...)` 已改为 `gm_load(cumsumMMPtr + offset)`。
- 已验证：
  - build PASS
  - small PASS rank0/rank1
  - large PASS rank0/rank1

## 8. 下一批建议执行任务

### Task A：Epilogue GM wrapper 清零

文件：

- `block_epilogue_pertoken_row.hpp`
- `block_epilogue_pertoken_v2.hpp`
- `block_epilogue_pertoken_swiglu.hpp`
- `pto_vector_ops.hpp`

内容：

1. 在 `pto_vector_ops.hpp` 加 raw pointer GM load/store overload。
2. 三个 epilogue 删除 operator 内 `AscendC::GlobalTensor` 临时 wrapper。
3. `block_epilogue_pertoken_v2.hpp` 的 `tokenPerExpert` 成员改 raw pointer。
4. build + small + large。

### Task B：MMAD soft-flag 与 C/S 存储链路 raw pointer 化

文件：

- `block_mmad_preload_async_fixpipe_quant.hpp`
- `dispatch_policy_custom.hpp`

内容：

1. `PtoLoadSoftFlagL1` / `PtoStoreSoftFlagL1` raw pointer 化。
2. `L1TileMmadParams` 中 `gmBlockC/gmBlockS` 改 raw pointer。
3. `PtoStoreAccToGm` / `StagePerChannelScale` / `StoreAccumulator` 的 GM 参数 raw pointer 化。
4. build + small + large。

### Task C：MMAD A/B GM wrapper 去除

文件：

- `dispatch_policy_custom.hpp`
- `block_mmad_preload_async_fixpipe_quant.hpp`

内容：

1. `CopyGmToL1` 增加 raw pointer source overload。
2. `BlockMmad::operator()` 删除 `gmBlockA/gmBlockB` 临时 wrapper。
3. build + small + large。

### Task D：UB LocalTensor offset 化试点

优先文件：

- `block_epilogue_pertoken_row.hpp`

原因：

- row epilogue 数学链路最短。
- UB buffer 数量少。
- 易验证。

内容：

1. 成员 `ubCList/ubDList/ubCFp32List/ubMulList` 改成 UB offset 数组。
2. 使用 `pto_vector_ops.hpp` 的 UB offset overload。
3. build + small + large。

### Task E：Swiglu / V2 UB LocalTensor offset 化

前置：Task D 通过。

内容：

- `block_epilogue_pertoken_swiglu.hpp`：UB 数学链路长，最后改。
- `block_epilogue_pertoken_v2.hpp`：有多行矩阵 load/store 和 remote TPUT，排在 swiglu 前或后均可，但必须单独验证。

## 9. 禁止事项

- 不把长期成员改成长期持有的 `pto::GlobalTensor`。
- 不把 L0A/L0B/L0C 的 `LocalTensor` 直接当 Vec Tile 替换。
- 不跳过 small/large 验证。
- 不同时改 epilogue UB offset 和 MMAD substrate。
- 不因 build 过了就认为完成，必须 rank0/rank1 compare PASS。
