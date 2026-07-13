# kirin9030 tmov_acc2mat NZ2ND 测试用例失败分析

## 1. 问题概述

在 kirin9030 平台的 `tmov_acc2mat` 测试中,部分 NZ2ND 测试用例失败。失败用例均为 `half`(fp16) 数据类型,ND 布局输出,无 ReLU。

## 2. 测试结果汇总

### 失败用例(7 个)

| 用例 | validM | validK | validN | M(对齐) | N(对齐) | row | col | col×2(bytes) |
|------|--------|--------|--------|---------|---------|-----|-----|-------------|
| case_nz2nd_2 | 111 | 48 | 88 | 112 | 96 | 112 | 96 | 192 |
| case_nz2nd_6 | 97 | 33 | 65 | 112 | 80 | 112 | 80 | 160 |
| case_nz2nd_12 | 112 | 48 | 88 | 112 | 96 | 112 | 96 | 192 |
| case_nz2nd_14 | 112 | 48 | 65 | 112 | 80 | 112 | 80 | 160 |
| case_nz2nd_17 | 97 | 48 | 97 | 112 | 112 | 112 | 112 | 224 |
| case_nz2nd_21 | 64 | 48 | 65 | 64 | 80 | 64 | 80 | 160 |
| case_nz2nd_22 | 48 | 48 | 65 | 48 | 80 | 48 | 80 | 160 |

### 通过用例(N 非对齐,5 个)

| 用例 | validM | validK | validN | M(对齐) | N(对齐) | row | col | col×2(bytes) |
|------|--------|--------|--------|---------|---------|-----|-----|-------------|
| case_nz2nd_4 | 6 | 7 | 8 | 16 | 16 | 32 | 32 | 64 |
| case_nz2nd_15 | 64 | 48 | 33 | 64 | 48 | 64 | 48 | 96 |
| case_nz2nd_16 | 80 | 48 | 49 | 80 | 64 | 80 | 64 | 128 |
| case_nz2nd_19 | 80 | 48 | 57 | 80 | 64 | 80 | 64 | 128 |
| case_nz2nd_20 | 80 | 48 | 63 | 80 | 64 | 80 | 64 | 128 |

### 通过用例(N 对齐,所有其他用例)

case_nz2nd_1(ReLU), case_nz2nd_3, case_nz2nd_5, case_nz2nd_7~11, case_nz2nd_13, case_nz2nd_18 等均通过。

### 观察规律

- **所有失败用例**: `col ≥ 80`(即 `burstSrcStride = col × 2 ≥ 160 bytes`)
- **所有通过用例(N 非对齐)**: `col ≤ 64`(即 `burstSrcStride = col × 2 ≤ 128 bytes`)
- **N 对齐的用例**: 无论 col 多大均通过(因为 `validN == col`,无 stride 不匹配)

## 3. 代码执行流程分析

### 3.1 完整数据流

```
GM(src0, src1)
  │ TLOAD (MTE2 pipe)
  ▼
MatTile_A [M×K, cbuf]    MatTile_B [K×N, cbuf]
  │ TMOV (MTE1 pipe)       │ TMOV (MTE1 pipe)
  ▼                        ▼
LeftTile [M×K, cbuf]     RightTile [K×N, cbuf]
  │                          │
  └──── TMATMUL (M pipe) ────┘
           ▼
      AccTile [M×N, cc, fractal格式]
           │ TMOV Acc→Mat (FIX pipe)
           │ copy_matrix_cc_to_cbuf(enableNz2Nd=true)
           ▼
      MatTile [row×col, cbuf, RowMajor+NoneBox]
           │ TMOVMat2Vec (MTE1 pipe)
           │ copy_cbuf_to_ubuf(nBurst=1, 整块拷贝)
           ▼
      VecTile [row×col, ubuf, RowMajor+NoneBox]
           │ TSTORE (MTE3 pipe)
           │ copy_ubuf_to_gm_align_v2
           ▼
      GM(out, ND布局, shape=[validM, validN])
```

### 3.2 关键步骤详解

#### Step 1: TMOV Acc→Mat (`TMovCcToCb`)

**文件**: `include/pto/npu/kirin9030/TMov.hpp:99-141`

```cpp
constexpr uint32_t dstStride = GetTmovAccDstStride<DstTile, SrcTile>();
// 对于 ND 布局 (RowMajor + NoneBox): dstStride = DstTile::Cols = col

auto srcStride = CeilAlignment(validRow, BLOCK_LEN);  // = CeilAlignment(validM, 16) = M

pto_copy_matrix_cc_to_cbuf(dstAddr, srcData, 0, validCol, validRow, dstStride, srcStride,
                           0, 0, 0, QuantPre, reluMode, channelSplitEnable, enableNz2Nd, ...);
```

- **功能**: 硬件格式转换,从 AccTile(NZ fractal, cc 缓冲区)转为 MatTile(RowMajor, cbuf 缓冲区)
- **`enableNz2Nd = true`**: 告诉硬件执行 NZ→ND 格式转换
- **`dstStride = col`**: 目标 MatTile 每行 `col` 个元素
- **`validCol = validN`**: 每行写入 `validN` 个有效元素
- **`SetLoop3Para()`**: 配置硬件 loop3 参数(ndNum=1, dstNdStride=0, srcNdStride=0)

#### Step 2: TMOVMat2Vec (`copy_cbuf_to_ubuf`)

**文件**: `tmov_acc2mat_kernel.cpp:227-239`

```cpp
uint16_t nBurst = 1;
uint16_t lenBurst = row * col * sizeof(T) / 32;  // 整块拷贝,单位为32字节

copy_cbuf_to_ubuf(dstTileAddr, srcTileAddr, 0, nBurst, lenBurst, 0, 0);
```

- **功能**: 将 MatTile 从 cbuf 整块拷贝到 ubuf
- **假设**: cbuf 和 ubuf 的数据布局完全一致(均为 flat row-major, stride=col)
- **拷贝大小**: `row × col × 2` bytes(包含有效数据和 padding)

#### Step 3: TSTORE Vec→GM (`TStoreVecND`)

**文件**: `include/pto/common/arch/register/tstore_common.hpp:276-322`

```cpp
uint32_t nBurst = gShape3;                    // = validM (行数)
uint32_t lenBurst = validCol * sizeof(T);     // = validN × 2 (每行有效字节数)
uint64_t burstDstStride = gStride3 * sizeof(T); // = validN × 2 (GM 行间距)
uint32_t burstSrcStride = TileData::Cols * sizeof(T); // = col × 2 (UB 行间距)

pto_copy_ubuf_to_gm_align_v2(dst, src, 0, nBurst, lenBurst, 0, burstDstStride, burstSrcStride);
```

- **功能**: 从 UB 写入 GM,处理 stride 不匹配(UB stride=col×2, GM stride=validN×2)
- **`burstSrcStride`**: UB 中每行间距 = `col × 2` bytes
- **`lenBurst`**: 每行实际写入 = `validN × 2` bytes
- **`burstDstStride`**: GM 中每行间距 = `validN × 2` bytes

## 4. 各层代码正确性分析

### 4.1 TSTORE ND 路径 — 代码逻辑正确

`TStoreVecND` 正确使用 `TileData::Cols`(即 `col`)作为 UB 源 stride,使用 `gStride3`(即 `validN`)作为 GM 目标 stride。DMA 参数 `burstSrcStride ≠ burstDstStride` 时,硬件应自动跳过 UB 中的 padding 列。

**结论**: TSTORE 代码逻辑本身无问题。

### 4.2 TMOV Acc→Mat — 硬件格式转换

`copy_matrix_cc_to_cbuf` 是硬件 CCE 指令,负责将 AccTile 的 NZ fractal 格式转换为 MatTile 的 RowMajor 格式。`dstStride = col` 告诉硬件每行写入 `col` 个元素宽度的空间,其中前 `validN` 个为有效数据。

**结论**: TMOV 代码逻辑本身无问题。

### 4.3 TMOVMat2Vec — 关键嫌疑点

`copy_cbuf_to_ubuf` 以 `nBurst=1` 做整块线性拷贝,假设 cbuf 中 MatTile 的物理存储布局是 flat row-major(stride=col)。

**关键问题**: cbuf(L1 Cache)的物理存储是否真的是 flat row-major?

cbuf 是为 Cube 运算设计的硬件缓冲区,其物理存储可能采用 block-based 格式(如 32 字节块)。如果 `copy_matrix_cc_to_cbuf` 写入 cbuf 时,硬件按 block 格式存储(而非 flat row-major),则 `copy_cbuf_to_ubuf` 的线性拷贝会将 block 格式数据原样搬到 ubuf。

此时 TSTORE 按 flat row-major(stride=col)读取 ubuf 数据,就会读到错误结果。

### 4.4 VecTile 内部布局 — NoneBox 应为 flat

根据 `pto_tile.hpp` 和 CPU 模拟器代码,`SLayout::NoneBox` 的 VecTile 逻辑上是 flat row-major。`sfractalSize=512` 对 NoneBox 无效。

**结论**: VecTile 的逻辑布局无问题,但物理布局取决于 `copy_cbuf_to_ubuf` 的拷贝结果。

## 5. 可疑点分析

### 可疑点 1: cbuf 物理存储格式与 copy_cbuf_to_ubuf 的交互

**假设**: cbuf 的物理存储不是 flat row-major,而是 block-based 格式。

`copy_matrix_cc_to_cbuf` 将 AccTile 的 fractal 数据转换为 RowMajor 写入 cbuf。但 cbuf 硬件可能以 32 字节(BLOCK_BYTE_SIZE)为单位组织数据。对于 `col=80`(每行 160 bytes = 5 blocks),cbuf 可能将每行存储为 5 个独立 block。

`copy_cbuf_to_ubuf` 做线性拷贝时,如果 cbuf 的 block 排列顺序与 flat row-major 不一致(例如 block 按列优先排列),拷贝到 ubuf 的数据就不是 flat row-major。

**为什么 col≤64 通过**: col=64 时每行 128 bytes = 4 blocks。cbuf 的 block 排列可能恰好与 flat row-major 一致(或硬件对 ≤4 blocks/行有特殊优化)。col=48 时每行 96 bytes = 3 blocks,同理。

**为什么 col≥80 失败**: col=80 时每行 160 bytes = 5 blocks。cbuf 的 block 排列可能与 flat row-major 不一致,导致线性拷贝产生错误布局。

**验证方法**: 在 TMOVMat2Vec 之后、TSTORE 之前,将 ubuf 数据 dump 到 GM,检查数据布局是否为 flat row-major。

### 可疑点 2: copy_ubuf_to_gm_align_v2 硬件 DMA 限制

**假设**: `copy_ubuf_to_gm_align_v2` 硬件 DMA 在 `burstSrcStride > 128` bytes 时存在 bug 或行为差异。

DMA 参数:
- `burstSrcStride = col × 2` bytes(UB 行间距)
- `lenBurst = validN × 2` bytes(每行有效数据)

当 `burstSrcStride > 128`(即 `col > 64`)时,DMA 可能需要跨多个 UB bank 读取数据,可能存在 bank conflict 或地址对齐问题。

**验证方法**: 对比 a5 平台的相同测试用例(a5 也使用 `copy_ubuf_to_gm_align_v2`)。如果 a5 通过但 kirin9030 失败,说明是 kirin9030 特有的 DMA 硬件差异。

### 可疑点 3: copy_matrix_cc_to_cbuf 的 dstStride 处理

**假设**: `copy_matrix_cc_to_cbuf` 硬件在处理 `dstStride`(目标行间距)时,对非 2 的幂次值有特殊行为。

- col=48: dstStride=48(非 2 的幂)→ 通过
- col=64: dstStride=64(2 的幂)→ 通过
- col=80: dstStride=80(非 2 的幂)→ 失败
- col=96: dstStride=96(非 2 的幂)→ 失败

col=48 和 col=80 都不是 2 的幂,但一个通过一个失败。所以 dstStride 是否为 2 的幂不是决定因素。

但 col=48 时 `dstStride × sizeof(half) = 96 bytes = 3 × 32`,col=80 时 `dstStride × sizeof(half) = 160 bytes = 5 × 32`。3 blocks 和 5 blocks 的区别可能触发不同的硬件路径。

### 可疑点 4: TMOVMat2Vec 应改为逐行拷贝

当前 `TMOVMat2Vec` 用 `nBurst=1` 做整块拷贝,假设 cbuf 和 ubuf 布局完全一致。如果 cbuf 的物理布局不是 flat row-major,这个假设就不成立。

**替代方案**: 将 `TMOVMat2Vec` 改为逐行拷贝(每行 `col` 个元素),或者直接使用 TMOV Acc→Vec(`TMovCcToUb`)跳过 MatTile 中间步骤。

**注意**: 内核代码中已有 `TMovCcToUb` 函数(`TMov.hpp:143-192`),它使用 `copy_matrix_cc_to_ub` 硬件指令直接将 AccTile 转换为 VecTile,不经过 cbuf 中间步骤。这可能是一个更可靠的路径。

### 可疑点 5: gen_data.py golden 数据正确性

**验证**: gen_data.py 使用 `np.matmul(x1, x2)` 计算 golden 数据,对于 ND 布局无额外格式转换。golden 数据应为正确的 matmul 结果。

**结论**: golden 数据生成逻辑无问题。

## 6. 建议排查步骤

### 优先级 1: 验证 cbuf 物理布局

在 `TMOVMat2Vec` 之后,将 VecTile(ubuf)数据 dump 到 GM,检查数据是否为 flat row-major 格式:

```cpp
// 在 TMOVMat2Vec 之后添加:
// 将 ubuf 数据直接拷贝到 GM 用于调试
copy_ubuf_to_gm(debugGmAddr, dstTileAddr, 0, 1, row * col * sizeof(T) / 32, 0, 0);
```

对比 dump 数据与期望的 flat row-major 数据,确认 cbuf→ubuf 拷贝是否产生正确布局。

### 优先级 2: 尝试 TMOV Acc→Vec 直接路径

修改 kernel,跳过 MatTile 中间步骤,直接使用 `TMOV(VecTile, AccTile)`:

```cpp
// 替换:
// TMOV(srcTileData, cTile);        // Acc→Mat
// TMOVMat2Vec(dstTileData, srcTileData);  // Mat→Vec

// 为:
TMOV(dstTileData, cTile);           // Acc→Vec (直接)
```

如果直接路径通过,说明问题确实在 cbuf 的物理布局或 `copy_cbuf_to_ubuf` 的行为上。

### 优先级 3: 对比 a5 平台

在 a5 平台上运行相同的测试用例(特别是 col≥80 的用例)。a5 也使用 `copy_ubuf_to_gm_align_v2`,但 `copy_matrix_cc_to_cbuf` 的参数可能略有不同。

- 如果 a5 也失败: 说明是 PTO 库的通用问题
- 如果 a5 通过: 说明是 kirin9030 特有的硬件行为差异

### 优先级 4: 检查 copy_matrix_cc_to_cbuf 的 c0PadEn 参数

在 `TMovCcToCb` 中,`c0PadEn` 参数被设为 `false`。对于 ND 布局,可能需要启用 C0 padding 来确保 cbuf 中的数据布局与 flat row-major 一致。

```cpp
// 当前代码 (TMov.hpp:138-140):
pto_copy_matrix_cc_to_cbuf(dstAddr, srcData, 0, validCol, validRow, dstStride, srcStride, 0, 0, 0, QuantPre,
                           reluMode, channelSplitEnable, enableNz2Nd, 0, 0, false, false, 0,
                           false, false, false, false, false, enableNz2Dn);
//                                                                                     ^^^^^
//                                                                               c0PadEn = false
```

## 7. 关键代码文件索引

| 文件 | 关键内容 |
|------|---------|
| `include/pto/npu/kirin9030/TMov.hpp:99-141` | `TMovCcToCb` — TMOV Acc→Mat 实现 |
| `include/pto/npu/kirin9030/TMov.hpp:143-192` | `TMovCcToUb` — TMOV Acc→Vec 实现(替代路径) |
| `include/pto/npu/kirin9030/TMov.hpp:70-78` | `SetLoop3Para` — 硬件 loop3 配置 |
| `include/pto/npu/kirin9030/TMov.hpp:80-97` | `GetTmovAccDstStride` — dstStride 计算 |
| `include/pto/common/arch/register/tstore_common.hpp:276-322` | `TStoreVecND` — TSTORE ND 路径实现 |
| `include/pto/common/arch/register/tstore_common.hpp:267-274` | `TStoreInstr` — DMA 封装 |
| `include/pto/common/arch/register/tstore_common.hpp:410-429` | `TStore` — VecTile 分发逻辑 |
| `include/pto/npu/kirin9030/TStore.hpp:126-155` | `TSTORE_IMPL` — kirin9030 TSTORE 入口 |
| `include/pto/common/arch_cce_intrinsic.hpp:248-259` | `pto_copy_ubuf_to_gm_align_v2` — DMA  intrinsic 封装 |
| `include/pto/common/arch_cce_intrinsic.hpp:262-281` | `pto_copy_matrix_cc_to_cbuf` — CC→CBUF intrinsic 封装 |
| `include/pto/common/pto_tile.hpp:1394-1533` | Tile 模板定义,InnerRows/InnerCols 计算 |
| `include/pto/common/constants.hpp` | BLOCK_BYTE_SIZE=32, FRACTAL_NZ_ROW=16 等常量 |
| `tests/npu/kirin9030/src/st/testcase/tmov_acc2mat/tmov_acc2mat_kernel.cpp:227-239` | `TMOVMat2Vec` — cbuf→ubuf 拷贝 |
| `tests/npu/kirin9030/src/st/testcase/tmov_acc2mat/tmov_acc2mat_kernel.cpp:241-311` | `RunTMOV` — 完整 kernel 流程 |
