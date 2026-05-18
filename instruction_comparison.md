# PTO 指令支持对比分析

## 对比范围

基于 `include/pto/npu/{a5,kirin9030,kirinX90}/header.hpp` 及实际实现文件。

- **a5**: 昇腾 A5 架构，指令最全的参考基线
- **kirin9030**: 麒麟9030，通过复用 a5 实现 + 部分原生实现覆盖大部分指令
- **kirinX90**: 麒麟X90，通过复用 a5/kirin9030/a2a3 + 少量原生实现覆盖基础指令集

---

## 1. a5 vs kirin9030

### 1.1 两架构都支持的指令

#### kirin9030 原生实现（10 个文件，11 条指令）

| # | 指令 | 文件 | 说明 |
|---|------|------|------|
| 1 | TLOAD | `kirin9030/TLoad.hpp` | 从 Global Memory 加载数据到 Tile（ND/DN/NZ 布局） |
| 2 | TSTORE | `kirin9030/TStore.hpp` | 从 Tile 存储到 Global Memory |
| 3 | TMATMUL | `kirin9030/TMatmul.hpp` | 矩阵乘法 (cube)，含 Bias 变体 |
| 4 | TGEMV | `kirin9030/TMatmul.hpp` | 矩阵-向量乘 |
| 5 | TSETTF32MODE | `kirin9030/TMatmul.hpp` | 设置 TF32 模式（kirin9030 始终关闭） |
| 6 | TCVT | `kirin9030/TCvt.hpp` | 数据类型转换（float/half/int8/uint8/int16/int32 等） |
| 7 | TMOV | `kirin9030/TMov.hpp` | Tile 间数据搬移（Vec↔Vec, Acc↔Cb, Cc↔Cb/Ub 等） |
| 8 | TEXTRACT | `kirin9030/TExtract.hpp` | 从 Tile 中抽取数据（Vec→Vec, Acc→Mat, Cube→L0A/L0B 等） |
| 9 | TGATHER | `kirin9030/TGather.hpp` | 使用索引收集数据 |
| 10 | TQUANT | `kirin9030/TQuant.hpp` | 量化（INT8_SYM, INT8_ASYM），MXFP8/MXFP4 被 stub |
| 11 | TSYNC | `kirin9030/TSync.hpp` | 流水线同步 + 事件同步 |

#### kirin9030 包装 a5（1 个文件）

| # | 指令 | 来源 | 说明 |
|---|------|------|------|
| 12 | TINSERT | `kirin9030/TInsert.hpp` → `a5/TInsert.hpp` | 将数据插入 Tile，宏 COPY_CC_TO_CUBF 重写 |

#### 直接复用 a5（81 条指令）

**二元运算（9）**

| # | 指令 | a5 源文件 |
|---|------|-----------|
| 13 | TAdd | `a5/TAdd.hpp` |
| 14 | TSub | `a5/TSub.hpp` |
| 15 | TMul | `a5/TMul.hpp` |
| 16 | TDiv | `a5/TDiv.hpp` |
| 17 | TMax | `a5/TMax.hpp` |
| 18 | TMin | `a5/TMin.hpp` |
| 19 | TAxpy | `a5/TAxpy.hpp` |
| 20 | TPrelu | `a5/TPrelu.hpp` |
| 21 | TLRelu | `a5/TLRelu.hpp` |

**标量运算（8）**

| # | 指令 | a5 源文件 |
|---|------|-----------|
| 22 | TAddS | `a5/TAddS.hpp` |
| 23 | TSubS | `a5/TSubS.hpp` |
| 24 | TMulS | `a5/TMulS.hpp` |
| 25 | TDivS | `a5/TDivS.hpp` |
| 26 | TMaxS | `a5/TMaxS.hpp` |
| 27 | TMins / TMinS | `a5/TMins.hpp`（与 TMaxs 对应） |
| 28 | TMaxs | `a5/TMaxs.hpp` |

**位运算（12）**

| # | 指令 | a5 源文件 |
|---|------|-----------|
| 29 | TAnd | `a5/TAnd.hpp` |
| 30 | TOr | `a5/TOr.hpp` |
| 31 | TXor | `a5/TXor.hpp` |
| 32 | TShl | `a5/TShl.hpp` |
| 33 | TShr | `a5/TShr.hpp` |
| 34 | TAndS | `a5/TAndS.hpp` |
| 35 | TOrS | `a5/TOrS.hpp` |
| 36 | TXorS | `a5/TXorS.hpp` |
| 37 | TShlS | `a5/TShlS.hpp` |
| 38 | TShrS | `a5/TShrS.hpp` |

**比较（2）**

| # | 指令 | a5 源文件 |
|---|------|-----------|
| 39 | TCmp | `a5/TCmp.hpp` |
| 40 | TCmps | `a5/TCmps.hpp` |

**选择（2）**

| # | 指令 | a5 源文件 |
|---|------|-----------|
| 41 | TSel | `a5/TSel.hpp` |
| 42 | TSels | `a5/TSels.hpp` |

**一元运算（7）**

| # | 指令 | a5 源文件 |
|---|------|-----------|
| 43 | TAbs | `a5/TUnaryOp.hpp` |
| 44 | TExp | `a5/TUnaryOp.hpp` |
| 45 | TLog | `a5/TUnaryOp.hpp` |
| 46 | TSqrt | `a5/TUnaryOp.hpp` |
| 47 | TRsqrt | `a5/TRsqrt.hpp` |
| 48 | TRelu | `a5/TUnaryOp.hpp` |
| 49 | TNeg | `a5/TUnaryOp.hpp` |

**数据变换（8）**

| # | 指令 | a5 源文件 |
|---|------|-----------|
| 50 | TAssign | `a5/TAssign.hpp` |
| 51 | TSubView | `a5/TSubView.hpp` |
| 52 | TReshape | `a5/TReshape.hpp` |
| 53 | TConcat | `a5/TConcat.hpp` |
| 54 | TTrans | `a5/TTrans.hpp` |
| 55 | TFillPad | `a5/TFillPad.hpp` |
| 56 | TExpandS | `a5/TExpandS.hpp` |
| 57 | TTri | `a5/TTri.hpp` |

**行广播（8）**

| # | 指令 | a5 源文件 |
|---|------|-----------|
| 58 | TRowExpand | `a5/TRowExpand.hpp` |
| 59 | TRowExpandAdd | `a5/TRowExpandAdd.hpp` |
| 60 | TRowExpandSub | `a5/TRowExpandSub.hpp` |
| 61 | TRowExpandMul | `a5/TRowExpandMul.hpp` |
| 62 | TRowExpandDiv | `a5/TRowExpandDiv.hpp` |
| 63 | TRowExpandMax | `a5/TRowExpandMax.hpp` |
| 64 | TRowExpandMin | `a5/TRowExpandMin.hpp` |

**列广播（8）**

| # | 指令 | a5 源文件 |
|---|------|-----------|
| 65 | TColExpand | `a5/TColExpand.hpp` |
| 66 | TColExpandAdd | `a5/TColExpandAdd.hpp` |
| 67 | TColExpandSub | `a5/TColExpandSub.hpp` |
| 68 | TColExpandMul | `a5/TColExpandMul.hpp` |
| 69 | TColExpandDiv | `a5/TColExpandDiv.hpp` |
| 70 | TColExpandMax | `a5/TColExpandMax.hpp` |
| 71 | TColExpandMin | `a5/TColExpandMin.hpp` |

**行规约（6）**

| # | 指令 | a5 源文件 |
|---|------|-----------|
| 72 | TRowSum | `a5/TRowReduce.hpp` |
| 73 | TRowMax | `a5/TRowReduce.hpp` |
| 74 | TRowMin | `a5/TRowReduce.hpp` |
| 75 | TRowProd | `a5/TRowProd.hpp` |
| 76 | TRowArgMax | `a5/TRowReduceIdx.hpp` |
| 77 | TRowArgMin | `a5/TRowReduceIdx.hpp` |

**列规约（6）**

| # | 指令 | a5 源文件 |
|---|------|-----------|
| 78 | TColSum | `a5/TColSum.hpp` |
| 79 | TColProd | `a5/TColProd.hpp` |
| 80 | TColMax | `a5/TColMax.hpp` |
| 81 | TColMin | `a5/TColMin.hpp` |
| 82 | TColReduceIdx (ColArgMax) | `a5/TColReduceIdx.hpp` |
| 83 | TColReduceIdx (ColArgMin) | `a5/TColReduceIdx.hpp` |

**部分运算（6）**

| # | 指令 | a5 源文件 |
|---|------|-----------|
| 84 | TPartAdd | `a5/TPartAdd.hpp` |
| 85 | TPartMul | `a5/TPartMul.hpp` |
| 86 | TPartMax | `a5/TPartMax.hpp` |
| 87 | TPartMin | `a5/TPartMin.hpp` |
| 88 | TPartArgMax | `a5/TPartArgMax.hpp` |
| 89 | TPartArgMin | `a5/TPartArgMin.hpp` |

**量化（1）**

| # | 指令 | a5 源文件 |
|---|------|-----------|
| 90 | TDeQuant | `a5/TDeQuant.hpp` |

**Gather/Scatter（2）**

| # | 指令 | a5 源文件 |
|---|------|-----------|
| 91 | TGatherB | `a5/TGatherB.hpp` |
| 92 | TScatter | `a5/TScatter.hpp` |

**图像处理（4）**

| # | 指令 | a5 源文件 |
|---|------|-----------|
| 93 | TImg2col | `a5/TImg2col.hpp` |
| 94 | SetFmatrix | `a5/SetFmatrix.hpp` |
| 95 | SetImg2colRpt | `a5/SetImg2colRpt.hpp` |
| 96 | SetImg2colPadding | `a5/SetImg2colPadding.hpp` |

**排序（2）**

| # | 指令 | a5 源文件 |
|---|------|-----------|
| 97 | TSort32 | `a5/TSort32.hpp` |
| 98 | TMrgSort | `a5/TMrgSort.hpp` |

**其他（5）**

| # | 指令 | a5 源文件 |
|---|------|-----------|
| 99 | TPow | `a5/TPow.hpp` |
| 100 | Tci | `a5/Tci.hpp` |
| 101 | THistogram | `a5/THistogram.hpp` |
| 102 | TGetScaleAddr | `a5/TGetScaleAddr.hpp` |
| 103 | TBinSOp | `a5/TBinSOp.hpp` |

### 1.2 a5 有但 kirin9030 缺失的指令（15 条）

| # | 指令 | a5 源文件 | kirin9030 中状态 |
|---|------|-----------|------------------|
| 1 | TFMod | `a5/TFmod.hpp` | 已注释（line 49） |
| 2 | TFModS | `a5/TFModS.hpp` | 已注释（line 42） |
| 3 | TRem | `a5/TRem.hpp` | 已注释（line 50） |
| 4 | TRemS | `a5/TRemS.hpp` | 已注释（line 43） |
| 5 | TRowExpandExpdif | `a5/TRowExpandExpdif.hpp` | 已注释（line 91） |
| 6 | TColExpandExpdif | `a5/TColExpandExpdif.hpp` | 已注释（line 118） |
| 7 | TRandom | `a5/TRandom.hpp` | `// TRandom to be evaluated`（line 105） |
| 8 | TPrefetch | `a5/TPrefetch.hpp` | `// TPrefetch to be evaluated`（line 123） |
| 9 | TPush | `a5/TPush.hpp` | `// TPush to be evaluated`（line 124） |
| 10 | TPop | `a5/TPop.hpp` | `// TPop to be evaluated`（line 125） |
| 11 | TAlloc | `a5/TAlloc.hpp` | `// TAlloc to be evaluated`（line 126） |
| 12 | TFree | `a5/TFree.hpp` | `// TFree to be evaluated`（line 127） |
| 13 | MGather | `a5/MGather.hpp` | `// MGather to be evaluated`（line 113） |
| 14 | MScatter | `a5/MScatter.hpp` | `// MScatter to be evaluated`（line 114） |
| 15 | TMATMUL_MX | `a5/TMatmul.hpp`（MX 变体） | stub `static_assert("no support instruction.")` |

---

## 2. kirin9030 vs kirinX90

### 2.1 两架构都支持的指令

#### kirinX90 原生实现（4 条）

| # | 指令 | kirinX90 源文件 | 说明 |
|---|------|-----------------|------|
| 1 | TLOAD | `kirinX90/TLoad.hpp` | 原生 VecTile/MatTile/ConvTile 加载 |
| 2 | TSTORE | `kirinX90/TStore.hpp` | 原生 VecTile/AccTile/MatTile 存储 |
| 3 | TMOV | `kirinX90/TMov.hpp` | 原生 Tile 间搬移 |
| 4 | TCVT | `kirinX90/TCvt.hpp` | 原生类型转换（条件：`__DAV_VEC__`） |

#### 复用 kirin9030（3 条）

| # | 指令 | 来源 | 说明 |
|---|------|------|------|
| 5 | TSYNC | `kirin9030/TSync.hpp` | 同步 |
| 6 | TMATMUL (+TGEMV/TSETTF32MODE) | `kirin9030/TMatmul.hpp` | 矩阵乘法 |
| 7 | TGATHER | `kirin9030/TGather.hpp` | 收集 |

#### 复用 a2a3（3 条）

| # | 指令 | 来源 | 说明 |
|---|------|------|------|
| 8 | TSubView | `a2a3/TSubView.hpp` | 子 Tile 视图 |
| 9 | TAssign | `a2a3/TAssign.hpp` | Tile 地址赋值 |
| 10 | TExtract | `a2a3/TExtract.hpp` | 抽取数据（与 kirin9030 原生 TExtract 不同来源） |

#### 复用 a5（27 条）

| # | 指令 | a5 源文件 |
|---|------|-----------|
| 11 | TAdd | `a5/TAdd.hpp` |
| 12 | TSub | `a5/TSub.hpp` |
| 13 | TMul | `a5/TMul.hpp` |
| 14 | TDiv | `a5/TDiv.hpp` |
| 15 | TAddS | `a5/TAddS.hpp` |
| 16 | TDivS | `a5/TDivS.hpp` |
| 17 | TMulS | `a5/TMulS.hpp` |
| 18 | TMax | `a5/TMax.hpp` |
| 19 | TMin | `a5/TMin.hpp` |
| 20 | TMrgSort | `a5/TMrgSort.hpp` |
| 21 | TCmps | `a5/TCmps.hpp` |
| 22 | TColSum | `a5/TColSum.hpp` |
| 23 | TReshape | `a5/TReshape.hpp` |
| 24 | TRowReduce (TRowSum/TRowMax/TRowMin) | `a5/TRowReduce.hpp` |
| 25 | TFillPad | `a5/TFillPad.hpp` |
| 26 | TTrans | `a5/TTrans.hpp` |
| 27 | Tci | `a5/Tci.hpp` |
| 28 | TSel | `a5/TSel.hpp` |
| 29 | TSort32 | `a5/TSort32.hpp` |
| 30 | TRowExpand | `a5/TRowExpand.hpp` |
| 31 | TPartAdd | `a5/TPartAdd.hpp` |
| 32 | TPartMax | `a5/TPartMax.hpp` |
| 33 | TPartMin | `a5/TPartMin.hpp` |
| 34 | TRsqrt | `a5/TRsqrt.hpp` |
| 35 | TUnaryOp (TAbs/TExp/TLog/TSqrt/TNot/TRelu/TNeg) | `a5/TUnaryOp.hpp` |
| 36 | TBinSOp | `a5/TBinSOp.hpp` |

### 2.2 kirin9030 有但 kirinX90 缺失的指令

**位运算（10）**

| # | 指令 | kirin9030 来源 | kirinX90 中状态 |
|---|------|---------------|-----------------|
| 1 | TAnd | 复用 a5 | 未 include |
| 2 | TOr | 复用 a5 | 未 include |
| 3 | TXor | 复用 a5 | 未 include |
| 4 | TShl | 复用 a5 | 未 include |
| 5 | TShr | 复用 a5 | 未 include |
| 6 | TAndS | 复用 a5 | 未 include |
| 7 | TOrS | 复用 a5 | 未 include |
| 8 | TXorS | 复用 a5 | 未 include |
| 9 | TShlS | 复用 a5 | 未 include |
| 10 | TShrS | 复用 a5 | 未 include |

**标量运算（3）**

| # | 指令 | kirin9030 来源 |
|---|------|---------------|
| 11 | TSubS | 复用 a5 |
| 12 | TMaxS/TMaxs | 复用 a5 |
| 13 | TMins | 复用 a5 |

**激活函数（2）**

| # | 指令 | kirin9030 来源 |
|---|------|---------------|
| 14 | TPrelu | 复用 a5 |
| 15 | TLRelu | 复用 a5 |

**线性运算（1）**

| # | 指令 | kirin9030 来源 |
|---|------|---------------|
| 16 | TAxpy | 复用 a5 |

**列展开（7）**

| # | 指令 | kirin9030 来源 |
|---|------|---------------|
| 17 | TColExpand | 复用 a5 |
| 18 | TColExpandAdd | 复用 a5 |
| 19 | TColExpandSub | 复用 a5 |
| 20 | TColExpandMul | 复用 a5 |
| 21 | TColExpandDiv | 复用 a5 |
| 22 | TColExpandMax | 复用 a5 |
| 23 | TColExpandMin | 复用 a5 |

**行展开扩展（6）**

| # | 指令 | kirin9030 来源 |
|---|------|---------------|
| 24 | TRowExpandAdd | 复用 a5 |
| 25 | TRowExpandSub | 复用 a5 |
| 26 | TRowExpandMul | 复用 a5 |
| 27 | TRowExpandDiv | 复用 a5 |
| 28 | TRowExpandMax | 复用 a5 |
| 29 | TRowExpandMin | 复用 a5 |

**行规约扩展（3）**

| # | 指令 | kirin9030 来源 |
|---|------|---------------|
| 30 | TRowProd | 复用 a5 |
| 31 | TRowArgMax | 复用 a5 |
| 32 | TRowArgMin | 复用 a5 |

**列规约扩展（4）**

| # | 指令 | kirin9030 来源 |
|---|------|---------------|
| 33 | TColProd | 复用 a5 |
| 34 | TColMax | 复用 a5 |
| 35 | TColMin | 复用 a5 |
| 36 | TColReduceIdx | 复用 a5 |

**部分运算（3）**

| # | 指令 | kirin9030 来源 |
|---|------|---------------|
| 37 | TPartMul | 复用 a5 |
| 38 | TPartArgMax | 复用 a5 |
| 39 | TPartArgMin | 复用 a5 |

**数据变换（3）**

| # | 指令 | kirin9030 来源 |
|---|------|---------------|
| 40 | TConcat | 复用 a5 |
| 41 | TExpandS | 复用 a5 |
| 42 | TTri | 复用 a5 |

**选择（1）**

| # | 指令 | kirin9030 来源 |
|---|------|---------------|
| 43 | TSels | 复用 a5 |

**比较（1）**

| # | 指令 | kirin9030 来源 |
|---|------|---------------|
| 44 | TCmp | 复用 a5（kirinX90 只有 TCmps） |

**量化相关（3）**

| # | 指令 | kirin9030 来源 |
|---|------|---------------|
| 45 | TQuant | kirin9030 原生 |
| 46 | TDeQuant | 复用 a5 |
| 47 | TGetScaleAddr | 复用 a5 |

**Gather/Scatter（2）**

| # | 指令 | kirin9030 来源 |
|---|------|---------------|
| 48 | TGatherB | 复用 a5 |
| 49 | TScatter | 复用 a5 |

**图像处理（4）**

| # | 指令 | kirin9030 来源 |
|---|------|---------------|
| 50 | TImg2col | 复用 a5 |
| 51 | SetFmatrix | 复用 a5 |
| 52 | SetImg2colRpt | 复用 a5 |
| 53 | SetImg2colPadding | 复用 a5 |

**其他（3）**

| # | 指令 | kirin9030 来源 |
|---|------|---------------|
| 54 | TInsert | kirin9030 包装 a5（kirinX90 注释了对应 include line 57） |
| 55 | TPow | 复用 a5 |
| 56 | THistogram | 复用 a5 |

### 2.3 两架构都从 a5 缺失的指令（与 a5→kirin9030 缺失清单相同，15 条）

| # | 指令 | 说明 |
|---|------|------|
| 57 | TFMod | 两架构均未支持 |
| 58 | TFModS | 两架构均未支持 |
| 59 | TRem | 两架构均未支持 |
| 60 | TRemS | 两架构均未支持 |
| 61 | TRowExpandExpdif | 两架构均未支持 |
| 62 | TColExpandExpdif | 两架构均未支持 |
| 63 | TRandom | 待评估 |
| 64 | TPrefetch | 待评估 |
| 65 | TPush | 待评估 |
| 66 | TPop | 待评估 |
| 67 | TAlloc | 待评估 |
| 68 | TFree | 待评估 |
| 69 | MGather | 待评估 |
| 70 | MScatter | 待评估 |
| 71 | TMATMUL_MX | kirin9030 stub, kirinX90 复用 kirin9030 因此同样 stub |

---

## 3. 汇总统计

### 3.1 架构支持指令数

| 架构 | 支持指令数 | 原生实现 | 复用 a5 | 复用 kirin9030 | 复用 a2a3 |
|------|:---------:|:--------:|:-------:|:--------------:|:---------:|
| **a5** | ~110+ | 全部 | - | - | - |
| **kirin9030** | 103 | 11（10 文件） | 91（含 TInsert 包装） | - | - |
| **kirinX90** | ~34 | 4 | 27 | 3 | 3 |

### 3.2 各层差异

| 对比维度 | 差异数 | 说明 |
|---------|:-----:|------|
| a5 有 → kirin9030 无 | 15 | TFMod/TRem 系(4), Expdif(2), Random(1), Prefetch(1), FIFO(4), MGather/Scatter(2), TMATMUL_MX(1) |
| kirin9030 有 → kirinX90 无 | 56 | 位运算(10), 标量运算(3), 激活(2), Axpy(1), 列展开(7), 行展开扩展(6), 行规约扩展(3), 列规约扩展(4), 部分运算(3), 数据变换(3), TSels(1), TCmp(1), 量化(3), Gather/Scatter(2), 图像处理(4), TInsert(1), TPow(1), THistogram(1) |
| 三者都无 | 15 | 与 a5→kirin9030 缺失清单相同 |

### 3.3 header.hpp include 行差异

| 架构 | header.hpp 行数 | 有效 include | 注释 include |
|------|:--------------:|:-----------:|:-----------:|
| kirin9030 | 137 | 95（10 原生 + 85 a5） | 14（注释 + to be evaluated） |
| kirinX90 | 71 | 37（4 原生 + 3 a2a3 + 3 kirin9030 + 27 a5） | 1（TInsert 注释） |
