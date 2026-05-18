# Kirin9030 与 KirinX90 指令集支持分析

## 背景

根据 README.md 的描述：

> PTO（Parallel Tile Operation）是昇腾 CANN 定义的一套面向 tile 编程的虚拟 ISA。本仓库提供 PTO Tile 指令的实现、示例、测试与文档。
>
> include/pto/npu目录下有不同硬件架构对应同一个指令的不同实现，其中 a2a3 与 a5 的硬件架构差异较大，kirin9030 以及 kirinX90 和 a5 是相似但不完全相同的硬件架构，kirin9030 以及 kirinX90 的硬件架构也不完全相同。
>
> 现在需要做的是将 kirin9030 的指令对齐到 a5，kirinX90 的指令对齐到 kirin9030，由于三者的硬件架构比较相似，因此部分指令可以复用，体现在 include/pto/npu/kirin9030/header.hpp 和 include/pto/npu/kirinX90/header.hpp 中。
>
> 不同硬件对应的已经支持的指令的测试用例在 tests/npu 路径下，目前 kirin9030 和 kirinX90 共用一套测试用例。

## 信息源

- `include/pto/npu/kirin9030/header.hpp` — kirin9030 的指令包含清单
- `include/pto/npu/kirinX90/header.hpp` — kirinX90 的指令包含清单
- `include/pto/npu/kirin9030/` 目录下各指令实现文件
- `include/pto/npu/kirinX90/` 目录下各指令实现文件
- `tests/npu/kirin9030/src/st/testcase/` — 共享的 119 个测试用例目录

## Kirin9030 支持的指令

### 原生实现（kirin9030/ 目录，10 个文件）

| 指令 | 文件 | 说明 |
|------|------|------|
| **TLoad** | `TLoad.hpp` | 从全局内存加载数据到 tile，支持多种数据排布格式（ND/ND2NZ/DN2ZN 等） |
| **TStore** | `TStore.hpp` | 将 tile 数据写回全局内存，支持量化/ReLU/AtomicAdd 等预处理 |
| **TMov** | `TMov.hpp` | tile 间数据搬运（Mat→Left/Right/Bias/Scaling, Acc→Vec/Mat, Vec→Vec/Mat） |
| **TCvt** | `TCvt.hpp` | 数据类型转换，支持 F32/F16/I8/U8 等互转，多种舍入模式 |
| **TExtract** | `TExtract.hpp` | 从源 tile 提取子块到目标 tile（L1→L0A/L0B, Vec→Mat, Acc→Mat/Vec） |
| **TMatmul** | `TMatmul.hpp` | 矩阵乘法（cube 单元），支持 fp16→fp16、int8→int32 精度路径，含 bias/gemv 变体 |
| **TQuant** | `TQuant.hpp` | 量化操作（INT8_SYM/INT8_ASYM） |
| **TGather** | `TGather.hpp` | 基于索引的 gather/scatter 操作 |
| **TSync** | `TSync.hpp` | 同步原语（管道 barrier、跨管道事件同步） |
| **TInsert** | `TInsert.hpp`（薄封装，下层复用 a5） | 将子块插入目标 tile |

### 从 a5 复用的指令

#### 算术运算
TAdd, TAddS, TSub, TSubS, TMul, TMulS, TDiv, TDivS, TAxpy, TPow

#### 位运算
TAnd, TAndS, TOr, TOrS, TXor, TXorS, TShl, TShlS, TShr, TShrS

#### 比较与选择
TCmp, TCmps, TMin, TMins, TMax, TMaxs, TSel, TSels

#### 激活函数
TPrelu, TLRelu

#### 数学函数
TRsqrt, TUnaryOp（abs/neg/exp/log/sqrt 等）, TBinSOp

#### 规约操作
TColSum, TColProd, TColMax, TColMin, TRowReduce, TRowReduceIdx, TRowProd, TColReduceIdx

#### 行列扩展
TRowExpand, TColExpand, TRowExpandAdd/Sub/Mul/Div/Max/Min, TColExpandAdd/Sub/Mul/Div/Max/Min

#### 数据变换
TReshape, TTrans, TConcat, TExpandS, TTri, TFillPad, TScatter, TGatherB, Tci, TSubView, TAssign

#### 分区操作
TPartAdd, TPartMul, TPartMax, TPartMin, TPartArgMax, TPartArgMin

#### 排序
TMrgSort, TSort32

#### 其他
TDeQuant, TImg2col, SetFmatrix, SetImg2colRpt, SetImg2colPadding, TGetScaleAddr, THistogram

### 不支持（注释/断言拦截）
TFMod, TFModS, TRem, TRemS, TRandom, MGather, MScatter, TPrefetch, TPush, TPop, TAlloc, TFree, TRowExpandExpdif, TColExpandExpdif, TMatmul_MX, TQuant_MX

### 总计
约 **92 个**指令族

---

## KirinX90 支持的指令

### 原生实现（kirinX90/ 目录，5 个文件）

| 指令 | 文件 | 说明 |
|------|------|------|
| **TLoad** | `TLoad.hpp` | 从全局内存加载数据到 tile |
| **TStore** | `TStore.hpp` | 将 tile 数据写回全局内存 |
| **TMov** | `TMov.hpp` | tile 间数据搬运 |
| **TCvt** | `TCvt.hpp` | 数据类型转换（仅在 `__DAV_VEC__` 宏下编译） |
| **TExtract** | — | 无 kirinX90 原生实现，复用 a2a3/TExtract.hpp |

### 从 kirin9030 复用的指令

| 指令 | 来源 | 说明 |
|------|------|------|
| **TMatmul** | `kirin9030/TMatmul.hpp` | 矩阵乘法 |
| **TSync** | `kirin9030/TSync.hpp` | 同步原语 |
| **TGather** | `kirin9030/TGather.hpp` | 基于索引的 gather |
| **TQuant** | `kirin9030/TQuant.hpp` | 量化操作 |
| **utils** | `kirin9030/utils.hpp` | 工具函数 |

### 从 a5 复用的指令

#### 算术运算
TAdd, TAddS, TSub, TSubS, TMul, TMulS, TDiv, TDivS, TAxpy, TPow

#### 位运算
TAnd, TAndS, TOr, TOrS, TXor, TXorS, TShl, TShlS, TShr, TShrS

#### 比较与选择
TCmp, TCmps, TMin, TMins, TMax, TMaxs, TSel, TSels

#### 激活函数
TPrelu, TLRelu

#### 数学函数
TRsqrt, TUnaryOp, TBinSOp

#### 规约操作
TColSum, TColProd, TColMax, TColMin, TRowReduce, TRowReduceIdx, TRowProd, TColReduceIdx

#### 行列扩展
TRowExpand, TColExpand, TRowExpandAdd/Sub/Mul/Div/Max/Min, TColExpandAdd/Sub/Mul/Div/Max/Min

#### 数据变换
TReshape, TTrans, TConcat, TExpandS, TTri, TFillPad, TScatter, TGatherB, Tci

#### 分区操作
TPartAdd, TPartMul, TPartMax, TPartMin, TPartArgMax, TPartArgMin

#### 排序
TMrgSort, TSort32

### 从 a2a3 复用的指令

| 指令 | 来源 | 说明 |
|------|------|------|
| **TSubView** | `a2a3/TSubView.hpp` | 创建 tile 的子视图 |
| **TAssign** | `a2a3/TAssign.hpp` | tile 间赋值 |
| **TExtract** | `a2a3/TExtract.hpp` | 提取子块（通用实现） |

### 不支持
- **TInsert**（注释掉：`// #include "pto/npu/kirin9030/TInsert.hpp"`）
- **THistogram**（注释掉：`// #include "pto/npu/a5/THistogram.hpp"`）
- **TDeQuant**（无 include）
- **TImg2col / SetFmatrix / SetImg2colRpt / SetImg2colPadding**（无 include）
- **TGetScaleAddr**（无 include）
- 同 kirin9030 不支持的其他指令

### 总计
约 **85 个**指令族

---

## KirinX90 vs Kirin9030 关键差异对比

| 指令 | Kirin9030 | KirinX90 | 备注 |
|------|-----------|----------|------|
| **TInsert** | ✅ 支持 | ❌ 不支持 | kirinX90 的 header.hpp 中注释掉了 include |
| **THistogram** | ✅ 支持 | ❌ 不支持 | kirinX90 的 header.hpp 中注释掉了 include |
| **TDeQuant** | ✅ 支持 | ❌ 不支持 | kirinX90 没有 include |
| **TImg2col / SetFmatrix / SetImg2colRpt / SetImg2colPadding** | ✅ 支持 | ❌ 不支持 | 卷积相关指令，kirinX90 缺失 |
| **TGetScaleAddr** | ✅ 支持 | ❌ 不支持 | 量化缩放地址获取 |
| **TExtract** | ✅ 原生实现（功能完整） | ⚠️ 复用 a2a3 通用实现 | a2a3 的实现可能不如 kirin9030 原生实现功能完整 |
| **TCvt** | ✅ 无条件编译 | ⚠️ 条件编译（`__DAV_VEC__`） | 在不支持 `__DAV_VEC__` 的环境下可能不可用 |
| **TSubView** | ✅ 复用 a5 | ⚠️ 复用 a2a3 | 不同来源，实现可能不同 |
| **TAssign** | ✅ 复用 a5 | ⚠️ 复用 a2a3 | 不同来源，实现可能不同 |
| **TMatmul** | ✅ 原生实现 | ✅ 复用 kirin9030 | 功能对齐 |
| **TQuant** | ✅ 原生实现 | ✅ 复用 kirin9030 | 功能对齐 |
| **TGather** | ✅ 原生实现 | ✅ 复用 kirin9030 | 功能对齐 |
| **TSync** | ✅ 原生实现 | ✅ 复用 kirin9030 | 功能对齐 |
| **TLoad/TStore/TMov** | ✅ 原生实现 | ✅ 原生实现 | 各自有独立的硬件适配实现 |

## 测试用例覆盖情况

两硬件架构共用 `tests/npu/kirin9030/src/st/testcase/` 下的 **119 个**测试目录。值得注意的是，测试用例覆盖了 `thistogram`、`tinsert` 等指令的测试，但 **kirinX90 的 header.hpp 中并未包含这些指令**，因此这些测试在 kirinX90 上会编译失败或运行失败。当前测试脚本（`run_st.py`/`build_st.py`）对 `kirin9030` 和 `kirinX90` 均映射到同一测试源码目录，未做差异过滤。

---

## KirinX90 缺失指令分析

### 对比说明

以 kirinX90/header.hpp 为基准，对比 kirin9030/header.hpp 和 a5 全部 117 个 hpp 文件。

### 一、KirinX90 缺失 vs Kirin9030（对齐目标，共 8 个）

| 缺失指令 | kirin9030 来源 | 说明 |
|----------|----------------|------|
| **TInsert** | `kirin9030/TInsert.hpp` | kirinX90 的 header 中注释掉了 include |
| **TDeQuant** | `a5/TDeQuant.hpp` | 反量化，kirinX90 完全没包含 |
| **TGetScaleAddr** | `a5/TGetScaleAddr.hpp` | 获取量化缩放地址，kirinX90 完全没包含 |
| **TImg2col** | `a5/TImg2col.hpp` | 图像转列（卷积辅助），kirinX90 缺失 |
| **SetFmatrix** | `a5/SetFmatrix.hpp` | 卷积 filter 矩阵设置，kirinX90 缺失 |
| **SetImg2colRpt** | `a5/SetImg2colRpt.hpp` | img2col 重复参数设置，kirinX90 缺失 |
| **SetImg2colPadding** | `a5/SetImg2colPadding.hpp` | img2col padding 设置，kirinX90 缺失 |
| **THistogram** | `a5/THistogram.hpp` | 直方图，kirinX90 的 header 中注释掉了 include |

此外，**TExtract** 在 kirin9030 中有原生实现（`kirin9030/TExtract.hpp`），而 kirinX90 只用了 `a2a3/TExtract.hpp`（通用版本）；**TCvt** 在 kirin9030 是无条件编译，kirinX90 需要 `__DAV_VEC__` 宏。

### 二、KirinX90（及 Kirin9030）均缺失 vs a5（共 15 个）

以下指令在 a5 中存在，但 kirinX90 和 kirin9030 的 header 中都注释掉或完全没包含：

| 缺失指令 | 说明 |
|----------|------|
| **TFMod / TFModS** | 浮点数取模（两架构均注释掉） |
| **TRem / TRemS** | 取余（两架构均注释掉） |
| **TRandom** | 随机数生成（标注"to be evaluated"） |
| **MGather / MScatter** | 矩阵 gather/scatter（标注"to be evaluated"） |
| **TPrefetch** | 预取（标注"to be evaluated"） |
| **TPush / TPop** | 栈操作（标注"to be evaluated"） |
| **TAlloc / TFree** | tile 内存分配/释放（标注"to be evaluated"） |
| **TPrint** | 打印调试（仅 a5 有，两架构均无） |
| **TRowExpandExpdif / TColExpandExpdif** | exp 差值扩展（两架构均注释掉） |

### 总结

- **KirinX90 对齐 kirin9030** 需补齐：**8 个**指令（TInsert、TDeQuant、TGetScaleAddr、TImg2col、SetFmatrix、SetImg2colRpt、SetImg2colPadding、THistogram）
- **Kirin9030 对齐 a5** 需补齐（同时也是 kirinX90 和 kirin9030 共同的差距）：**15 个**指令
- 当前脚本 `run_kirinX90_tests.py` 只跑了 kirinX90 **已有**的 90 个测试，上述缺失指令对应的测试（`tinsert`、`tdequant`、`thistogram`、`timg2col` 等）不在列表中，运行时自然被排除。
