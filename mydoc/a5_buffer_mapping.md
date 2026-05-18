# A5 Buffer Mapping with Kirin9030

口径：`include/pto/npu/a5/*.hpp` 中，文件名去掉扩展名视为一个指令；`common.hpp`、`datatype.hpp`、`utils.hpp` 排除。

`kirin9030` 列说明：`支持` = `include/pto/npu/kirin9030/` 下有本地实现；`复用` = 通过 `kirin9030/header.hpp` 复用 A5/A2A3；`不支持` = `header.hpp` 未纳入。

`内部调用其他指令` 列说明：列出该文件实现内部显式 `#include` 的其他指令头；无则记为 `无`。

| 指令文件 | buffer | kirin9030 | 内部调用其他指令 |
|---|---|---|---|
| `MGather.hpp` | `__gm__`, `__ubuf__` | 不支持 | 无 |
| `MScatter.hpp` | `__gm__`, `__ubuf__` | 不支持 | 无 |
| `SetFmatrix.hpp` | <none> | 复用 | 无 |
| `SetImg2colPadding.hpp` | <none> | 复用 | 无 |
| `SetImg2colRpt.hpp` | <none> | 复用 | 无 |
| `SyncAll.hpp` | `__gm__`, `__ubuf__`, `__cbuf__` | 不支持 | 无 |
| `TAdd.hpp` | `__ubuf__` | 复用 | 无 |
| `TAddS.hpp` | `__ubuf__` | 复用 | `TBinSOp.hpp` |
| `TAlias.hpp` | <none> | 不支持 | `pto/npu/a2a3/TAlias.hpp` |
| `TAlloc.hpp` | <none> | 不支持 | 无 |
| `TAnd.hpp` | `__ubuf__` | 复用 | 无 |
| `TAndS.hpp` | `__ubuf__` | 复用 | `TBinSOp.hpp` |
| `TAssign.hpp` | <none> | 复用 | `pto/npu/a2a3/TAssign.hpp` |
| `TAxpy.hpp` | `__ubuf__` | 复用 | 无 |
| `TBinOp.hpp` | `__ubuf__` | 不支持 | 无 |
| `TBinSOp.hpp` | `__ubuf__` | 复用 | 无 |
| `TCmp.hpp` | `__ubuf__` | 复用 | 无 |
| `TCmps.hpp` | `__ubuf__` | 复用 | 无 |
| `TColExpand.hpp` | `__ubuf__` | 复用 | 无 |
| `TColExpandAdd.hpp` | `__ubuf__` | 复用 | `TColExpandBinOp.hpp` |
| `TColExpandBinOp.hpp` | `__ubuf__` | 不支持 | 无 |
| `TColExpandDiv.hpp` | `__ubuf__` | 复用 | `TColExpandBinOp.hpp` |
| `TColExpandExpdif.hpp` | `__ubuf__` | 不支持 | `TColExpandBinOp.hpp` |
| `TColExpandMax.hpp` | `__ubuf__` | 复用 | `TColExpandBinOp.hpp` |
| `TColExpandMin.hpp` | `__ubuf__` | 复用 | `TColExpandBinOp.hpp` |
| `TColExpandMul.hpp` | `__ubuf__` | 复用 | `TColExpandBinOp.hpp` |
| `TColExpandSub.hpp` | `__ubuf__` | 复用 | `TColExpandBinOp.hpp` |
| `TColMax.hpp` | `__ubuf__` | 复用 | `TColReduceOps.hpp` |
| `TColMin.hpp` | `__ubuf__` | 复用 | `TColReduceOps.hpp` |
| `TColProd.hpp` | `__ubuf__` | 复用 | `TColReduceOps.hpp` |
| `TColReduceIdx.hpp` | `__ubuf__` | 复用 | 无 |
| `TColReduceOps.hpp` | `__ubuf__` | 不支持 | 无 |
| `TColSum.hpp` | `__ubuf__` | 复用 | `TColReduceOps.hpp` |
| `TConcat.hpp` | `__ubuf__` | 复用 | 无 |
| `TCvt.hpp` | `__ubuf__` | 支持 | 无 |
| `TDeQuant.hpp` | `__ubuf__` | 复用 | 无 |
| `TDiv.hpp` | `__ubuf__` | 复用 | 无 |
| `TDivS.hpp` | `__ubuf__` | 复用 | `TBinSOp.hpp` |
| `TExpandS.hpp` | `__ubuf__`, `__cbuf__` | 复用 | `TBinSOp.hpp` |
| `TExtract.hpp` | `__ubuf__`, `__cbuf__`, `__ca__`, `__cb__`, `__cc__`, `__fbuf__` | 支持 | 无 |
| `TFMod.hpp` | `__ubuf__` | 不支持 | 无 |
| `TFModS.hpp` | `__ubuf__` | 不支持 | `TBinSOp.hpp` |
| `TFillPad.hpp` | `__ubuf__`, `__cbuf__` | 复用 | `TLoad.hpp` |
| `TFree.hpp` | <none> | 不支持 | 无 |
| `TGather.hpp` | `__ubuf__` | 支持 | 无 |
| `TGatherB.hpp` | `__ubuf__` | 复用 | 无 |
| `TGetScaleAddr.hpp` | <none> | 复用 | 无 |
| `THistogram.hpp` | `__ubuf__` | 复用 | 无 |
| `TImg2col.hpp` | `__cbuf__`, `__ca__` | 复用 | 无 |
| `TInsert.hpp` | `__ubuf__`, `__cbuf__`, `__cc__`, `__fbuf__` | 支持 | 无 |
| `TLRelu.hpp` | `__ubuf__` | 复用 | `TBinSOp.hpp` |
| `TLoad.hpp` | `__gm__`, `__ubuf__`, `__cbuf__` | 支持 | 无 |
| `TMatmul.hpp` | `__ca__`, `__cb__`, `__cc__` | 支持 | 无 |
| `TMax.hpp` | `__ubuf__` | 复用 | 无 |
| `TMaxs.hpp` | `__ubuf__` | 复用 | `TBinSOp.hpp` |
| `TMin.hpp` | `__ubuf__` | 复用 | 无 |
| `TMins.hpp` | `__ubuf__` | 复用 | `TBinSOp.hpp` |
| `TMov.hpp` | `__gm__`, `__ubuf__`, `__cbuf__`, `__ca__`, `__cb__`, `__cc__`, `__fbuf__` | 支持 | `TExtract.hpp`, `TPartAdd.hpp` |
| `TMrgSort.hpp` | `__ubuf__` | 复用 | 无 |
| `TMul.hpp` | `__ubuf__` | 复用 | 无 |
| `TMulS.hpp` | `__ubuf__` | 复用 | `TBinSOp.hpp` |
| `TOr.hpp` | `__ubuf__` | 复用 | 无 |
| `TOrS.hpp` | `__ubuf__` | 复用 | `TBinSOp.hpp` |
| `TPartAdd.hpp` | `__ubuf__` | 复用 | 无 |
| `TPartArgBinOps.hpp` | `__ubuf__` | 不支持 | `TPartBinOps.hpp` |
| `TPartArgMax.hpp` | `__ubuf__` | 复用 | `TPartBinOps.hpp`, `TPartArgBinOps.hpp` |
| `TPartArgMin.hpp` | `__ubuf__` | 复用 | `TPartBinOps.hpp`, `TPartArgBinOps.hpp` |
| `TPartBinOps.hpp` | `__ubuf__` | 不支持 | 无 |
| `TPartMax.hpp` | `__ubuf__` | 复用 | `TPartBinOps.hpp` |
| `TPartMin.hpp` | `__ubuf__` | 复用 | `TPartBinOps.hpp` |
| `TPartMul.hpp` | <none> | 复用 | 无 |
| `TPop.hpp` | <none> | 不支持 | 无 |
| `TPow.hpp` | `__ubuf__` | 复用 | 无 |
| `TPrefetch.hpp` | `__gm__`, `__ubuf__`, `__cbuf__` | 不支持 | `TLoad.hpp` |
| `TPrelu.hpp` | `__ubuf__` | 复用 | 无 |
| `TPrint.hpp` | <none> | 不支持 | `pto/npu/a2a3/TPrint.hpp` |
| `TPush.hpp` | `__gm__` | 不支持 | 无 |
| `TQuant.hpp` | `__ubuf__` | 支持 | `TReshape.hpp` |
| `TRandom.hpp` | `__ubuf__` | 不支持 | 无 |
| `TRem.hpp` | `__ubuf__` | 不支持 | 无 |
| `TRemS.hpp` | `__ubuf__` | 不支持 | `TBinSOp.hpp` |
| `TReshape.hpp` | <none> | 复用 | `pto/npu/a2a3/TReshape.hpp` |
| `TRowExpand.hpp` | `__ubuf__` | 复用 | 无 |
| `TRowExpandAdd.hpp` | `__ubuf__` | 复用 | `TRowExpandBinOp.hpp` |
| `TRowExpandBinOp.hpp` | `__ubuf__` | 不支持 | 无 |
| `TRowExpandDiv.hpp` | `__ubuf__` | 复用 | `TRowExpandBinOp.hpp` |
| `TRowExpandExpdif.hpp` | `__ubuf__` | 不支持 | `TRowExpandBinOp.hpp` |
| `TRowExpandMax.hpp` | `__ubuf__` | 复用 | `TRowExpandBinOp.hpp` |
| `TRowExpandMin.hpp` | `__ubuf__` | 复用 | `TRowExpandBinOp.hpp` |
| `TRowExpandMul.hpp` | `__ubuf__` | 复用 | `TRowExpandBinOp.hpp` |
| `TRowExpandSub.hpp` | `__ubuf__` | 复用 | `TRowExpandBinOp.hpp` |
| `TRowProd.hpp` | `__ubuf__` | 复用 | 无 |
| `TRowReduce.hpp` | `__ubuf__` | 复用 | `TPartBinOps.hpp` |
| `TRowReduceIdx.hpp` | `__ubuf__` | 复用 | `TPartBinOps.hpp` |
| `TRsqrt.hpp` | `__ubuf__` | 复用 | 无 |
| `TScatter.hpp` | `__ubuf__` | 复用 | 无 |
| `TSel.hpp` | `__ubuf__` | 复用 | 无 |
| `TSels.hpp` | `__ubuf__` | 复用 | 无 |
| `TShl.hpp` | `__ubuf__` | 复用 | 无 |
| `TShlS.hpp` | `__ubuf__` | 复用 | `TBinSOp.hpp` |
| `TShr.hpp` | `__ubuf__` | 复用 | 无 |
| `TShrS.hpp` | `__ubuf__` | 复用 | `TBinSOp.hpp` |
| `TSort32.hpp` | `__ubuf__` | 复用 | 无 |
| `TStore.hpp` | `__gm__`, `__ubuf__`, `__cc__`, `__fbuf__` | 支持 | 无 |
| `TSub.hpp` | `__ubuf__` | 复用 | 无 |
| `TSubS.hpp` | `__ubuf__` | 复用 | `TBinSOp.hpp` |
| `TSubView.hpp` | <none> | 复用 | `pto/npu/a2a3/TSubView.hpp` |
| `TSync.hpp` | <none> | 支持 | 无 |
| `TTrans.hpp` | `__ubuf__` | 复用 | 无 |
| `TTri.hpp` | `__ubuf__` | 复用 | 无 |
| `TUnaryOp.hpp` | `__ubuf__` | 复用 | 无 |
| `TXor.hpp` | `__ubuf__` | 复用 | 无 |
| `TXorS.hpp` | `__ubuf__` | 复用 | `TBinSOp.hpp` |
| `Tci.hpp` | `__ubuf__` | 复用 | 无 |

## Statistics

基于当前表格统计，共 114 行，对应 A5 顶层非辅助头文件全集。

### 1. 指令状态分布

- `复用`: 82
- `不支持`: 22
- `支持`: 10

解读：
- `复用` 占绝对多数，说明 A5 目录里大量文件只是把公共族头或 A2A3 旧实现组装到一起。
- `支持` 只有 10 个，主要是 A5 自己的本地实现核心入口。
- `不支持` 22 个，主要集中在辅助型、占位型或未纳入 `header.hpp` 的指令。

### 2. buffer 使用分布

按表中出现次数统计：

- `__ubuf__`: 97
- `__cbuf__`: 9
- `__gm__`: 8
- `__cc__`: 5
- `__fbuf__`: 4
- `__ca__`: 4
- `__cb__`: 3

解读：
- `__ubuf__` 是 A5 指令实现的绝对主力，几乎覆盖所有向量/标量/中间态处理。
- `__gm__` 主要出现在搬运、预取、同步、存储等 GM 相关指令中。
- `__cbuf__` 主要集中在 `TLoad`、`TStore`、`TMov`、`TFillPad`、`TPrefetch` 等涉及 L1/CBUF 的路径。
- `__ca__`、`__cb__`、`__cc__` 和 `__fbuf__` 出现次数少，但都集中在 cube/matmul/extract/mov 这类硬件语义更强的文件里。

### 3. 内部调用其他指令分布

- `无`: 65
- 1 个依赖: 47
- 2 个依赖: 2

解读：
- 65 个文件没有显式 `#include` 其他指令头，说明它们是相对独立的实现单元。
- 47 个文件只依赖 1 个公共族头，说明 A5 的代码结构高度模块化，常见模式是“文件名对应入口，核心算法放到公共族头”。
- 只有 2 个文件依赖 2 个头：`TPartArgMax.hpp` 和 `TPartArgMin.hpp`。

### 4. 最常见的被复用头文件

按依赖次数统计：

- `TBinSOp.hpp`: 15
- `TColExpandBinOp.hpp`: 7
- `TPartBinOps.hpp`: 7
- `TRowExpandBinOp.hpp`: 7
- `TColReduceOps.hpp`: 4
- `TLoad.hpp`: 2
- `TPartArgBinOps.hpp`: 2

解读：
- `TBinSOp.hpp` 是最典型的基础算子族头，承载了大量“标量/双目”变体：`TAddS`、`TAndS`、`TDivS`、`TExpandS`、`TFModS`、`TLRelu`、`TMaxs`、`TMins`、`TMulS`、`TOrS`、`TRemS`、`TShlS`、`TShrS`、`TSubS`、`TXorS`。
- `TColExpandBinOp.hpp` 和 `TRowExpandBinOp.hpp` 分别对应列展开和行展开的公共二元模板，说明这两类算子的实现模式高度一致。
- `TPartBinOps.hpp` 及其派生族覆盖了局部归约、局部 arg 计算等逻辑，是分块/分段算子的核心。

### 5. 文件类型分布观察

#### 5.1 纯 `__ubuf__` 文件最多

只有 `__ubuf__` 的文件数量很多，典型包括：
- 算术/逻辑类：`TAdd`、`TSub`、`TMul`、`TDiv`、`TAnd`、`TOr`、`TXor`
- 标量变体：`TAddS`、`TMulS`、`TDivS`、`TXorS` 等
- 行/列展开和归约：`TRowExpand*`、`TColExpand*`、`TRowReduce*`、`TColReduce*`
- 选择/比较/排序/随机相关：`TCmp`、`TCmps`、`TSel`、`TSels`、`TSort32`、`TRandom`

这类文件说明 A5 的大部分算子是围绕 UB 向量处理组织的。

#### 5.2 多 buffer 文件集中在少数关键算子

使用 3 个以上 buffer 的文件只有少数：
- `TExtract.hpp`: `__ubuf__`, `__cbuf__`, `__ca__`, `__cb__`, `__cc__`, `__fbuf__`
- `TMov.hpp`: `__gm__`, `__ubuf__`, `__cbuf__`, `__ca__`, `__cb__`, `__cc__`, `__fbuf__`
- `TStore.hpp`: `__gm__`, `__ubuf__`, `__cc__`, `__fbuf__`
- `TLoad.hpp`: `__gm__`, `__ubuf__`, `__cbuf__`
- `TPrefetch.hpp`: `__gm__`, `__ubuf__`, `__cbuf__`

这些文件通常对应“内存搬运 + 格式转换 + 分形适配”的复杂逻辑，是 A5 与普通 UB 算子差异最大的区域。

### 6. `kirin9030` 支持情况观察

- `支持`: 10
- `复用`: 82
- `不支持`: 22

解读：
- `kirin9030` 已支持的本地实现集中在 `TLoad`、`TStore`、`TMatmul`、`TExtract`、`TMov`、`TCvt`、`TQuant`、`TGather`、`TSync`、`TInsert`。
- 大量 `复用` 表明 `kirin9030` 的策略是尽量复用 A5 / A2A3 的成熟实现，而不是重新实现全部指令。
- `不支持` 的文件多数是辅助、占位或 A5 才新增的族头，这也符合 `kirin9030` 更保守的能力边界。

### 7. 额外校验结果

- 当前表格行数已修正为 114，与 A5 顶层非辅助头文件数一致。
- 之前缺失的 `TPartBinOps.hpp` 已补回。


## Summary

- A5 中显式依赖其他指令头的文件主要集中在 `TBinSOp`、`TRowExpandBinOp`、`TColExpandBinOp`、`TColReduceOps`、`TPartBinOps`、`TPartArgBinOps`、`TLoad`、`TReshape` 等公共族头。
- 直接复用 A2A3 头的文件是 `TAlias.hpp`、`TAssign.hpp`、`TPrint.hpp`、`TReshape.hpp`、`TSubView.hpp`。
