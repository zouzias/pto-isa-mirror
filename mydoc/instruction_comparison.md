# A5 vs Kirin9030 vs KirinX90 指令支持对照表

## 背景

以 a5 已实现的全部指令为参考（共 104 个指令族），对比 kirin9030 和 kirinX90 的支持情况。

图例：
- ✅ = header.hpp 中已 include，可直接使用
- ⚠️ = 有条件/有限支持（见备注）
- ❌ = header.hpp 中未 include 或被注释掉

---

## 一、数据搬运指令

| 指令 | A5 | Kirin9030 | KirinX90 | 备注 |
|------|:--:|:---------:|:---------:|------|
| **TLoad** | ✅ | ✅ 原生 | ✅ 原生 | 三架构各自有独立实现 |
| **TStore** | ✅ | ✅ 原生 | ✅ 原生 | 三架构各自有独立实现 |
| **TMov** | ✅ | ✅ 原生 | ✅ 原生 | 三架构各自有独立实现 |
| **TAssign** | ✅ | ✅ 复用 a5 | ⚠️ 复用 a2a3 | kirinX90 使用 a2a3 版本 |
| **TSubView** | ✅ | ✅ 复用 a5 | ⚠️ 复用 a2a3 | kirinX90 使用 a2a3 版本 |
| **TExtract** | ✅ | ✅ 原生 | ⚠️ 复用 a2a3 | kirinX90 使用 a2a3 通用版，非原生 |
| **TInsert** | ✅ | ✅ 原生(复用a5) | ❌ 注释掉 | kirinX90 header 中 `// #include` |
| **TReshape** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TTrans** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TConcat** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TExpandS** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TFillPad** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TScatter** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TGather** | ✅ | ✅ 原生 | ✅ 复用 kirin9030 | |
| **TGatherB** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **Tci** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TTri** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TPrefetch** | ✅ | ❌ to be evaluated | ❌ to be evaluated | |
| **TPush** | ✅ | ❌ to be evaluated | ❌ to be evaluated | |
| **TPop** | ✅ | ❌ to be evaluated | ❌ to be evaluated | |
| **TAlloc** | ✅ | ❌ to be evaluated | ❌ to be evaluated | |
| **TFree** | ✅ | ❌ to be evaluated | ❌ to be evaluated | |

## 二、算术运算指令

| 指令 | A5 | Kirin9030 | KirinX90 | 备注 |
|------|:--:|:---------:|:---------:|------|
| **TAdd** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TAddS** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TSub** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TSubS** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TMul** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TMulS** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TDiv** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TDivS** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TAxpy** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TPow** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TFMod** | ✅ | ❌ 注释掉 | ❌ 未包含 | |
| **TFModS** | ✅ | ❌ 注释掉 | ❌ 未包含 | |
| **TRem** | ✅ | ❌ 注释掉 | ❌ 未包含 | |
| **TRemS** | ✅ | ❌ 注释掉 | ❌ 未包含 | |

## 三、位运算指令

| 指令 | A5 | Kirin9030 | KirinX90 | 备注 |
|------|:--:|:---------:|:---------:|------|
| **TAnd** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TAndS** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TOr** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TOrS** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TXor** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TXorS** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TShl** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TShlS** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TShr** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TShrS** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |

## 四、比较与选择指令

| 指令 | A5 | Kirin9030 | KirinX90 | 备注 |
|------|:--:|:---------:|:---------:|------|
| **TCmp** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TCmps** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TMin** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TMins** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TMax** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TMaxs** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TSel** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TSels** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |

## 五、激活与数学函数指令

| 指令 | A5 | Kirin9030 | KirinX90 | 备注 |
|------|:--:|:---------:|:---------:|------|
| **TPrelu** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TLRelu** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TUnaryOp** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | abs/neg/exp/log/sqrt 等 |
| **TBinSOp** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | 二元标量操作 |
| **TRsqrt** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |

## 六、规约操作指令

| 指令 | A5 | Kirin9030 | KirinX90 | 备注 |
|------|:--:|:---------:|:---------:|------|
| **TColSum** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TColProd** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TColMax** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TColMin** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TColReduceIdx** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TRowReduce** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TRowReduceIdx** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TRowProd** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |

## 七、行列扩展指令

| 指令 | A5 | Kirin9030 | KirinX90 | 备注 |
|------|:--:|:---------:|:---------:|------|
| **TRowExpand** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TRowExpandAdd** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TRowExpandSub** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TRowExpandMul** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TRowExpandDiv** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TRowExpandMax** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TRowExpandMin** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TRowExpandExpdif** | ✅ | ❌ 注释掉 | ❌ 未包含 | |
| **TColExpand** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TColExpandAdd** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TColExpandSub** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TColExpandMul** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TColExpandDiv** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TColExpandMax** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TColExpandMin** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TColExpandExpdif** | ✅ | ❌ 注释掉 | ❌ 未包含 | |

## 八、分区操作指令

| 指令 | A5 | Kirin9030 | KirinX90 | 备注 |
|------|:--:|:---------:|:---------:|------|
| **TPartAdd** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TPartMul** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TPartMax** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TPartMin** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TPartArgMax** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TPartArgMin** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |

## 九、排序指令

| 指令 | A5 | Kirin9030 | KirinX90 | 备注 |
|------|:--:|:---------:|:---------:|------|
| **TSort32** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |
| **TMrgSort** | ✅ | ✅ 复用 a5 | ✅ 复用 a5 | |

## 十、类型转换指令

| 指令 | A5 | Kirin9030 | KirinX90 | 备注 |
|------|:--:|:---------:|:---------:|------|
| **TCvt** | ✅ | ✅ 原生 | ⚠️ 条件编译 | kirinX90 需 `__DAV_VEC__` 宏 |
| **TQuant** | ✅ | ✅ 原生 | ✅ 复用 kirin9030 | |
| **TDeQuant** | ✅ | ✅ 复用 a5 | ❌ 未包含 | |

## 十一、矩阵计算指令

| 指令 | A5 | Kirin9030 | KirinX90 | 备注 |
|------|:--:|:---------:|:---------:|------|
| **TMatmul** | ✅ | ✅ 原生 | ✅ 复用 kirin9030 | |

## 十二、同步指令

| 指令 | A5 | Kirin9030 | KirinX90 | 备注 |
|------|:--:|:---------:|:---------:|------|
| **TSync** | ✅ | ✅ 原生 | ✅ 复用 kirin9030 | |

## 十三、卷积辅助指令

| 指令 | A5 | Kirin9030 | KirinX90 | 备注 |
|------|:--:|:---------:|:---------:|------|
| **TImg2col** | ✅ | ✅ 复用 a5 | ❌ 未包含 | |
| **SetFmatrix** | ✅ | ✅ 复用 a5 | ❌ 未包含 | |
| **SetImg2colRpt** | ✅ | ✅ 复用 a5 | ❌ 未包含 | |
| **SetImg2colPadding** | ✅ | ✅ 复用 a5 | ❌ 未包含 | |

## 十四、其他专用指令

| 指令 | A5 | Kirin9030 | KirinX90 | 备注 |
|------|:--:|:---------:|:---------:|------|
| **THistogram** | ✅ | ✅ 复用 a5 | ❌ 注释掉 | |
| **TGetScaleAddr** | ✅ | ✅ 复用 a5 | ❌ 未包含 | MX 缩放地址获取 |
| **TPrint** | ✅ | ❌ 未包含 | ❌ 未包含 | 调试打印 |
| **TRandom** | ✅ | ❌ to be evaluated | ❌ to be evaluated | 随机数生成 |
| **MGather** | ✅ | ❌ to be evaluated | ❌ to be evaluated | 矩阵 gather |
| **MScatter** | ✅ | ❌ to be evaluated | ❌ to be evaluated | 矩阵 scatter |

---

## 统计汇总

| 架构 | 支持数 | 不支持/待评估 | 总指令数（a5参考） |
|------|:------:|:------------:|:-----------------:|
| **A5** | 106 | 0 | 106 |
| **Kirin9030** | 93 | 13 | 106 |
| **KirinX90** | 84 | 22 | 106 |

### KirinX90 vs Kirin9030 差距（8个）

KirinX90 不支持但 kirin9030 支持的 8 个指令：
TInsert, TDeQuant, TGetScaleAddr, TImg2col, SetFmatrix, SetImg2colRpt, SetImg2colPadding, THistogram

### 两者共同不支持 vs A5（12个）

KirinX90 和 kirin9030 均不支持的 12 个指令：
TFMod, TFModS, TRem, TRemS, TRowExpandExpdif, TColExpandExpdif, TRandom, MGather, MScatter, TPrefetch, TPush, TPop, TAlloc, TFree, TPrint
（其中 TPrint 两架构均未包含，其余为注释/待评估）
