# A5 架构各指令硬件缓冲区速查

> 基于 `include/pto/npu/a5/` 全量源码分析。

---

## 硬件缓冲区一览

| 缩写 | 硬件单元 | 属性 | 容量 | 角色 |
|------|---------|------|:----:|------|
| **GM** | 全局内存 (HBM/DDR) | `__gm__` | 大 | 主存，host 可见 |
| **UB** | 统一缓冲区 | `__ubuf__` | >=256KB | 向量计算工作区 |
| **L1** | Cube 缓冲区 | `__cbuf__` | <=512KB | Cube 数据中转站 |
| **L0A** | Cube 左矩阵输入 | `__ca__` | 小 | A 矩阵 |
| **L0B** | Cube 右矩阵输入 | `__cb__` | 小 | B 矩阵 |
| **L0C** | Cube 累加器 | `__cc__` | 小 | C 矩阵（累加/结果） |
| **Fixpipe (Fb)** | 定点管道缓冲区 | `__fbuf__` | 4KB | 量化/反量化参数表 |
| **Bias (BT)** | 偏置表 | -- | 4KB | 偏置加法专用表 |
| **C2** | 备用累加器 | -- | -- | `cmatrixSource=1` 时替代 L0C |

---

## 各指令缓冲区使用

### GM ↔ UB / L1（数据搬运）

| 指令 | 方向 | 读 | 写 |
|------|------|:--:|:--:|
| **TLOAD** (Vec) | GM -> UB | GM | UB |
| **TLOAD** (Cube) | GM -> L1 | GM | L1 |
| **TSTORE** (Vec) | UB -> GM | UB | GM |
| **TSTORE** (Acc) | L0C -> GM | L0C | GM |
| **TStoreAccFp** | L0C+FB -> GM | L0C, Fixpipe | GM |
| **MGather** | GM -> UB | GM | UB |
| **MScatter** | UB -> GM | UB | GM |
| **TPrefetch** | GM -> L1（预取） | GM | L1 |

### L1 -> L0A / L0B（Cube 输入准备）

| 指令 | 读 | 写 |
|------|:--:|:--:|
| **TExtract** ToA | L1 | L0A |
| **TExtract** ToB | L1 | L0B |
| **TMovToLeft** | L1 | L0A |
| **TMovToRight** | L1 | L0B |
| **TImg2col** | L1 | L0A |

### L0C -> L1 / UB（累加器读出）

| 指令 | 读 | 写 |
|------|:--:|:--:|
| **TMovCcToCb** | L0C | L1 |
| **TMovCcToUb** | L0C | UB |
| **TExtractAccToMat** | L0C | L1 |
| **TExtractAccToVec** | L0C | UB |
| **TInsertAccToMat** | L0C | L1 |
| **TInsertAccToVec** | L0C | UB |

### L1 -> Bias / Fixpipe（参数表加载）

| 指令 | 读 | 写 |
|------|:--:|:--:|
| **TMovToBt** | L1 | Bias |
| **TMovToFb** | L1 | Fixpipe |

### L0A x L0B -> L0C（Cube 矩阵乘）

| 指令 | 读 | 写 | 备注 |
|------|:--:|:--:|------|
| **TMatmul** | L0A, L0B | L0C | A x B -> C |
| **TMatmulBias** | L0A, L0B, Bias | L0C | A x B + bias -> C |
| **TMatmulMx** | L0A(MX), L0B(MX) | L0C | MXFP8/FP4 |
| **TMatmulMxBias** | L0A(MX), L0B(MX), Bias | L0C | MX + bias |

### UB -> UB 纯向量指令

以下指令**仅在 UB 内**读写，不涉及其他缓冲区：

**代数：** TAdd, TSub, TMul, TDiv, TMax, TMin, TAxpy, TPrelu, TLRelu
**标量：** TAddS, TSubS, TMulS, TDivS, TMaxS, TMins
**位：** TAnd, TOr, TXor, TShl, TShr, TAndS, TOrS, TXorS, TShlS, TShrS
**比较/选择：** TCmp, TCmps, TSel, TSels
**一元：** TAbs, TExp, TLog, TSqrt, TRsqrt, TRelu, TNeg
**行/列广播：** TRowExpand 系(7), TColExpand 系(7)
**行/列规约：** TRowSum/Max/Min/Prod/ArgMax/ArgMin(6), TColSum/Prod/Max/Min/ReduceIdx(5)
**部分运算：** TPartAdd/Mul/Max/Min/ArgMax/ArgMin(6)
**变换：** TConcat, TTrans, TFillPad, TExpandS, TTri, TAssign
**视图：** TSubView, TReshape
**量化：** TQuant（UB only）, TDeQuant（读 Fixpipe）
**类型转换：** TCvt
**收集/散列：** TGather, TGatherB, TScatter
**排序：** TSort32, TMrgSort
**填充/索引：** Tci, THistogram, TGetScaleAddr
**其他：** TPow, TRandom, TBinSOp
**ND->ZZ/NZ 搬移：** TMovNdTo2Zz, TMovToVecNd2Nz

### 其他

| 指令 | 涉及缓冲区 | 说明 |
|------|-----------|------|
| **TSync** | 无 | pipe_barrier / set_flag，纯同步控制 |
| **SyncAll** | L1（中转） | 多核同步，L1 做 workspace |
| **TPush / TPop** | UB / L1 | FIFO 流水线操作 |
| **TAlloc / TFree** | 无 | FIFO slot 管理，无数据移动 |

---

## 总览矩阵

| 指令 | UB | L1 | L0A | L0B | L0C | Fixpipe | Bias | GM |
|------|:--:|:--:|:---:|:---:|:---:|:-------:|:----:|:--:|
| **TLOAD**(Vec) | W | -- | -- | -- | -- | -- | -- | R |
| **TLOAD**(Cube) | -- | W | -- | -- | -- | -- | -- | R |
| **TSTORE**(Vec) | R | -- | -- | -- | -- | -- | -- | W |
| **TSTORE**(Acc) | -- | -- | -- | -- | R | -- | -- | W |
| **TMATMUL** | -- | -- | R | R | R/W | -- | -- | -- |
| **TMATMULBias** | -- | -- | R | R | R/W | -- | R | -- |
| **TMATMULMx** | -- | -- | R | R | R/W | -- | -- | -- |
| **TMovCcToCb** | -- | W | -- | -- | R | -- | -- | -- |
| **TMovCcToUb** | W | -- | -- | -- | R | -- | -- | -- |
| **TMovToBt** | -- | R | -- | -- | -- | -- | W | -- |
| **TMovToFb** | -- | R | -- | -- | -- | W | -- | -- |
| **TMovToLeft** | -- | R | W | -- | -- | -- | -- | -- |
| **TMovToRight** | -- | R | -- | W | -- | -- | -- | -- |
| **TExtract->L0A** | -- | R | W | -- | -- | -- | -- | -- |
| **TExtract->L0B** | -- | R | -- | W | -- | -- | -- | -- |
| **TExtract Acc->Mat** | W | -- | -- | -- | R | -- | -- | -- |
| **TExtract Acc->Vec** | W | -- | -- | -- | R | -- | -- | -- |
| **TInsert Acc->Mat** | W | -- | -- | -- | R | -- | -- | -- |
| **TInsert Acc->Vec** | W | -- | -- | -- | R | R(FP) | -- | -- |
| **TImg2col** | -- | R | W | (W) | -- | -- | -- | -- |
| **TQuant** | R/W | -- | -- | -- | -- | -- | -- | -- |
| **TCvt** | R/W | -- | -- | -- | -- | -- | -- | -- |
| **TGather/Scatter** | R/W | -- | -- | -- | -- | -- | -- | -- |
| **UB 向量指令** | R/W | -- | -- | -- | -- | -- | -- | -- |
| **TSync** | -- | -- | -- | -- | -- | -- | -- | -- |
| **TPush/Pop** | R/W | R/W | -- | -- | -- | -- | -- | -- |
| **MGather/MScatter** | R/W | -- | -- | -- | -- | -- | -- | R/W |

> R=读, W=写, R/W=读写, --=不涉及

---

## 数据流全景

```
GM ---TLOAD---> UB ---向量指令---> UB
 |                |                  |
 |                +--TMovToLeft----> L0A ---+
 |                +--TMovToRight---> L0B ---+
 |                |                  |      TMATMUL
 |                +--TExtractAccToVec<-- L0C<-+
 |                                      |
 |    TSTORE(Vec)<---------UB <---------+
 |                          |
 +---TLOAD---> L1 ---TExtract->L0A ---+
                                       |  TMATMUL
                  L0C <---------------+
                    |
                    +--TSTORE(Acc)----> GM
                    +--TMovCcToCb----> L1
                    +--TMovCcToUb----> UB
                    +--TMovToBt------> Bias Table
                    +--TMovToFb------> Fixpipe Buffer
```
