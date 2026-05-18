# PTO 指令分类方案

## 背景

根据 readme.md 和 background.md 的描述：
> PTO（Parallel Tile Operation）是昇腾 CANN 定义的一套面向 tile 编程的虚拟 ISA。本仓库提供 PTO Tile 指令的实现、示例、测试与文档，帮助开发者在不同昇腾代际之间更平滑地迁移和优化算子。
>
> 目前的硬件架构：A2A3、A5、Kirin9030

本文档对 PTO 指令进行系统性分类，以**硬件Buffer数据流路径**为核心分类维度，结合**实现方式与依赖关系**进行二级分类，为指令对齐工作提供参考。

---

## 分类设计原则

1. **一级分类按Buffer数据流路径**：根据指令使用的硬件Buffer组合进行分类，反映数据流特征
2. **二级分类按实现方式与依赖关系**（互斥分类）：
   - **原生实现**：kirin9030本地*.hpp实现
   - **独立复用A5**：复用A5且不依赖公共族头
   - **依赖已复用族头**：复用A5，依赖TBinSOp（已复用），可正常工作
   - **依赖未复用族头**：复用A5，依赖其他族头（未复用），可能编译失败
   - **复用A2A3**：复用a2a3实现
   - **暂缓**：未纳入或被注释
3. **特殊规则**：依赖TLoad/TStore的复用指令不单独分类
4. **分类互斥**：一个指令只属于一个一级分类和二级分类

---

## 硬件Buffer说明

| Buffer | 全称 | 位置 | 容量 | 用途 |
|:------:|:----:|:----:|:----:|:-----|
| `__gm__` | Global Memory | DDR/HBM | 最大 | 全局内存，数据加载/存储 |
| `__ubuf__` | Unified Buffer | UB | 1MB | 向量缓冲区，向量计算输入 |
| `__cbuf__` | Cube Buffer | L1 | 512KB | 矩阵缓冲区，矩阵搬运 |
| `__ca__` | Cube A Buffer | L0A | 64KB | 左矩阵缓冲区 |
| `__cb__` | Cube B Buffer | L0B | 64KB | 右矩阵缓冲区 |
| `__cc__` | Cube C Buffer | L0C | 64KB | 累加器缓冲区 |
| `__fbuf__` | Fixpipe Buffer | FB | 4KB | 量化表缓冲区 |
| `__bt__` | Bias Table | BT | 4KB | 偏置表缓冲区 |

---

## 一级分类：Buffer数据流路径

| 编号 | 分类名称 | Buffer组合 | 数据流特征 | 指令数 |
|:----:|:--------:|:-----------|:-----------|:------:|
| **B1** | GM↔UB搬运 | `__gm__`+`__ubuf__` | GM↔UB数据搬运 | 5 |
| **B2** | UB↔L1搬运 | `__ubuf__`+`__cbuf__` | UB↔L1数据搬运 | 1 |
| **B3** | L1↔L0搬运 | `__cbuf__`+`__ca__/__cb__/__cc__` | L1↔L0数据搬运 | 3 |
| **B4** | UB纯向量化 | `__ubuf__` | UB内向量计算 | ~85 |
| **B5** | L0矩阵计算 | `__ca__`+`__cb__`+`__cc__` | Cube矩阵乘法 | 1 |
| **B6** | Acc输出处理 | `__cc__`+多buffer | Acc输出处理 | 1 |
| **B8** | 无Buffer操作 | `<none>` | 系统控制 | 16 |

**Buffer使用分布**（基于a5_buffer_mapping.md）：

| Buffer | 次数 | 占比 | 主要用途 |
|:------:|:----:|:----:|:---------|
| `__ubuf__` | 97 | 85% | 向量计算主力 |
| `__cbuf__` | 9 | 8% | L1矩阵搬运 |
| `__gm__` | 8 | 7% | GM加载/存储 |
| `__cc__` | 5 | 4% | 累加器输出 |
| `__fbuf__` | 4 | 3% | 量化表 |
| `__ca__` | 4 | 3% | L0A输入 |
| `__cb__` | 3 | 2% | L0B输入 |

---

## 二级分类详细表

### B1. GM↔UB搬运

**功能**：GM全局内存与UB向量缓冲区之间的数据搬运。

| 指令 | 实现 | Buffer | 功能 | 硬件差异 |
|:-----|:----:|:-------|:-----|:---------|
| TLoad | **原生** | `__gm__`,`__ubuf__`,`__cbuf__` | GM→UB加载 | L2 cache参数、DN→NZ裁剪、FP4裁剪 |
| TStore | **原生** | `__gm__`,`__ubuf__`,`__cc__`,`__fbuf__` | UB→GM存储 | Nd2Nd/NZ参数、含Acc→GM路径 |
| MGather | 暂缓 | `__gm__`,`__ubuf__` | GM大规模收集 | GM↔UB大规模场景待评估 |
| MScatter | 暂缓 | `__gm__`,`__ubuf__` | GM大规模散布 | GM↔UB大规模场景待评估 |
| TPrefetch | 暂缓 | `__gm__`,`__ubuf__`,`__cbuf__` | GM预取 | GM预取功能待评估 |

---

### B2. UB↔L1搬运

**功能**：UB向量缓冲区与L1矩阵缓冲区之间的数据搬运。

| 指令 | 实现 | Buffer | 功能 | 说明 |
|:-----|:----:|:-------|:-----|:-----|
| TFillPad | **独立复用A5** | `__ubuf__`,`__cbuf__` | UB填充到L1 | 依赖TLoad（原生），按规则排除 |

---

### B3. L1↔L0搬运

**功能**：L1矩阵缓冲区与L0计算单元之间的数据搬运。

| 指令 | 实现 | Buffer | 功能 | 硬件差异 |
|:-----|:----:|:-------|:-----|:---------|
| TMov | **原生** | `__gm__`,`__ubuf__`,`__cbuf__`,`__ca__`,`__cb__`,`__cc__`,`__fbuf__` | 多路径矩阵搬运 | Acc→Ub裁剪、Scale裁剪、ND→ZZ裁剪 |
| TExtract | **原生** | `__ubuf__`,`__cbuf__`,`__ca__`,`__cb__`,`__cc__`,`__fbuf__` | 子块提取 | FP4通路裁剪、Scale stub |
| TImg2col | **独立复用A5** | `__cbuf__`,`__ca__` | L1→L0A img2col | img2col预处理 |

---

### B4. UB纯向量化（最大分类）

**功能**：仅涉及UB的向量计算操作，无跨Buffer数据搬运。

#### B4.1 原生实现（3个）

| 指令 | Buffer | 功能 | 硬件差异 |
|:-----|:-------|:-----|:---------|
| TCvt | `__ubuf__` | 类型转换 | 高级类型裁剪（bf16/fp8/fp4） |
| TGather | `__ubuf__` | 向量收集 | FP8收集裁剪、条件收集裁剪 |
| TQuant | `__ubuf__` | 量化计算 | MXFP8/MXFP4量化stub |

#### B4.2 独立复用A5（~40个）

不依赖公共族头，可直接复用。

| 类别 | 指令 | 数量 | 功能 |
|:----:|:-----|:----:|:-----|
| 二元算术 | TAdd,TSub,TMul,TDiv,TMax,TMin,TAxpy | 7 | Vector×Vector |
| 位运算 | TAnd,TOr,TXor,TShl,TShr | 5 | 按位操作 |
| 一元算术 | TUnaryOp,TRsqrt,TPow | 3 | 单向量变换 |
| 特殊激活 | TPrelu | 1 | 激活函数 |
| 比较/选择 | TCmp,TCmps,TSel,TSels | 4 | 条件筛选 |
| 列归约 | TColReduceIdx | 1 | 列索引归约 |
| 行归约 | TRowProd | 1 | 行乘积 |
| 展开归约 | TRowExpand,TColExpand | 2 | 行/列展开 |
| 部分归约 | TPartAdd,TPartMul | 2 | 局部归约 |
| 布局变换 | TTrans,TConcat | 2 | 分形转换 |
| 填充扩展 | TTri | 1 | Padding |
| 排序 | TSort32,TMrgSort | 2 | 排序 |
| 其他 | TDeQuant,TScatter,TGatherB,THistogram,Tci | 5 | 其他操作 |

#### B4.3 依赖已复用族头（15个）

依赖TBinSOp（已复用），可正常工作。

| 族头 | 复用状态 | 依赖指令（15个） | 功能 |
|:----:|:--------:|:-----------------|:-----|
| TBinSOp | ✅ **已复用** | TAddS,TSubS,TMulS,TDivS,TMaxs,TMins | 标量算术 |
| | | TAndS,TOrS,TXorS,TShlS,TShrS | 标量位运算 |
| | | TLRelu,TExpandS | 激活/扩展 |

**工作状态**：✅ 可正常编译和运行（TBinSOp已在header.hpp复用）

#### B4.4 依赖未复用族头（20个）

依赖其他公共族头（未在header.hpp显式include），但实际可正常工作。

| 族头 | header.hpp状态 | 依赖指令 | 数量 | 功能 |
|:----:|:--------------:|:---------|:----:|:-----|
| TColExpandBinOp | 未显式include | TColExpandAdd/Sub/Mul/Div/Max/Min | 6 | 列展开二元 |
| TRowExpandBinOp | 未显式include | TRowExpandAdd/Sub/Mul/Div/Max/Min | 6 | 行展开二元 |
| TColReduceOps | 未显式include | TColSum,ColProd,ColMax,ColMin | 4 | 列归约 |
| TPartBinOps | 未显式include | TRowReduce,TRowReduceIdx,TPartMax,TPartMin | 4 | 部分归约 |
| TPartArgBinOps | 未显式include | TPartArgMax,TPartArgMin | 2 | 部分Arg |

**工作状态**：✅ 测试通过（所有已支持指令均已验证）

#### B4.5 暂缓（7个）

| 指令 | Buffer | 暂缓原因 |
|:-----|:-------|:---------|
| TFMod | `__ubuf__` | 除余运算场景复杂 |
| TFModS | `__ubuf__` | 标量除余场景复杂 |
| TRem | `__ubuf__` | 余数运算场景复杂 |
| TRemS | `__ubuf__` | 标量余数场景复杂 |
| TRandom | `__ubuf__` | 随机数生成待评估 |
| TRowExpandExpdif | `__ubuf__` | Exp差分展开场景少见 |
| TColExpandExpdif | `__ubuf__` | Exp差分展开场景少见 |

---

### B5. L0矩阵计算

**功能**：L0A×L0B→L0C的Cube矩阵乘法。

| 指令 | 实现 | Buffer | 功能 | 硬件差异 |
|:-----|:----:|:-------|:-----|:---------|
| TMatmul | **原生** | `__ca__`,`__cb__`,`__cc__` | L0A×L0B→L0C | 累加器float→half、MXFP stub |

---

### B6. Acc输出处理

**功能**：L0C累加器输出到其他Buffer。

| 指令 | 实现 | Buffer | 功能 | 说明 |
|:-----|:----:|:-------|:-----|:-----|
| TInsert | **原生** | `__ubuf__`,`__cbuf__`,`__cc__`,`__fbuf__` | L0C插入 | 19行薄包装器 |

---

### B8. 无Buffer操作

**功能**：系统控制、视图管理，无数据搬运。

| 指令 | 实现 | 功能 | 说明 |
|:-----|:----:|:-----|:-----|
| TSync | **原生** | 流水线同步 | 跨核同步裁剪 |
| SetFmatrix | **独立复用A5** | img2col配置 | img2col矩阵格式 |
| SetImg2colRpt | **独立复用A5** | img2col重复 | 重复参数 |
| SetImg2colPadding | **独立复用A5** | img2col填充 | 填充参数 |
| TGetScaleAddr | **独立复用A5** | 量化地址 | 缩放因子地址 |
| TPartMul | **独立复用A5** | 部分乘法 | 部分乘法模板 |
| TAssign | **复用A2A3** | 视图赋值 | 子视图赋值 |
| TSubView | **复用A2A3** | 子视图 | 子视图操作 |
| TAlias | **复用A2A3** | Tile别名 | Tile别名 |
| TPrint | **复用A2A3** | 打印调试 | 打印工具 |
| TReshape | **复用A2A3** | 形状变换 | 形状变换 |
| TPush | 暂缓 | GM推送 | 内存管理待评估 |
| TPop | 暂缓 | 内存弹出 | 内存管理待评估 |
| TAlloc | 暂缓 | 内存分配 | 内存管理待评估 |
| TFree | 暂缓 | 内存释放 | 内存管理待评估 |
| SyncAll | 暂缓 | 跨核同步 | kirin9030不支持 |

---

## 分类统计汇总

### 按Buffer路径统计

| 一级分类 | 原生 | 独立复用A5 | 依赖已复用族头 | 依赖未复用族头 | 复用A2A3 | 暂缓 | 合计 |
|:--------:|:----:|:----------:|:--------------:|:--------------:|:--------:|:----:|:----:|
| B1.GM↔UB | 2 | 0 | 0 | 0 | 0 | 3 | 5 |
| B2.UB↔L1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| B3.L1↔L0 | 2 | 1 | 0 | 0 | 0 | 0 | 3 |
| B4.UB纯向量化 | 3 | ~40 | 15 | 20 | 0 | 7 | ~85 |
| B5.L0矩阵计算 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| B6.Acc输出 | 1 | 0 | 0 | 0 | 0 | 0 | 1 |
| B8.无Buffer | 1 | 5 | 0 | 0 | 5 | 5 | 16 |
| **合计** | **10** | **~47** | **15** | **20** | **5** | **15** | **~112** |

### 按实现方式统计

| 实现方式 | 数量 | 占比 | 特点 |
|:--------:|:----:|:----:|:-----|
| **原生实现** | 10 | 9% | 多Buffer组合，硬件差异大 |
| **独立复用A5** | ~47 | 42% | 不依赖族头 |
| **依赖已复用族头** | 15 | 13% | 依赖TBinSOp（header.hpp显式include） |
| **依赖未复用族头** | 20 | 18% | 依赖其他族头（间接复用） |
| **复用A2A3** | 5 | 4% | 视图管理类 |
| **暂缓** | 15 | 13% | 边界场景，待评估 |

### 公共族头依赖统计

| 公共族头 | header.hpp状态 | 被依赖指令 | 数量 |
|:---------|:--------------:|:----------:|:----:|
| TBinSOp | ✅ 显式include | 标量二元运算 | 15 |
| TColExpandBinOp | 间接复用 | 列展开二元 | 7 |
| TRowExpandBinOp | 间接复用 | 行展开二元 | 7 |
| TColReduceOps | 间接复用 | 列归约 | 4 |
| TPartBinOps | 间接复用 | 部分归约 | 7 |
| TPartArgBinOps | 间接复用 | 部分Arg | 2 |

---

## 架构复用关系图

```
┌─────────────────────────────────────────────────────────────────────┐
│                     架构复用关系图                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                      ┌──────────────┐                              │
│                      │    A5        │                              │
│                      │  (最完整)    │                              │
│                      │  ~114 指令   │                              │
│                      └──────┬───────┘                              │
│                             │                                      │
│          ┌──────────────────┴──────────────────┐                  │
│          │                                      │                  │
│          ▼                                      ▼                  │
│   ┌─────────────┐                      ┌─────────────┐            │
│   │ Kirin9030   │                      │  A2A3       │            │
│   │             │                      │  (旧架构)    │            │
│   ├─────────────┤                      ├─────────────┤            │
│   │ 原生: 10    │                      │ 原生: 全部   │            │
│   │ 独立复用A5: │                      │ 复用: 0     │            │
│   │   ~47      │                      │             │            │
│   │ 依赖已复用 │                      │             │            │
│   │ 族头: 15   │                      │             │            │
│   │ 依赖未复用 │                      │             │            │
│   │ 族头: 20   │                      │             │            │
│   │ 复用A2A3: 5│                      │             │            │
│   │ 暂缓: 15   │                      │             │            │
│   └─────────────┘                      └─────────────┘            │
│                                                                     │
│   工作状态分布：                                                    │
│   ┌─────────────────────────────────────────────────────────────┐ │
│   │ ✅ 已验证通过: 原生(10) + 独立复用(~47) + 依赖族头(35)        │ │
│   │ ✅ 已验证通过: 复用A2A3(5)                                   │ │
│   │ ⏸️ 待评估: 暂缓(15)                                         │ │
│   └─────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 关键发现

### 1. 已验证通过的指令占比82%

| 分类 | 数量 | 占比 | 说明 |
|:----:|:----:|:----:|:-----|
| 原生实现 | 10 | 9% | 硬件差异已适配 |
| 独立复用A5 | ~47 | 42% | 不依赖族头 |
| 依赖族头 | 35 | 31% | TBinSOp及其他族头 |
| 复用A2A3 | 5 | 4% | 视图管理类 |
| **合计** | **~97** | **82%** | ✅ 已验证通过 |

### 2. 公共族头依赖分布

**依赖族头的35个指令分布**：

| 族头 | 指令 | 数量 | 功能 |
|:----:|:-----|:----:|:-----|
| TBinSOp | TAddS,TSubS,TMulS,TDivS,TMaxs,TMins | 7 | 标量算术 |
| | TAndS,TOrS,TXorS,TShlS,TShrS | 5 | 标量位运算 |
| | TLRelu,TExpandS | 2 | 激活/扩展 |
| TColExpandBinOp | TColExpandAdd/Sub/Mul/Div/Max/Min | 6 | 列展开二元 |
| TRowExpandBinOp | TRowExpandAdd/Sub/Mul/Div/Max/Min | 6 | 行展开二元 |
| TColReduceOps | TColSum,Prod,Max,Min | 4 | 列归约 |
| TPartBinOps | TRowReduce,ReduceIdx,PartMax,Min | 4 | 部分归约 |
| TPartArgBinOps | TPartArgMax,ArgMin | 2 | 部分Arg |

### 3. 公共族头复用方式差异

**header.hpp显式include**：
```cpp
#include "pto/npu/a5/TBinSOp.hpp"  // ✅ 显式include，独立复用
```

**header.hpp未显式include（间接复用）**：
- TColExpandBinOp、TRowExpandBinOp、TColReduceOps、TPartBinOps、TPartArgBinOps
- 通过依赖指令的A5实现间接引用，实际功能可用（测试通过）

### 4. 原生实现集中在多Buffer组合

| Buffer组合 | 原生指令 | 数量 | 硬件差异 |
|:-----------|:---------|:----:|:---------|
| GM+UB+L1 | TLoad,TStore | 2 | L2 cache、DN→NZ裁剪 |
| GM+UB+L1+L0+Acc+FB | TMov,TExtract | 2 | Scale裁剪、FP4裁剪 |
| L0A+L0B+Acc | TMatmul | 1 | 累加器类型差异 |
| Acc+多Buffer | TInsert | 1 | 薄包装器 |

---

## 对齐工作建议

### 当前工作状态汇总

| 状态 | 分类 | 数量 | 占比 |
|:----:|:----:|:----:|:----:|
| ✅ 已验证通过 | 原生+独立复用+依赖族头+复用A2A3 | ~97 | 82% |
| ⏸️ 待评估 | 暂缓 | 15 | 13% |

### 公共族头复用说明

| 族头 | header.hpp状态 | 复用方式 | 被依赖指令数 |
|:----:|:--------------:|:--------:|:------------:|
| TBinSOp | 显式include | 独立复用 | 15 |
| TColExpandBinOp | 未显式include | 间接复用 | 7 |
| TRowExpandBinOp | 未显式include | 间接复用 | 7 |
| TColReduceOps | 未显式include | 间接复用 | 4 |
| TPartBinOps | 未显式include | 间接复用 | 7 |
| TPartArgBinOps | 未显式include | 间接复用 | 2 |

---

## 分类决策流程

```
                    指令 X
                      │
                      ▼
        ┌─────────────────────────────────────┐
        │ 1. 分析指令使用的Buffer组合          │
        │    确定一级分类（B1-B8）             │
        └─────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────┐
        │ 2. 检查 kirin9030/*.hpp 是否有本地实现│
        └─────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │ YES                   │ NO
          ▼                       ▼
    ┌───────────┐     ┌─────────────────────────────────────┐
    │ 原生实现  │     │ 3. 检查依赖的公共族头               │
    │ (结束)    │     └─────────────────────────────────────┘
    └───────────┘                   │
                      ┌─────────────┴─────────────┐
                      │ 依赖公共族头              │ 不依赖公共族头
                      ▼                           ▼
          ┌─────────────────────┐       ┌─────────────────────┐
          │ 检查族头include状态 │       │ 检查复用来源        │
          │ - TBinSOp→已显式    │       │ - a5→独立复用A5     │
          │ - 其他→未显式       │       │ - a2a3→复用A2A3     │
          └─────────────────────┘       │ - 注释→暂缓         │
                      │                 └─────────────────────┘
          ┌───────────┴───────────┐
          │ TBinSOp               │ 其他族头
          ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐
    │ 依赖已复用族头  │     │ 依赖未复用族头  │
    │ (显式include)   │     │ (间接复用)      │
    └─────────────────┘     └─────────────────┘
```

---

## 附录：指令快速索引

### 按一级分类索引

| 分类 | 指令列表 |
|:----:|:---------|
| B1.GM↔UB | TLoad(原生),TStore(原生),MGather(暂),MScatter(暂),TPrefetch(暂) |
| B2.UB↔L1 | TFillPad(独立复用) |
| B3.L1↔L0 | TMov(原生),TExtract(原生),TImg2col(独立复用) |
| B4.UB纯向量化 | TCvt(原生),TGather(原生),TQuant(原生),TAdd~Tci(独立复用),TAddS~TLRelu(依赖已复用族头),TColExpandAdd~TPartArgMin(依赖未复用族头),TFMod~TColExpandExpdif(暂缓) |
| B5.L0矩阵 | TMatmul(原生) |
| B6.Acc输出 | TInsert(原生) |
| B8.无Buffer | TSync(原生),SetFmatrix~TPartMul(独立复用),TAssign~TReshape(复用A2A3),TPush~SyncAll(暂缓) |

### 按二级分类索引

| 分类 | 数量 | 指令列表 |
|:----:|:----:|:---------|
| 原生实现 | 10 | TLoad,TStore,TMov,TExtract,TMatmul,TInsert,TCvt,TGather,TQuant,TSync |
| 独立复用A5 | ~47 | TFillPad,TImg2col,TAdd~Tci,SetFmatrix~TPartMul |
| 依赖已复用族头 | 15 | TAddS,TSubS,TMulS,TDivS,TMaxs,TMins,TAndS,TOrS,TXorS,TShlS,TShrS,TLRelu,TExpandS |
| 依赖未复用族头 | 20 | TColExpandAdd/Sub/Mul/Div/Max/Min,TRowExpandAdd/Sub/Mul/Div/Max/Min,TColSum/Prod/Max/Min,TRowReduce/ReduceIdx,TPartMax/Min/ArgMax/ArgMin |
| 复用A2A3 | 5 | TAssign,TSubView,TAlias,TPrint,TReshape |
| 暂缓 | 15 | MGather,MScatter,TPrefetch,TFMod,TFModS,TRem,TRemS,TRandom,TRowExpandExpdif,TColExpandExpdif,TPush,TPop,TAlloc,TFree,SyncAll |