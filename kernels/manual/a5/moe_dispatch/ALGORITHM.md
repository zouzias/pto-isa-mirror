# MoE Dispatch 算子算法详解

本文档对 PTO-ISA MoE Dispatch 算子的完整算法逻辑进行图文解析，力求通俗易懂。

---

## 1. 背景：MoE 是什么？

在大语言模型中，**MoE（Mixture of Experts）** 结构将每个 token 路由到不同的"专家"进行计算。
当模型跨多卡部署时，每张卡只负责一部分专家，因此需要 **跨卡通信** 把 token 发送到正确的卡上。

```
         Token 路由（每个 token 选择一个或多个专家）
         ┌──────────────────────────────────────────┐
         │ Token 0 → Expert 0 (Rank 0)             │
         │ Token 1 → Expert 1 (Rank 1)             │
         │ Token 2 → Expert 0 (Rank 0)             │
         │ Token 3 → Expert 1 (Rank 1)             │
         └──────────────────────────────────────────┘
```

**Dispatch** 就是将 token 从源卡"取"到目标卡的过程（本算子实现的是"拉取"模式）。

---

## 2. 整体架构：三条 Kernel 路径

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MoE Dispatch Kernel                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  mode = direct   │  │  mode = viagm    │  │  mode = sync     │  │
│  ├──────────────────┤  ├──────────────────┤  ├──────────────────┤  │
│  │ MoeDispatchDirect│  │ MoeDispatchViaGM │  │MoeDispatchWithSync│  │
│  │                  │  │                  │  │                  │  │
│  │ 远端GM → UB →   │  │ 远端GM → 本地GM  │  │ Phase A: TPE交换 │  │
│  │ 本地GM           │  │ → UB → 本地GM    │  │ Phase B: 计算路由│  │
│  │                  │  │                  │  │ Phase C: Direct  │  │
│  │ (2步快速路径)    │  │ (4步兼容路径)    │  │ (自包含完整路径) │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

- **Direct**：最快，直接从远端共享内存读取
- **ViaGM**：先 TGET 到本地 GM，再拆分（兼容 MegaMoE 方式）
- **WithSync**：先在设备侧自行计算路由表，再调用 Direct 完成数据搬运

---

## 3. 数据格式：远端共享内存中的 Token 布局

每张卡将自己的 token 数据存放在共享内存（shmem）中，供其他卡来"拉取"。

```
每行数据格式（远端 shmem 中的一个 token）：

┌────────────── hiddenSize (128 bytes) ──────────────┬─── 32 bytes ───┐
│                                                    │                │
│     int8_t token[128]（量化后的 token 数据）        │  float scale   │
│                                                    │  (+ padding)   │
└────────────────────────────────────────────────────┴────────────────┘
                    总计 160 bytes/行 (copyInNum)

输出拆分为两个独立的紧凑数组：

输出 gmA:                              输出 gmPerTokenScale:
┌────────────── 128 bytes ──────┐      ┌──── 32 bytes ────┐
│       token 数据 (int8)       │      │   scale (float)   │
├───────────────────────────────┤      ├───────────────────┤
│       token 数据 (int8)       │      │   scale (float)   │
├───────────────────────────────┤      ├───────────────────┤
│          ...                  │      │       ...         │
└───────────────────────────────┘      └───────────────────┘
```

---

## 4. Direct 路径算法详解

### 4.1 核心思想

每个 AIV 核负责从一个或多个远端卡拉取 token。使用 **ping-pong 流水线** 重叠数据加载和存储，隐藏延迟。

### 4.2 算法流程

```
对于每个本地专家 (groupIdx = 0, 1, ...):
    对于每个远端卡 (dstEpIdx，按核跨步分配):
        ① 计算该远端卡发给当前专家多少行 token
        ② 计算远端源地址（在对方 shmem 中的位置）
        ③ 计算本地目标地址（输出 gmA/gmPerTokenScale 中的行号）
        ④ 分批拉取（每批 MOVE_NUM 行），使用 ping-pong
```

### 4.3 Ping-Pong 流水线图示

Ping-Pong 的核心是：**加载第 N+1 批数据的同时，存储第 N 批数据**。

```
时间 ─────────────────────────────────────────────────────────────────▶

MTE2 (Load)  │ Load Batch0 │             │ Load Batch2 │             │
引擎         │ → Ping UB   │             │ → Ping UB   │             │
             │             │ Load Batch1 │             │ Load Batch3 │
             │             │ → Pong UB   │             │ → Pong UB   │
             ├─────────────┼─────────────┼─────────────┼─────────────┤
MTE3 (Store) │             │ Store Bat0  │ Store Bat1  │ Store Bat2  │
引擎         │             │ Ping → GM   │ Pong → GM   │ Ping → GM   │
             │             │ (token)     │ (token)     │ (token)     │
             │             │ (scale)     │ (scale)     │ (scale)     │
             └─────────────┴─────────────┴─────────────┴─────────────┘

UB 空间布局:
  ┌─────────────────────────────┬─────────────────────────────┐
  │        Ping Buffer          │        Pong Buffer          │
  │  [MOVE_NUM × TILE_COLS]     │  [MOVE_NUM × TILE_COLS]     │
  │                             │                             │
  │  ┌─ token ─┬─ scale ─┐     │  ┌─ token ─┬─ scale ─┐     │
  │  │ 128B    │  32B     │×N  │  │ 128B    │  32B     │×N  │
  │  └─────────┴──────────┘     │  └─────────┴──────────┘     │
  └─────────────────────────────┴─────────────────────────────┘
```

### 4.4 跨 Rank 连续流水

关键优化：ping-pong 状态 **不在 rank 边界重置**，而是连续编号（`globalChunkIdx`）。

```
                          Rank 0 的数据            Rank 2 的数据
                      ┌─────┬─────┬─────┐     ┌─────┬─────┐
globalChunkIdx:       │  0  │  1  │  2  │     │  3  │  4  │
使用缓冲区:            │Ping │Pong │Ping │     │Pong │Ping │
                      └─────┴─────┴─────┘     └─────┴─────┘

        ⬆ 传统做法会在 Rank 边界重置，产生"气泡"
        ✓ 本实现连续编号，无缝衔接
```

### 4.5 拆分操作

UB 中的数据是交织格式 `[token | scale]`。利用 TSTORE 的视图（View）机制，
对同一块 UB 数据创建不同的"窗口"来分别写出 token 和 scale：

```
UB 中一批数据（MOVE_NUM 行）:
┌─────────────────────────────────────────────────────────────┐
│ Row0: [token 128B][scale 32B]                               │
│ Row1: [token 128B][scale 32B]                               │
│ ...                                                         │
│ RowN: [token 128B][scale 32B]                               │
└─────────────────────────────────────────────────────────────┘

tokenView（从偏移 0 开始，宽 128B）     scaleView（从偏移 128 开始，宽 32B）
┌──────────────────────┐                ┌────────────┐
│ token Row0           │                │ scale Row0 │
│ token Row1           │                │ scale Row1 │
│ ...                  │                │ ...        │
└──────────────────────┘                └────────────┘
       │                                       │
       ▼ TSTORE                                ▼ TSTORE
    gmA (紧凑token输出)               gmPerTokenScale (紧凑scale输出)
```

---

## 5. ViaGM 路径算法详解

### 5.1 核心思想

先用 TGET 把远端数据搬到本地 GM（中间缓冲区），再从本地 GM 拆分。适合远端访问延迟高的场景。

### 5.2 两阶段流程

```
阶段1: TGET（远端 GM → 本地 tempGmBuffer）
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  远端 Rank                          本地 Rank                          │
│  ┌──────────────────┐              ┌──────────────────┐               │
│  │ shmem            │    TGET      │ tempGmBuffer     │               │
│  │ [token|scale]    │ ──────────▶  │ [token|scale]    │               │
│  │ [token|scale]    │  (ping-pong  │ [token|scale]    │               │
│  │ [token|scale]    │   staging)   │ [token|scale]    │               │
│  └──────────────────┘              └──────────────────┘               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘

阶段2: TLOAD + TSTORE（本地 tempGmBuffer → UB → gmA + gmPerTokenScale）
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│  tempGmBuffer               UB (ping/pong)          输出              │
│  ┌────────────┐            ┌─────────────┐         ┌──────┐          │
│  │[token|scale]│  TLOAD    │ 交织数据     │ TSTORE  │ gmA  │          │
│  │[token|scale]│ ────────▶ │ (ping/pong) │ ──────▶ │      │          │
│  │    ...      │           │             │         ├──────┤          │
│  └────────────┘            └─────────────┘         │scale │          │
│                                                     └──────┘          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 6. WithSync 路径算法详解（最完整的自包含路径）

WithSync 是完整的"自包含"路径：**不依赖 host 预计算路由表**，在设备侧自行完成
CrossRankSync（路由信息交换和计算）+ Dispatch（数据搬运）的全过程。

### 6.1 总体流程

```
┌──────────────────────────────────────────────────────────────────────┐
│                      MoeDispatchWithSync                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ Phase A: TPE AllGather（交换每卡的 tokenPerExpert）            │  │
│  │   各卡把自己的路由信息写给所有其他卡                            │  │
│  └──────────────────────────────────┬─────────────────────────────┘  │
│                                     ▼                                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ Phase B: 计算路由表（设备侧）                                  │  │
│  │   B.1 去除标志位 → B.2 前缀和 → B.3 起始位置                  │  │
│  └──────────────────────────────────┬─────────────────────────────┘  │
│                                     ▼                                │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ Phase C: 调用 MoeDispatchDirect 完成实际数据搬运               │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.2 Phase A: TPE AllGather（数据+标志位 方式）

**目标**：每张卡需要知道所有卡的 `tokenPerExpert`（每个专家收到多少 token）。

**方法**：每张卡将自己的 TPE 数据加上一个大偏移量（DataAsFlag = 0x800000），然后写到所有远端卡的共享内存中。接收方通过检测数据是否非零来判断数据是否到达。

```
示例：EP=4（4张卡），expertPerRank=2（每卡2个专家）

Rank 0 的本地 TPE = [3, 5]  （含义：Expert0 收到 3 个 token，Expert1 收到 5 个）
Rank 0 发送给所有其他卡的数据 = [3+0x800000, 5+0x800000] = [8388611, 8388613]

各卡的 TPE 交换区（shmem 中）：
                    Rank 0 的 shmem          Rank 1 的 shmem
                    ┌─────────────┐          ┌─────────────┐
来自 Rank 0 的数据: │ 3, 5        │          │ 8388611,613 │ ← Rank0 远程写入
来自 Rank 1 的数据: │ 8388612,610 │ ← 等待  │ 4, 2        │
来自 Rank 2 的数据: │ 8388615,617 │ ← 等待  │ 8388615,617 │ ← 等待
来自 Rank 3 的数据: │ 8388609,614 │ ← 等待  │ 8388609,614 │ ← 等待
                    └─────────────┘          └─────────────┘
```

**TWAIT 等待机制**：

```
┌──────────────────────────────────────────────────────────────┐
│ 初始状态: 交换区全零                                          │
│                                                              │
│ Rank 0 写入: [3+FLAG, 5+FLAG]   ← 值一定非零（因为加了FLAG）  │
│                                                              │
│ 接收方轮询: signalAddr → 等待 != 0                            │
│                                                              │
│ 当 *signalAddr != 0 → 数据到达！                              │
└──────────────────────────────────────────────────────────────┘
```

**Phase A 代码流程**（以单核简化视角）：

```
① TLOAD 本地 TPE → UB
② TADDS 加上 DataAsFlag 偏移 (0x800000)
③ 对每个远端 Rank i:
       TSTORE UB → Rank i 的 shmem[myRank 行]
④ 对每个远端 Rank i:
       TWAIT 检查 localShmem[Rank i 行] 直到非零
```

### 6.3 Phase B: 计算路由表

Phase B 的目标是从收到的 TPE 数据，计算出三个路由表。先看这三个表是什么：

```
输入: tokenPerExpert (TPE) — 每卡每专家的 token 数量

         Expert0  Expert1
Rank 0:  [  3  ,    5  ]
Rank 1:  [  4  ,    2  ]
Rank 2:  [  7  ,    9  ]
Rank 3:  [  1  ,    6  ]

输出1: cumsumMM — 按行前缀和（累计到每行为止的 token 总数）

         Expert0  Expert1
Row 0:   [  3  ,    5  ]    ← 就是 Rank0 本身
Row 1:   [  7  ,    7  ]    ← 3+4=7, 5+2=7
Row 2:   [ 14  ,   16  ]    ← 7+7=14, 7+9=16
Row 3:   [ 15  ,   22  ]    ← 14+1=15, 16+6=22

用途: cumsumMM[i][j] 告诉你 "Rank 0 到 Rank i 的所有卡总共发了多少 token 给 Expert j"
     这用来确定每个 token 在输出数组中的起始位置。

输出2: preSumBeforeRank — "我的每个专家，该去对方卡的哪个位置拉取数据"

假设当前是 Rank 0，看 Rank 1 的数据:
  Rank 1 总共发出 token: Expert0(4个)给Rank0, Expert1(2个)给Rank0, ...
  preSumBeforeRank[Rank1][Expert0] = Rank1发给(Rank<0的所有卡)和(Rank0,Expert<0)的总和
                                   = 0  (没有比 Rank0 序号更小的目标卡)

用途: 这是"在对方 shmem 中从第几行开始读"的起始偏移。
```

#### Phase B.1: 去除 DataAsFlag 偏移（向量化）

```
对每行 TPE 数据:
  ┌─────────────────────────────────────────────────────────┐
  │ TLOAD: shmem TPE 行 → UB                               │
  │ TADDS: 减去 0x800000（恢复原始值）                        │
  │ TSTORE: UB → workspace TPE                              │
  └─────────────────────────────────────────────────────────┘

  [8388611, 8388613]   ─── TADDS(-0x800000) ───▶  [3, 5]
```

#### Phase B.2: 计算 cumsumMM（向量化前缀和）

使用 **accumulator 累加器模式**：逐行加载 TPE，用 TADD 指令做向量加法，然后存储。

```
迭代过程（EP=4, expertPerRank=2）:

步骤1: i=0
  TLOAD TPE[0] = [3, 5] → accumTile
  TSTORE accumTile → cumsumMM[0] = [3, 5]

步骤2: i=1
  TLOAD TPE[1] = [4, 2] → tmpTile
  TADD accumTile = accumTile + tmpTile = [3+4, 5+2] = [7, 7]
  TSTORE accumTile → cumsumMM[1] = [7, 7]

步骤3: i=2
  TLOAD TPE[2] = [7, 9] → tmpTile
  TADD accumTile = [7+7, 7+9] = [14, 16]
  TSTORE accumTile → cumsumMM[2] = [14, 16]

步骤4: i=3
  TLOAD TPE[3] = [1, 6] → tmpTile
  TADD accumTile = [14+1, 16+6] = [15, 22]
  TSTORE accumTile → cumsumMM[3] = [15, 22]

UB 空间布局:
  ┌──────────────────────────────────────────────────────────┐
  │  accumTile [paddedExpNum 宽度]   │  tmpTile [同宽]        │
  │  用于累加                         │  临时加载              │
  └──────────────────────────────────────────────────────────┘
```

**流水线时序**（TLOAD → TADD → TSTORE 之间的同步）：

```
时间 ──────────────────────────────────────────────────────▶

MTE2 (Load): │ Load row[i]    │                │ Load row[i+1] │
             └────────────────┘                └───────────────┘
                    │ set_flag(MTE2→V)                │
                    ▼                                 │
PIPE_V (计算):      │ TADD accum+tmp │                │
                    └────────────────┘                │
                           │ set_flag(V→MTE3)         │
                           ▼                          │
MTE3 (Store):              │ Store accum    │         │
                           └────────────────┘         │
                                  │ set_flag(MTE3→MTE2)
                                  ▼
                           （下一轮 TLOAD 开始）
```

#### Phase B.3: 计算 preSumBeforeRank（标量循环）

这一步计算"对于每个远端卡 srcRank，它发给本卡每个专家的数据在其 shmem 中的偏移"。

```
对于每个 srcRank:
    offset = 0
    按目标卡×专家的顺序遍历:
        if (目标卡 == myRank):
            preSumBeforeRank[srcRank][expert] = offset   ← 记录偏移
        offset += TPE[srcRank][目标卡 * expertPerRank + expert]

图示（myRank=0, srcRank=1, EP=2, expertPerRank=2）:

  srcRank=1 的发送顺序:
    → Rank0 Expert0: 4个token (offset=0→记录, offset+=4)
    → Rank0 Expert1: 2个token (offset=4→记录, offset+=2)
    → Rank1 Expert0: ...
    → Rank1 Expert1: ...

  结果: preSumBeforeRank[1][0] = 0    (从 shmem 第 0 行开始读)
        preSumBeforeRank[1][1] = 4    (从 shmem 第 4 行开始读)
```

### 6.4 Phase C: 实际数据搬运

Phase B 计算完路由表后，执行 SYNCALL 确保所有核可见，然后直接调用 `MoeDispatchDirect`。

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  SYNCALL (所有核同步，确保路由表对所有核可见)                      │
│                         │                                       │
│                         ▼                                       │
│  MoeDispatchDirect(cumsumMM, tokenPerExpert, preSumBeforeRank)  │
│     │                                                           │
│     ├─ 根据 cumsumMM 确定输出行号                                │
│     ├─ 根据 preSumBeforeRank 确定远端源地址                      │
│     └─ 使用 ping-pong 流水线搬运数据                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 多核并行策略

所有路径都使用 **跨步分配（strided assignment）** 策略将工作分配给多核：

```
假设 EP=8, coreNum=2:

Core 0 处理: Rank 0, 2, 4, 6
Core 1 处理: Rank 1, 3, 5, 7

┌───────────────────────────────────────────────────┐
│  Core 0:  │ R0 │    │ R2 │    │ R4 │    │ R6 │   │
│  Core 1:  │    │ R1 │    │ R3 │    │ R5 │    │R7 │
└───────────────────────────────────────────────────┘
```

WithSync 路径中的 Phase B 仅由 Core 0 执行（数据量小，避免多核竞争）：

```
Phase A: 所有核并行写入 + 等待
Phase B: 仅 Core 0 计算路由表，其他核等在 SYNCALL
Phase C: 所有核并行搬运 token
```

---

## 8. 路由寻址：如何找到远端数据？

### 8.1 "在对方 shmem 的哪里读？"

```
源地址计算:

  otherRankBase = HcclRemotePtr(shmem, dstRank)    ← 获取对方 shmem 基地址
  rowSrc = preSumBeforeRank[dstRank][groupIdx]     ← 第几行开始
  remoteSrcPtr = otherRankBase + offsetA + rowSrc * copyInNum

图示:
  Rank 1 的 shmem:
  ┌───────────────────────┐  ← otherRankBase + offsetA
  │ [token|scale] row 0   │
  │ [token|scale] row 1   │
  │ [token|scale] row 2   │  ← rowSrc=2 → 从这里开始读
  │ [token|scale] row 3   │
  │ [token|scale] row 4   │  ← 读 rows 行
  │ ...                   │
  └───────────────────────┘
```

### 8.2 "写到本地输出的哪里？"

```
目标地址计算:

  rowStart = cumsumMM[dstEpIdx-1][groupIdx] + prevGroupSum
  gmA目标   = gmA + rowStart * HIDDEN_SIZE
  scale目标 = gmPerTokenScale + rowStart * UB_ALIGN

图示:
  gmA 输出数组:
  ┌────────────────────────────┐
  │ (之前的 expert 的数据)      │ ← rows 0..prevGroupSum-1
  │ Rank0 发来的 token         │ ← rowStart for Rank0
  │ Rank1 发来的 token         │ ← rowStart for Rank1 = cumsumMM[0][g]
  │ Rank2 发来的 token         │ ← rowStart for Rank2 = cumsumMM[1][g]
  │ ...                        │
  │ (下一个 expert 的数据)      │ ← prevGroupSum += currentM
  └────────────────────────────┘
```

---

## 9. Workspace 内存布局

WithSync 路径使用一块 workspace 存放计算出的路由表：

```
workspace 布局 (int32_t 数组):

偏移 0                          EP*paddedExpNum          +EP*expertPerRank
│                                     │                        │
▼                                     ▼                        ▼
┌─────────────────────────────┬──────────────────┬──────────────────────────┐
│      cumsumMM               │ preSumBeforeRank │    tokenPerExpert        │
│  [EP 行 × paddedExpNum 列]   │ [EP × expPerRank] │ [EP 行 × paddedExpNum]  │
│                             │                  │                          │
│  向量化操作需 32B 对齐       │  标量存储         │  向量化操作需 32B 对齐    │
└─────────────────────────────┴──────────────────┴──────────────────────────┘

paddedExpNum = ((EP × expertPerRank) + 7) & ~7  （向上对齐到 8 的倍数 = 32 bytes）
```

---

## 10. 关键技术点总结

| 技术点 | 说明 |
|--------|------|
| DataAsFlag | 数据本身作为"到达标志"，避免额外信号通道 |
| Ping-Pong 流水 | 双缓冲重叠 Load 和 Store，隐藏延迟 |
| 向量化前缀和 | TADD 指令一次处理一整行，避免逐元素循环 |
| 32B 对齐 padding | DMA 引擎要求传输地址和大小 32 字节对齐 |
| 软件 SYNCALL | 通过 GM 轮询实现跨核同步（绕过硬件 FFTS 限制） |
| 跨 Rank 连续流水 | ping-pong 计数器不在 rank 边界重置 |
| UB View 拆分 | 对同一 UB 区域创建不同窗口，一次加载拆分写出 |
| 跨步多核分配 | 各核处理不相邻的 rank，避免冲突 |

---

## 11. 与 MegaMoE 的对应关系

```
MegaMoE (AscendC + Catlass)              PTO-ISA MoeDispatch
═══════════════════════════              ════════════════════
CrossRankSync...V2()                     Phase A + Phase B
  ├─ DataCopy + DataAsFlag              ├─ TSTORE + TADDS
  ├─ TWAIT polling                      ├─ TWAIT (相同)
  ├─ GetCumsumForMMAIV()               ├─ Phase B.2 (TADD prefix sum)
  └─ GetSumPreRank()                    └─ Phase B.3 (scalar loop)

DispatchAndCombine()                     MoeDispatchDirect / ViaGM
  └─ DispatchCopyPerToken()             └─ TLOAD + TSTORE (ping-pong)

主要差异:
  MegaMoE: AIC+AIV 双核、Catlass MatMul 融合、AscendC 接口
  PTO-ISA: 纯 AIV 核、PTO 指令集、无融合（独立算子）
```

---

## 12. 端到端执行示例

以 `EP=2, expertPerRank=1, hiddenSize=128, mode=sync` 为例：

```
═══════════════ Rank 0 ═══════════════    ═══════════════ Rank 1 ═══════════════

Phase A: 我有 3 个 token 给 Expert 0       Phase A: 我有 5 个 token 给 Expert 0
  TLOAD localTPE=[3]                         TLOAD localTPE=[5]
  TADDS → [3+FLAG]=[8388611]                 TADDS → [5+FLAG]=[8388613]
  TSTORE → Rank1.shmem[row0]                 TSTORE → Rank0.shmem[row1]
  TWAIT Rank1 数据到达                        TWAIT Rank0 数据到达

Phase B (Core 0 only):                     Phase B (Core 0 only):
  B.1: 去除 FLAG                             B.1: 去除 FLAG
    TPE = [[3], [5]]                           TPE = [[3], [5]]
  B.2: cumsumMM                              B.2: cumsumMM
    row0 = [3], row1 = [3+5] = [8]             row0 = [3], row1 = [8]
  B.3: preSumBeforeRank                      B.3: preSumBeforeRank
    Rank0 → offset for Rank0 = 0               Rank0 → offset for Rank1 = 3
    Rank1 → offset for Rank0 = 0               Rank1 → offset for Rank1 = 5

Phase C: Direct Dispatch                   Phase C: Direct Dispatch
  从 Rank0 shmem 拉 0 行 (自己给自己=0？)    从 Rank0 shmem 拉取 row[3..4]
  从 Rank1 shmem 拉取 row[0..4]              从 Rank1 shmem 拉取 row[5..9]
  写到 gmA[0..7]                             写到 gmA[0..7]
```

---

*文档生成时间: 2026-05-27*
*对应代码: `code/pto-isa-main/kernels/manual/{a2a3,a5}/moe_dispatch/`*
