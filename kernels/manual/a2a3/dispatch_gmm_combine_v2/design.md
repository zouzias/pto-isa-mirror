# dispatch_combine_moe_v2 设计文档

> Last refreshed: 2026-05-24 (重定位为 dispatch_gmm_combine 的 PTO 风味化翻译)
>
> 本文件是当前有效设计基线。

---

## 1. 项目定位

`dispatch_combine_moe_v2` = `dispatch_gmm_combine` 的 PTO 风味化翻译。

对标原文：`/home/ntlab/zy/code/zhangyuan/shmem_zy/examples/dispatch_gmm_combine/`

**翻译原则：**
- 以 shmem 的全链路数据流、数据类型、tile 策略、overlap 模式为"原文"
- 翻译为 PTO 编程范式（Vec/Cube target 分离、PTO 指令集、two-kernel GM queue pipeline）
- 保持性能同级：int8 Cube 满算力、大 tile、double-buffer、Fixpipe (TSTORE_FP)
- 不照搬 Catlass/AscendC 代码，只翻译语义

---

## 2. shmem 原文：全链路数据流

### 2.1 数据类型链路
```text
fp16 input
  → [Device Routing] per-token dynamic int8 quant → int8 expandedX + float scale1(per-token)
  → [Dispatch] CopyGMToGM: remote peermem → workspace gmA (int8) + gmPerTokenScale1
  → [GMM1 AIC] int8 A × int8 B1 → int32 L0C → Fixpipe(uint64 per-channel scale1) → fp16 gmC
  → [Epilogue1 AIV] fp16 C1 × perTokenScale1 → SwiGLU → int8 gmPermutedToken + float scale2
  → [GMM2 AIC] int8 A × int8 B2 → int32 L0C → Fixpipe(uint64 per-channel scale2) → fp16 gmC2
  → [Epilogue2 AIV] fp16 C2 × perTokenScale2 → fp16/bf16 → 直接写远端 ptrD
  → [Unpermute] expandedRowIdx + topK probs 加权求和 → fp16 output
```

### 2.2 buffer 布局与复用
```text
对称内存 (peermem):
  ptrPeerTokenPerExpert  → int32 [EP × (EP×expertPerRank + 8)]
  ptrA                   → int8 [m×topK, K]    (routing quant 输出，dispatch 源)
  ptrPeerPerTokenScale   → float [m×topK]
  ptrD                   → fp16 [m×topK, K]    (combine 目标，unpermute 源)

workspace (device malloc):
  expandedRowIdx   → int32
  cumsumMM/Send    → int32
  perTokenScale1/2 → float
  ptrC = ptrC2     → fp16 (GMM1 输出 = GMM2 输出，互斥复用)
  ptrA = ptrPermutedToken → int8 (dispatch A = SwiGLU 输出，互斥复用)

Host GM (不进对称内存/workspace):
  B1 → int8 zN [expertPerPe, K, N]
  B2 → int8 zN [expertPerPe, N/2, K]
  scale1 → uint64 per-channel [expertPerPe, N]
  scale2 → uint64 per-channel [expertPerPe, K]
```

### 2.3 硬件 tile 约束（A3 AtlasA2, int8）

| 缓存 | A3 容量 | shmem 使用 |
|---|---|---|
| L1 | 512 KB | L1A(64KB)+L1B(128KB)+L1S(2KB) × 2 stages = 388KB |
| L0A | 64 KB | 16KB × 2 stages = 32KB |
| L0B | 64 KB | 32KB × 2 stages = **64KB (极限)** |
| L0C | 128 KB | **128KB (极限)** |

```text
L1 Tile: 128×256×512 (M,N,K)  — int8, double-buffer
L0 Tile: 128×256×128 (M,N,K)  — int8, double-buffer
Stages: L1=2, L0A=2, L0B=2, L0C=1, preload=1
```

L0B 和 L0C 在 int8 场景下都用到了 A3 硬件极限。

### 2.4 overlap 模式
```text
AIV: [Routing][CountEx][Dispatch grp0][grp1]...[Epilogue1 batch][FLAG_VALUE=2][Combine grp0][grp1]...[Unpermute]
AIC: ----等flag0----[GMM1 grp0][grp1]...------等FLAG_VALUE=2------[GMM2 grp0][grp1]...→flag3→

同步点:
  flag 0             : AIV dispatch 每 group → AIC GMM1 可开始
  flag syncLoopIdx/8+1: AIC GMM1 Finalize → AIV Epilogue1 可开始
  FLAG_VALUE=2       : AIV 全部 Epilogue1 完成 → AIC GMM2 可开始
  flag groupIdx/8+3  : AIC GMM2 Finalize 每 group → AIV Combine 可开始
```

### 2.5 关键设计要点
- **workspace 激进复用**：ptrC/ptrC2 同址，ptrA/ptrPermutedToken 同址
- **Fixpipe per-channel dequant**：int32 L0C → fp16 GM 一步到位，零额外 buffer
- **Epilogue1 在 Dispatch 函数内**：不是独立 kernel，是 AIV 在 GMM1 完成后就地执行
- **Epilogue2 直写远端**：combine 不经过中间 buffer，epilogue 带 scale2 直接 TPUT 远端 ptrD
- **默认配置 (2 rank, expertPerRank=2)**：GMM1→Epilogue1 退化为批量模式（先完成全部 GMM1 再 Epilogue1）

---

## 3. v2 当前实现 vs shmem 原文：逐阶段对比

### 3.1 数据类型链路对比

```text
shmem:  int8 → [int8 Cube MMAD + Fixpipe] → fp16 → [SwiGLU] → int8 → [int8 Cube MMAD + Fixpipe] → fp16 → 远端
v2:     int8 → [Vec dequant → float] → [float Cube MMAD] → float → [SwiGLU] → int8 → [int8 Cube MMAD] → int32 → [Vec dequant → float] → float → TPUT
```

**核心偏离：v2 在 dispatch 后插入了独立 Vec dequant (int8→float)，导致 GMM1 整条变成 float 路径。**

### 3.2 逐阶段 gap

| 阶段 | shmem 原文 | v2 现状 | 需要改什么 |
|---|---|---|---|
| **Routing** | Device 侧 moe_init_routing_quant_v2 | Host 预算 + toy publication | 保持 host 预算；替换 toy publication 为真实 int8+scale1 数据 |
| **Dispatch** | remote peermem → workspace int8 A + scale1 | HCCL window TGET → compute region int8 + scale1 | **保持现状**，dispatch 本身 OK |
| **Dequant** | **无此阶段**（int8 直入 Cube） | Vec dequant int8→float | **删除**此阶段 |
| **GMM1** | int8×int8→int32, Fixpipe→fp16, tile 128×256×512 | float×float→float, 8行小tile, 无Fixpipe | **重写**：int8 MMAD + TSTORE_FP + 大tile + double-buffer |
| **SwiGLU** | fp16 C1 × perTokenScale1 → sigmoid → int8 + scale2 | float gmm1Out → sigmoid → int8 + scale2 | **调整输入**：从 fp16 C1 读（GMM1 Fixpipe 输出）；加上 perTokenScale1 乘回 |
| **GMM2** | int8×int8→int32, Fixpipe→fp16 | int8×int8→int32, 无Fixpipe, Vec dequant bridge | **加 TSTORE_FP**；删除 Vec dequant bridge |
| **Combine** | Epilogue2(×scale2) 直写远端 ptrD fp16 | Vec dequant→localPartial float→TPUT float | **改为 fp16 + 直写远端**（或 TPUT fp16 到 combine region） |
| **Restore** | 多核并行 unpermute | 单 block 串行 | 后续优化 |

### 3.3 需要删除的 buffer/阶段

| 删除项 | 原因 |
|---|---|
| `dequantOutDev` (float) | 不再需要独立 dequant |
| `weight1Dev` (float) | 改为 int8 weight + uint64 channel scale |
| `gmm1OutDev` (float) | 改为 fp16（或直接复用 SwiGLU 输入区） |
| `gmm2AccDev` (int32 全量 GM) | Fixpipe 直接出 fp16，不需要 int32 中间 GM buffer |
| `localPartialDev` (float) | combine 直接从 Fixpipe 输出的 fp16 C2 读取 |
| `dispatchToGmm1Queue` 上的 dequant 生产者 | dispatch 完成直接 publish int8 ready |

### 3.4 需要新增的能力

| 新增项 | 说明 |
|---|---|
| **TSTORE_FP** | PTO 的 Fixpipe 等价：Acc(int32/float) + Scaling(uint64 per-channel) → int8/fp16 GM |
| **int8 weight1 + uint64 channel scale** | 替换 float weight1 |
| **L1/L0 double-buffer** | L1 stages=2, L0A/L0B stages=2 |
| **K-loop preload pipeline** | preload stages=1 |
| **大 tile** | 目标 L1 128×256×512, L0 128×256×128（与 shmem 一致） |

---

## 4. PTO 风味化翻译方案

### 4.1 PTO 等价映射表

| shmem 概念 | PTO 等价 |
|---|---|
| Mix Kernel (AIC+AIV 同 kernel) | Two-kernel: Vec kernel + Cube kernel，GM queue 串接 |
| SHMEM peermem + CopyGMToGM | HCCL window + TGET/TPUT |
| Catlass BlockMmad int8 | PTO `TMATMUL`/`TMATMUL_ACC` int8 |
| Catlass Fixpipe (L0C→GM + per-channel dequant) | PTO **`TSTORE_FP`** (Acc + Scaling → GM) |
| CrossCoreFlag (AIC↔AIV 同 kernel) | GM queue publish/pop (跨 kernel) |
| Catlass L1/L0 double-buffer | PTO manual L1/L0 tile ping-pong |
| workspace 复用 (ptrC=ptrC2, ptrA=ptrPermutedToken) | 同样可以复用，host 分配一份 |

### 4.2 目标数据类型链路（PTO 翻译后）
```text
int8 input (host 预算或 dispatch 从远端拉来)
  → [Dispatch Vec] TGET/TLOAD-TSTORE → compute region int8 A + float perTokenScale1
  → [GMM1 Cube] int8 A × int8 B1 → int32 L0C → TSTORE_FP(uint64 channel scale1) → fp16 gmC
  → [SwiGLU Vec] fp16 C1 × perTokenScale1 → SwiGLU → int8 swigluQ + float scale2
  → [GMM2 Cube] int8 swigluQ × int8 B2 → int32 L0C → TSTORE_FP(uint64 channel scale2) → fp16 gmC2
  → [Combine Vec] fp16 C2 × perTokenScale2 → fp16 → TPUT 远端 combine region
  → [Restore Vec] expandedRowIdx + topK probs 加权求和 → fp16/float output
```

### 4.3 目标 buffer 布局
```text
HCCL window (per rank):
  dispatchRegion   → int8 publication 源（远端读）
  computeRegion    → int8 dispatch 目标 A + float perTokenScale1
  combineRegion    → fp16 combine 目标（restore 源）
  signalRegion     → ready/done signals

Device buffers (aclrtMalloc):
  weight1Q        → int8 [expertPerPe, K, N]     (不再 float)
  scale1Channel   → uint64 [expertPerPe, N]       (per-channel, 新增)
  weight2Q        → int8 [expertPerPe, N/2, K]    (已有)
  scale2Channel   → uint64 [expertPerPe, K]       (per-channel, 新增)
  gmC = gmC2      → fp16 [maxRows, max(N, K)]    (GMM1/GMM2 输出, 互斥复用)
  swigluQ         → int8 [maxRows, N/2]           (已有, 可复用 computeRegion A 区)
  scale2PerToken  → float [maxRows]               (已有)

删除:
  dequantOutDev   → 不再需要
  weight1Dev(float) → 改为 int8
  gmm1OutDev(float) → 改为 fp16 gmC
  gmm2AccDev(int32) → 不再需要（Fixpipe 直出 fp16）
  localPartialDev → 不再需要（从 fp16 gmC2 直接 combine）
```

### 4.4 GMM1 Cube 设计（int8 + TSTORE_FP）

```text
for each queue entry (tile of rows):
  for kTile in [0, K, L1_K=512):
    L1A: TLOAD int8 A[rows, kTile:kTile+512] from computeRegion
    L1B: TLOAD int8 B1[expert, kTile:kTile+512, nBase:nBase+256] from weight1Q
    for kSub in [0, 512, L0_K=128):
      L0A: TLOAD from L1A[., kSub:kSub+128]
      L0B: TLOAD from L1B[kSub:kSub+128, .]
      TMATMUL_ACC acc, L0A, L0B   (int8×int8→int32 累加)
    load channel scale1[expert, nBase:nBase+256] → Scaling tile
    TSTORE_FP(gmC[rows, nBase:nBase+256], acc, scaleTile)  → fp16 直写 GM
```

**L1 double-buffer**：两个 L1 stage 交替加载，当前 tile compute 时预加载下一个 K-tile。
**L0 double-buffer**：两个 L0A/L0B stage 交替，当前 K-sub compute 时预加载下一个。

### 4.5 GMM2 Cube 设计（int8 + TSTORE_FP）

与 GMM1 对称：
```text
A = swigluQ [int8, rows × N/2]
B = weight2Q [int8, expert, N/2 × K]
→ int32 L0C → TSTORE_FP(channel scale2) → fp16 gmC2
```

### 4.6 SwiGLU Vec 设计调整

当前 SwiGLU 逻辑正确（gate×sigmoid(gate)×up → absmax → scale2 → int8），需要调整：
- **输入从 float gmm1Out 改为 fp16 gmC**（GMM1 TSTORE_FP 的 fp16 输出）
- **加上 perTokenScale1 乘回**：`fp16 → fp32 → ×scale1 → SwiGLU → quant`
- 这与 shmem Epilogue1 完全一致

### 4.7 Combine Vec 设计调整

- **输入从 float localPartial 改为 fp16 gmC2**（GMM2 TSTORE_FP 的 fp16 输出）
- **加上 perTokenScale2 乘回**：`fp16 → ×scale2 → fp16/bf16`
- **直接 TPUT fp16 到远端 combine region**（消除 localPartial float 中间 buffer）

### 4.8 pipeline queue 调整

当前 4 条 queue 保持，但 producer/consumer 变化：

```text
dispatchToGmm1Queue:  dispatch 完成 int8 ready → Cube GMM1 消费
                      (不再经过 Vec dequant 生产者)
gmm1ToSwiGluQueue:    Cube GMM1 fp16 C1 ready → Vec SwiGLU 消费
swiGluToGmm2Queue:    Vec SwiGLU int8 swigluQ ready → Cube GMM2 消费
gmm2ToCombineQueue:   Cube GMM2 fp16 C2 ready → Vec combine 消费
```

---

## 5. 与 shmem overlap 的翻译对比

| shmem overlap | PTO 翻译 |
|---|---|
| AIV dispatch grp → flag0 → AIC GMM1 grp | Vec dispatch tile → queue publish → Cube GMM1 tile |
| AIC GMM1 Finalize → flag1 → AIV Epilogue1 | Cube GMM1 tile → queue publish → Vec SwiGLU tile |
| AIV Epilogue1 done → FLAG_VALUE=2 → AIC GMM2 | Vec SwiGLU tile → queue publish → Cube GMM2 tile |
| AIC GMM2 Finalize → flag3 → AIV Combine | Cube GMM2 tile → queue publish → Vec combine tile |

**PTO 版本实际上粒度更细**（tile 级 vs expert-group 级），overlap 潜力更好。
但 shmem 在大 expert 数场景下用 group-flag 也能实现很好的 overlap。

---

## 6. 硬件 tile 约束详解

### 6.1 int8 path（目标）

```text
L1 Tile 128×256×512 (M,N,K), int8, double-buffer:
  L1A = 128×512×1 = 64KB,  L1B = 256×512×1 = 128KB,  L1S = 256×8 = 2KB
  (64+128+2)×2 = 388KB ≤ 512KB  ✓

L0 Tile 128×256×128 (M,N,K), int8, double-buffer:
  L0A = 128×128×1 = 16KB,  ×2 = 32KB ≤ 64KB  ✓
  L0B = 128×256×1 = 32KB,  ×2 = 64KB ≤ 64KB  ✓ (极限)
  L0C = 128×256×4(int32) = 128KB ≤ 128KB  ✓ (极限)
```

### 6.2 TSTORE_FP 约束（A3）

从 PTO ISA 文档：
- 源 dtype: int32 或 float（L0C accumulator）
- 目标 dtype: int8/fp16（GM）
- FpTile: Scaling type, uint64 per-channel scale
- 列约束: 1 ≤ ValidCol ≤ 4095
- ND layout: 1 ≤ ValidRow ≤ 8192

对于 128×256 tile：ValidRow=128, ValidCol=256，满足约束。

---

## 7. 改动优先级

### P0: 性能决定性改动（必须做）

| # | 改动 | 影响 |
|---|---|---|
| P0-1 | **删除独立 dequant 阶段**，dispatch 直接 publish int8 给 Cube | 消除一次 GM 读写 + float 4× 带宽 |
| P0-2 | **GMM1 改 int8 MMAD + TSTORE_FP** | Cube 满算力 + Fixpipe 零额外 buffer |
| P0-3 | **weight1 改 int8 + uint64 channel scale** | 与 int8 MMAD 配合 |
| P0-4 | **大 tile 128×256×512 + L1/L0 double-buffer + preload** | 缓存利用率对齐 shmem |
| P0-5 | **GMM2 加 TSTORE_FP**，删除 Vec dequant bridge 和 gmm2Acc/localPartial buffer | 同上 |

### P1: 语义对齐改动

| # | 改动 |
|---|---|
| P1-1 | SwiGLU 输入改为 fp16 C1 + perTokenScale1 乘回 |
| P1-2 | Combine 输入改为 fp16 C2 + perTokenScale2，直写远端 fp16 |
| P1-3 | 删除 float 中间 buffer（dequantOut, gmm1Out float, gmm2Acc, localPartial） |
| P1-4 | 真实 int8+scale1 publication payload（替换 toy） |
| P1-5 | K/N tail 全覆盖（当前 silent return 问题） |

### P2: 工程完善

| # | 改动 |
|---|---|
| P2-1 | workspace 复用设计（gmC=gmC2, computeRegion A = swigluQ） |
| P2-2 | Restore 多核并行 |
| P2-3 | C2 layout 绑定 combine ptrD |
| P2-4 | overlap 结构复验 |
| P2-5 | 性能对标 shmem baseline |

---

## 7.1 已完成的缺失补齐（2026-05-24）

| # | 缺失项 | 状态 | 实现位置 |
|---|---|---|---|
| G1 | Combine perTokenScale2 乘回 | ✅ 已完成 | `combine_push.hpp`: `ScaleRowsInPlace` (fp16→fp32→×scale→fp16 in-place) |
| G2 | L0 double-buffer (ping/pong) | ✅ 已完成 | `pto_gmm_block_int8.hpp`: L0A/L0B ping/pong + K-loop prefetch overlap |
| G3 | 大 tile M=128 | ⏳ 后续规划 | 需改 queue 消费策略，影响面大，待 profile 后决定 |

**G1 设计要点：**
- `CombineOnlyParams` 新增 `perTokenScale2` (GM 基址) 和 `outputElems` 字段
- `RunCombinePushTaskBody` / `RunCombineRangeTaskBody` 在 TPUT/Copy 前逐行 scale
- 流程：TLOAD fp16 → TCVT fp32 → TMULS(scale) → TCVT fp16 → TSTORE (in-place)

**G2 设计要点：**
- L0A/L0B 各声明 ping/pong 两份 tile，通过 `TASSIGN` 绑定不同 offset
- K-loop 流水：iter N 的 TMATMUL 与 iter N+1 的 TMOV(L1→L0) 重叠
- L1 prefetch 提前 2 个 iteration（iter N 发起 iter N+2 的 GM→L1 加载）
- 同步：PIPE_MTE1→PIPE_M (L0 ready for compute) + PIPE_M→PIPE_MTE2 (L0 released)

**G3 后续规划：**
- 大 M 需要在 Cube kernel 内聚合多个 queue entries（8×N 行合并为单次 MMAD）
- 需引入"batch window"等待策略，可能增加 overlap 延迟
- 建议先 profile G2 效果后再决策最优 M 值

---

## 8. 风险与待确认

| # | 风险/待确认 | 说明 |
|---|---|---|
| R1 | TSTORE_FP 在 A3 int8→fp16 per-channel 路径是否已经过仓内验证 | 需要查找 testcase 或自行验证 |
| R2 | PTO 大 tile (128×256) 是否需要特殊 GlobalTensor 声明方式 | 当前 v2 的 Tile 最大只有 8 行 |
| R3 | L1 double-buffer 在 PTO manual mode 下的实现方式 | PTO 有 TASSIGN 绑定 L1 offset，需要手动 ping-pong |
| R4 | weight1/weight2 从 float 改 int8 后 host 数据生成链的改动量 | 需要改 host golden oracle |
| R5 | shmem baseline 环境依赖（torch_npu/triton） | 可能不具备 |
