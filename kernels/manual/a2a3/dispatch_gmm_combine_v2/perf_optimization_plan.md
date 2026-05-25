# dispatch_combine_moe_v2 性能优化详细方案

> 目标：收回与 shmem baseline 的 ~280x/token 性能差距
> 
> 当前瓶颈排序：tile 利用率(~16x) > queue 开销(~3-5x) > 通算 overlap(~2x) > Vec/Cube 融合(~1.5x)
>
> 硬件平台：A3 (910B), CANN 8.5, PTO Manual 模式

---

## 优化 1：推大 Cube Tile 尺寸（收益 ~16-50x）

### 1.1 目标 tile 参数

| 参数 | 当前值 | 目标值 | 说明 |
|------|--------|--------|------|
| ValidM (MMAD 行数) | 8 | **128** | 与 shmem 一致 |
| ValidN (MMAD 列数) | 16/32 | **256** | N 方向分 tile 循环 |
| BaseK (单次 K 步长) | 16 | **128** | L0 K-sub 粒度 |
| L1_K (L1 tile K 维) | 128 (ValidK) | **512** | L1 预加载粒度 |
| kComputeTileRows | 8 | **128** | queue entry 覆盖行数 |

### 1.2 容量验证 (A3 int8)

```text
L1 (512KB 总量, double-buffer):
  L1A ping: 128 × 512 × 1B = 64KB
  L1A pong: 128 × 512 × 1B = 64KB
  L1B ping: 512 × 256 × 1B = 128KB   ← 注意：B 的 K=512, N=256
  L1B pong: 512 × 256 × 1B = 128KB
  L1 Scale: 256 × 8B = 2KB × 2 = 4KB
  总计: 64+64+128+128+4 = 388KB ≤ 512KB  ✓

L0A (64KB 总量, double-buffer):
  L0A ping: 128 × 128 × 1B = 16KB
  L0A pong: 128 × 128 × 1B = 16KB
  总计: 32KB ≤ 64KB  ✓

L0B (64KB 总量, double-buffer):
  L0B ping: 128 × 256 × 1B = 32KB
  L0B pong: 128 × 256 × 1B = 32KB
  总计: 64KB ≤ 64KB  ✓ (极限)

L0C (128KB 总量, single-buffer):
  L0C: 128 × 256 × 4B(int32) = 128KB ≤ 128KB  ✓ (极限)
```

### 1.3 代码改动清单

#### 文件: `pto_gmm_block_int8.hpp`

**改动 1: 新增大 tile MMAD 函数模板**

```cpp
// 新的大 tile GMM block：M=128, N=256, 外部传入 ValidK 和 BaseK=128
template <uint32_t ValidM,    // = 128
          uint32_t ValidK,    // = 真实K (7168等), 取 min(K, 512) 作为 L1 tile
          uint32_t ValidN,    // = 256
          uint32_t BaseK,     // = 128 (L0 K-sub 粒度)
          uint32_t L1K>       // = 512 (L1 K 粒度)
void RunPtoGmmBlockInt8_Large(
    __gm__ half* dst,
    __gm__ int8_t* lhs,        // [M, K] ND
    __gm__ int8_t* rhs,        // [K, N] NZ/ND
    __gm__ uint64_t* channelScale,
    uint32_t lhsStrideK,       // = K
    uint32_t rhsStrideN,       // = N (或 NZ stride)
    uint32_t dstStrideN)       // = N
```

**核心循环结构 (三层嵌套)**:

```text
for l1Iter in [0, ValidK, L1K):          // L1 tile K-loop
    // L1 double-buffer: ping/pong 加载 A[M, l1Iter..l1Iter+L1K] 和 B[l1Iter..l1Iter+L1K, N]
    L1A_cur ← TLOAD A[., l1Iter:l1Iter+L1K]      // 64KB
    L1B_cur ← TLOAD B[l1Iter:l1Iter+L1K, .]      // 128KB
    // 预加载下一 L1 tile
    L1A_next ← async TLOAD A[., l1Iter+L1K:l1Iter+2*L1K]
    L1B_next ← async TLOAD B[l1Iter+L1K:l1Iter+2*L1K, .]

    for l0Iter in [0, L1K, BaseK):        // L0 K-sub loop (4次: 512/128=4)
        // L0 double-buffer: ping/pong
        L0A_cur ← TMOV L1A[., l0Iter:l0Iter+BaseK]     // 16KB
        L0B_cur ← TMOV L1B[l0Iter:l0Iter+BaseK, .]     // 32KB
        TMATMUL_ACC acc, L0A_cur, L0B_cur               // 128×256×128 int8 MMAD

    // K 维全部累加完后 TSTORE_FP
    TSTORE_FP(dst[., nBase:nBase+N], acc, scaleTile)    // int32 → fp16 + per-channel dequant
```

**改动 2: N-loop 外层**

```cpp
// GMM1: K=7168, N=4096  →  N方向 4096/256 = 16 次 TSTORE_FP
// GMM2: K_in=N/2=2048, N_out=4096(=K_hidden)  →  类似
void RunGmm1Int8Tile_Large(__gm__ int8_t* input,
                           __gm__ int8_t* weight1,
                           __gm__ half* gmm1Out,
                           __gm__ uint64_t* channelScale,
                           uint32_t rows,         // 实际行数 ≤ 128
                           uint32_t hiddenK,
                           uint32_t gmm1N,
                           uint32_t outputStrideN)
{
    // 设置 ValidM = rows (dynamic valid region)
    for (uint32_t nBase = 0; nBase < gmm1N; nBase += 256) {
        uint32_t nTile = min(gmm1N - nBase, 256u);
        RunPtoGmmBlockInt8_Large<128, ..., 256, 128, 512>(
            gmm1Out + nBase,
            input,
            weight1 + nBase,  // B 的 nBase 偏移
            channelScale + nBase,
            hiddenK, gmm1N, outputStrideN);
    }
}
```

#### 文件: `compute_cube_queue_kernel.cpp`

**改动 3: 修改 kComputeTileRows**

```cpp
constexpr uint32_t kComputeTileRows = 128;  // 从 8 改为 128
```

**改动 4: queue 消费函数适配大 tile**

- `TryConsumeDispatchAndProduceGmm1`: 内部调用改为 `RunGmm1Int8Tile_Large`
- `TryConsumeSwiGluAndProduceGmm2`: 内部调用改为 `RunGmm2Int8Tile_Large`
- MakeRowsVisible: rows 变大后可能需要按 cacheline 批量 invalidate

#### 文件: `compute_vec_views.hpp`

**改动 5: 同步 kComputeTileRows = 128**

SwiGLU 的 UB 使用需重新规划:
- 当前 SwiGLU 处理 8 行 × 16 列 tile，UB 占用 ~18KB
- 改为 128 行后，SwiGLU 需分多次迭代（每次仍处理 8-16 行），或增大 UB tile
- 推荐策略: SwiGLU 仍按 16 行子 tile 处理，但外层 tile 是 128 行（N-loop 照常）

### 1.4 关键设计决策

**Q: 如果 M < 128 怎么办（如 M=64）？**

A: 使用 PTO dynamic valid region:
```cpp
L0A lhsTile(actualRows, BaseK);  // actualRows = min(128, remainingRows)
```
TMATMUL 支持 m ∈ [1, 4095]，不需要 pad 到 128。Cube 利用率 = actualRows/128。

**Q: L0B/L0C 在极限容量，是否有余量放 Scaling tile？**

A: TSTORE_FP 的 Scaling tile 在 FBuffer(2KB) 中，不占 L0B/L0C。容量无冲突。

**Q: tail 处理——K 不是 512 或 128 的倍数？**

A: `kLoop = (ValidK + BaseK - 1) / BaseK`，最后一次 BaseK 用 dynamic valid cols 处理尾部。PTO TMATMUL 支持 k ∈ [1, 4095] 的任意值。

---

## 优化 2：减少 Queue Pipeline 开销（收益 ~3-5x）

### 2.1 核心改动

推大 tile 后，queue entry 数量**直接下降 16 倍**:
- 当前: M=64, kComputeTileRows=8 → 8 entries/expert → 16 entries/rank
- 优化后: M=64, kComputeTileRows=128 → 1 entry/expert → 2 entries/rank
- 标准 shape M=128: 1 entry/expert → 2 entries/rank

### 2.2 Queue 批量化设计

#### 方案 A: 单 entry 覆盖整个 expert（推荐）

当 kComputeTileRows ≥ maxTokensPerExpert 时，每个 expert 只产生 1 个 queue entry:

```cpp
// dispatch 侧: 不再逐 8 行发布，而是积累到 expert 完整后发布
PipelineQueueEntry entry;
entry.windowId = expertId;
entry.rowBegin = expertRowBase;
entry.rowEnd = expertRowBase + expertRows;  // 整个 expert 的行数
entry.localExpert = expertId;
// 单次 publish
v2_pipeline_queue::PublishNext(dispatchToGmm1Queue, dispatchCursor, entry);
```

#### 方案 B: M > 128 时仍分 tile，但粒度为 128 行

```cpp
constexpr uint32_t kComputeTileRows = 128;
const uint32_t tileCount = (rowCount + kComputeTileRows - 1) / kComputeTileRows;
// tileCount 在典型 shape 下很小 (1-4 per expert)
```

### 2.3 MakeRowsVisible 优化

当前逐行 invalidate 每行的全部字节:
```cpp
for (localRow = 0; localRow < rows; ++localRow)
    MakeCacheRangeVisible(ptr + row * stride, stride * sizeof(T));  // 逐行
```

优化为批量区域 invalidate:
```cpp
// 连续内存区域: 一次 invalidate 覆盖 rows * stride 字节
MakeCacheRangeVisible(ptr, static_cast<uint64_t>(rows) * stride * sizeof(T));
```

### 2.4 预期效果

| 指标 | 当前 | 优化后 |
|------|------|--------|
| queue entries (M=64, 2 experts) | 16 | 2 |
| GM 原子操作/iter | 16 × 3(pop+publish×2) = 48 | 2 × 3 = 6 |
| MakeRowsVisible 调用 | 16 × 8 = 128 次 | 2 × 64 = 2 次(批量) |
| PublishGmWrites 调用 | 16 次 | 2 次 |

---

## 优化 3：Compute-Comm Overlap（收益 ~2x）

### 3.1 当前串行模型

```text
时间线:
  Vec: [dispatch all] ─────────────────── [SwiGLU all] ──────── [combine all] [restore]
  Cube:               [GMM1 tile0][tile1]...[tileN] ──── [GMM2 tile0][tile1]...[tileN]

问题: GMM2 的全部 tile 完成后 combine 才开始；dispatch 必须全部完成 GMM1 才开始
```

### 3.2 目标 overlap 模型 (Expert-Group 级交叠)

```text
时间线 (2 experts, A & B):
  Vec:  [dispatch A] [dispatch B] [SwiGLU A] [SwiGLU B] [combine A] [combine B] [restore]
  Cube:     idle     [GMM1 A]     [GMM1 B]   [GMM2 A]   [GMM2 B]     idle

期望重叠:
  - dispatch B 与 GMM1 A 并行
  - SwiGLU A 与 GMM1 B 并行  
  - combine A 与 GMM2 B 并行
```

### 3.3 实现方案: 基于现有 Group-First 调度增强

#### Step 1: Cube 侧改为 per-expert 交替 GMM1/GMM2

当前 Cube loop:
```cpp
while (gmm2Cursor.tail < assignedTileCount) {
    TryConsumeDispatchAndProduceGmm1(...);  // 尽可能多地做 GMM1
    TryConsumeSwiGluAndProduceGmm2(...);    // 尽可能多地做 GMM2
}
```

优化为**优先级策略**:
```cpp
while (gmm2Cursor.tail < assignedTileCount) {
    // 优先 GMM2: 如果有 ready 的 SwiGLU tile，先做 GMM2（让 combine 尽早开始）
    bool gmm2Done = TryConsumeSwiGluAndProduceGmm2(...);
    if (!gmm2Done) {
        // GMM2 无可消费，做 GMM1
        TryConsumeDispatchAndProduceGmm1(...);
    }
}
```

#### Step 2: Vec 侧 combine 改为 per-expert 边触发

当前 combine 逻辑:
```cpp
// combine block 启动时等 gmm2ToCombineQueue 的对应 entry
WaitGmm2QueueEntryForCombineWork(gmm2ToCombineQueue, combineParams, workIndex, readyEntry);
// 然后执行 push/copy
RunCombinePipelineWorkBlock(remoteCtx, combineParams, workIndex);
```

改为 **per-expert 流水触发**:
```cpp
// combine block 按 expert 顺序工作
for (uint32_t expertIdx = 0; expertIdx < expertCount; ++expertIdx) {
    // 等待该 expert 的 GMM2 完成信号
    WaitGmm2ExpertReady(signalBase, expertIdx);
    // 立即执行该 expert 的 combine push（不等其他 expert）
    RunCombineForExpert(ctx, params, expertIdx);
}
```

#### Step 3: Ready 信号粒度从 per-tile 提升到 per-expert

当前 `PublishGmm2WindowReady` 在每个 tile (8行) 完成后发信号。
推大 tile + per-expert 语义后:

```cpp
// Cube 完成一个 expert 的全部 N-tiles 后发布一次 ready
if (currentExpert 全部 N-loop 完成) {
    PublishExpertGmm2Ready(signalBase, expertReadyIndex + expertId, expertRows);
}
```

Vec combine 侧用 `TWAIT` 或 `WaitVisibleGmValue` 等该 expert ready:
```cpp
v2_pto_sync::WaitVisibleGmValue(
    signalBase + expertReadySlot(expertIdx),
    expectedRows);
```

### 3.4 TNOTIFY/TWAIT 使用方案

对于 per-expert 信号（跨 Cube/Vec 两个 kernel）:
- **不能** 用 PTO Event（Event 只在同 kernel 内有效）
- **不能** 用 TNOTIFY/TWAIT（这是跨 rank 通信信号，不是跨 kernel）
- **正确方案**: GM signal（当前 `v2_pto_sync::PublishWindowReadyStore` + `WaitVisibleGmValue`）

当前已有的 GM signal 机制完全足够，只需调整信号粒度从 per-8-row-tile → per-expert。

### 3.5 预期收益

| 场景 | 当前 E2E | 优化后 E2E | 加速比 |
|------|----------|------------|--------|
| 2 experts 串行 | T(dispatch) + T(GMM1) + T(SwiGLU) + T(GMM2) + T(combine) | max(T(GMM1+GMM2), T(dispatch+SwiGLU+combine)) | ~1.8-2x |

---

## 优化 4：Vec/Cube 中间层优化（收益 ~1.2-1.5x）

### 4.1 当前瓶颈

Cube GMM2 完成后 `TSTORE_FP` 写 fp16 到 GM，combine Vec 再从 GM 读该 fp16 做 `ScaleRowsInPlace` 和 `TPUT`。这引入了一次完整的 GM write → GM read round-trip。

### 4.2 方案 A: 消除 ScaleRowsInPlace 的 in-place 读写（低风险，推荐先做）

当前 `ScaleRowsInPlace` 的问题:
1. 从 GM 读 fp16 → UB
2. UB 内 fp16→fp32→×scale→fp16
3. 写回 GM 同地址
4. 再从 GM 读该结果做 TPUT

**优化: 合并 scale + TPUT 为单次操作**

```cpp
// 不再 ScaleRowsInPlace 写回 GM，而是 scale 后直接 TPUT
void CombineScaleAndPut(__gm__ half* gmm2Src,
                        __gm__ half* remoteDst,
                        __gm__ float* perTokenScale2,
                        uint32_t rows,
                        uint32_t outputElems)
{
    for (uint32_t row = 0; row < rows; ++row) {
        float scale = *perTokenScale2++;
        for (uint32_t offset = 0; offset < outputElems; offset += tileElems) {
            // 1. TLOAD fp16 from GM (gmm2 output)
            pto::TLOAD(halfTile, gmm2SrcView);
            // 2. TCVT fp16 → fp32
            pto::TCVT(floatTile, halfTile);
            // 3. TMULS × scale
            pto::TMULS(floatTile, floatTile, scale);
            // 4. TCVT fp32 → fp16
            pto::TCVT(resultHalfTile, floatTile);
            // 5. TPUT fp16 到远端（跳过 GM 中间写回）
            pto::comm::TPUT(remoteDstView, localSrcView, resultHalfTile);
        }
    }
}
```

**收益**: 省去一次 GM write (ScaleRowsInPlace 写回) + 一次 GM read (TPUT 前重读)。对于 K=7168 的 row，节省 7168×2B×2(读写) = 28KB/row 的 GM 带宽。

### 4.3 方案 B: TSTORE_FP 目标直接为 combine region（中风险）

如果 TSTORE_FP 可以直接写到远端 HCCL window 地址:
```text
int32 L0C → TSTORE_FP(channelScale) → fp16 直接写远端 combine region
```

**约束检查**:
- TSTORE_FP 的 dst 是 `GlobalTensor<half>` 指向 GM 地址
- HCCL window 地址本质是 GM 地址（通过 windowIn[rank] 映射）
- 理论上可行，但需要验证 TSTORE_FP 是否支持写跨 NUMA 的远端地址

**如果可行**: 完全消除 combine 阶段的 TLOAD + TPUT，GMM2 直接输出到远端。

**风险**: 
1. TSTORE_FP 可能不支持远端 window 地址（需要 RDMA 引擎，不是 MTE3）
2. 需要额外的 per-token scale 乘回，TSTORE_FP 的 Scaling tile 只支持 per-channel

**结论**: 方案 B 因为 per-token scale 的存在，不能完全替代 combine。TSTORE_FP 只做 per-channel dequant，per-token scale2 仍需 Vec 侧处理。

### 4.4 方案 C: 单 Kernel MPMD（高风险，远期目标）

将 Vec 和 Cube 合并为单个 kernel:
```text
block 0-3: Cube role (GMM1/GMM2)
block 4-7: Vec role (dispatch/SwiGLU/combine/restore)
```

**优势**: 
- Cube 写 L1 shared buffer → Vec 从 L1 直接读，跳过 GM
- Cross-core event（`set_flag`/`wait_flag`）替代 GM signal
- 减少 kernel launch overhead

**约束**:
- PTO 支持 MPMD 模型（docs/pto-knowledge.md 明确说明）
- 但当前 two-kernel 架构已经稳定且调试方便
- 单 kernel MPMD 的调试复杂度显著增加

**建议**: 作为 Phase 3 远期目标，当前先做方案 A。

### 4.5 推荐优先级

| 子方案 | 难度 | 收益 | 优先级 |
|--------|------|------|--------|
| A: 合并 scale+TPUT | 低 | ~1.3x | **P0** |
| B: TSTORE_FP 直写远端 | 中 | ~1.5x | P2 (需验证) |
| C: 单 kernel MPMD | 高 | ~1.5x | P3 (远期) |

---

## 实施计划

### Phase 1: 推大 Tile（预期 ~16-50x 收益）

| 步骤 | 改动 | 文件 | 预计工作量 |
|------|------|------|-----------|
| 1.1 | 新建 `RunPtoGmmBlockInt8_Large` 模板，支持 M=128, N=256, BaseK=128, L1K=512 | `pto_gmm_block_int8.hpp` | 重写 |
| 1.2 | 改 `kComputeTileRows = 128` | `compute_cube_queue_kernel.cpp` + `compute_vec_views.hpp` | 小改 |
| 1.3 | 新建 `RunGmm1Int8Tile_Large` 和 `RunGmm2Int8Tile_Large` 封装 N-loop | `pto_gmm_block_int8.hpp` | 新增 |
| 1.4 | 改 Cube 主循环中的 MMAD 调用 | `compute_cube_queue_kernel.cpp` | 中改 |
| 1.5 | 调整 host 端 computeTileCount/workspace 分配 | `main.cpp` | 小改 |
| 1.6 | SwiGLU 适配 128 行输入（内部仍分 16 行子 tile） | `swiglu_fp16_vec.hpp` | 小改 |

### Phase 2: Queue 批量化 + 通算 Overlap（预期 ~4-10x 收益）

| 步骤 | 改动 | 文件 | 预计工作量 |
|------|------|------|-----------|
| 2.1 | dispatch producer 改为 per-expert 粒度发布 | `comm_vec_queue_kernel.cpp` | 中改 |
| 2.2 | Cube loop 改优先级策略（GMM2 优先） | `compute_cube_queue_kernel.cpp` | 小改 |
| 2.3 | combine 改为 per-expert 边触发 | `comm_vec_queue_kernel.cpp` + `combine_push.hpp` | 中改 |
| 2.4 | ready 信号粒度改为 per-expert | `compute_cube_queue_kernel.cpp` | 小改 |
| 2.5 | MakeRowsVisible 批量化 | `compute_cube_queue_kernel.cpp` | 小改 |

### Phase 3: Vec/Cube 中间层优化（预期 ~1.3x 收益）

| 步骤 | 改动 | 文件 | 预计工作量 |
|------|------|------|-----------|
| 3.1 | 新增 `CombineScaleAndPut` 合并 scale+TPUT | `combine_push.hpp` | 中改 |
| 3.2 | 移除 `ScaleRowsInPlace` 的 GM 写回 | `combine_push.hpp` | 小改 |
| 3.3 | 调整 combine 流程：TLOAD gmm2 → scale → TPUT 远端 | `combine_push.hpp` | 中改 |

---

## 预期总收益

```text
Phase 1 (tile):     当前 17.5ms / 64 tokens → 预计 ~0.3-1ms / 64 tokens  (16-50x)
Phase 2 (overlap):  → 预计 ~0.15-0.5ms  (2-4x from phase 1)
Phase 3 (fusion):   → 预计 ~0.1-0.4ms  (1.3x from phase 2)

shmem baseline:     969 us / 512 tokens = 1.9 us/token
v2 目标:            300-500 us / 64 tokens = 4.7-7.8 us/token  (差距收窄到 ~3-4x)
```

注: 剩余 3-4x 差距来自:
- shmem 单 kernel 的 Fixpipe 硬件直通（零 GM round-trip）
- shmem topK=8 (512 routed tokens) 的 batch 摊平效应
- shmem 多核 restore 并行

---

## 风险与依赖

| # | 风险 | 缓解措施 |
|---|------|----------|
| R1 | L0B/L0C 极限容量，可能需要微调 N=256→192 | 先按 256 实现，profile 后按需缩小 |
| R2 | L1 388KB 接近极限，Scaling tile 可能有额外开销 | Scaling 在 FBuffer(2KB) 中，不占 L1 |
| R3 | kComputeTileRows=128 时 M<128 的 tail 行利用率 | 使用 dynamic valid region，不影响正确性 |
| R4 | Pipeline reset bug（多 iteration 挂起）仍存在 | 先用单 iteration 验证性能，reset bug 独立修复 |
| R5 | weight NZ layout 下的 TLOAD 对齐约束 | 确认 weight stride 满足 32B 对齐 |
