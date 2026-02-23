# 2026-02-23

## Server Migration & Run Guide (No Auto-Stop Needed)

### 1) Pull latest code from ywangmu repo

```bash
cd /path/to/pto-isa
git remote -v
# origin should be https://gitcode.com/ywangmu/pto-isa.git
git pull origin master
```

### 2) Environment setup

```bash
source /home/ywangmu/Ascend/ascend-toolkit/set_env.sh
```

### 3) Direct run commands on server simulator

Server-side simulator can exit normally, so use `run.sh` directly (no monitor script required).

#### FA4 (example: baseline case64)

```bash
cd kernels/manual/a5/flash_atten_fa4
FA4_SKIP_RESCALE_ENABLE=OFF \
bash run.sh -r sim -v Ascend910_9599 --cases "64,128,1024,128,128" -p 2 -m 1
```

#### BLASST (example: ratio50 + skip-vload, case64)

```bash
cd kernels/manual/a5/flash_atten_blasst
BLASST_SKIP_SOFTMAX_ENABLE=ON \
BLASST_SKIP_SOFTMAX_FORCE=OFF \
BLASST_SKIP_SOFTMAX_FORCE_RATIO=0.5f \
BLASST_SKIP_VLOAD_ENABLE=ON \
bash run.sh -r sim -v Ascend910_9599 --cases "64,128,1024,128,128" -p 2 -m 1
```

### 4) Output/check points

- Build and logs are generated under each kernel directory `build/`.
- Completion markers: simulator exits normally; log contains `send_stars_interrupt` and `build/core0_summary_log` is generated.

### 5) Portability note

- `kernels/manual/a5/flash_atten_fa4/run.sh` now supports fallback build root:
  - if `FA4_BUILD_ROOT_DIR` is set, use it;
  - else prefer `/hddata/...` only when writable;
  - otherwise fallback to local kernel directory.
  This avoids server failures when `/hddata` is unavailable.

# pto-isa Development Notes

## 2026-02-10

### MODE1 Bottleneck Analysis (case 64,128,1024,128,128, -p 2 -m 1)

**Config**: S0=128, S1=1024, HEAD=64, CUBE_S0=128, CUBE_S1=128, TILE_S1=128, qk_preload=2, FIFO_MODE=1 (ALL_UB_PATH)
**Kernel file**: [fa_performance_kernel.cpp](kernels/manual/a5/flash_atten/fa_performance_kernel.cpp)
**Log source**: `build/core0.{cubecore0,veccore0}.instr_{popped_log,log}.dump`
**Analysis scripts**: [run_bottleneck.sh](kernels/manual/a5/flash_atten/scripts/run_bottleneck.sh) (one-click), [bottleneck_analysis.py](kernels/manual/a5/flash_atten/scripts/bottleneck_analysis.py), [run_timeline.sh](kernels/manual/a5/flash_atten/scripts/run_timeline.sh), [run_swimlane.sh](kernels/manual/a5/flash_atten/scripts/run_swimlane.sh)
**num_tiles**: 8 (1024/128), kTileFactor=1

#### Overall Timing
- Total kernel wall time: **30,834 cycles**
- Cube span: 29,272 cycles
- Vector span: 30,814 cycles
- MMAD (Cube compute) utilization: **14.6%** (4,496 / 30,834)

#### Pipeline Utilization Breakdown

| Core   | Compute   | Load     | Store   | Sync (stall) | Other (stall) |
|--------|-----------|----------|---------|--------------|---------------|
| Cube   | 0.2%      | 1.4%    | 0.2%   | 55.3%        | 42.9%         |
| Vector | 17.0%     | 0.0%    | 0.2%   | 52.7%        | 30.0%         |

> Cube 核心 97% 以上的时间在同步等待，只有不到 2% 时间在做有效计算和搬运。

#### Per-Tile Steady-State Breakdown (Cube 视角)

每个 tile 迭代中 Cube 的时间分解（单位: cycles）：

| Stage              | Tile 0 | Tile 1 | Tile 2 | Tile 3 | Tile 4 | Tile 5 |
|--------------------|--------|--------|--------|--------|--------|--------|
| QK MMAD            | 281    | 281    | 281    | 281    | 281    | 281    |
| **UB buffer stall**| 1559   | 1197   | 1197   | 1600   | 1805   | 1829   |
| QK TMOV (L0C→UB)  | 338    | 340    | 314    | 378    | 375    | 378    |
| **SM sync stall**  | 349    | 346    | 336    | 382    | 382    | 382    |
| PV MMAD            | 281    | 281    | 281    | 281    | 281    | 281    |
| PV TMOV (L0C→UB)  | 231    | 231    | 187    | 195    | 195    | 195    |

- **Average tile iteration: ~3,048 cycles** (ideal 562 cycles for 2× MMAD)
- **Efficiency: 18.4%**
- **Stall ratio: 63%** per tile

#### Bottleneck 1: UB Buffer Stall (ubBufSync) — 平均 1,531 cycles/tile

**位置**: [fa_performance_kernel.cpp L446-448](kernels/manual/a5/flash_atten/fa_performance_kernel.cpp)
```cpp
if (sub_tile_id == 0 && tile_id >= static_cast<int>(SRC_VEC_TN_BUFFERS)) {
    ubBufSync.allocate();  // ← Cube blocks here ~1,531 cycles
}
```
**原因**: QK MMAD 完成后，Cube 要 TMOV 结果到 Vec 的 UB。但 UB 只有 2 个 ping-pong buffer (`srcVecTNBuffers=2`)，Vec 还在处理之前的 tile 的 softmax，UB 没释放。Cube 必须等 Vec 执行 `ubBufSync.free()` ([L727-729](kernels/manual/a5/flash_atten/fa_performance_kernel.cpp))。

**根本原因**: Vec 的 softmax 计算延迟（~2000+ cycles per tile）远超 Cube MMAD（281 cycles），ping-pong depth=2 不足以掩盖。

#### Bottleneck 2: Softmax Sync Stall (sm2pvSync) — 平均 363 cycles/tile

**位置**: [fa_performance_kernel.cpp L528-529](kernels/manual/a5/flash_atten/fa_performance_kernel.cpp)
```cpp
if (sub_tile_id == 0)
    sm2pvSync.wait();  // ← Cube waits for Vec softmax completion
```
**原因**: Cube 释放 QK TMOV 后，还需等 Vec 完成 softmax 并将 P 写入 L1（TINSERT），才能开始 PV matmul。

#### Bottleneck 3: Tail Tiles (tile 6, 7) — 无 QK 流水线覆盖

| Tile   | Gap before PV | Note               |
|--------|--------------|---------------------|
| tile 6 | 2,873 cycles | 无 next QK 可预取    |
| tile 7 | 3,073 cycles | 无 next QK 可预取    |

最后两个 tile 没有下一轮 QK 来掩盖 PV 等待，Cube 完全闲置等待 Vec softmax。

#### Vec 计算密度分析

Vec 在整个 kernel 期间持续高密度工作（每 500 cycles 窗口内 600-1200 条计算指令），没有显著空闲期。这说明 **Vec softmax 是计算瓶颈**，Cube 的所有 stall 都源于等待 Vec。

#### Vec P (Softmax) vs GU (Global Update) 瓶颈分析

**分析方法**: 由于 Ascend A5 的 Vec 核心采用深度流水线架构（指令发射远超执行，WAIT 异步解析），无法用时间窗口分离 P 和 GU 阶段。采用 **opcode 分类法**，根据 softmax 和 GU 宏的代码结构将指令归类。详见 [phase_analysis.py](kernels/manual/a5/flash_atten/scripts/phase_analysis.py)。

**指令分类**:
| 类别 | 所属阶段 | 代表指令 | 对应 PTO 宏操作 |
|------|---------|---------|----------------|
| P-unique | Softmax | VCMAX, VCADD, VCVT_F2F, VSSTB | TROWMAX, TROWSUM, TCVT, NZ scatter |
| P-macro-only | Softmax | VSUB, VEXP, VMAX | TROWEXPANDSUB/TSUB, TEXP, TMAX |
| GU-unique | GU last | VDIV | TROWEXPANDDIV (最后一个 tile 的归一化) |
| Shared heavy | Both | VDUP, VMULS, VADD, VMUL | TROWEXPANDSUB/MUL, TMULS, TADD, TMUL |
| Overhead | Both | VLD*, VST*, PLT, VMOV | 数据搬运、谓词操作 |

**Vec 全局指令统计** (48,247 ops, 340,396 instruction-cycles):

| 分类 | Instruction-cycles | 占比 |
|------|-------------------|------|
| P-unique | 52,224 | 15.3% |
| P-macro-only | 29,899 | 8.8% |
| GU-unique | 1,088 | 0.3% |
| Shared heavy | 33,561 | 9.9% |
| Overhead (LD/ST/PLT) | 205,613 | 60.4% |
| Scalar/control | 16,795 | 4.9% |

**Shared ops 分配估算** (基于宏结构):
- P 每 tile 有 3 个 [32,128] 大 tile 操作: TROWEXPANDSUB + TMULS + TEXP
- GU 每 tile 有 2 个 [32,128] 大 tile 操作: TROWEXPANDMUL + TADD
- 8 P tiles × 3 = 24 单位 vs 7 GU tiles × 2 = 14 单位 → P 分得 63.2%

**最终结果**（双口径）:

| 口径 | P (Softmax) | GU (Global Update) |
|------|-------------|--------------------|
| Conservative | ~222,933 (68.9%) | ~100,668 (31.1%) |
| Macro-strict (VSUB/VEXP/VMAX 固定归 P) | ~252,832 (71.5%) | ~100,668 (28.5%) |

**P / GU 比值 = 2.21x ~ 2.51x → Softmax 是 Vec 核心的主要计算瓶颈。**

交叉验证:
- P-unique: VCMAX 1024 条 (128/tile × 8 tiles) ✓, VCADD 1024 条 ✓, VCVT_F2F 1024 条 ✓, VSSTB 512 条 (64/tile × 8 tiles) ✓
- GU-unique: VDIV 64 条 (仅 last tile 的 TROWEXPANDDIV) ✓

**Softmax 瓶颈内部分解** ([pto_macro_fa_softmax.hpp](kernels/manual/a5/flash_atten/pto_macro_fa_softmax.hpp)):
1. TROWMAX → VCMAX: 16,384 icyc (行最大值归约)
2. TROWSUM → VCADD: 22,528 icyc (行求和归约，最重!)
3. TEXP [32,128]: ~16,496 icyc (指数运算)
4. TCVT → VCVT_F2F: 7,168 icyc (fp32→fp16 类型转换)
5. NZ scatter → VSSTB: 6,144 icyc (L1 写入格式转换)
6. TROWEXPANDSUB [32,128]: VDUP + VSUB ~18,481 icyc (广播减法)

**微架构发现**: Vec 指令发射采用深度流水线，多个 tile 的 P 和 GU 指令在同一时间窗口内交叉发射。WAIT_INTRA_BLOCK 的 `ts_start` 和 `ts_end` 跨度可达 10,000+ cycles（例如 [15015, 25229]），表示信号从发射到解析的等待时间。这意味着 Vec 可能已经发射了 5+ 个 tile 的完整指令流，但实际执行受限于 Cube 的 pv2guSync 信号。

#### 优化方向

1. **增大 TILE_S1 / 减少 tiles 数量**: 每个 tile 更大的 S1 维度可以减少 tile 迭代次数，从而减少跨核同步开销。但受限于 L1/UB 容量。
2. **增大 qk_preload**: 当前 `qk_preload=2`，增大可以让更多 QK 提前完成，减少 UB buffer 竞争。但 Mode 1 要求 `qkPreloadNum <= pMatTNBuffers(=2)`，需考虑放松此约束或增加 pMat buffer 数量。
3. **切换到 MODE 2 (QK_PV_UB_ONLY)**: P 路径走 GM 可以消除 TINSERT 对 L1 buffer 的独占需求，允许更大的 `qk_preload`。代价是 P 路径增加 GM roundtrip 延迟。
4. **优化 Vec softmax** (占 Vec 总计算 ~69%，是绝对瓶颈):
   - **TROWSUM (VCADD)** 是 softmax 内部最重的单项 (22,528 icyc)，可考虑与 TEXP 融合
   - **TROWMAX (VCMAX)** 次之 (16,384 icyc)，可考虑近似或分层归约
   - 减少 softmax 中的精度转换 (VCVT_F2F: 7,168 icyc)，探索全 fp16 softmax
   - 合并 reduce 操作（VCMAX + VCADD 在两次 pass 扫描，可尝试单 pass online 计算）
   - NZ scatter (VSSTB: 6,144 icyc) 用于 L1 insert 的格式转换，若能避免 NZ 格式则可省去
   - 利用 Vec SIMD 更宽的向量化
5. **增大 srcVecTNBuffers**: 从 2 增到 3 可能减少 UB stall，但需确认 256KB UB 空间足够。

#### 泳道图可视化验证

泳道图脚本 ([pipeline_swimlane_svg.py](mytools/pipeline_swimlane_svg.py)) **不受深度流水线影响**:
- 10000+ cycle 的 WAIT/SET 指令全部在 `FLOWCTRL` 管线，已被 `EXCLUDED_PIPELINES` 排除
- 700+ cycle 的 `VF` (Vector Flush) 指令在 `PUSHQ` 管线，同样已排除
- 实际渲染的 RVEC* 计算指令 (48,247 条) duration 全部 ≤ 100 cycles，泳道图准确
- 受影响的仅是基于 timeline CSV 的时间窗口分析方法（P/GU 边界判断），已改用 opcode 分类法解决

#Agent Notes
 To resume this session: agent --continue or agent --resume=50f9ffca-74b6-4f8e-ba4a-4e8580720812

  To resume this session: agent --continue or agent --resume=50f9ffca-74b6-4f8e-ba4a-4e8580720812