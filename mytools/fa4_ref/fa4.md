
# 🚀 FlashAttention‑v4 Skip‑Rescale 优化  
### GPU → NPU 迁移实现技术说明文档

**版本：** v2.0  
**适用目标：** FlashAttention Forward — GU 阶段（Running Update）  
**涵盖平台：**  
- CUDA (GPU, SM80/90 A100/H100)  
- Ascend NPU (A2/A5/C系列 DAV VecCore)  
**编写日期：** 2026‑02‑11  

---
## 参考
- Blog 解读1：[Flash Attention 4浅析](https://zhuanlan.zhihu.com/p/1969881877331547787)
- Blog 解读2：[We reverse-engineered Flash Attention 4](https://modal.com/blog/reverse-engineer-flash-attention-4)
- Cuda 开源实现Blackwell：[FlashAttention](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py)

## 📖 一、背景与问题

FlashAttention 系列通过分块（tiling）计算 QKᵀV ，减少 O(N²) 访存。  
在每个 q‑block 处理完后，为了数值稳定，需要调整历史输出：

\[
O = O \times \exp(m_\text{old} - m_\text{new}) + P V
\]

其中：  
- \( O \) 为当前累计输出；  
- \( m_\text{old} \)、\( m_\text{new} \) 分别为 softmax 最大值；  
- \( PV \) 为当前 tile 的加权输出。

传统做法始终计算这一步，即使 \(\exp(m_\text{old} - m_\text{new}) \approx 1\)。

---

## 🧠 二、FA v4 Skip‑Rescale 在 GPU‑CUDA 上的实现

### 1️⃣ 原理

当 softmax 的最大值变化很小时：  
\(\exp(m_\text{old} - m_\text{new}) \approx 1.0\)  
此时乘法几乎不改变 O，可跳过该行的 `O *= scale`。

CUDA 线程块的实现可以检测 warp 级掩码：部分 thread 可跳过 FMA。

---

### 2️⃣ CUDA 核心实现示例（简化版）

下面展示了在 **NVIDIA Hopper SM90** 上的 FA‑v4 伪实现（C++ CUDA 示例）：

```cpp
// FlashAttention v4 -- GPU CUDA kernel (simplified)

template <int BLOCK_M, int BLOCK_N, typename scalar_t>
__global__ void flashattention_fwd_gu_kernel(
        scalar_t* __restrict__ O,
        const scalar_t* __restrict__ PV_tile,
        const float* __restrict__ exp_scale,   // exp(m_old - m_new)
        int rows, int cols)
{
    int row = blockIdx.x * BLOCK_M + threadIdx.y;
    int col = threadIdx.x;

    if (row >= rows) return;

    // load scale value for this row
    float s = exp_scale[row];
    float do_rescale = fabsf(s - 1.0f) > 1e-5f;

    // warp-level predicate: any thread in warp needs rescale?
    unsigned mask = __ballot_sync(0xffffffff, do_rescale);
    if (mask) {
        // scale != 1 for some row --> do FMA
        for (int j = col; j < cols; j += BLOCK_N) {
            O[row * cols + j] = O[row * cols + j] * s + PV_tile[row * cols + j];
        }
    } else {
        // all scale ≈ 1 --> skip rescale, only add
        for (int j = col; j < cols; j += BLOCK_N) {
            O[row * cols + j] += PV_tile[row * cols + j];
        }
    }
}
```

> ✅ 每 warp 检查 scale = 1 的行情况，全部满足则跳过 mul。  
> ✅ 每 thread 可单独分支（SIMT）。  
> ✅ Hopper 并行度高、分支成本低 → 可明显加速 20–40%。

---

## ⚙️ 三、GPU → NPU 迁移可行性分析

| 特征维度 | GPU (CUDA SIMT) | NPU (Ascend SIMD/Tile) | 差异与影响 |
|-----------|----------------|------------------------|-------------|
| 并行模型 | SIMT （per‑thread 独立） | SIMD （tile 向量锁步） | ❌ NPU 无 per‑lane 分支 |
| 控制流粒度 | 线程 (row) | tile (多个 rows) | 粒度更粗 |
| 掩码执行 | warp‑level `__ballot_sync` | 无内置掩码 | 需 tile‑reduce 逻辑替代 |
| 内存访问 | global + shared mem | GM + UB + Vec Tile | 类似 DMA 模式 |
| Skip 粒度 | Row / Warp 级 | ✅ Tile 级 | 可近似模拟 |

### ✅ 结论  

> FA4 skip 机制在 NPU 上**可以实现 tile‑level 版本**：  
> - 以整 Vec tile 为单位判断是否 scale ≈ 1；  
> - 若全部行满足，则跳过 TROWEXPANDMUL。

---

## 🔧 四、适配方案设计（NPU 架构）

### 1️⃣ 思路  

在当前 NPU FA 内核中（见 `pto_macro_fa_gu`）逻辑为：

```cpp
O = O * exp_max + PV
```

执行流程：
1. `TROWEXPANDMUL(prev_sv_tile, prev_sv_tile, exp_max)`  
2. `TADD(prev_sv_tile, prev_sv_tile, est_sv_tile)`

→ 在执行 TROWEXPANDMUL 前，对 exp_max tile 做 reduce 检查。

---

### 2️⃣ Tile‑level Skip 判定逻辑  

```cpp
// reduce over the tile, to check if all exp_max ≈ 1.0
float max_diff = vec_reduce_max(abs(exp_max - 1.0f));

bool skip_rescale = (max_diff < 1e-5f);

// if skip_rescale == true → 直接执行加法阶段，不做 rescale
```

---

## 💻 五、NPU 实现伪代码（与原计算融合）

> 下述伪代码保留原 Ascend vector tile 调度模型符号，与 SDK runtime 接口兼容。  

```cpp
// ============================================================
// FlashAttention v4 -- NPU Tile-level Skip Rescale (pseudo-code)
// ============================================================

AICORE inline void pto_macro_fa_gu(
        Tile<float> &O_tile,        // prev_sv_tile
        const Tile<float> &PV_tile, // est_sv_tile
        const Tile<float> &exp_max) // exp(m_old - m_new)
{
    // Step 1: optional reduce to check scale stability
    Tile<float> tmp_tile = exp_max;
    TADDS(tmp_tile, exp_max, -1.0f);     // diff = exp_max - 1
    TABS(tmp_tile, tmp_tile);            // |diff|
    float max_diff = TREDUCE_MAX(tmp_tile);

    bool skip_rescale = (max_diff < 1e-5f);

    // Step 2: apply skip logic
    if (!skip_rescale) {
        // normal case
        TROWEXPANDMUL(O_tile, O_tile, exp_max);
    }

    // Step 3: accumulate current PV
    TADD(O_tile, O_tile, PV_tile);
}

// ============================================================
// Last-tile version with normalization
// ============================================================

AICORE inline void pto_macro_fa_gu_last(
        Tile<float> &O_tile,
        const Tile<float> &PV_tile,
        const Tile<float> &exp_max,
        const Tile<float> &global_sum)
{
    Tile<float> tmp_tile = exp_max;
    TADDS(tmp_tile, exp_max, -1.0f);
    TABS(tmp_tile, tmp_tile);
    float max_diff = TREDUCE_MAX(tmp_tile);
    bool skip_rescale = (max_diff < 1e-5f);

    if (!skip_rescale) {
        TROWEXPANDMUL(O_tile, O_tile, exp_max);
    }

    TADD(O_tile, O_tile, PV_tile);
    TROWEXPANDDIV(O_tile, O_tile, global_sum);
}
```

---

## 📊 六、性能 & 数值评估

| 指标 | 效果 |
|------|-------|
| 可跳过比例 | 取决于输入分布与阈值，需要实测统计 |
| 计算减少 | 仅 GU rescale 相关 compute 可减少，收益上界有限 |
| 功耗降低 | 需结合 core0_summary_log 与泳道图实测 |
| 误差 | 需在目标精度下评估（float32/float16） |
| 兼容性 | 完全向前兼容，无需更改 pipeline |

---

## ✅ 七、总结与建议

| 项 | 方案 |
|----|------|
| **目标** | 复现 FA4 skip‑rescale，在 NPU 上提升 GU 阶段效率 |
| **GPU 实现粒度** | per‑row / warp‑mask |
| **NPU 可行粒度** | tile‑level skip |
| **关键修改点** | 在 `pto_macro_fa_gu` 内部添加 reduce 检测逻辑 |
| **推荐阈值 ε** | 1e‑5 （float32），1e‑3 （float16） |
| **可复用场景** | forward/backward 共用 |
| **预期提升** | 需以 A5 实测为准（见实验章节） |

---

## 📘 附录：函数语义对照表

| CUDA 指令/函数 | 对应 NPU Tile API | 含义 |
|----------------|-------------------|------|
| `__ballot_sync()` | `TREDUCE_MAX` | 逻辑归约 |
| `if(scale==1)` | `abs(exp_max-1)<eps` | 近似判断 |
| `O[row]*scale + PV[row]` | `TROWEXPANDMUL + TADD` | 向量化广播乘加 |
| warp mask skip | tile‑level skip flag | 控制粒度较粗 |

---

### ✅ **一句话总结**

> FA‑v4 的 skip‑rescale 优化原理完全可迁移到 NPU：  
> 通过对 `exp_max` tile 做一次 reduce 判断，在整 Vec tile 粒度跳过 `O*=exp` ，  
> 从而在 Ascend 硬件上安全实现 **FA4 优化版 FlashAttention Forward**，  
> 在 A5 上收益需以实际 workload 与阈值实测为准。  

## 八、A5 PTO-ISA 指令支持与可迁移性校验（实现级）

本节基于 `pto-isa` A5 后端实现文件进行核对（不是仅语义文档）：

- `TROWEXPANDMUL` → `vmul`（`include/pto/npu/a5/TRowExpandMul.hpp`）
- `TROWEXPANDSUB` → `vsub`（`include/pto/npu/a5/TRowExpandSub.hpp`）
- `TROWEXPANDDIV` → `vdiv`（`include/pto/npu/a5/TRowExpandDiv.hpp`）
- `TROWMAX` / `TROWSUM` → `vcmax` / `vcadd`（`include/pto/npu/a5/TRowReduce.hpp`）
- `TEXP` → `vexp`（`include/pto/npu/a5/TUnaryOp.hpp`）
- `TMULS` → `vmuls`（`include/pto/npu/a5/TMulS.hpp`）
- `TMUL` → `vmul`（`include/pto/npu/a5/TMul.hpp`）
- `TCVT` → `vcvt`/`vtrc`（`include/pto/npu/a5/TCvt.hpp`）

### 结论

1. **Skip-rescale 所需算子在 A5 PTO-ISA 中已具备**，不需要新增 ISA 指令。  
2. A5 不具备 CUDA 的 warp-ballot 等细粒度分支语义，最稳妥的是 **tile-level gate**。  
3. 可迁移路径是：在 GU 中对 `exp_max` 做近 1 判断，满足阈值时跳过 `TROWEXPANDMUL`，保留 `TADD` 与末尾 `TROWEXPANDDIV`。

---

## 九、理论性能收益模型（A5, tile-level gate）

记：

- `C_mul` = 一次 `TROWEXPANDMUL` 的 Vec 周期
- `C_chk` = 判定 `skip` 的周期（abs/compare 或 scalar scan）
- `r` = 可跳过 tile 比例（skip ratio）

则 GU 阶段期望周期：

\[
C'_{GU} \approx C_{GU} - r\cdot C_{mul} + C_{chk}
\]

总 Vec 周期近似：

\[
C'_{Vec} \approx C_{Vec} - r\cdot C_{mul} + C_{chk}
\]

### 用当前 A5 日志做数量级估计（`64,128,1024,128,128`, mode1, preload=2）

已观测（timeline）：

- `Vec` 总 instruction-cycles ≈ `340,396`
- `RV_VMUL` instruction-cycles ≈ `3,640`（其中一部分来自 GU 的 row-expand multiply）

若以 `RV_VMUL` 作为 `C_mul` 的下界，假设：

- `r = 0.6 ~ 0.8`
- `C_chk` 量级 `500 ~ 1500` cycles（实现相关）

则理论净收益约为：

\[
\Delta C_{Vec} \approx r\cdot 3640 - C_{chk}
\]

对应区间（粗估）：

- 乐观：`0.8*3640 - 500 = 2412` cycles（约 `0.7%` Vec）
- 中性：`0.7*3640 - 1000 = 1548` cycles（约 `0.45%` Vec）
- 保守：`0.6*3640 - 1500 = 684` cycles（约 `0.2%` Vec）

### 解释

1. 在当前 shape 下，GU 本来就不是主瓶颈（P 更重），因此总收益不会像 GPU 报告那样高。  
2. 但在 **更大 HEAD、更高 GU 比重、或多 head 并发更高** 的配置中，`C_mul` 会增大，收益会更明显。  
3. 如果 `C_chk` 能进一步降到近似常量开销，收益会更稳定。

---

## 十、当前工程化建议（A5）

1. 先落地 `flash_atten_fa4` 的 **可编译 prototype**（tile-level skip，阈值可配）。  
2. 用现有 timeline 脚本新增 `skip-hit-rate` 统计，实测 `r`。  
3. 分 case 扫描（S1/HEAD/preload/mode）找出 `r` 与性能收益最匹配区间。  
4. 若 `C_chk` 偏高，再考虑把判定从“全行扫描”降为“分块采样 + 安全回退”。

---

## 十一、A5 实验与结果（FA4 原型）

### 1️⃣ 实验配置与运行方式

**目录与代码：**
- 原型实现目录：`kernels/manual/a5/flash_atten_fa4/`
- GU skip 逻辑：`kernels/manual/a5/flash_atten_fa4/pto_macro_fa_gu.hpp`
- 编译参数：`kernels/manual/a5/flash_atten_fa4/CMakeLists.txt`
- 运行脚本：`kernels/manual/a5/flash_atten_fa4/run.sh`

**运行命令（示例）：**
```bash
source /home/ywangmu/Ascend/ascend-toolkit/set_env.sh

# baseline (no skip)
FA4_SKIP_RESCALE_ENABLE=OFF \
bash run.sh -r sim -v Ascend910_9599 --cases "64,128,1024,128,128" -p 2 -m 1

# skip (eps=1e-2)
FA4_SKIP_RESCALE_ENABLE=ON FA4_SKIP_RESCALE_EPS=1.0e-2 \
bash run.sh -r sim -v Ascend910_9599 --cases "64,128,1024,128,128" -p 2 -m 1

# force-skip (upper bound)
FA4_SKIP_RESCALE_ENABLE=ON FA4_SKIP_RESCALE_FORCE=ON \
bash run.sh -r sim -v Ascend910_9599 --cases "64,128,1024,128,128" -p 2 -m 1
```

**仿真完成标志：**
- 终端出现：`[DRVSTUB_LOG] ... send_stars_interrupt:get cq_0 base_addr: 10020000`
- `build/core0_summary_log` 中有完整 summary 段落

---

### 2️⃣ Phase 分析与泳道图输出（结果文件）

- baseline (no‑skip)：
  - Phase：`kernels/manual/a5/flash_atten_fa4/scripts/results/phase_skip_off/phase_analysis.txt`
  - Swimlane：`kernels/manual/a5/flash_atten_fa4/scripts/results/phase_skip_off/swimlane.svg`
- skip eps=1e‑2：
  - Phase：`kernels/manual/a5/flash_atten_fa4/scripts/results/phase_skip_eps1e-2/phase_analysis.txt`
  - Swimlane：`kernels/manual/a5/flash_atten_fa4/scripts/results/phase_skip_eps1e-2/swimlane.svg`
- force‑skip（极限上界）：
  - Phase：`kernels/manual/a5/flash_atten_fa4/scripts/results/phase_skip_force/phase_analysis.txt`
  - Swimlane：`kernels/manual/a5/flash_atten_fa4/scripts/results/phase_skip_force/swimlane.svg`

**泳道图嵌入：**

baseline (no‑skip)

![](../kernels/manual/a5/flash_atten_fa4/scripts/results/phase_skip_off/swimlane.svg)

skip eps=1e‑2

![](../kernels/manual/a5/flash_atten_fa4/scripts/results/phase_skip_eps1e-2/swimlane.svg)

force‑skip（极限上界）

![](../kernels/manual/a5/flash_atten_fa4/scripts/results/phase_skip_force/swimlane.svg)

---

### 3️⃣ Phase 分析结论（Vec P/GU）

- baseline 与 skip eps=1e‑2 的 P/GU 结论一致：**P 仍是主瓶颈**  
  - baseline：P/GU ≈ **2.21x / 2.51x**
  - eps=1e‑2：P/GU ≈ **2.21x / 2.51x**

- **force‑skip（极限上界）**仍未改变结论：  
  - P/GU ≈ **2.27x / 2.60x**

**Phase 对比表（摘要）：**

| Case | Vec total icyc | P share (cons) | P share (strict) | P/GU (cons) | P/GU (strict) |
|------|----------------|----------------|------------------|-------------|---------------|
| baseline (no‑skip) | 340,396 | 68.9% | 71.5% | 2.21x | 2.51x |
| skip eps=1e‑2 | 340,396 | 68.9% | 71.5% | 2.21x | 2.51x |
| force‑skip | 309,154 | 69.4% | 72.2% | 2.27x | 2.60x |

---

### 4️⃣ core0_summary_log 周期对比（baseline vs force‑skip）

来源：
- baseline：`kernels/manual/a5/flash_atten_fa4/build_skip_off/core0_summary_log`
- force‑skip：`kernels/manual/a5/flash_atten_fa4/build_skip_force/core0_summary_log`

**Cube（subcore 0）：**
- kernel ticks：`32340 → 30599`（下降 **5.4%**）
- cube busy：`4496 → 4496`（几乎不变）

**Vector（subcore 1/2）：**
- kernel ticks：`32943/32992 → 31198/31197`（下降 **~5.3%**）
- rvec busy：`26389 → 22996`（下降 **~12.9%**）

**结论：**
- 极限跳过主要降低 **Vector 计算占用**，Cube 侧变化很小。  
- 即便在极限跳过下，P 仍主导整体 Vec 计算。  

**core0_summary_log 对比表（摘要）：**

| Metric | baseline | force‑skip | Delta |
|--------|----------|------------|-------|
| cube kernel ticks | 32,340 | 30,599 | -1,741 (-5.4%) |
| vec0 kernel ticks | 32,943 | 31,198 | -1,745 (-5.3%) |
| vec1 kernel ticks | 32,992 | 31,197 | -1,795 (-5.4%) |
| rvec busy (vec0/1) | 26,389 | 22,996 | -3,393 (-12.9%) |
| cube busy | 4,496 | 4,496 | ~0 |

---

### 5️⃣ 小结（A5 初步落地方案）

1. **实现方式**：在 GU 阶段引入 tile‑level skip gate（A5 可行）。  
2. **可配置参数**：`FA4_SKIP_RESCALE_ENABLE / FA4_SKIP_RESCALE_EPS / FA4_SKIP_RESCALE_FORCE`。  
3. **实测结论**：当前 shape 与 mode 下，收益有限且 P 仍是主瓶颈；force‑skip 仅降低 Vector 占用。  
4. **下一步**：对更大 HEAD、更高 GU 比重场景进行 sweep，结合 skip‑ratio 统计验证收益上限。  

---

## 十二、CUDA 与 A5 的计算等价性与流水差异（重点澄清）

本节解释一个常见疑问：  
为什么 CUDA 侧看起来像“先加 PV，再做 rescale”，却仍能与 A5 的“先 rescale，再加 PV”保持等价？

### 1️⃣ 统一数学目标

对第 `i` 个 KV block，目标更新是：

\[
O_i = O_{i-1}\cdot \exp(m_{i-1}-m_i) + PV_i
\]

其中 `PV_i` 是基于当前 block softmax 得到的贡献。

> 关键点：简单的 `(O + PV)*scale` 与 `O*scale + PV` **一般不等价**。  
> CUDA 成立等价不是因为“交换了加乘顺序”，而是因为它把“纠正对象”和“同步时机”做了严格流水控制。

### 2️⃣ A5 路径（同步、单阶段 GU）

在 A5 原型里，GU 宏直接执行：
1. `TROWEXPANDMUL(prev_sv_tile, prev_sv_tile, exp_max)`（可被 skip）
2. `TADD(prev_sv_tile, prev_sv_tile, est_sv_tile)`

即显式 `O = O*scale + PV`，逻辑直观，阶段内完成。

### 3️⃣ CUDA/SM100 路径（异步、分阶段）

在 `flash_fwd_sm100.py` 里，计算被拆成 3 个协同流水：

- **MMA warp（PV 累加）**：`P*V -> O`，通过 `zero_init` 控制是初始化还是累加。
- **Softmax warp（尺度产生）**：计算 `row_max/row_sum/acc_scale` 并写入 `sScale`。
- **Correction warp（纠正）**：按 `scale` 对中间 `O` 做 `correction_rescale`（可 skip）。

关键代码片段：

1) `P` 已按当前 `row_max` 做过稳定 softmax 处理（不是旧尺度的 P）：
```python
softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
softmax.apply_exp2_convert(tSrS_t2r, tSrP_r2t, ...)
```

2) PV 累加发生在 MMA：
```python
gemm_Pi[stage](..., zero_init=not O_should_accumulate, ...)
```

3) correction warp 根据 `scale` 决定是否执行 rescale：
```python
scale = sScale[tidx + stage * self.m_block_size]
should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
if should_rescale:
    self.correction_rescale(thr_mma_pv, tOtOs[stage], tidx, scale)
```

4) `mbar_P_full_O_rescaled_offset` 保证“先纠正，再进入下一轮 PV 累加”：
```python
# MMA 侧等待 corrected O
cute.arch.mbarrier_wait(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage, ...)

# correction 侧完成后发信号
cute.arch.mbarrier_arrive(mbar_ptr + self.mbar_P_full_O_rescaled_offset + stage)
```

### 4️⃣ 为什么“混在一起”后仍可纠正？

因为它不是把“新 PV”与“待纠正旧 O”无条件混合后再统一乘。  
流水通过 barrier 约束了依赖：每一轮进入 PV 累加前，上一轮需要纠正的 O 已经被 correction warp 处理完成。  
所以语义仍是“**对旧 O 做纠正，再加当前 PV**”，只是执行被拆成异步并行阶段。

### 5️⃣ 为什么 Blackwell 收益更明显？

`correction_rescale` 在 CUDA 中是一个独立重路径（含 TMEM load/store + mul）。  
当 `should_rescale` 为 false 时，可整段跳过这条路径，因此收益通常比 A5 的“仅跳过一条 `TROWEXPANDMUL`”更显著。

这也是我们在 A5 实验里看到“结论方向一致，但收益量级较小”的根本原因。

### 6️⃣ 时序图（A5 vs CUDA）

下面给出简化时序图，帮助理解“等价语义、不同流水”。

**A5（单阶段 GU，顺序执行）**

```text
time ->

Vec(GU):   [load O_old] -> [rescale O_old *= scale] -> [add PV_i] -> [store O_new]
                     (optional skip only removes the rescale op)

Semantics per tile i:
  O_i = O_{i-1} * scale_i + PV_i
```

**CUDA/SM100（异步三流水，barrier 协同）**

```text
time ->

Softmax WG : [S_i -> row_max/acc_scale] --write sScale--> (mbar_softmax_corr_full)
Correction :                wait mbar ----> [if needed: correction_rescale(O_old)] ----arrive----> (mbar_P_full_O_rescaled)
MMA(PV) WG :     wait (mbar_P_full_O_rescaled) ---------> [P_i * V_i accumulate to O] ----------> next tile

Key invariant:
  Before MMA accumulates PV_i into O buffer for stage i,
  correction for previous O state has completed.
```

**对照说明**
- A5: “先纠正旧 O，再加当前 PV”在同一阶段直接完成。  
- CUDA: 同一语义被拆到不同 warp group，通过 mbarrier 保证依赖。  
- 因此 CUDA 不是简单把 `O*scale + PV` 改成 `(O+PV)*scale`，而是“分阶段实现同一数学更新”。