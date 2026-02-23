以下是一个**简明的 BLASST Skip‑Softmax 机制与实现说明文档**，基于你提供的 NVIDIA 源代码与上下文分析。

---

# 🧩 BLASST Skip‑Softmax 机制概览

**BLASST** （Block‑Level Attention Skip Softmax Technique）是一种在 Hopper Warp‑Specialized 内核中实现的**动态软跳过机制**，旨在在注意力计算中识别和跳过数值可忽略的块（tile），以减少算力与访存负担。

具体体现在 `fmha::ws::Compute` 内核模板及 `fmha::ws::Softmax_base` 函数中，通过在 softmax 计算阶段引入**阈值判断与warp‑group级投票机制**，决定是否跳过整个 softmax 子块的计算。

---

## 1️⃣ 设计思路与目标

### 🎯 目标
- 减少在 Flash‑Attention 型内核中 softmax 部分的冗余计算；
- 当 \( e^{(m_{\text{block}} - m_{\text{global}})} \) 很小（数值几乎为0）时跳过；
- 保持数值稳定与精度；
- 支持 warp‑specialized 并行流水（Producer/Consumer 双 warp group）；

---

## 2️⃣ 判定逻辑与原理

在 在线 softmax 算法中维护：
\[
m_i = \text{running max}, \quad s_i = \sum_j e^{x_j - m_i}
\]

对于当前处理块（tile），BLASST 使用阈值判断：

\[
\exp(m_{\text{local}} - m_{\text{global}}) < \text{skip\_threshold}
\]

若满足条件，则此 tile 的贡献在 softmax 归一化后几乎为 0，允许跳过。

- 阈值由运行时参数 `skip_softmax_threshold = params.skip_softmax_threshold_scale_factor / actual_kv_seqlen` 动态缩放；
- 实际比较逻辑在 `Softmax_base::compute_and_update_scale()` 中实现。

---

## 3️⃣ 实现机制（Softmax 部分）

### 🔍 关键路径：
文件：`fmha/softmax.h`

函数：  
```cpp
template <bool IS_FIRST_COL>
bool Softmax_base::compute_and_update_scale(
    float (&global_max)[M], float (&global_sum)[M], uint32_t* skip_softmax_vote)
```

### ⚙️ 主要流程：
1. **计算当前tile行最大值**
   ```cpp
   local_max_[mi] = max(elt_[mi]);
   ```

2. **与全局最大值比较是否可跳过**
   ```cpp
   skip &= expf(local_max_[mi] - global_max[mi]) < skip_softmax_threshold;
   ```

3. **warp内归约：**
   - 使用 `__all_sync()` 判断整个warp内是否均应跳过；
   - 每个warp 的leader 用 `atomicAnd()` 投票写入 `skip_softmax_vote` ；
   - WarpGroup间同步通过 `named_barrier_wait()`；

4. **warp_group 同步后检查结果：**
   - 若 `skip_softmax_vote == 1` → 跳过后续 exp/sum/V 加载；
   - 否则继续 softmax 计算与归一化累积。

5. **统计（DEBUG选项）**
   ```cpp
   atomicAdd(params.skip_softmax_total_blocks, total_blocks);
   atomicAdd(params.skip_softmax_skipped_blocks, skipped_blocks);
   ```

---

## 4️⃣ 内核级整合（Compute 部分）

文件：`fmha/ws/compute.h`  
函数：`Compute::compute_single_tile()`

### ✅ 执行流程：

| 阶段 | 操作 | 含 skip 逻辑 |
|------|------|---------------|
| (1) | Q×K′ → logits | 无 |
| (2) | unpack → apply mask/Alibi | 无 |
| (3) | **`softmax.compute_and_update_scale()`** | ✅ 判定是否跳过 |
| (4) | 若跳过 → 提前 return；跳过 V load 及 S×V MMA |
| (5) | 若未跳过 → 继续 softmax.pack 和 BMM2 (S×V) 计算 |

在 warp‑group 交叠流水结构中，  
每个 kv tile 先由 wg0 执行 QK ，再由 wg1 进行 softmax 与 SV；  
skip 机制内嵌在 softmax 阶段，确保跳过的tile不会进入 V 加载与 MMA。

---

## 5️⃣ 同步与线程协作细节

### 🧵 （1）Warp内与WarpGroup间通信：

- Warp内 ： `__all_sync(0xffffffff, skip)` 归约；
- WarpGroup间 ：  
  - `atomicAnd(skip_softmax_vote, skip)` 投票；
  - `named_barrier_wait(SKIP_SOFTMAX_BARRIER + id, 128)` 保证同步；
  - 主warp读取 vote 结果。

### 🪩 （2）Barrier 与 Mutex：

- 通过定义常量 `SKIP_SOFTMAX_BARRIER_ID` 指定；
- 确保 warp_group 间跳过决策一致；
- 允许与 HGMMA/QGMMA 互斥交叠执行。

---

## 6️⃣ 性能与结构特征

| 类别 | 说明 |
|------|------|
| **粒度** | Tile 级（16–64 列） |
| **触发条件** | \( e^{\Delta m} < \text{threshold} \) |
| **计算节省** | 跳过 expf / reduce / V‑load / MMA |
| **同步开销** | 极低（仅warp内布尔归约与atomic） |
| **数值安全** | 阈值与实际序列长度成反比，稳定性强 |
| **统计支持** | `SKIP_SOFTMAX_STAT` 可记录跳过比率 |

---

## 7️⃣ 对比 FlashAttention‑4 的 Rescale Skip

| 项目 | FA4 Rescale Skip | BLASST Skip‑Softmax |
|------|------------------|---------------------|
| 作用层 | 行内 softmax rescale | 块级 softmax 跳过 |
| 粒度 | row‑wise | tile‑wise (16×64) |
| 控制逻辑 | 静态 constexpr | 动态 runtime 判定 |
| 跳过代价 | exp/reduce 操作 | exp + V load + MMA 全部 |
| 对性能贡献 | 减少 fp 运算 | 提升 访存 与 compute 效率 |

---

## 🔚 总结

> **BLASST Skip‑Softmax 机制**  
> 在 Hopper Warp‑Specialized FlashAttention 实现中，引入 **动态tile级跳过判断**，由warp并行投票与barrier同步协作控制。  
> 当局部 QK 块的logit上界远低于前序块（贡献可忽略）时，  
> 整个块的 softmax 与 SV 操作被跳过。  
>  
> ➜ 优点：极低开销实现稀疏化跳过。  
> ➜ 特点：与FA4 rescale skip 不同，属于宏粒度动态pruning机制。  
> ➜ 硬件耦合：深度融合Hopper WarpGroup 架构、mbarrier 和GMMA 流水。

---

# 8️⃣ A5 迁移方案与落地实现（flash_atten_blasst）

## 8.1 迁移目标与边界

目标是在 A5 的固定 QK/P/PV/GU 流水中加入 BLASST 风格的 tile 级 skip-softmax，优先保证：

- 编译可通过；
- 流水同步（`TSync`/`SET_FLAG`/`WAIT_FLAG`）不被破坏；
- 仿真可启动；
- 可统计 skip 数量与比例，便于后续性能评估。

本轮实现采用**最小侵入原型**：先在 P 阶段实现 skip 判定与 `x_exp` 置零，不改变已有跨阶段握手拓扑；后续再迭代到 PV/GU 的更激进跳过。

## 8.2 A5 适配设计（完整方案）

### 1) 判定粒度

- CUDA BLASST 的判定是 tile 级、基于 `exp(local_max - global_max) < threshold`；
- A5 原型采用等价的 raw-logit 差值形式近似：  
  `local_max - running_global_max <= delta_threshold`（`delta_threshold < 0`）；
- 判定在 P 阶段 `softmax_not_init` 路径执行，保持首 tile（init tile）不参与 skip。

### 2) 同步与流水不变性

- 保留 `qk2smSync` / `sm2pvSync` / `pv2guSync` 的 record/wait/free 节奏；
- 即使 tile 被 skip，P 阶段依然输出合法 `x_exp`（全零）并完成下游握手；
- PV/GU 阶段读到的该 tile 贡献自然接近 no-op（`P=0`）。

### 3) 数值语义

当判定为 skip 时：

- `x_exp = 0`（该 tile 不参与 `SV`）；
- `local_sum = 0`；
- `exp_max = 1`（保证 GU 的 rescale 不引入额外缩放）；
- running `global_max/global_sum` 保持上一 tile 的状态（近似忽略该 tile 贡献）。

### 4) 参数化与可控性

新增 3 个 CMake/编译宏：

- `BLASST_SKIP_SOFTMAX_ENABLE`：总开关；
- `BLASST_SKIP_SOFTMAX_FORCE`：强制跳过（用于上界/联调）；
- `BLASST_SKIP_SOFTMAX_DELTA`：判定阈值（raw-logit delta）。
- `BLASST_SKIP_SOFTMAX_FORCE_RATIO`：按比例强制跳过（`[0,1]`，用于流水联调；例如 `0.5` 约 50% tile 被跳过）。

### 5) 统计与可观测

- 统计 `p_total_cnt`（非 init tile 计数）与 `p_skip_cnt`；
- 写入 profile entry 的扩展槽位；
- host 侧汇总并打印 skip ratio。

## 8.3 本次代码落地（已实现）

### 目录与基线

- 从 `kernels/manual/a5/flash_atten` 复制出 `kernels/manual/a5/flash_atten_blasst`；
- 复制时排除 `build/`，避免大体积仿真产物复制。

### 修改文件

- `kernels/manual/a5/flash_atten_blasst/pto_macro_fa_softmax.hpp`
  - 增加 `BLASST_SKIP_SOFTMAX_*` 宏；
  - 新增 `blasst_should_skip_softmax(...)`；
  - `pto_macro_fa_softmax(...)` 改为返回 `bool`（是否 skip）；
  - 在 not-init 路径加入 skip 处理（`x_exp=0/local_sum=0/exp_max=1`）。
  - 新增按比例强制跳过逻辑（基于 `s0_index/s1_index` 的哈希分桶，近似随机覆盖）。

- `kernels/manual/a5/flash_atten_blasst/fa_performance_kernel.cpp`
  - `compute_p(...)` 增加 `p_total_cnt/p_skip_cnt` 引用参数；
  - 接收 `pto_macro_fa_softmax(...)` 的 skip 返回值并统计；
  - `runTFA(...)` 增加计数器并写入 profile。

- `kernels/manual/a5/flash_atten_blasst/main.cpp`
  - 汇总 profile 中的 P skip 统计并打印：
    `[SUMMARY] BLASST skip-softmax: skip X / Y (Z%)`。

- `kernels/manual/a5/flash_atten_blasst/CMakeLists.txt`
  - 新增 `BLASST_SKIP_SOFTMAX_ENABLE/FORCE/DELTA` 选项并透传到编译宏。

- `kernels/manual/a5/flash_atten_blasst/run.sh`
  - 新增环境变量读取与打印；
  - 新增向 CMake 透传上述 skip 参数。

## 8.4 推荐验证步骤

在 `kernels/manual/a5/flash_atten_blasst` 下：

1. baseline（关闭 skip）
   - `BLASST_SKIP_SOFTMAX_ENABLE=OFF ./run.sh ...`
2. 原型开启（非强制）
   - `BLASST_SKIP_SOFTMAX_ENABLE=ON BLASST_SKIP_SOFTMAX_FORCE=OFF BLASST_SKIP_SOFTMAX_DELTA=-8.0f ./run.sh ...`
3. 强制跳过（联调/上界）
   - `BLASST_SKIP_SOFTMAX_ENABLE=ON BLASST_SKIP_SOFTMAX_FORCE=ON ./run.sh ...`

关注点：

- 编译成功；
- 仿真启动成功；
- 输出包含 skip summary；
- 与 baseline 的数值偏差在可接受范围内（原型阶段可先容忍偏差并以稳定性优先）。

## 8.5 下一步增强（建议）

- 将 skip 决策共享到 PV 阶段，避免无效 `S*V` 计算；
- 增加按 tile 的 skip bitmap（UB/GM）用于跨阶段更精细 no-op；
- 与 `phase_analysis.py` 联动，量化 vector 指令周期与 stall 的改善比例；
- 对阈值进行按 `HEAD_SIZE` 标准化标定（向 `exp(local-global)` 判据对齐）。

## 8.6 BT Buffer / MMAD 疑问澄清（结合 Ascend 文档与本项目代码）

本节总结本轮关键疑问：

- 文档示例里 `cSize = m * n`，是否意味着 MMAD bias 也可以是 `m * n` 全矩阵？
- 如果 BT Buffer 是“矩阵”，为什么当前实现里无法把 `O_old` 直接作为 MMAD bias 做 `O_old + PV`？
- 为什么 CUDA 的 FlashAttention 可以把 `O_old + PV` 融合在 MMA，而 A5 侧当前做不到同样语义？

### 1) 文档示例中 `C` 是 `m*n`，但 bias 仍是 `1*n` 行向量

Ascend 文档“通过 BT Buffer 实现高效的 bias 计算”示例里：

- 输出 `C` 的确是 `m*n`；
- 但 bias 缓冲按 `n` 分配与搬运（不是 `m*n`）；
- `SplitBias` 到 BT 的拷贝参数为 `{1, ...}`，对应单行语义；
- 随后 `Mmad(c1Local, a2Local, b2Local, bias2Local, mmadParams)` 使用该 bias 参与计算。

结论：该实践展示的是“`1*n` 行 bias 广播到 `m*n` 输出”，而不是“`m*n` 全矩阵 bias”。

### 2) BT Buffer 资源形态 vs 指令/接口消费语义

这里要区分两层概念：

- **资源层**：BT Buffer（C2）在硬件上是表/块状存储资源；
- **接口语义层**：当前 AscendC/PTO 暴露的 `TMatmulBias` 用法是行 bias 语义。

在本项目 A5 PTO 头文件中（`include/pto/npu/a5/TMatmul.hpp`），`TMATMUL_BIAS_IMPL` 对 bias 有编译期约束：

- `TileBias::Loc == TileType::Bias`
- `TileBias::Rows == 1`
- `TileBias::isRowMajor`

这与文档示例是一致的：当前可用路径是“行 bias 融合 MMAD”，不是“任意 `m*n` bias tile”。

### 3) 对当前 FlashAttention A5 实现的直接影响

- `O_old` 在当前实现中是作为输出累积路径的一部分存在，不是通过 BT 的全矩阵 bias 注入；
- 因为 bias 语义受限为 `1*n`，无法直接把 `O_old(m*n)` 作为 MMAD bias 一次性完成 `O_old + PV`；
- 因此当前 BLASST/FA4 侧仍采用：
  - PV 通过 MMAD 计算；
  - `O_old` 相关更新通过既有 GU / 向量路径完成；
  - 结合 skip-softmax、skip-rescale、skip-vload 减少冗余开销。

### 4) 与 CUDA 的差异（为什么 CUDA 看起来“可以”）

- CUDA Tensor Core 的 MMA 常见语义是 `D = A * B + C`，`C` 为同形状矩阵 fragment（寄存器累加器）；
- 在 FlashAttention CUDA 内核中，`O` 常驻 MMA accumulator，下一轮 `P*V` 可直接累加；
- Ascend 当前这条公开 PTO bias 路径是“MMAD + 行 bias（BT）”，不等价于“MMAD + 任意矩阵 C”。

因此，两边“都在做矩阵乘加”，但可直接利用的 `+C` 语义和数据通路不同，这就是当前行为差异的根因。