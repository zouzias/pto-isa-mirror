# FA4 + BLASST A5 迁移汇报（PPT 直贴文案）

> 使用方式：每个 `##` 对应 1 页 PPT，标题和要点可直接粘贴。  
> 配图与表格已整理到 `assets/` 目录下。

---

## 0. 封面
**标题**  
FA4 与 BLASST 在 A5 NPU 上的迁移与性能评估

**副标题**  
从 GPU CUTE 优化范式到 NPU 架构适配的工程实践

**汇报人信息（可改）**  
项目：FlashAttention A5 Migration  
时间：2026-02

**本页配图**  
- `assets/00_cover/cover_arch_overview.svg`

---

## 1. 研究背景与目标
- 本轮目标是完成 FA4 与 BLASST 在 A5 NPU 上的可运行迁移，并形成可复现实验体系。  
- 迁移优先级为：编译通过 > 仿真可启动并正常退出 > 可观测统计 > 性能优化。  
- 核心挑战在于 GPU 优化路径与 A5 NPU 数据通路、同步协议、MMAD 语义存在差异。  
- 最终输出包括：机制实现、对比数据、瓶颈解释、后续优化方向。

---

## 2. FA4 CUTE 实现分析（GPU 侧）
- FA4 在 GPU 上依赖 CUTE/Tensor Core 的异步流水重叠，实现高吞吐注意力计算。  
- 关键优化之一是 skip-rescale：在 `exp(m_old-m_new)` 接近 1 时跳过 `O` 的重缩放。  
- 该机制成立前提是 MMA 累加器路径与寄存器驻留策略可以高效承载 `O + PV` 更新。  
- 因此 GPU 上优化是“算子语义 + 存储布局 + 异步调度”联合设计的结果。

---

## 3. 优化点分析（迁移视角）
- 可迁移：条件跳过逻辑（skip-rescale / skip-softmax）与统计闭环。  
- 半可迁移：部分访存优化（如 skip V-load）需要保持流水握手不变。  
- 不可直接平移：GPU MMA accumulator 风格矩阵累加语义。  
- 迁移策略采用“先功能与稳定，再做收益优化”的迭代方式。

---

## 4. TMEM Layout（GPU 与 A5 的约束差异）
- GPU 的 TMEM/SMEM/寄存器布局天然服务于 MMA 累加器语义与高重叠流水。  
- A5 侧需要显式管理 GM/L1/UB/L0C 之间的数据搬运，并遵守固定阶段同步协议。  
- 在当前公开 PTO 路径中，MMAD bias 暴露为行 bias 语义，不等价于任意矩阵累加。  
- 因此迁移重点不是“照搬布局”，而是“在 A5 约束下重建等价收益路径”。

**本页配图**  
- `assets/04_tmem_layout/tmem_vs_a5_layout.svg`

---

## 5. Skip Rescale 机制说明（FA4）
- 数学上，当 `exp(m_old-m_new)` 接近 1 时，`O` 的重缩放贡献接近空操作。  
- skip-rescale 通过阈值判定跳过该步骤，目标是降低 GU 阶段 vector 开销。  
- 为保证正确性与稳定性，需要保留必要的数据依赖与同步结构。  
- 该机制通常对 GU 压力明显的场景更有效。

---

## 6. Skip Rescale A5 迁移
- 在 `flash_atten_fa4` 中实现了 skip-rescale 开关、阈值控制和 force 模式。  
- 迁移过程中保持 QK/P/PV/GU 的固定握手关系，不破坏既有流水安全性。  
- 实验已验证：可编译、可启动仿真、可输出统计与时序日志。  
- 该实现提供了 OFF/ON/FORCE 三组对照，支持后续上界与敏感性分析。

**本页重点图**  
- `assets/06_skip_rescale_a5/fa4_skip_off_swimlane.svg`  
- `assets/06_skip_rescale_a5/fa4_skip_force_swimlane.svg`  
- `assets/06_skip_rescale_a5/fa4_skip_off_timeline.svg`  
- `assets/06_skip_rescale_a5/fa4_skip_force_timeline.svg`

**本页重点表（Excel）**  
- `assets/tables/case64_mechanisms_report.xlsx`（sheet: `summary`, `bandwidth`）

---

## 7. FA4 Performance Analysis（收益为何不明显）
- 实测显示 skip-rescale 有收益，但总收益有限。  
- 原因是当前 case 下主瓶颈更多落在 P 阶段（vector-bound），并非 GU 主导。  
- 因此“只优化 GU”难以显著拉动总周期，体现为局部改善大于整体改善。  
- 结论：优化命中瓶颈的匹配度，决定总收益上限。

---

## 8. 与 GPU CUTE 的差异（MMAD 语义）
- CUDA MMA 常见语义是 `D=A*B+C`，其中 `C` 可为同形状矩阵 fragment。  
- 当前 A5 PTO 公共 bias 路径语义为“MMAD + 行 bias”，接口约束为 `Rows==1`。  
- 因此不能直接将 `O_old(m*n)` 作为 MMAD bias 做一次性 `O_old + PV`。  
- 这是 GPU 与 A5 在算子融合能力表现差异的重要根因之一。

---

## 9. BLASST 的 CUDA 实现原理与分析
- BLASST 在 CUDA Hopper 的 warp-specialized 内核中实现 tile 级 skip-softmax，不是静态剪枝。  
- 判定核心是在线 softmax 的上界比较：当 `exp(local_max - global_max) < threshold` 时，该 tile 贡献可忽略。  
- 线程协作采用“warp 内归约 + warp-group 投票 + barrier 同步”：先 `__all_sync`，再 `atomicAnd` 写共享票，再 `named_barrier_wait` 汇合。  
- 一旦投票为 skip，CUDA 路径会直接跳过后续 `exp/sum`、`V load` 和 `S*V MMA`，属于“计算+访存”联合跳过。  

**本页讲解建议（30 秒）**  
- 重点不是某条指令，而是“判定-共识-早退”三步闭环。  
- 这也是 A5 迁移时要保留语义、但必须重写执行路径的原因。  

**本页配图**  
- `assets/09_cuda_blasst/blasst_cuda_principle.svg`

---

## 10. CUDA 机制对比：FA4 vs BLASST（表格）
| 对比维度 | FA4 CUDA（skip-rescale） | BLASST CUDA（skip-softmax） |
|---|---|---|
| 优化位置 | 输出在线更新路径中的 rescale 环节 | warp-specialized softmax tile 处理环节 |
| 核心思想 | 当 `exp(m_old-m_new)` 近似 1 时跳过重缩放 | 当 tile 贡献可忽略时直接跳过整块后续路径 |
| 判定粒度 | row/tile 更新条件 | tile 级 block 稀疏化决策 |
| 判定信号 | rescale 因子是否近似恒等 | `exp(local_max-global_max)` 是否低于阈值 |
| 同步协作 | 主要沿用 FA4 原有依赖 | `__all_sync` + `atomicAnd` + `named_barrier_wait` |
| 跳过范围 | 主要是 rescale 乘法步骤 | 可联合跳过 `exp/sum`、`V load`、`SxV MMA` |
| 潜在收益形态 | 当更新路径占比高时收益较稳 | 当低贡献 tile 比例高时收益更大 |
| 主要风险 | 一般较低（保守阈值下） | 阈值过激可能影响精度，需 profiling |
| 工程复杂度 | 相对局部 | 更高（判定-共识-早退全链路） |

**本页重点表**  
- `assets/tables/cuda_fa4_vs_blasst_mechanism_comparison.csv`  
- `assets/tables/case64_mechanisms_report.xlsx`（sheet: `cuda_mechanism_compare`）

---

## 11. BLASST Skip Softmax A5 迁移（目标与设计）
- 目标：在 A5 固定流水中实现 tile 级 skip-softmax，且保证编译/仿真/同步安全。  
- 设计：在 P 阶段做判定，支持 enable / force / ratio-force 多模式。  
- 策略：即使跳过也输出合法 no-op 语义，保证下游阶段接口行为稳定。  
- 可观测性：增加 skip 计数与比例输出，支持跨配置定量分析。

**本页重点图**  
- `assets/09_blasst_migration_design/blasst_design_flow.svg`  
- `assets/09_blasst_migration_design/blasst_skip_off_swimlane.svg`  
- `assets/09_blasst_migration_design/blasst_skip_ratio50_swimlane.svg`  
- `assets/09_blasst_migration_design/blasst_skip_force_swimlane.svg`

---

## 12. BLASST 实现细节（如何实现）
- P 阶段 skip 语义：`x_exp=0, local_sum=0, exp_max=1`，保持数值路径可控。  
- 增加 skip 统计：`p_total_cnt / p_skip_cnt`，host 侧汇总输出 skip ratio。  
- 实现 skip V-load：在跳过集合中减少 PV 阶段无效 `TLOAD(V)`。  
- 保持 MMAD 与同步协议不变，优先保证流水安全与可运行性。

**本页重点图**  
- `assets/10_blasst_impl/blasst_impl_flow.svg`  
- `assets/10_blasst_impl/blasst_skip_off_timeline.svg`  
- `assets/10_blasst_impl/blasst_skip_force_timeline.svg`  
- `assets/10_blasst_impl/blasst_skip_force_vload_timeline.svg`

**本页重点表（Excel）**  
- `assets/tables/case64_mechanisms_report.xlsx`（sheet: `summary`, `bandwidth`）

---

## 13. BLASST Performance Analysis
- 在对齐 case 下，skip-softmax 与 force/ratio 模式带来显著的 vector 计算降幅。  
- skip-vload 进一步降低了部分无效访存，带宽相关指标有改善趋势。  
- 但总体收益仍受“主瓶颈位置 + 跳过覆盖率 + 阈值策略”共同约束。  
- 结论：BLASST 在 A5 上具备优化潜力，但需分场景调参和机制组合。

**本页重点表（Excel）**  
- `assets/tables/case64_mechanisms_report.xlsx`

---

## 14. 收益点总结
- 机制层：已完成 FA4/BLASST 关键跳过机制在 A5 的工程可用迁移。  
- 数据层：已形成可复现的对比数据链路（timeline/swimlane/phase/summary）。  
- 方法层：验证了“先稳态迁移再收益优化”的路径可行。  
- 价值层：为后续面向真实模型 workload 的策略优化打下基础。

---

## 15. Concern：Sparse Attention 阈值 profiling
- BLASST skip-softmax 本质属于稀疏化思路，收益与阈值强相关。  
- 阈值过激会带来数值风险，阈值保守则收益不足。  
- 需要建立 workload-aware profiling：按 layer/head/seq/batch 做分桶评估。  
- 建议输出阈值-收益-误差的 Pareto 曲线，形成可部署策略。

---

## 16. Takeaways（NPU 视角）
- 面向 GPU 的 FA 优化往往依赖 warp-specialized 的细粒度 pipeline 设计；在 FA4 持久化内核中，重点是通过固定角色分工与时序编排提升 MMA 路径与 softmax/vector 路径的重叠效率，而不是运行时动态重分配算力资源；因此在 A5 侧单独迁移某个局部技巧（例如 skip-rescale）通常只能拿到有限收益。  
- BLASST skip-softmax 属于稀疏化机制，收益与阈值设定、logit 分布和 tile 顺序强相关；它在合适分布下收益更大，但机制通用性弱于 FA4 类“保守近似”优化，必须依赖 profiling 才能稳定落地。  
- 稀疏化落地不只是“跳过多少”，还涉及执行负载均衡与流水稳定性：skip 会改变阶段负载形态，可能引入新的阶段不均衡，这对 NPU 固定流水尤其关键。  
- 针对当前 A5 的 vector-bound 现象，优先级应放在 compute-P（softmax）阶段的细粒度优化，而不仅是 PTO 宏指令层面的开关；后续需要更靠近调度/切分/数据布局层面的优化策略。  
- 方法论上应坚持“瓶颈定位 -> 机制选择 -> 分布校准 -> 稳定性验证”的闭环：先确认主瓶颈阶段，再设计针对性机制，最后用统一指标体系（cycles + busy + bandwidth + skip ratio）做跨配置验证。

