# A5 FA4 + BLASST PPT Kit

## 内容说明
- `slides_cn_ppt_copy.md`：每页可直接粘贴到 PPT 的中文文案。
- `assets/`：重点插图与表格文件（SVG/CSV/XLSX）。

## 重点页面资源
- `## 0. 封面`  
  - `assets/00_cover/cover_arch_overview.svg`
- `## 4. TMEM Layout`  
  - `assets/04_tmem_layout/tmem_vs_a5_layout.svg`
- `## 6. Skip Rescale A5`  
  - `assets/06_skip_rescale_a5/*.svg`
- `## 9. BLASST 的 CUDA 实现原理与分析`  
  - `assets/09_cuda_blasst/blasst_cuda_principle.svg`
  - 文案页（`slides_cn_ppt_copy.md`），可配合 `skip_softmax/blasst.md` 代码/机制说明讲解
- `## 10. FA4 vs BLASST 机制对比（表格）`  
  - `assets/tables/cuda_fa4_vs_blasst_mechanism_comparison.csv`
  - `assets/tables/case64_mechanisms_report.xlsx`（sheet: `cuda_mechanism_compare`）
- `## 11. BLASST Skip Softmax A5 迁移（目标与设计）`  
  - `assets/09_blasst_migration_design/*.svg`
- `## 12. BLASST 实现细节（如何实现）`  
  - `assets/10_blasst_impl/*.svg`

## 表格（Excel）
- `assets/tables/case64_mechanisms_report.xlsx`
- `assets/tables/cuda_fa4_vs_blasst_mechanism_comparison.csv`
- `assets/tables/fa4_vs_blasst_mechanism_comparison.csv`（A5 实现对比，保留）
- `assets/tables/case64_all_mechanisms_summary.csv`
- `assets/tables/case64_all_mechanisms_bandwidth.csv`
- `assets/tables/case64_bandwidth_key_metrics.csv`（带宽关键指标精简版）

## 带宽精简指标说明（main datapath）
- `mte2_busy_total` / `mte3_busy_total`：主搬运通路 busy 规模（load/store 主路径）
- `mte2_busy_pct_of_kernel` / `mte3_busy_pct_of_kernel`：相对内核总周期的占比
- `ub_rd_B_per_kernel_cycle` / `ub_wr_B_per_kernel_cycle` / `ub_total_B_per_kernel_cycle`：UB 读写与总带宽强度
- `gain_mte2_busy_total_pct_vs_family_off` / `gain_ub_total_B_per_kernel_cycle_pct_vs_family_off`：对各 family-off 基线的相对变化

