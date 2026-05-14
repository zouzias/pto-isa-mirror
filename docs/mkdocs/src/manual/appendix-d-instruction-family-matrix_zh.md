# 附录 D. 指令族矩阵

## D.1 范围

本附录由 `docs/isa/manifest.yaml` 自动生成，用于给出 PTO 虚拟指令族的源同步矩阵。

## D.2 覆盖统计

| 分类 | 指令数量 |
|---|---:|
| 同步 | 1 |
| 手动 / 资源绑定 | 4 |
| 逐元素（Tile-Tile） | 29 |
| Tile-标量 / Tile-立即数 | 20 |
| 轴归约 / 扩展 | 24 |
| 内存（GM <-> Tile） | 6 |
| 矩阵乘 | 8 |
| 数据搬运 / 布局 | 13 |
| 复杂指令 | 15 |
| 通信 | 11 |
| 总计 | 129 |

## D.3 头文件同步状态

- 头文件清单来源：`include/pto/common/pto_instr.hpp`（115 个唯一指令 API）
- Manifest 清单来源：`docs/isa/manifest.yaml`（115 条目）
- 头文件有但 manifest 缺失：无
- manifest 有但头文件缺失：无

## D.4 指令族矩阵

| 分类 | 指令 | 图示模板 | 操作数契约 | 语义页面 |
|---|---|---|---|---|
| 同步 | [TSYNC](/docs/isa/tile/ops/sync-and-config/tsync_zh.md) | `sync` | `producer, consumer` | `docs/isa/TSYNC_zh.md` |
| 手动 / 资源绑定 | [TASSIGN](/docs/isa/tile/ops/sync-and-config/tassign_zh.md) | `config` | `config, state` | `docs/isa/TASSIGN_zh.md` |
| 手动 / 资源绑定 | [SETFMATRIX](/docs/isa/tile/ops/sync-and-config/setfmatrix.md) | `config` | `config, state` | `docs/isa/SETFMATRIX_zh.md` |
| 逐元素（Tile-Tile） | [TADD](/docs/isa/tile/ops/elementwise-tile-tile/tadd_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TADD_zh.md` |
| 逐元素（Tile-Tile） | [TABS](/docs/isa/tile/ops/elementwise-tile-tile/tabs_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TABS_zh.md` |
| 逐元素（Tile-Tile） | [TAND](/docs/isa/tile/ops/elementwise-tile-tile/tand_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TAND_zh.md` |
| 逐元素（Tile-Tile） | [TOR](/docs/isa/tile/ops/elementwise-tile-tile/tor_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TOR_zh.md` |
| 逐元素（Tile-Tile） | [TSUB](/docs/isa/tile/ops/elementwise-tile-tile/tsub_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TSUB_zh.md` |
| 逐元素（Tile-Tile） | [TMUL](/docs/isa/tile/ops/elementwise-tile-tile/tmul_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TMUL_zh.md` |
| 逐元素（Tile-Tile） | [TMIN](/docs/isa/tile/ops/elementwise-tile-tile/tmin_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TMIN_zh.md` |
| 逐元素（Tile-Tile） | [TMAX](/docs/isa/tile/ops/elementwise-tile-tile/tmax_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TMAX_zh.md` |
| 逐元素（Tile-Tile） | [TCMP](/docs/isa/tile/ops/elementwise-tile-tile/tcmp_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TCMP_zh.md` |
| 逐元素（Tile-Tile） | [TDIV](/docs/isa/tile/ops/elementwise-tile-tile/tdiv_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TDIV_zh.md` |
| 逐元素（Tile-Tile） | [TSHL](/docs/isa/tile/ops/elementwise-tile-tile/tshl_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TSHL_zh.md` |
| 逐元素（Tile-Tile） | [TSHR](/docs/isa/tile/ops/elementwise-tile-tile/tshr_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TSHR_zh.md` |
| 逐元素（Tile-Tile） | [TXOR](/docs/isa/tile/ops/elementwise-tile-tile/txor_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TXOR_zh.md` |
| 逐元素（Tile-Tile） | [TLOG](/docs/isa/tile/ops/elementwise-tile-tile/tlog_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TLOG_zh.md` |
| 逐元素（Tile-Tile） | [TRECIP](/docs/isa/tile/ops/elementwise-tile-tile/trecip_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TRECIP_zh.md` |
| 逐元素（Tile-Tile） | [TPRELU](/docs/isa/tile/ops/elementwise-tile-tile/tprelu_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TPRELU_zh.md` |
| 逐元素（Tile-Tile） | [TADDC](/docs/isa/tile/ops/elementwise-tile-tile/taddc_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TADDC_zh.md` |
| 逐元素（Tile-Tile） | [TSUBC](/docs/isa/tile/ops/elementwise-tile-tile/tsubc_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TSUBC_zh.md` |
| 逐元素（Tile-Tile） | [TCVT](/docs/isa/tile/ops/elementwise-tile-tile/tcvt_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TCVT_zh.md` |
| 逐元素（Tile-Tile） | [TSEL](/docs/isa/tile/ops/elementwise-tile-tile/tsel_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TSEL_zh.md` |
| 逐元素（Tile-Tile） | [TRSQRT](/docs/isa/tile/ops/elementwise-tile-tile/trsqrt_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TRSQRT_zh.md` |
| 逐元素（Tile-Tile） | [TSQRT](/docs/isa/tile/ops/elementwise-tile-tile/tsqrt_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TSQRT_zh.md` |
| 逐元素（Tile-Tile） | [TEXP](/docs/isa/tile/ops/elementwise-tile-tile/texp_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TEXP_zh.md` |
| 逐元素（Tile-Tile） | [TNOT](/docs/isa/tile/ops/elementwise-tile-tile/tnot_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TNOT_zh.md` |
| 逐元素（Tile-Tile） | [TRELU](/docs/isa/tile/ops/elementwise-tile-tile/trelu_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TRELU_zh.md` |
| 逐元素（Tile-Tile） | [TNEG](/docs/isa/tile/ops/elementwise-tile-tile/tneg_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TNEG_zh.md` |
| 逐元素（Tile-Tile） | [TREM](/docs/isa/tile/ops/elementwise-tile-tile/trem_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TREM_zh.md` |
| 逐元素（Tile-Tile） | [TFMOD](/docs/isa/tile/ops/elementwise-tile-tile/tfmod_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TFMOD_zh.md` |
| 逐元素（Tile-Tile） | [TPOW](/docs/isa/tile/ops/elementwise-tile-tile/tpow_zh.md) | `elementwise` | `dst, src0, src1` | `docs/isa/TPOW_zh.md` |
| Tile-标量 / Tile-立即数 | [TEXPANDS](/docs/isa/tile/ops/tile-scalar-and-immediate/texpands_zh.md) | `scalar` | `dst, src, scalar` | `docs/isa/TEXPANDS_zh.md` |
| Tile-标量 / Tile-立即数 | [TCMPS](/docs/isa/tile/ops/tile-scalar-and-immediate/tcmps_zh.md) | `scalar` | `dst, src, scalar` | `docs/isa/TCMPS_zh.md` |
| Tile-标量 / Tile-立即数 | [TSELS](/docs/isa/tile/ops/tile-scalar-and-immediate/tsels_zh.md) | `scalar` | `dst, src, scalar` | `docs/isa/TSELS_zh.md` |
| Tile-标量 / Tile-立即数 | [TMINS](/docs/isa/tile/ops/tile-scalar-and-immediate/tmins_zh.md) | `scalar` | `dst, src, scalar` | `docs/isa/TMINS_zh.md` |
| Tile-标量 / Tile-立即数 | [TADDS](/docs/isa/tile/ops/tile-scalar-and-immediate/tadds_zh.md) | `scalar` | `dst, src, scalar` | `docs/isa/TADDS_zh.md` |
| Tile-标量 / Tile-立即数 | [TSUBS](/docs/isa/tile/ops/tile-scalar-and-immediate/tsubs_zh.md) | `scalar` | `dst, src, scalar` | `docs/isa/TSUBS_zh.md` |
| Tile-标量 / Tile-立即数 | [TDIVS](/docs/isa/tile/ops/tile-scalar-and-immediate/tdivs_zh.md) | `scalar` | `dst, src, scalar` | `docs/isa/TDIVS_zh.md` |
| Tile-标量 / Tile-立即数 | [TMULS](/docs/isa/tile/ops/tile-scalar-and-immediate/tmuls_zh.md) | `scalar` | `dst, src, scalar` | `docs/isa/TMULS_zh.md` |
| Tile-标量 / Tile-立即数 | [TFMODS](/docs/isa/tile/ops/tile-scalar-and-immediate/tfmods_zh.md) | `scalar` | `dst, src, scalar` | `docs/isa/TFMODS_zh.md` |
| Tile-标量 / Tile-立即数 | [TREMS](/docs/isa/tile/ops/tile-scalar-and-immediate/trems_zh.md) | `scalar` | `dst, src, scalar` | `docs/isa/TREMS_zh.md` |
| Tile-标量 / Tile-立即数 | [TMAXS](/docs/isa/tile/ops/tile-scalar-and-immediate/tmaxs_zh.md) | `scalar` | `dst, src, scalar` | `docs/isa/TMAXS_zh.md` |
| Tile-标量 / Tile-立即数 | [TANDS](/docs/isa/tile/ops/tile-scalar-and-immediate/tands_zh.md) | `scalar` | `dst, src, scalar` | `docs/isa/TANDS_zh.md` |
| Tile-标量 / Tile-立即数 | [TORS](/docs/isa/tile/ops/tile-scalar-and-immediate/tors_zh.md) | `scalar` | `dst, src, scalar` | `docs/isa/TORS_zh.md` |
| Tile-标量 / Tile-立即数 | [TSHLS](/docs/isa/tile/ops/tile-scalar-and-immediate/tshls_zh.md) | `scalar` | `dst, src, scalar` | `docs/isa/TSHLS_zh.md` |
| Tile-标量 / Tile-立即数 | [TSHRS](/docs/isa/tile/ops/tile-scalar-and-immediate/tshrs_zh.md) | `scalar` | `dst, src, scalar` | `docs/isa/TSHRS_zh.md` |
| Tile-标量 / Tile-立即数 | [TXORS](/docs/isa/tile/ops/tile-scalar-and-immediate/txors_zh.md) | `scalar` | `dst, src, scalar` | `docs/isa/TXORS_zh.md` |
| Tile-标量 / Tile-立即数 | [TLRELU](/docs/isa/tile/ops/tile-scalar-and-immediate/tlrelu_zh.md) | `scalar` | `dst, src, scalar` | `docs/isa/TLRELU_zh.md` |
| Tile-标量 / Tile-立即数 | [TADDSC](/docs/isa/tile/ops/tile-scalar-and-immediate/taddsc_zh.md) | `scalar` | `dst, src, scalar` | `docs/isa/TADDSC_zh.md` |
| Tile-标量 / Tile-立即数 | [TSUBSC](/docs/isa/tile/ops/tile-scalar-and-immediate/tsubsc_zh.md) | `scalar` | `dst, src, scalar` | `docs/isa/TSUBSC_zh.md` |
| Tile-标量 / Tile-立即数 | [TPOWS](/docs/isa/tile/ops/tile-scalar-and-immediate/tpows_zh.md) | `scalar` | `dst, src, scalar` | `docs/isa/TPOWS_zh.md` |
| 轴归约 / 扩展 | [TROWSUM](/docs/isa/tile/ops/reduce-and-expand/trowsum_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWSUM_zh.md` |
| 轴归约 / 扩展 | [TROWPROD](/docs/isa/tile/ops/reduce-and-expand/trowprod_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWPROD_zh.md` |
| 轴归约 / 扩展 | [TCOLSUM](/docs/isa/tile/ops/reduce-and-expand/tcolsum_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLSUM_zh.md` |
| 轴归约 / 扩展 | [TCOLPROD](/docs/isa/tile/ops/reduce-and-expand/tcolprod_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLPROD_zh.md` |
| 轴归约 / 扩展 | [TCOLMAX](/docs/isa/tile/ops/reduce-and-expand/tcolmax_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLMAX_zh.md` |
| 轴归约 / 扩展 | [TROWMAX](/docs/isa/tile/ops/reduce-and-expand/trowmax_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWMAX_zh.md` |
| 轴归约 / 扩展 | [TROWMIN](/docs/isa/tile/ops/reduce-and-expand/trowmin_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWMIN_zh.md` |
| 轴归约 / 扩展 | [TCOLARGMAX](/docs/isa/tile/ops/reduce-and-expand/tcolargmax_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLARGMAX_zh.md` |
| 轴归约 / 扩展 | [TCOLARGMIN](/docs/isa/tile/ops/reduce-and-expand/tcolargmin_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLARGMIN_zh.md` |
| 轴归约 / 扩展 | [TROWEXPAND](/docs/isa/tile/ops/reduce-and-expand/trowexpand_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPAND_zh.md` |
| 轴归约 / 扩展 | [TROWEXPANDDIV](/docs/isa/tile/ops/reduce-and-expand/trowexpanddiv_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDDIV_zh.md` |
| 轴归约 / 扩展 | [TROWEXPANDMUL](/docs/isa/tile/ops/reduce-and-expand/trowexpandmul_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDMUL_zh.md` |
| 轴归约 / 扩展 | [TROWEXPANDSUB](/docs/isa/tile/ops/reduce-and-expand/trowexpandsub_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDSUB_zh.md` |
| 轴归约 / 扩展 | [TROWEXPANDADD](/docs/isa/tile/ops/reduce-and-expand/trowexpandadd_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDADD_zh.md` |
| 轴归约 / 扩展 | [TROWEXPANDMAX](/docs/isa/tile/ops/reduce-and-expand/trowexpandmax_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDMAX_zh.md` |
| 轴归约 / 扩展 | [TROWEXPANDMIN](/docs/isa/tile/ops/reduce-and-expand/trowexpandmin_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDMIN_zh.md` |
| 轴归约 / 扩展 | [TROWEXPANDEXPDIF](/docs/isa/tile/ops/reduce-and-expand/trowexpandexpdif_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TROWEXPANDEXPDIF_zh.md` |
| 轴归约 / 扩展 | [TCOLMIN](/docs/isa/tile/ops/reduce-and-expand/tcolmin_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLMIN_zh.md` |
| 轴归约 / 扩展 | [TCOLEXPAND](/docs/isa/tile/ops/reduce-and-expand/tcolexpand_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPAND_zh.md` |
| 轴归约 / 扩展 | [TCOLEXPANDDIV](/docs/isa/tile/ops/reduce-and-expand/tcolexpanddiv_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDDIV_zh.md` |
| 轴归约 / 扩展 | [TCOLEXPANDMUL](/docs/isa/tile/ops/reduce-and-expand/tcolexpandmul_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDMUL_zh.md` |
| 轴归约 / 扩展 | [TCOLEXPANDADD](/docs/isa/tile/ops/reduce-and-expand/tcolexpandadd_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDADD_zh.md` |
| 轴归约 / 扩展 | [TCOLEXPANDMAX](/docs/isa/tile/ops/reduce-and-expand/tcolexpandmax_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDMAX_zh.md` |
| 轴归约 / 扩展 | [TCOLEXPANDMIN](/docs/isa/tile/ops/reduce-and-expand/tcolexpandmin_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDMIN_zh.md` |
| 轴归约 / 扩展 | [TCOLEXPANDSUB](/docs/isa/tile/ops/reduce-and-expand/tcolexpandsub_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDSUB_zh.md` |
| 轴归约 / 扩展 | [TCOLEXPANDEXPDIF](/docs/isa/tile/ops/reduce-and-expand/tcolexpandexpdif_zh.md) | `reduce_expand` | `dst, src` | `docs/isa/TCOLEXPANDEXPDIF_zh.md` |
| 内存（GM <-> Tile） | [TLOAD](/docs/isa/tile/ops/memory-and-data-movement/tload_zh.md) | `memory` | `tile, global` | `docs/isa/TLOAD_zh.md` |
| 内存（GM <-> Tile） | [TPREFETCH](/docs/isa/tile/ops/memory-and-data-movement/tprefetch_zh.md) | `memory` | `tile, global` | `docs/isa/TPREFETCH_zh.md` |
| 内存（GM <-> Tile） | [TSTORE](/docs/isa/tile/ops/memory-and-data-movement/tstore_zh.md) | `memory` | `tile, global` | `docs/isa/TSTORE_zh.md` |
| 内存（GM <-> Tile） | [TSTORE_FP](/docs/isa/tile/ops/memory-and-data-movement/tstore_zh.md) | `memory` | `tile, global` | `docs/isa/TSTORE_FP_zh.md` |
| 内存（GM <-> Tile） | [MGATHER](/docs/isa/tile/ops/memory-and-data-movement/mgather_zh.md) | `memory` | `tile, global` | `docs/isa/MGATHER_zh.md` |
| 内存（GM <-> Tile） | [MSCATTER](/docs/isa/tile/ops/memory-and-data-movement/mscatter_zh.md) | `memory` | `tile, global` | `docs/isa/MSCATTER_zh.md` |
| 矩阵乘 | [TGEMV_MX](/docs/isa/tile/ops/matrix-and-matrix-vector/tgemv-mx_zh.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TGEMV_MX_zh.md` |
| 矩阵乘 | [TMATMUL_MX](/docs/isa/tile/ops/matrix-and-matrix-vector/tmatmul-mx_zh.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TMATMUL_MX_zh.md` |
| 矩阵乘 | [TMATMUL](/docs/isa/tile/ops/matrix-and-matrix-vector/tmatmul_zh.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TMATMUL_zh.md` |
| 矩阵乘 | [TMATMUL_ACC](/docs/isa/tile/ops/matrix-and-matrix-vector/tmatmul-acc_zh.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TMATMUL_ACC_zh.md` |
| 矩阵乘 | [TMATMUL_BIAS](/docs/isa/tile/ops/matrix-and-matrix-vector/tmatmul-bias_zh.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TMATMUL_BIAS_zh.md` |
| 矩阵乘 | [TGEMV](/docs/isa/tile/ops/matrix-and-matrix-vector/tgemv_zh.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TGEMV_zh.md` |
| 矩阵乘 | [TGEMV_ACC](/docs/isa/tile/ops/matrix-and-matrix-vector/tgemv-acc_zh.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TGEMV_ACC_zh.md` |
| 矩阵乘 | [TGEMV_BIAS](/docs/isa/tile/ops/matrix-and-matrix-vector/tgemv-bias_zh.md) | `matmul` | `dst, lhs, rhs` | `docs/isa/TGEMV_BIAS_zh.md` |
| 数据搬运 / 布局 | [TEXTRACT](/docs/isa/tile/ops/layout-and-rearrangement/textract_zh.md) | `reshape_move` | `dst, src` | `docs/isa/TEXTRACT_zh.md` |
| 数据搬运 / 布局 | [TEXTRACT_FP](/docs/isa/tile/ops/layout-and-rearrangement/textract_zh.md) | `reshape_move` | `dst, src` | `docs/isa/TEXTRACT_FP_zh.md` |
| 数据搬运 / 布局 | [TIMG2COL](/docs/isa/tile/ops/layout-and-rearrangement/timg2col_zh.md) | `reshape_move` | `dst, src` | `docs/isa/TIMG2COL_zh.md` |
| 数据搬运 / 布局 | [TINSERT](/docs/isa/tile/ops/layout-and-rearrangement/tinsert_zh.md) | `reshape_move` | `dst, src` | `docs/isa/TINSERT_zh.md` |
| 数据搬运 / 布局 | [TINSERT_FP](/docs/isa/tile/ops/layout-and-rearrangement/tinsert_zh.md) | `reshape_move` | `dst, src` | `docs/isa/TINSERT_FP_zh.md` |
| 数据搬运 / 布局 | [TFILLPAD](/docs/isa/tile/ops/layout-and-rearrangement/tfillpad_zh.md) | `reshape_move` | `dst, src` | `docs/isa/TFILLPAD_zh.md` |
| 数据搬运 / 布局 | [TFILLPAD_INPLACE](/docs/isa/tile/ops/layout-and-rearrangement/tfillpad-inplace_zh.md) | `reshape_move` | `dst, src` | `docs/isa/TFILLPAD_INPLACE_zh.md` |
| 数据搬运 / 布局 | [TFILLPAD_EXPAND](/docs/isa/tile/ops/layout-and-rearrangement/tfillpad-expand_zh.md) | `reshape_move` | `dst, src` | `docs/isa/TFILLPAD_EXPAND_zh.md` |
| 数据搬运 / 布局 | [TMOV](/docs/isa/tile/ops/layout-and-rearrangement/tmov_zh.md) | `reshape_move` | `dst, src` | `docs/isa/TMOV_zh.md` |
| 数据搬运 / 布局 | [TMOV_FP](/docs/isa/tile/ops/layout-and-rearrangement/tmov_zh.md) | `reshape_move` | `dst, src` | `docs/isa/TMOV_FP_zh.md` |
| 数据搬运 / 布局 | [TRESHAPE](/docs/isa/tile/ops/layout-and-rearrangement/treshape_zh.md) | `reshape_move` | `dst, src` | `docs/isa/TRESHAPE_zh.md` |
| 数据搬运 / 布局 | [TTRANS](/docs/isa/tile/ops/layout-and-rearrangement/ttrans_zh.md) | `reshape_move` | `dst, src` | `docs/isa/TTRANS_zh.md` |
| 数据搬运 / 布局 | [TSUBVIEW](/docs/isa/tile/ops/sync-and-config/subview.md) | `reshape_move` | `dst, src, rowOffset, colOffset` | `docs/isa/TSUBVIEW_zh.md` |
| 数据搬运 / 布局 | [TGET_SCALE_ADDR](/docs/isa/tile/ops/sync-and-config/get-scale-addr.md) | `reshape_move` | `dst, src` | `docs/isa/TGET_SCALE_ADDR_zh.md` |
| 数据搬运 / 布局 | [TCONCAT](/docs/isa/tile/ops/layout-and-rearrangement/tconcat.md) | `reshape_move` | `dst, src0, src1` | `docs/isa/TCONCAT_zh.md` |
| 复杂指令 | [TPRINT](/docs/isa/tile/ops/irregular-and-complex/tprint_zh.md) | `complex` | `dst, src0, src1` | `docs/isa/TPRINT_zh.md` |
| 复杂指令 | [TMRGSORT](/docs/isa/tile/ops/irregular-and-complex/tmrgsort_zh.md) | `complex` | `dst, src0, src1` | `docs/isa/TMRGSORT_zh.md` |
| 复杂指令 | [TSORT32](/docs/isa/tile/ops/irregular-and-complex/tsort32_zh.md) | `complex` | `dst, src0, src1` | `docs/isa/TSORT32_zh.md` |
| 复杂指令 | [TGATHER](/docs/isa/tile/ops/irregular-and-complex/tgather_zh.md) | `complex` | `dst, src0, src1` | `docs/isa/TGATHER_zh.md` |
| 复杂指令 | [TCI](/docs/isa/tile/ops/irregular-and-complex/tci_zh.md) | `complex` | `dst, src0, src1` | `docs/isa/TCI_zh.md` |
| 复杂指令 | [TTRI](/docs/isa/tile/ops/irregular-and-complex/ttri_zh.md) | `complex` | `dst, src0, src1` | `docs/isa/TTRI_zh.md` |
| 复杂指令 | [TPARTADD](/docs/isa/tile/ops/irregular-and-complex/tpartadd_zh.md) | `complex` | `dst, src0, src1` | `docs/isa/TPARTADD_zh.md` |
| 复杂指令 | [TPARTMUL](/docs/isa/tile/ops/irregular-and-complex/tpartmul_zh.md) | `complex` | `dst, src0, src1` | `docs/isa/TPARTMUL_zh.md` |
| 复杂指令 | [TPARTMAX](/docs/isa/tile/ops/irregular-and-complex/tpartmax_zh.md) | `complex` | `dst, src0, src1` | `docs/isa/TPARTMAX_zh.md` |
| 复杂指令 | [TPARTMIN](/docs/isa/tile/ops/irregular-and-complex/tpartmin_zh.md) | `complex` | `dst, src0, src1` | `docs/isa/TPARTMIN_zh.md` |
| 复杂指令 | [TPARTARGMAX](/docs/isa/TPARTARGMAX_zh.md) | `complex` | `dstVal, dstIdx, src0Val, src1Val, src0Idx, src1Idx` | `docs/isa/TPARTARGMAX_zh.md` |
| 复杂指令 | [TPARTARGMIN](/docs/isa/TPARTARGMIN_zh.md) | `complex` | `dstVal, dstIdx, src0Val, src1Val, src0Idx, src1Idx` | `docs/isa/TPARTARGMIN_zh.md` |
| 复杂指令 | [TGATHERB](/docs/isa/tile/ops/irregular-and-complex/tgatherb_zh.md) | `complex` | `dst, src0, src1` | `docs/isa/TGATHERB_zh.md` |
| 复杂指令 | [TSCATTER](/docs/isa/tile/ops/irregular-and-complex/tscatter_zh.md) | `complex` | `dst, src0, src1` | `docs/isa/TSCATTER_zh.md` |
| 复杂指令 | [TQUANT](/docs/isa/tile/ops/irregular-and-complex/tquant_zh.md) | `complex` | `dst, src0, src1` | `docs/isa/TQUANT_zh.md` |
| 通信 | [TPUT](/docs/isa/comm/TPUT_zh.md) | `comm` | `dst, src, staging` | `docs/isa/comm/TPUT_zh.md` |
| 通信 | [TGET](/docs/isa/comm/TGET_zh.md) | `comm` | `dst, src, staging` | `docs/isa/comm/TGET_zh.md` |
| 通信 | [TPUT_ASYNC](/docs/isa/comm/TPUT_ASYNC_zh.md) | `comm` | `dst, src, session` | `docs/isa/comm/TPUT_ASYNC_zh.md` |
| 通信 | [TGET_ASYNC](/docs/isa/comm/TGET_ASYNC_zh.md) | `comm` | `dst, src, session` | `docs/isa/comm/TGET_ASYNC_zh.md` |
| 通信 | [TNOTIFY](/docs/isa/comm/TNOTIFY_zh.md) | `comm` | `signal, value, op` | `docs/isa/comm/TNOTIFY_zh.md` |
| 通信 | [TWAIT](/docs/isa/comm/TWAIT_zh.md) | `comm` | `signal, value, cmp` | `docs/isa/comm/TWAIT_zh.md` |
| 通信 | [TTEST](/docs/isa/comm/TTEST_zh.md) | `comm` | `signal, value, cmp` | `docs/isa/comm/TTEST_zh.md` |
| 通信 | [TGATHER](/docs/isa/comm/TGATHER_zh.md) | `comm` | `group, dst, staging` | `docs/isa/comm/TGATHER_zh.md` |
| 通信 | [TSCATTER](/docs/isa/comm/TSCATTER_zh.md) | `comm` | `group, src, staging` | `docs/isa/comm/TSCATTER_zh.md` |
| 通信 | [TREDUCE](/docs/isa/comm/TREDUCE_zh.md) | `comm` | `group, dst, acc, recv` | `docs/isa/comm/TREDUCE_zh.md` |
| 通信 | [TBROADCAST](/docs/isa/comm/TBROADCAST_zh.md) | `comm` | `group, src, staging` | `docs/isa/comm/TBROADCAST_zh.md` |

## D.5 说明

- 逐条指令语义仍以 `docs/isa/*_zh.md` 为准。
- 本附录用于分类与覆盖追踪，不替代逐条指令的规范化语义描述。
