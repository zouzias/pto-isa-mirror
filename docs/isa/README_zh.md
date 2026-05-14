<p align="center">
  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO ISA 参考

本目录是 PTO Tile Lib ISA 的指令参考（每条指令一页）。

- 权威来源：`include/pto/common/pto_instr.hpp`
- 通用约定（操作数、事件、修饰符）：`docs/isa/conventions_zh.md`

## 同步
- [TSYNC](tile/ops/sync-and-config/tsync_zh.md) - 同步 PTO 执行（等待事件或插入每操作流水线屏障）。

## 手动 / 资源绑定
- [TASSIGN](tile/ops/sync-and-config/tassign_zh.md) - 将 Tile 对象绑定到实现定义的片上地址（手动放置）。
- [SETHF32MODE](tile/ops/sync-and-config/sethf32mode_zh.md) - 设置 HF32 变换模式（实现定义）。
- [SETTF32MODE](tile/ops/sync-and-config/settf32mode_zh.md) - 设置 TF32 变换模式（实现定义）。
- [SETFMATRIX](tile/ops/sync-and-config/setfmatrix.md) - 为类 IMG2COL 操作设置 FMATRIX 寄存器。
- [SET_IMG2COL_RPT](tile/ops/sync-and-config/set-img2col-rpt.md) - 从 IMG2COL 配置 Tile 设置 IMG2COL 重复次数元数据。
- [SET_IMG2COL_PADDING](tile/ops/sync-and-config/set-img2col-padding.md) - 从 IMG2COL 配置 Tile 设置 IMG2COL 填充元数据。
- [TSUBVIEW](tile/ops/sync-and-config/subview.md) - 表达一个tile是另一个tile的subview。
- [TGET_SCALE_ADDR](tile/ops/sync-and-config/get-scale-addr.md) - 将输出tile的片上内存值绑定为扩展后的输入tile内存的值。

## 逐元素（Tile-Tile）
- [TADD](tile/ops/elementwise-tile-tile/tadd_zh.md) - 两个 Tile 的逐元素加法。
- [TABS](tile/ops/elementwise-tile-tile/tabs_zh.md) - Tile 的逐元素绝对值。
- [TAND](tile/ops/elementwise-tile-tile/tand_zh.md) - 两个 Tile 的逐元素按位与。
- [TOR](tile/ops/elementwise-tile-tile/tor_zh.md) - 两个 Tile 的逐元素按位或。
- [TSUB](tile/ops/elementwise-tile-tile/tsub_zh.md) - 两个 Tile 的逐元素减法。
- [TMUL](tile/ops/elementwise-tile-tile/tmul_zh.md) - 两个 Tile 的逐元素乘法。
- [TMIN](tile/ops/elementwise-tile-tile/tmin_zh.md) - 两个 Tile 的逐元素最小值。
- [TMAX](tile/ops/elementwise-tile-tile/tmax_zh.md) - 两个 Tile 的逐元素最大值。
- [TCMP](tile/ops/elementwise-tile-tile/tcmp_zh.md) - 比较两个 Tile 并写入一个打包的谓词掩码。
- [TDIV](tile/ops/elementwise-tile-tile/tdiv_zh.md) - 两个 Tile 的逐元素除法。
- [TSHL](tile/ops/elementwise-tile-tile/tshl_zh.md) - 两个 Tile 的逐元素左移。
- [TSHR](tile/ops/elementwise-tile-tile/tshr_zh.md) - 两个 Tile 的逐元素右移。
- [TXOR](tile/ops/elementwise-tile-tile/txor_zh.md) - 两个 Tile 的逐元素按位异或。
- [TLOG](tile/ops/elementwise-tile-tile/tlog_zh.md) - Tile 的逐元素自然对数。
- [TRECIP](tile/ops/elementwise-tile-tile/trecip_zh.md) - Tile 的逐元素倒数。
- [TPRELU](tile/ops/elementwise-tile-tile/tprelu_zh.md) - 带逐元素斜率 Tile 的逐元素参数化 ReLU (PReLU)。
- [TADDC](tile/ops/elementwise-tile-tile/taddc_zh.md) - 三元逐元素加法：`src0 + src1 + src2`。
- [TSUBC](tile/ops/elementwise-tile-tile/tsubc_zh.md) - 三元逐元素运算：`src0 - src1 + src2`。
- [TCVT](tile/ops/elementwise-tile-tile/tcvt_zh.md) - 带指定舍入模式的逐元素类型转换。
- [TSEL](tile/ops/elementwise-tile-tile/tsel_zh.md) - 使用掩码 Tile 在两个 Tile 之间进行选择（逐元素选择）。
- [TRSQRT](tile/ops/elementwise-tile-tile/trsqrt_zh.md) - 逐元素倒数平方根。
- [TSQRT](tile/ops/elementwise-tile-tile/tsqrt_zh.md) - 逐元素平方根。
- [TEXP](tile/ops/elementwise-tile-tile/texp_zh.md) - 逐元素指数运算。
- [TNOT](tile/ops/elementwise-tile-tile/tnot_zh.md) - Tile 的逐元素按位取反。
- [TRELU](tile/ops/elementwise-tile-tile/trelu_zh.md) - Tile 的逐元素 ReLU。
- [TNEG](tile/ops/elementwise-tile-tile/tneg_zh.md) - Tile 的逐元素取负。
- [TREM](tile/ops/elementwise-tile-tile/trem_zh.md) - 两个 Tile 的逐元素余数，余数符号与除数相同。
- [TFMOD](tile/ops/elementwise-tile-tile/tfmod_zh.md) - 两个 Tile 的逐元素余数，余数符号与被除数相同。

## Tile-标量 / Tile-立即数
- [TEXPANDS](tile/ops/tile-scalar-and-immediate/texpands_zh.md) - 将标量广播到目标 Tile 中。
- [TCMPS](tile/ops/tile-scalar-and-immediate/tcmps_zh.md) - 将 Tile 与标量比较并写入逐元素比较结果。
- [TSELS](tile/ops/tile-scalar-and-immediate/tsels_zh.md) - 使用标量 `selectMode` 在两个源 Tile 中选择一个（全局选择）。
- [TMINS](tile/ops/tile-scalar-and-immediate/tmins_zh.md) - Tile 与标量的逐元素最小值。
- [TADDS](tile/ops/tile-scalar-and-immediate/tadds_zh.md) - Tile 与标量的逐元素加法。
- [TSUBS](tile/ops/tile-scalar-and-immediate/tsubs_zh.md) - 从 Tile 中逐元素减去一个标量。
- [TDIVS](tile/ops/tile-scalar-and-immediate/tdivs_zh.md) - 与标量的逐元素除法（Tile/标量 或 标量/Tile）。
- [TMULS](tile/ops/tile-scalar-and-immediate/tmuls_zh.md) - Tile 与标量的逐元素乘法。
- [TFMODS](tile/ops/tile-scalar-and-immediate/tfmods_zh.md) - 与标量的逐元素余数：`fmod(src, scalar)`。
- [TREMS](tile/ops/tile-scalar-and-immediate/trems_zh.md) - 与标量的逐元素余数：`remainder(src, scalar)`。
- [TMAXS](tile/ops/tile-scalar-and-immediate/tmaxs_zh.md) - Tile 与标量的逐元素最大值：`max(src, scalar)`。
- [TANDS](tile/ops/tile-scalar-and-immediate/tands_zh.md) - Tile 与标量的逐元素按位与。
- [TORS](tile/ops/tile-scalar-and-immediate/tors_zh.md) - Tile 与标量的逐元素按位或。
- [TSHLS](tile/ops/tile-scalar-and-immediate/tshls_zh.md) - Tile 按标量逐元素左移。
- [TSHRS](tile/ops/tile-scalar-and-immediate/tshrs_zh.md) - Tile 按标量逐元素右移。
- [TXORS](tile/ops/tile-scalar-and-immediate/txors_zh.md) - Tile 与标量的逐元素按位异或。
- [TLRELU](tile/ops/tile-scalar-and-immediate/tlrelu_zh.md) - 带标量斜率的 Leaky ReLU。
- [TADDSC](tile/ops/tile-scalar-and-immediate/taddsc_zh.md) - 与标量和第二个 Tile 的融合逐元素加法：`src0 + scalar + src1`。
- [TSUBSC](tile/ops/tile-scalar-and-immediate/tsubsc_zh.md) - 融合逐元素运算：`src0 - scalar + src1`。

## 轴归约 / 扩展
- [TROWSUM](tile/ops/reduce-and-expand/trowsum_zh.md) - 通过对列求和来归约每一行。
- [TCOLSUM](tile/ops/reduce-and-expand/tcolsum_zh.md) - 通过对行求和来归约每一列。
- [TCOLPROD](tile/ops/reduce-and-expand/tcolprod_zh.md) - 通过跨行乘积来归约每一列。
- [TCOLMAX](tile/ops/reduce-and-expand/tcolmax_zh.md) - 通过取行间最大值来归约每一列。
- [TROWMAX](tile/ops/reduce-and-expand/trowmax_zh.md) - 通过取列间最大值来归约每一行。
- [TROWMIN](tile/ops/reduce-and-expand/trowmin_zh.md) - 通过取列间最小值来归约每一行。
- [TROWARGMAX](tile/ops/reduce-and-expand/trowargmax_zh.md) - 获取每行最大值对应列索引。
- [TROWARGMIN](tile/ops/reduce-and-expand/trowargmin_zh.md) - 获取每行最大值对应列索引。
- [TROWEXPAND](tile/ops/reduce-and-expand/trowexpand_zh.md) - 将每个源行的第一个元素广播到目标行中。
- [TROWEXPANDDIV](tile/ops/reduce-and-expand/trowexpanddiv_zh.md) - 行广播除法：将 `src0` 的每一行除以一个每行标量向量 `src1`。
- [TROWEXPANDMUL](tile/ops/reduce-and-expand/trowexpandmul_zh.md) - 行广播乘法：将 `src0` 的每一行乘以一个每行标量向量 `src1`。
- [TROWEXPANDSUB](tile/ops/reduce-and-expand/trowexpandsub_zh.md) - 行广播减法：从 `src0` 的每一行中减去一个每行标量向量 `src1`。
- [TROWEXPANDADD](tile/ops/reduce-and-expand/trowexpandadd_zh.md) - 行广播加法：加上一个每行标量向量。
- [TROWEXPANDMAX](tile/ops/reduce-and-expand/trowexpandmax_zh.md) - 行广播最大值：与每行标量向量取最大值。
- [TROWEXPANDMIN](tile/ops/reduce-and-expand/trowexpandmin_zh.md) - 行广播最小值：与每行标量向量取最小值。
- [TROWEXPANDEXPDIF](tile/ops/reduce-and-expand/trowexpandexpdif_zh.md) - 行指数差运算：计算 exp(src0 - src1)，其中 src1 为每行标量。
- [TCOLMIN](tile/ops/reduce-and-expand/tcolmin_zh.md) - 通过取行间最小值来归约每一列。
- [TCOLEXPAND](tile/ops/reduce-and-expand/tcolexpand_zh.md) - 将每个源列的第一个元素广播到目标列中。
- [TCOLEXPANDDIV](tile/ops/reduce-and-expand/tcolexpanddiv_zh.md) - 列广播除法：将每一列除以一个每列标量向量。
- [TCOLEXPANDMUL](tile/ops/reduce-and-expand/tcolexpandmul_zh.md) - 列广播乘法：将每一列乘以一个每列标量向量。
- [TCOLEXPANDADD](tile/ops/reduce-and-expand/tcolexpandadd_zh.md) - 列广播加法：对每一列加上每列标量向量。
- [TCOLEXPANDMAX](tile/ops/reduce-and-expand/tcolexpandmax_zh.md) - 列广播最大值：与每列标量向量取最大值。
- [TCOLEXPANDMIN](tile/ops/reduce-and-expand/tcolexpandmin_zh.md) - 列广播最小值：与每列标量向量取最小值。
- [TCOLEXPANDSUB](tile/ops/reduce-and-expand/tcolexpandsub_zh.md) - 列广播减法：从每一列中减去一个每列标量向量。
- [TCOLEXPANDEXPDIF](tile/ops/reduce-and-expand/tcolexpandexpdif_zh.md) - 列指数差运算：计算 exp(src0 - src1)，其中 src1 为每列标量。

## 内存（GM <-> Tile）
- [TLOAD](tile/ops/memory-and-data-movement/tload_zh.md) - 从 GlobalTensor (GM) 加载数据到 Tile。
- [TPREFETCH](tile/ops/memory-and-data-movement/tprefetch_zh.md) - 将数据从全局内存预取到 Tile 本地缓存/缓冲区（提示）。
- [TSTORE](tile/ops/memory-and-data-movement/tstore_zh.md) - 将 Tile 中的数据存储到 GlobalTensor (GM)，可选使用原子写入或量化参数。
- [TSTORE_FP](tile/ops/memory-and-data-movement/tstore_zh.md) - 使用缩放 (`fp`) Tile 作为向量量化参数，将累加器 Tile 存储到全局内存。
- [MGATHER](tile/ops/memory-and-data-movement/mgather_zh.md) - 使用逐元素索引从全局内存收集加载元素到 Tile 中。
- [MSCATTER](tile/ops/memory-and-data-movement/mscatter_zh.md) - 使用逐元素索引将 Tile 中的元素散播存储到全局内存。

## 矩阵乘
- [TGEMV_MX](tile/ops/matrix-and-matrix-vector/tgemv-mx_zh.md) - 带缩放 Tile 的 GEMV 变体，支持混合精度/量化矩阵向量计算。
- [TMATMUL_MX](tile/ops/matrix-and-matrix-vector/tmatmul-mx_zh.md) - 带额外缩放 Tile 的矩阵乘法 (GEMM)，用于支持目标上的混合精度/量化矩阵乘法。
- [TMATMUL](tile/ops/matrix-and-matrix-vector/tmatmul_zh.md) - 矩阵乘法 (GEMM)，生成累加器/输出 Tile。
- [TMATMUL_ACC](tile/ops/matrix-and-matrix-vector/tmatmul-acc_zh.md) - 带累加器输入的矩阵乘法（融合累加）。
- [TMATMUL_BIAS](tile/ops/matrix-and-matrix-vector/tmatmul-bias_zh.md) - 带偏置加法的矩阵乘法。
- [TGEMV](tile/ops/matrix-and-matrix-vector/tgemv_zh.md) - 通用矩阵-向量乘法，生成累加器/输出 Tile。
- [TGEMV_ACC](tile/ops/matrix-and-matrix-vector/tgemv-acc_zh.md) - 带显式累加器输入/输出 Tile 的 GEMV。
- [TGEMV_BIAS](tile/ops/matrix-and-matrix-vector/tgemv-bias_zh.md) - 带偏置加法的 GEMV。

## 数据搬运 / 布局
- [TEXTRACT](tile/ops/layout-and-rearrangement/textract_zh.md) - 从源 Tile 中提取子 Tile。
- [TEXTRACT_FP](tile/ops/layout-and-rearrangement/textract_zh.md) - 带 fp/缩放 Tile 的提取（向量量化参数）。
- [TIMG2COL](tile/ops/layout-and-rearrangement/timg2col_zh.md) - 用于类卷积工作负载的图像到列变换。
- [TINSERT](tile/ops/layout-and-rearrangement/tinsert_zh.md) - 在 (indexRow, indexCol) 偏移处将子 Tile 插入到目标 Tile 中。
- [TINSERT_FP](tile/ops/layout-and-rearrangement/tinsert_zh.md) - 带 fp/缩放 Tile 的插入（向量量化参数）。
- [TFILLPAD](tile/ops/layout-and-rearrangement/tfillpad_zh.md) - 复制 Tile 并在有效区域外使用编译时填充值进行填充。
- [TFILLPAD_INPLACE](tile/ops/layout-and-rearrangement/tfillpad-inplace_zh.md) - 原地填充/填充变体。
- [TFILLPAD_EXPAND](tile/ops/layout-and-rearrangement/tfillpad-expand_zh.md) - 填充/填充时允许目标大于源。
- [TMOV](tile/ops/layout-and-rearrangement/tmov_zh.md) - 在 Tile 之间移动/复制，可选应用实现定义的转换模式。
- [TMOV_FP](tile/ops/layout-and-rearrangement/tmov_zh.md) - 使用缩放 (`fp`) Tile 作为向量量化参数，将累加器 Tile 移动/转换到目标 Tile。
- [TRESHAPE](tile/ops/layout-and-rearrangement/treshape_zh.md) - 将 Tile 重新解释为另一种 Tile 类型/形状，同时保留底层字节。
- [TTRANS](tile/ops/layout-and-rearrangement/ttrans_zh.md) - 使用实现定义的临时 Tile 进行转置。

## 复杂指令
- [TPRINT](tile/ops/irregular-and-complex/tprint_zh.md) - 调试/打印 Tile 中的元素（实现定义）。
- [TMRGSORT](tile/ops/irregular-and-complex/tmrgsort_zh.md) - 用于多个已排序列表的归并排序（实现定义的元素格式和布局）。
- [TSORT32](tile/ops/irregular-and-complex/tsort32_zh.md) - 对固定大小的 32 元素块进行排序并生成索引映射。
- [TGATHER](tile/ops/irregular-and-complex/tgather_zh.md) - 使用索引 Tile 或编译时掩码模式来收集/选择元素。
- [TCI](tile/ops/irregular-and-complex/tci_zh.md) - 生成连续整数序列到目标 Tile 中。
- [TTRI](tile/ops/irregular-and-complex/ttri_zh.md) - 生成三角（下/上）掩码 Tile。
- [TPARTADD](tile/ops/irregular-and-complex/tpartadd_zh.md) - 部分逐元素加法，对不匹配的有效区域具有实现定义的处理方式。
- [TPARTMUL](tile/ops/irregular-and-complex/tpartmul_zh.md) - 部分逐元素乘法，对有效区域不一致的处理为实现定义。
- [TPARTMAX](tile/ops/irregular-and-complex/tpartmax_zh.md) - 部分逐元素最大值，对不匹配的有效区域具有实现定义的处理方式。
- [TPARTMIN](tile/ops/irregular-and-complex/tpartmin_zh.md) - 部分逐元素最小值，对不匹配的有效区域具有实现定义的处理方式。
- [TGATHERB](tile/ops/irregular-and-complex/tgatherb_zh.md) - 使用字节偏移量收集元素。
- [TSCATTER](tile/ops/irregular-and-complex/tscatter_zh.md) - 使用逐元素行索引将源 Tile 的行散播到目标 Tile 中。
- [TQUANT](tile/ops/irregular-and-complex/tquant_zh.md) - 量化 Tile（例如 FP32 到 FP8），生成指数/缩放/最大值输出。
