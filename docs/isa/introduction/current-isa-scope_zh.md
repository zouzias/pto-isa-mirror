# 当前 PTO ISA 范围


当前 PTO ISA 手册定义如下指令清单与架构表面。本页是当前手册的范围说明。

## 指令集


PTO 定义五套具名指令集，并为每条指令提供明确参考页：

- **Tile 指令**：全部 `pto.t*` 操作，以及 `pto.mgather` 与 `pto.mscatter`
- **向量微指令**：全部 `pto.v*` 操作
- **标量与控制指令**：用于同步、DMA 控制、谓词构造及机器可见控制的 `pto.*` 操作
- **通信指令**：`docs/isa/comm/` 下的跨 NPU collective、点到点交换和通知操作
- **系统调度指令**：`docs/isa/system/` 下的 TPipe/TMPipe 生产者-消费者协议和资源生命周期操作

## 清单摘要


当前手册文档化了：

- **127** 条 tile 指令
- **100** 条向量微指令
- **48** 条标量与控制指令
- **11** 条通信指令
- **3** 条系统调度指令

合计为 **289 条具名指令**。

## Tile 指令清单


### 同步与配置


`tsync`、`tassign`、`talias`、`sethf32mode`、`settf32mode`、`setfmatrix`、`set_img2col_rpt`、`set_img2col_padding`、`subview`、`get_scale_addr`

### 逐元素 Tile-Tile


`tabs`、`tadd`、`taddc`、`tand`、`tcmp`、`tcvt`、`tdiv`、`texp`、`tpow`、`tfmod`、`tlog`、`tmax`、`tmin`、`tmul`、`tneg`、`tnot`、`tor`、`tprelu`、`trecip`、`trelu`、`trem`、`trsqrt`、`tsel`、`tshl`、`tshr`、`tsqrt`、`tsub`、`tsubc`、`txor`

### Tile-标量与立即数


`tadds`、`taddsc`、`taxpy`、`tands`、`tcmps`、`tdivs`、`texpands`、`tfmods`、`tlrelu`、`tmaxs`、`tmins`、`tmuls`、`tpows`、`tors`、`trems`、`tsels`、`tshls`、`tshrs`、`tsubs`、`tsubsc`、`txors`

### 归约与扩展


`tcolexpand`、`tcolexpandadd`、`tcolexpanddiv`、`tcolexpandexpdif`、`tcolexpandmax`、`tcolexpandmin`、`tcolexpandmul`、`tcolexpandsub`、`tcolargmax`、`tcolargmin`、`tcolmax`、`tcolmin`、`tcolprod`、`tcolsum`、`trowargmax`、`trowargmin`、`trowexpand`、`trowexpandadd`、`trowexpanddiv`、`trowexpandexpdif`、`trowexpandmax`、`trowexpandmin`、`trowexpandmul`、`trowexpandsub`、`trowmax`、`trowmin`、`trowprod`、`trowsum`

### 内存与数据搬运


`tload`、`tprefetch`、`tstore`、`mgather`、`mscatter`

### 矩阵与矩阵-向量


`tgemv`、`tgemv_acc`、`tgemv_bias`、`tgemv_mx`、`tmatmul`、`tmatmul_acc`、`tmatmul_bias`、`tmatmul_mx`

### 布局与重排


`tconcat`、`textract`、`tfillpad`、`tfillpad_expand`、`tfillpad_inplace`、`timg2col`、`tinsert`、`tmov`、`tpack`、`treshape`、`ttrans`

### 不规则与复杂操作


`tci`、`tdequant`、`tgather`、`tgatherb`、`thistogram`、`tmrgsort`、`tpartadd`、`tpartmax`、`tpartmin`、`tpartmul`、`tprint`、`tquant`、`trandom`、`tscatter`、`tsort32`、`ttri`

## 向量微指令清单


### 向量加载存储


`vgather2`、`vgather2_bc`、`vgatherb`、`vldas`、`vlds`、`vldus`、`vldx2`、`vscatter`、`vsld`、`vsldb`、`vsst`、`vsstb`、`vsta`、`vstar`、`vstas`、`vsts`、`vstu`、`vstur`、`vstus`、`vstx2`

### 谓词与物化


`vbr`、`vdup`

### 一元向量操作


`vabs`、`vbcnt`、`vcls`、`vexp`、`vln`、`vmov`、`vneg`、`vnot`、`vrec`、`vrelu`、`vrsqrt`、`vsqrt`

### 二元向量操作


`vadd`、`vaddc`、`vand`、`vdiv`、`vmax`、`vmin`、`vmul`、`vor`、`vshl`、`vshr`、`vsub`、`vsubc`、`vxor`

### 向量-标量操作


`vaddcs`、`vadds`、`vands`、`vlrelu`、`vmaxs`、`vmins`、`vmuls`、`vors`、`vshls`、`vshrs`、`vsubcs`、`vsubs`、`vxors`

### 转换操作


`vci`、`vcvt`、`vtrc`

### 归约操作


`vcadd`、`vcgadd`、`vcgmax`、`vcgmin`、`vcmax`、`vcmin`、`vcpadd`

### 比较与选择


`vcmp`、`vcmps`、`vsel`、`vselr`、`vselrv2`

### 数据重排


`vdintlv`、`vdintlvv2`、`vintlv`、`vintlvv2`、`vpack`、`vperm`、`vshift`、`vslide`、`vsqz`、`vsunpack`、`vusqz`、`vzunpack`

### SFU 与 DSA 操作


`vaddrelu`、`vaddreluconv`、`vaxpy`、`vexpdiff`、`vmrgsort`、`vmula`、`vmulconv`、`vmull`、`vprelu`、`vsort32`、`vsubrelu`、`vtranspose`

## 标量与控制指令清单


### 流水线同步


`get_buf`、`mem_bar`、`pipe_barrier`、`rls_buf`、`set_cross_core`、`set_flag`、`set_intra_block`、`wait_flag`、`wait_flag_dev`、`wait_intra_core`

### DMA 拷贝


`copy_gm_to_ubuf`、`copy_ubuf_to_gm`、`copy_ubuf_to_ubuf`、`set_loop_size_outtoub`、`set_loop_size_ubtoout`、`set_loop1_stride_outtoub`、`set_loop1_stride_ubtoout`、`set_loop2_stride_outtoub`、`set_loop2_stride_ubtoout`

### 谓词加载存储


`pld`、`pldi`、`plds`、`pst`、`psti`、`psts`、`pstu`

### 谓词生成与代数


`pand`、`pdintlv_b8`、`pge_b16`、`pge_b32`、`pge_b8`、`pintlv_b16`、`plt_b16`、`plt_b32`、`plt_b8`、`pnot`、`por`、`ppack`、`psel`、`pset_b16`、`pset_b32`、`pset_b8`、`punpack`、`pxor`

## 通信指令清单


`tbroadcast`、`tget`、`tget_async`、`tgather`、`tnotify`、`tput`、`tput_async`、`treduce`、`tscatter`、`ttest`、`twait`

## 系统调度指令清单


`tfree`、`tpop`、`tpush`

## 相关页面


- [指令集总览](../instruction-families/README_zh.md)
- [指令族总览](../instruction-families/README_zh.md)


# Current PTO ISA Scope
本节定义规范范围与边界条件，避免把实现细节误当成架构语义。

## Instruction Sets
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## Inventory Summary
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## Tile Instruction Inventory
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Sync And Config
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Elementwise Tile-Tile
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Tile-Scalar And Immediate
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Reduce And Expand
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Memory And Data Movement
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Matrix And Matrix-Vector
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Layout And Rearrangement
本节定义类型/布局/格式规则与合法性要求。

### Irregular And Complex
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## Vector Micro-Instruction Inventory
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Vector Load-Store
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Predicate And Materialization
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Unary Vector Operations
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Binary Vector Operations
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Vector-Scalar Operations
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Conversion Operations
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Reduction Operations
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Compare And Select
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Data Rearrangement
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### SFU And DSA Operations
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## Scalar And Control Instruction Inventory
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Pipeline Sync
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### DMA Copy
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Predicate Load-Store
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

### Predicate Generation And Algebra
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## Communication Instruction Inventory
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## System Scheduling Instruction Inventory
本节为与英文同名章节的中文说明位，后续可继续补充更细节内容。

## Related Pages
本节给出相关章节和上下游链接。
