# include/

PTO ISA 对外的 C/C++ 头文件（以模板化、基本 header-only 为主）。上层框架/算子可以通过这些头文件生成 PTO ISA 的 Tile 指令序列。

## 快速开始

推荐直接 include 统一入口头：

```cpp
#include <pto/pto-inst.hpp>
```

`pto/pto-inst.hpp` 会根据构建配置选择合适的后端（CPU 仿真/stub 或 NPU 实现）。详情见 [include/pto/README_zh.md](pto/README_zh.md)。

## 目录结构

- `include/pto/`：公共 PTO ISA API 与后端实现（common / cpu / npu / comm）

## 相关文档

- [ISA 指南](../docs/README_zh.md)
- [入门指南](../docs/getting-started_zh.md)

## PTO 指令实现状态（CPU / Costmodel / A2 / A3 / A5 / Kirin）

下表用于跟踪每条指令在不同后端的可用性：

- **CPU**：`__CPU_SIM`（CPU 仿真后端）。
- **Costmodel**：`__COSTMODEL`（A2 / A3 性能仿真后端，包含 `stub` 与 `fit` 两条路径；任一路径支持即记为支持）。
- **A2（Ascend 910B）/ A3（Ascend 910C）**：当前共享 `include/pto/npu/a2a3/` 的实现（因此两列状态相同）。
- **A5（Ascend 950）**：使用 `include/pto/npu/a5/` 的实现。
- **Kirin**：使用 `include/pto/npu/kirin9030/` 的实现。

| 指令 | CPU | Costmodel | A2 | A3 | A5 | Kirin |
|---|---:|---:|---:|---:|---:|---:|
| [`MGATHER`](../docs/isa/MGATHER_zh.md) | 是 | 否 | 是 | 是 | 是 | 否 |
| [`MSCATTER`](../docs/isa/MSCATTER_zh.md) | 是 | 否 | 是 | 是 | 是 | 否 |
| [`SYNCALL`](../docs/isa/SYNCALL_zh.md) | 否 | 否 | 是 | 是 | 是 | 否 |
| [`TABS`](../docs/isa/TABS_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TADD`](../docs/isa/TADD_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TADDS`](../docs/isa/TADDS_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TALLOC`](../docs/isa/TALLOC_zh.md) | 否 | 是 | 是 | 是 | 是 | 否 |
| [`TAND`](../docs/isa/TAND_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TANDS`](../docs/isa/TANDS_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TASSIGN`](../docs/isa/TASSIGN_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TAXPY`](../docs/isa/TAXPY_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TBROADCAST`](../docs/isa/comm/TBROADCAST_zh.md) | 是 | 否 | 是 | 是 | 是 | 否 |
| [`TCI`](../docs/isa/TCI_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TCMP`](../docs/isa/TCMP_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TCMPS`](../docs/isa/TCMPS_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TCOLARGMAX`](../docs/isa/TCOLARGMAX_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TCOLARGMIN`](../docs/isa/TCOLARGMIN_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TCOLEXPAND`](../docs/isa/TCOLEXPAND_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TCOLEXPANDADD`](../docs/isa/TCOLEXPANDADD_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TCOLEXPANDDIV`](../docs/isa/TCOLEXPANDDIV_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TCOLEXPANDEXPDIF`](../docs/isa/TCOLEXPANDEXPDIF_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TCOLEXPANDMAX`](../docs/isa/TCOLEXPANDMAX_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TCOLEXPANDMIN`](../docs/isa/TCOLEXPANDMIN_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TCOLEXPANDMUL`](../docs/isa/TCOLEXPANDMUL_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TCOLEXPANDSUB`](../docs/isa/TCOLEXPANDSUB_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TCOLMAX`](../docs/isa/TCOLMAX_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TCOLMIN`](../docs/isa/TCOLMIN_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TCOLPROD`](../docs/isa/TCOLPROD_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TCOLSUM`](../docs/isa/TCOLSUM_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TCONCAT`](../docs/isa/TCONCAT_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TCVT`](../docs/isa/TCVT_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TDIV`](../docs/isa/TDIV_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TDIVS`](../docs/isa/TDIVS_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TEXP`](../docs/isa/TEXP_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TEXPANDS`](../docs/isa/TEXPANDS_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TEXTRACT`](../docs/isa/TEXTRACT_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TFILLPAD`](../docs/isa/TFILLPAD_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TFMOD`](../docs/isa/TFMOD_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TFMODS`](../docs/isa/TFMODS_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TFREE`](../docs/isa/TFREE_zh.md) | 否 | 是 | 是 | 是 | 是 | 否 |
| [`TGATHER`](../docs/isa/TGATHER_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TGATHERB`](../docs/isa/TGATHERB_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TGEMV`](../docs/isa/TGEMV_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TGEMV_ACC`](../docs/isa/TGEMV_ACC_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TGEMV_BIAS`](../docs/isa/TGEMV_BIAS_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TGEMV_MX`](../docs/isa/TGEMV_MX_zh.md) | 是 | 否 | 否 | 否 | 是 | 否 |
| [`TGET`](../docs/isa/comm/TGET_zh.md) | 是 | 否 | 是 | 是 | 是 | 否 |
| [`TGET_ASYNC`](../docs/isa/comm/TGET_ASYNC_zh.md) | 是 | 否 | 是 | 是 | 是 | 否 |
| [`TIMG2COL`](../docs/isa/TIMG2COL_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TINSERT`](../docs/isa/TINSERT_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TLOAD`](../docs/isa/TLOAD_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TLOG`](../docs/isa/TLOG_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TLRELU`](../docs/isa/TLRELU_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMATMUL`](../docs/isa/TMATMUL_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMATMUL_ACC`](../docs/isa/TMATMUL_ACC_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMATMUL_BIAS`](../docs/isa/TMATMUL_BIAS_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMATMUL_MX`](../docs/isa/TMATMUL_MX_zh.md) | 是 | 否 | 否 | 否 | 是 | 是 |
| [`TMAX`](../docs/isa/TMAX_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMAXS`](../docs/isa/TMAXS_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMIN`](../docs/isa/TMIN_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMINS`](../docs/isa/TMINS_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMOV`](../docs/isa/TMOV_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMRGSORT`](../docs/isa/TMRGSORT_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMUL`](../docs/isa/TMUL_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TMULS`](../docs/isa/TMULS_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TNEG`](../docs/isa/TNEG_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TNOT`](../docs/isa/TNOT_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TNOTIFY`](../docs/isa/comm/TNOTIFY_zh.md) | 是 | 否 | 是 | 是 | 是 | 否 |
| [`TOR`](../docs/isa/TOR_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TORS`](../docs/isa/TORS_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TPARTADD`](../docs/isa/TPARTADD_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TPARTARGMAX`](../docs/isa/TPARTARGMAX_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TPARTARGMIN`](../docs/isa/TPARTARGMIN_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TPARTMAX`](../docs/isa/TPARTMAX_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TPARTMIN`](../docs/isa/TPARTMIN_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TPARTMUL`](../docs/isa/TPARTMUL_zh.md) | 否 | 否 | 是 | 是 | 是 | 是 |
| [`TPOP`](../docs/isa/TPOP_zh.md) | 是 | 是 | 是 | 是 | 是 | 否 |
| [`TPOW`](../docs/isa/TPOW_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TPOWS`](../docs/isa/TPOWS_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TPREFETCH`](../docs/isa/TPREFETCH_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TPRELU`](../docs/isa/TPRELU_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TPRINT`](../docs/isa/TPRINT_zh.md) | 是 | 是 | 是 | 是 | 是 | 否 |
| [`TPUSH`](../docs/isa/TPUSH_zh.md) | 是 | 是 | 是 | 是 | 是 | 否 |
| [`TPUT`](../docs/isa/comm/TPUT_zh.md) | 是 | 否 | 是 | 是 | 是 | 否 |
| [`TPUT_ASYNC`](../docs/isa/comm/TPUT_ASYNC_zh.md) | 是 | 否 | 是 | 是 | 是 | 否 |
| [`TQUANT`](../docs/isa/TQUANT_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TRANDOM`](../docs/isa/TRANDOM_zh.md) | 是 | 否 | 否 | 否 | 是 | 否 |
| [`TRECIP`](../docs/isa/TRECIP_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TREDUCE`](../docs/isa/comm/TREDUCE_zh.md) | 是 | 否 | 是 | 是 | 是 | 否 |
| [`TRELU`](../docs/isa/TRELU_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TREM`](../docs/isa/TREM_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TREMS`](../docs/isa/TREMS_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TROWARGMAX`](../docs/isa/TROWARGMAX_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TROWARGMIN`](../docs/isa/TROWARGMIN_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TROWEXPAND`](../docs/isa/TROWEXPAND_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TROWEXPANDADD`](../docs/isa/TROWEXPANDADD_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TROWEXPANDDIV`](../docs/isa/TROWEXPANDDIV_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TROWEXPANDEXPDIF`](../docs/isa/TROWEXPANDEXPDIF_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TROWEXPANDMAX`](../docs/isa/TROWEXPANDMAX_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TROWEXPANDMIN`](../docs/isa/TROWEXPANDMIN_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TROWEXPANDMUL`](../docs/isa/TROWEXPANDMUL_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TROWEXPANDSUB`](../docs/isa/TROWEXPANDSUB_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TROWMAX`](../docs/isa/TROWMAX_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TROWMIN`](../docs/isa/TROWMIN_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TROWPROD`](../docs/isa/TROWPROD_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TROWSUM`](../docs/isa/TROWSUM_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TRSQRT`](../docs/isa/TRSQRT_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSCATTER`](../docs/isa/TSCATTER_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSEL`](../docs/isa/TSEL_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSELS`](../docs/isa/TSELS_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TSHL`](../docs/isa/TSHL_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TSHLS`](../docs/isa/TSHLS_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TSHR`](../docs/isa/TSHR_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TSHRS`](../docs/isa/TSHRS_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TSORT32`](../docs/isa/TSORT32_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSQRT`](../docs/isa/TSQRT_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSTORE`](../docs/isa/TSTORE_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSUB`](../docs/isa/TSUB_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TSUBS`](../docs/isa/TSUBS_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TTEST`](../docs/isa/comm/TTEST_zh.md) | 是 | 否 | 是 | 是 | 是 | 否 |
| [`TTRANS`](../docs/isa/TTRANS_zh.md) | 是 | 是 | 是 | 是 | 是 | 是 |
| [`TTRI`](../docs/isa/TTRI_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TWAIT`](../docs/isa/comm/TWAIT_zh.md) | 是 | 否 | 是 | 是 | 是 | 否 |
| [`TXOR`](../docs/isa/TXOR_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |
| [`TXORS`](../docs/isa/TXORS_zh.md) | 是 | 否 | 是 | 是 | 是 | 是 |

说明：

- `是`：该后端已提供可用实现。
- `TODO`：该指令已进入公共 API 或文档范围，但对应后端实现暂未提供，或尚未在该后端接入。
- `否`：明确不支持或当前不计划在该后端提供实现。
- 空白：状态尚未最终确认，或该项仍在整理中。
