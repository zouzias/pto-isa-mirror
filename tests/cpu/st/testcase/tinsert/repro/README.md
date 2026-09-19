# TINSERT CPU 日志复现与修复

2026-09-19，仓库 `tinsert` 分支，基于 HEAD `3423977c1d2180b09bb8f75298ae3ce0bb0b1b79`。
输入材料：`/data/w00949583/log/tinsert` 中两份原始 CPU 日志、测试头文件和六份 Acc→Mat `.cce`。

## 原因与修复范围

| 原始失败 | 数量 | 原因与修复 |
| --- | ---: | --- |
| SPLIT2/4 | 31 | 公开重载没有向 `__CPU_SIM` 暴露。增加 CPU 条件，并将 CPU SPLIT 搬运对齐 NPU 的整列块、16 行补齐和 Compact 步长。 |
| CompactNullTLoad | 4 | CPU 强制 NZ Shape 等于有效形状，在 TINSERT 之前断言。仅针对 A5 NZ→Vec，按 NPU 的 Shape0/Shape1、GM 步长及物理 Tile 行距搬运。 |
| FP8/HIF8 Acc→Mat | 6 | 前置 CPU 普通 TMATMUL 不接受对应类型。补齐 A5 的四种 e4m3/e5m2 组合及 hif8×hif8→float，复用现有 float FMA。 |
| half boundary_maxarea | 1 | 缺少该 `.cce`、M/K/N 和报错详情，无法定因，未修改。 |

原普通 TINSERT 日志为 63 PASS / 35 FAIL，Acc→Mat 为 9 PASS / 7 FAIL。
这不是重跑外部框架全部 114 项；Acc→Mat 原日志没有 stderr，六个 FP8/HIF8 原因由附带参数实测确认。

修改文件：`include/pto/common/pto_instr.hpp`、`include/pto/cpu/{TInsert,TLoad,TMatmul,Hifloat8}.hpp`。
NPU 实现、普通非 SPLIT TINSERT 和不相关路径未修改。

## 与 NPU 对齐的行为

- 对照 `npu/a5/TInsert.hpp::TInsertSplitImpl`：SPLIT2/4 按完整 32 字节列块搬运，行数补齐至 16 的倍数；
  `Null` 取静态 Rows，`RowPlusOne` 取补齐行数加 1，其他 Compact 模式取补齐行数作为源列块间距。
  CPU 连续复制这些列块，结果与 NPU 分多次 DMA 搬运一致。
- 对照 `common/arch/register/tload_common.hpp::TLoadVecNZ2NZ`：每个 burst 为 `validRow * 32` 字节，
  列块数为 Shape1，GM 间距取 Stride1，UB 间距取 Tile 物理 Rows；外层按 Shape0/Stride0 搬运。
  有效列窗口外的列块同样被加载，行尾与未搬运列保持原值，NZ 路径不额外填充 PadValue。
  保留 A5 要求的静态 16 行、32 字节 fractal；A2/A3 和 Mat 路径的原有形状检查不变。
- 对照 `npu/a5/TMatmul.hpp::CheckMadValid`：新增类型仅允许 float 累加输出，CPU A5 模式运行。
  CPU A2/A3 模式在断言启用时继续拒绝这些新增类型。MX 路径未改变。

不能由 CPU 原断言推断 A5 测试 Shape 写错；A5 register 加载路径不要求 NZ Shape 等于有效窗口。

## 用例覆盖

原 NZ 用例保留以下物理形状和有效窗口：

| 类型 | 物理行列 | 有效行列 | 插入列偏移 |
| --- | --- | --- | --- |
| half | 128×64 | 64×32 | 0、32 |
| float | 64×32 | 32×16 | 8 |
| bfloat16 | 128×128 | 80×48 | 16 |

在既有 `main.cpp` 中新增 30 项回归用例，沿用同一 `tinsert` 构建目标：

- 四个原 NZ 窗口、三个完整加载对照；同时校验源 Tile 的有效列外数据、行尾和目标哨兵值。
- SPLIT2/4 公开接口、三个 Compact 模式的补齐行列尾部及不能整除的分块数。
- 两种 FP4 编码的 SPLIT2/4 按字节校验，包括三个列块分给四次 DMA 的情况。
- 六个 FP8/HIF8 Acc→Mat 链路，包含 e4m3 最大 M=2032；其余为 M/K/N=16/32/32。
- 带 GM 间隔和两个外层分组的 NZ 搬运，以及 A2/A3 原有形状检查和新增类型限制。

矩阵数值测试使用非均匀正数、负数和零，先确认输入可精确表示，再与独立 float 参考计算比较。
后续复核发现 HIF8 解码的前缀位数错误：外部编码 `0x18`（0.5）被 CPU 解码成 3，
导致 M/K/N=16/32/32 的链路结果期望 8、实际 288。`Hifloat8.hpp` 现按前缀长度分别取
5/4/3 位指数与尾数字段，替代统一取 5 位；未改动构造函数的舍入策略。

独立参考为 en_dtypes 0.0.4。修复前 44/256 个编码不一致，修复后全部 256 个解码一致，
包括零、NaN 和正负无穷；新增 ST 同时验证全部有限值的编码往返。
HIF8 矩阵 ST 已恢复带小数的非均匀输入，并增加原始 `0x18` 输入链路，均通过。
未进行 NPU 硬件数值精度验证，也未重跑缺失外部驱动和 golden 数据的原 114 项测试全集。

## 构建和运行

在仓库根目录执行；Debug 保留断言，BF16 开关覆盖全部用例：

```bash
cmake -S tests/cpu/st -B /tmp/tinsert-st \
  -DTEST_CASE=tinsert -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=/usr/local/bin/g++-15 \
  -DCMAKE_C_COMPILER=/usr/local/bin/gcc-15 \
  -DPTO_CPU_SIM_ENABLE_BF16=ON \
  -DCMAKE_PREFIX_PATH=/data/w00949583/tools/gtest-pic
cmake --build /tmp/tinsert-st --parallel 4
/tmp/tinsert-st/bin/tinsert --gtest_filter='TInsertLogRepro.*'

python3 tests/cpu/st/testcase/tinsert/repro/run.py \
  --cxx /usr/local/bin/g++-15 --output /tmp/tinsert-chain-check
```

独立脚本默认要求 11 项全部编译运行通过；对未修复代码复查原编译错误时使用 `--expect-original-failures`。
本环境 `run_cpu.py --enable-bf16` 存在 `tests.script` 导入冲突，因此构建直接使用仓内 CMake。
完整既有 ST 需先在构建目录运行相应 `gen_data.py`，再在 `build/bin` 中启动测试。

## 验证结果

使用 GCC 15.2.1、Debug、BF16 开启，CPU 默认环境，未上板。

| 检查 | 结果 |
| --- | --- |
| 修复前 9 项最小 ST | 6 FAILED / 3 PASSED |
| 最新完整 TINSERT（44 既有 + 30 专项） | 74 PASS |
| TLOAD | 50 PASS |
| TMATMUL | 12 PASS |
| TMATMUL_MX | 19 PASS |
| 独立 SPLIT / Acc→Mat 链路（前轮） | 11 PASS |
| 最新外部编码 Acc→Mat（含 HIF8） | 6 PASS |
| 最新 TTRANS（含 HIF8） | 2 PASS |
| HIF8 全部编码独立参考对照 | 256/256 一致 |

日志位于 `/data/w00949583/log/tinsert/st-repro-20260919/`：

- `cpu-st.log`、`compile/`：修复前结果。
- `tinsert-all-fixed.log`：最终完整 TINSERT 结果。
- `run-tload.log`、`run-tmatmul.log`、`run-tmatmul_mx.log`：相关回归。
- `compile-fixed/summary.json`：修复后独立链路结果及逐项诊断。

HIF8 后续修复日志位于 `/data/w00949583/log/tinsert/hif8-fix/`：
`decode-before-differences.json`、`decode-after-differences.json` 保存独立编码对照结果；
`run-tinsert.log`、`run-acc6.log`、`run-ttrans.log` 为最新运行结果。
`/data/w00949583/log/tinsert/final-log-audit/` 保留 HIF8 修复前的失败证据；旧日志没有覆盖。

写法整理后的重新验证记录位于 `/data/w00949583/log/tinsert/style-review/`：
全新 Debug/BF16 构建的 TINSERT 74 项、TLOAD 50 项、TMATMUL 12 项、TMATMUL_MX 19 项全部通过；
独立复现脚本 11 项、日志参数 SPLIT 31 项和外部随机编码 Acc→Mat 6 项全部通过。
