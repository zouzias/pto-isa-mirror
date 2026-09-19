# TINSERT CPU 回归与复现

本目录记录 TINSERT 日志中已确认问题的修复和复现方式。常规 CPU ST 位于
[`../main.cpp`](../main.cpp)，随既有 `tinsert` 目标构建；本目录的独立程序用于编译和运行检查。

## 修复范围

原始普通 TINSERT 日志为 63 PASS / 35 FAIL，Acc→Mat 为 9 PASS / 7 FAIL。

| 原始失败分组 | 数量 | 已确认问题与修复 |
| --- | ---: | --- |
| SPLIT2/4 | 31 | CPU 公开接口缺失；补齐接口，按 A5 的完整列块、行补齐和 Compact 步长搬运。 |
| CompactNullTLoad | 4 | CPU 错误要求 NZ 源形状等于 Tile 有效窗口；A5 NZ→Vec 改为按 GM 列块和物理行距加载。 |
| FP8/HIF8 Acc→Mat | 6 | CPU 普通 TMATMUL 缺少对应输入类型；补齐 A5 支持的组合，并修复 HIF8 解码。 |
| half boundary_maxarea | 1 | 缺少原始用例参数和详细失败输出，尚未确认根因或修复。 |

这些结论来自日志参数构造的复现，并非重新执行外部框架全部 114 项。
NPU 实现和普通非 SPLIT TINSERT 未修改。HIF8 使用 CPU 共用 LUT，修复也会影响其他
CPU 指令中的 HIF8 编码与解码；新增 SPLIT/TLOAD 搬运逻辑仅涉及上述路径。

## 与实现对应的行为

### SPLIT2/4

对应 `include/pto/npu/a5/TInsert.hpp::TInsertSplitImpl` 和 CPU 的带 `TInsertMode` 重载：

- 源为 Vec、目标为 Mat，源使用 NZ 布局，两端类型必须相同。CPU 与 A5 检查相同的类型列表：
  `half`、`bfloat16_t`、`float`、`int32_t`、`int8_t`、`hifloat8_t`、三种 FP8 类型
  （`float8_e4m3_t`、`float8_e5m2_t`、`float8_e8m0_t`）及两种打包 FP4 类型。
- 以完整 32 字节列块搬运，行数向上补齐到 16 的倍数，包含有效窗口外的补齐尾部。
  源和目标必须为实际搬运范围提供存储空间。
- `CompactMode::Null` 的源列块行距为静态 Rows，`RowPlusOne` 为补齐行数加 1，
  `Normal` 和 `RowAlignedPadding` 为补齐行数；目标列块行距为目标静态 Rows，乘以 32 即为字节步长。
- NPU 前几段各搬运 `totalBurstNum / SplitCount` 个列块，最后一段接收剩余列块。
  CPU 按相同顺序连续复制，模拟其数据结果，不模拟 DMA 调度或同步。
- FP4 的列数和列偏移按逻辑元素计数，每两个元素共用一个字节。

### A5 NZ→Vec TLOAD

对应 `include/pto/common/arch/register/tload_common.hpp::TLoadVecNZ2NZ`：

- 每个 burst 为 `validRow * 32` 字节；Shape0 是组数，Shape1 是每组列块数。
  GM 组步长和列块步长取 Stride0/Stride1；UB 列块步长取 Tile 物理 Rows，而不是有效行数。
- 列块不按 `validCol` 裁剪。有效列窗口外的列块也会加载；未搬运行、列块保留原值，
  即使 `PadValue::Zero` 也不额外填充。
- 静态内层形状要求为 16 行、32 字节列块（FP4 为 64 个逻辑元素）。
  NPU 编译期检查，CPU A5 模式在断言启用时于运行期检查。
- 此特殊路径仅用于 CPU A5 模式的 NZ→Vec；A2/A3 和 Mat 的原有形状检查保持不变。

### FP8/HIF8 Acc→Mat

对应 `include/pto/npu/a5/TMatmul.hpp::CheckMadValid` 的输入类型规则。CPU 的检查由
`TMATMUL`、`TMATMUL_ACC`、`TMATMUL_BIAS` 及其 `TGEMV` 包装路径共用，
因此这些入口的类型接受范围也随之改变；各自的累加器、偏置约束保持不变：

- float 累加器支持 e4m3/e5m2 的四种组合，以及 hif8×hif8。
- 输入转为 float 后复用现有 `std::fma` 累加，随后执行普通 Acc→Mat TINSERT。
  此验证不代表与 NPU 硬件的累加精度逐位一致。
- CPU 默认内存模型为 A2A3。运行 A5 用例前需调用 `NPU_MEMORY_INIT(NPUArch::A5)`，
  并在其后绑定 Tile。新增类型的 A5 检查是运行时断言，定义 `NDEBUG` 后不执行。

HIF8 的 LUT 解码原来对不同前缀统一取 5 位指数与尾数字段，现在按前缀长度分别取 5/4/3 位。
原始编码 `0x18` 表示 0.5，此前误解码为 3；M/K/N=16/32/32 的常量输入链路结果因此从 288 改正为 8。
编码和解码共用该 LUT，因此编码映射也随之修正，构造函数的舍入策略未改动。

## ST 覆盖

以下 37 项新增测试复用既有 `TINSERTTest` fixture，采用 `TEST_F` 和行为描述命名：

| 用例 | 数量 | 看护内容 |
| --- | ---: | --- |
| `NzHalf*`、`NzFloat*`、`NzBfloat16*` | 7 | 四个原有效窗口和三个完整加载对照；检查有效列外数据、行尾和目标哨兵值。 |
| `Split*PublicApi` | 2 | 公开接口可编译，逐元素校验搬运结果。 |
| `Split*NullAlignedTail`、`Split*NormalAlignedTail`、`Split*RowPlusOneTail`、`Split*RowAlignedPaddingTail` | 8 | 四种 Compact 模式、行列尾部以及不能整除的列块数。 |
| `Split*Fp4*` | 4 | 两种 FP4 编码的 SPLIT2/4，逐字节比较。 |
| `AccE4M3`、`AccE5M2`、`AccE4M3E5M2`、`AccE5M2E4M3`、`AccHif8`、`AccE4M3M2032` | 6 | 非均匀正负数和零的矩阵链路，包含 M=2032。 |
| `NzGappedOuterBlocks` | 1 | GM 间隔、两个外层分组、不额外填充。 |
| `NzFp4*DynamicGroups` | 2 | 两种 FP4 的动态组数和步长，逐字节检查额外列块、行尾和未搬运块。 |
| `A5NzRejectsInvalidInner*`、`A5MatNzRejectsMismatchedShape` | 3 | 拒绝非法 NZ 内层形状；A5 Mat 仍要求源形状与有效窗口一致。 |
| `A2A3*` | 2 | 原 NZ 形状检查和 FP8 架构限制仍触发断言。 |
| `AccHif8Raw0x18`、`Hif8AllEncodings` | 2 | 原始 `0x18` 链路、256 个编码和全部有限值的编码往返。 |

BF16 两项要求 `PTO_CPU_SIM_ENABLE_BF16=ON`，五项断言反向测试要求未定义 `NDEBUG`。
因此 **Debug + BF16** 下为 44 项既有 + 37 项新增，共 81 项；关闭 BF16 后为 79 项，
Release + BF16 下为 76 项，Release 且关闭 BF16 后为 74 项。
HIF8 golden 固化自 en_dtypes 0.0.4，测试运行时不依赖该 Python 包。

独立 `run.py` 共 12 项检查：11 项应编译运行成功；`split_unsupported_uint16` 应在编译期
被拒绝，并匹配指定类型诊断。后者覆盖 CPU 与 A5 SPLIT 类型约束的一致性。
脚本仅在全部检查符合预期时返回 0；负向用例的非零编译返回值表示预期结果。

## 构建和运行

以下命令在仓库根目录执行，需要 CMake、NumPy、GCC 14+（示例使用 GCC 15）和可用的 GTest。
若 GTest 安装在自定义位置，可通过 `CMAKE_PREFIX_PATH` 指定；未找到时仓库 CMake 会尝试下载。
首次使用新的构建目录，后续使用 `--clean-first` 重新编译。

```bash
repo_root="$PWD"
build_dir="$repo_root/build/tinsert-doc-check"
cmake -S tests/cpu/st -B "$build_dir" \
  -DTEST_CASE=tinsert -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_COMPILER=g++-15 -DCMAKE_C_COMPILER=gcc-15 \
  -DPTO_CPU_SIM_ENABLE_BF16=ON
cmake --build "$build_dir" --clean-first --parallel 4

# 全部 81 项：先生成既有用例所需数据，再从 bin 目录运行。
(cd "$build_dir" && python3 "$repo_root/tests/cpu/st/testcase/tinsert/gen_data.py")
(cd "$build_dir/bin" && ./tinsert)

# 排除既有 case_* 数据用例，仅运行 37 项新增用例，不依赖上述数据文件。
"$build_dir/bin/tinsert" --gtest_filter='TINSERTTest.*-TINSERTTest.case_*'

python3 tests/cpu/st/testcase/tinsert/repro/run.py \
  --cxx g++-15 --output build/tinsert-repro
```

`--expect-original-failures` 仅用于检查未修复版本的编译诊断：需将本目录的复现文件复制到
未修复 checkout 的相同位置，再从那里运行。本选项不应用于当前修复版本。

## 验证范围

- 仓内 TINSERT 回归：Debug 开启/关闭 BF16 分别为 81/79 项，Release 开启/关闭 BF16 分别为 76/74 项，全部通过。
- 相关算子此前 Debug + BF16 回归：TLOAD 50、TMATMUL 12、TMATMUL_MX 19 项通过。
- 独立 HIF8 参考对照：修复前 44/256 个编码不一致，修复后全部 256 个编码一致。
- 外部补充检查：日志参数 SPLIT 31 项、外部随机编码 Acc→Mat 6 项通过；
  这些检查依赖仓外材料，不是本目录 `run.py` 的 12 项，也不是原始 114 项全集重跑。
- 未进行此次修复的 NPU 硬件数值精度验证；`half boundary_maxarea` 仍未确认修复。
