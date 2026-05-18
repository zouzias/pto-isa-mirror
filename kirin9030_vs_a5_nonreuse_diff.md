# Kirin9030 vs A5 Non-Reuse Instructions

范围：`include/pto/npu/kirin9030/` 中**本地实现**的 10 个非复用指令。

结论先行：`kirin9030` 基本是 A5 的收敛子集，主要收窄在 `bf16/fp8/fp4/hifloat8`、MX 路径、更多 layout/format 转换、以及 cross-core sync。

| 指令 | 支持的数据类型 | 支持的数据格式 / 分形 | 硬件指令差异 | 功能差异 | 备注 |
|---|---|---|---|---|---|
| `TLoad` | kirin9030: 以字节宽度为主的 `b8/b16/b32/b64` 路径，conv 路径覆盖 `int8/u8/int16/u16/int32/u32/half/float`；A5: 额外支持 `bfloat16/fp8/fp4/hifloat8`。 | kirin9030: Vec 的 `ND/DN/NZ`，Cube 的 `ND2NZ/DN2NZ/ND2ND/DN2DN/NZ2NZ/DN2ZN`；A5: 还支持 MX scale、`NHWC/NCHW/NCDHW`、更多 3D/conv 转换。 | kirin9030 主要是 `copy_gm_to_ubuf_align_v2`、`copy_gm_to_cbuf_align_v2`、`copy_gm_to_cbuf_multi_*`；A5 还带 `copy_gm_to_cbuf_multi_dn2nz`、额外 L2 控制位和 loop reset。 | kirin9030 没有 MX-scale、没有更丰富的 conv layout 变换，底层 DN2NZ 也更受限。 | A5 明显更宽。 |
| `TStore` | kirin9030: Acc 源仅 `int32/half`，Acc 输出仅 `int32/float/half`，Vec 侧支持传统整型/`half/float`；A5: 额外支持 `bfloat16/fp8/fp4/hifloat8`，Acc 源也支持 `float`。 | kirin9030: Acc 存储仅 `ND/NZ`，Vec 存储仅 `ND/DN/NZ` + 1 行/1 列特例；A5: 额外支持 `NHWC/NCHW/NCDHW` 和更细的 NZ 变体。 | kirin9030 用较简单的 `copy_ubuf_to_gm_align_v2`；A5 的 `copy_matrix_cc_to_gm` / `copy_ubuf_to_gm_align_v2` 带更多控制参数。 | kirin9030 不支持 A5 的 NHWC/NCHW/NCDHW Acc store，也没有更丰富的 pre-quant / atomic 类型集合。 | `TStoreAcc*` 是差异核心。 |
| `TMatmul` | kirin9030: 标准 matmul 只走 `half` 或 `int32` Acc；A5: 标准 matmul 支持 `int32/float` Acc，输入可覆盖 `int8/half/bfloat16/float/fp8/hifloat8`，MX 还支持 `fp4/fp8` 组合。 | kirin9030: 只做标准 `Left/Right/Acc` fractal；A5: 标准 fractal 基本一致，但另有 MX fractal 约束和额外 K 维对齐要求。 | kirin9030 只用 `mad(...)`；A5 还有 `mad_mx(...)`。 | kirin9030 不支持 MX matmul，也不支持 A5 的更宽 dtype 组合。 | `TMatmul` 是最明确的 A5 子集。 |
| `TExtract` | kirin9030: 主要是 `int8/u8/int16/u16/int32/u32/half/float`；A5: 额外支持 `bfloat16/fp8/fp4/hifloat8`。 | kirin9030: 主要是 L0/Vec/Acc 的经典提取；A5: 额外支持 ScaleLeft/ScaleRight、conv-tile、`CA/CB/CC/FBUF` 路径，以及更多 fp4/fp8 对齐规则。 | kirin9030 主要是 `load_cbuf_to_ca/cb`、`copy_matrix_cc_to_*`；A5 还有 `*_mx`、`*_s4`、`vgather2`、`vlds/vsts` 路径。 | kirin9030 没有 A5 的 Scale 提取、conv-tile 提取和更宽的格式族。 | A5 的提取能力更像“全功能版”。 |
| `TMov` | kirin9030: Bias/Scaling/Acc 相关搬运以 `half/int32` 为主，普通路径也更窄；A5: 额外支持 `bfloat16/fp8/fp4/hifloat8/float` 等。 | kirin9030: 偏 Bias/Scaling、Acc→Vec/Mat；A5: 额外支持 ND→ZZ、ND/NZ 更多转换、MX scale 路径。 | kirin9030 主要是 `copy_cbuf_to_bt/fbuf`、`copy_matrix_cc_to_cbuf/ub`；A5 额外有 `vgather2`、`pto_copy_ubuf_to_ubuf`、更多控制位和 packed-fp4 处理。 | kirin9030 不支持 A5 的 fp8/fp4/bfloat16 广覆盖搬运，也没有 ND→ZZ 这类扩展路径。 | kirin9030 更像简化搬运器。 |
| `TInsert` | kirin9030 本身不重写核心插入逻辑，主要通过宏和本地 `TMov.hpp` 适配 A5 实现。 | 与 A5 基本一致，仍覆盖 Acc→Mat/Vec、ND/NZ 插入等。 | 差异主要在 `copy_matrix_cc_to_cbuf(...)` 的宏参数封装；kirin9030 还显式引入了 `pto/npu/kirin9030/TMov.hpp`。 | 功能上更像兼容薄封装，不是独立重写。 | 这是最接近 A5 的本地文件。 |
| `TCvt` | kirin9030: 支持经典 `float/half/int8/int16/int32/u8/u16/u32` 转换；A5: 额外支持 `bfloat16/int64/fp8/fp4/hifloat8`。 | 两者都主要是 UB 上的 1D/2D 规则转换；差异不在 layout，而在 dtype matrix。 | kirin9030 某些 32->16 路径用 `RS_ENABLE`；A5 对应实现用法更偏 `RS_DISABLE`/`__cce_simd::Round*Type` 体系。 | A5 的 rounding / conversion matrix 更完整，尤其是 BF16、FP8、FP4、INT64。 | `TCvt` 是 dtype 差异最集中的文件。 |
| `TQuant` | kirin9030: 只做 `INT8_SYM/INT8_ASYM`，输入基本是 `float32`；A5: 额外支持 `MXFP8`、`MXFP4_E2M1`，输入可扩到 `half/bfloat16`。 | kirin9030 只做普通 2D row-wise quant；A5 的 MX 路径要求 ND，且有 group/scale 的额外约束。 | kirin9030 主要是 `vmul/vcvt/vsts`；A5 还加入 `vabs/vcmax/vcgmax`、控制位设置和更重的中间态处理。 | kirin9030 明确不支持 MX；A5 支持 MX quant。 | A5 的量化路径明显更宽。 |
| `TGather` | kirin9030: 基础 indexed gather + mask gather，类型集中在传统整型/`half/float`；A5: 额外支持 `fp8`、`bfloat16`、`hifloat8`，还有 compare-gather。 | 两者都以 Vec/row-major 为主；A5 对 mask / compare 路径的类型和模式更丰富。 | kirin9030 主要是 `vgather2/vgather2_bc/vsqz/vstur/vstar/sprclr`；A5 额外有 `vci/vcmps_gt/eq/vcmp_gt/eq/sprsts`。 | A5 新增了 FP8 gather 和“比较后生成索引”的 gather 语义。 | `TGather` 在 A5 的功能面更大。 |
| `TSync` | 该指令不涉及 dtype。 | 该指令不涉及数据格式 / 分形。 | kirin9030 允许更宽的 pipe barrier 范围；A5 只允许 `PIPE_MTE2/MTE3/ALL`，并新增 `set_intra_block/wait_intra_block`。 | A5 增加 cross-core event，同 pipe / auto-token 的限制更严格。 | 这是同步语义上的差异，而不是数据面差异。 |

## 总结

- kirin9030 的本地实现整体偏“保守版”，大多数文件保留 A5 的基础语义，但收窄了 dtype 和 layout。
- 差异最大的集中在 `TLoad/TStore/TMatmul/TExtract/TMov/TQuant/TGather`。
- `TInsert` 基本是 A5 实现的兼容薄封装，不是独立重写。
- `TSync` 的差异主要体现在事件/pipe 约束和 cross-core 能力。
