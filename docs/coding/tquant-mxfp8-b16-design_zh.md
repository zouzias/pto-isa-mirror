# TQuant MXFP8 BF16/FP16 设计说明

本文档说明 `include/pto/npu/a5/TQuant.hpp` 中 BF16/FP16 到 MXFP8 E4M3 的量化设计。
目标是把每 32 个元素作为一个 MX block，计算一个 E8M0 block scale，并用该 scale 对
block 内元素做 FP8 E4M3 量化。

## 参考语义

OCP MX 规格中，MXFP8 使用 FP8 element 和 E8M0 block scale，block size 为 32。
E8M0 是 8-bit biased exponent，bias 为 127，表示 `2^(encoded - 127)`。
E8M0 没有 zero 和 infinity 编码，`0xFF` 是唯一 NaN 编码。

E4M3 的最大 finite 值是 `448 = 1.75 * 2^8`，因此从指数角度看，量化前希望把
block 最大值缩放到 real exponent 约为 8：

```text
scaled_value = input / block_scale
target: floor(log2(abs(scaled_value))) ~= 8
```

实现上，数据路径不直接除以 `block_scale`，而是预先构造 reciprocal scale：

```text
scaled_value = input * reciprocal_scale
reciprocal_scale = 1 / block_scale
```

注意：当前设计按 exponent 选择 scale，不使用 mantissa 进一步修正。因此它保证的是
指数位约束到 E4M3 的最大 exponent 8，而不是严格保证所有 mantissa 都不触发最终
FP8 饱和。例如 real exponent 为 8 但 mantissa 大于 E4M3 最大 mantissa 的值，最终
FP8 转换仍可能饱和。

## 总体流程

1. 每 32 个元素作为一个 block。
2. 对每个 block 计算 BF16 abs raw bit pattern 的最大值。
3. 从最大 BF16 abs raw bits 中拆出 exponent 和 mantissa，得到 E8M0 encoded scale。
4. 根据 E8M0 encoded scale 构造 BF16 格式的 reciprocal scale。
5. 量化阶段执行 `input * reciprocal_scale`，再转为 FP8 E4M3。

## 为什么 BF16 和 FP16 统一到 BF16 exponent 逻辑

BF16 和 E8M0 都使用 8-bit exponent，bias 都是 127。对于 BF16，直接提取 exponent
即可和 E8M0 公式对齐。

FP16 的 exponent 只有 5 bit，bias 是 15。如果直接用 FP16 biased exponent 计算：

```text
e8m0_biased = fp16_biased + 104
```

这个公式只对 FP16 normal 数有效。FP16 subnormal 的 exponent field 恒为 0，但真实
指数取决于 mantissa 的 leading bit。如果直接把所有 FP16 subnormal 都当成
`fp16_biased = 0`，会丢失 subnormal 之间的数量级差异。

因此 FP16 路径先把输入值转换成 BF16，再取 BF16 exponent。BF16 的 exponent 范围比
FP16 大得多，FP16 subnormal 转成 BF16 后通常会成为 BF16 normal，从而保留真实指数。
这也是当前设计相对旧方案的关键修正。

## 最大值计算

当前实现计算的是每个 block 的 BF16 abs raw bit pattern 最大值，而不是只保存
exponent。保留 raw bits 的原因是后续需要通过 mantissa 区分 `Inf` 和 `NaN`。

BF16 路径：

```text
bf16_bits -> bf16_bits & 0x7FFF -> reduce max
```

FP16 路径：

```text
fp16_bits
  -> fp16 value 转 BF16
  -> bf16_bits & 0x7FFF
  -> reduce max
```

这样做的原因：

- scale 选择主要需要 block 最大 exponent；
- BF16 abs raw bit pattern 可直接用于 unsigned max，且 NaN payload 会排在 Inf 之后；
- 保留 mantissa 后可以区分 `Inf` 和 `NaN`；
- FP16 subnormal 先转 BF16 后能保留真实数量级；
- FP16/BF16 的后续 scale 计算可以统一在 BF16 exponent 坐标系上。

## E8M0 exp 计算

定义：

```text
realexp:      实际指数
biasedexp:    源类型带 bias 的 exponent
e8m0_biased:  E8M0 encoded exponent
```

目标是：

```text
realexp - e8m0_real_exp = 8
```

也就是：

```text
e8m0_real_exp = realexp - 8
e8m0_biased = e8m0_real_exp + 127
```

### BF16

BF16:

```text
realexp = bf16_biased - 127
```

代入：

```text
e8m0_biased = (bf16_biased - 127 - 8) + 127
             = bf16_biased - 8
```

但 E8M0 的最小 finite scale 是 encoded 0：

```text
encoded 0 -> scale = 2^-127
```

所以当 `bf16_biased <= 8` 时，不能继续减到负数，只能 clamp 到 0：

```text
e8m0_biased = (bf16_biased <= 8) ? 0 : bf16_biased - 8
```

这覆盖了 BF16 zero、BF16 subnormal、以及极小 normal。它们都会使用最小 E8M0 scale。
数值效果不同：

```text
zero * 2^127 = 0
bf16 subnormal * 2^127 可能变成非零可量化值
```

### FP16

如果 FP16 是 normal：

```text
realexp = fp16_biased - 15
e8m0_biased = (fp16_biased - 15 - 8) + 127
             = fp16_biased + 104
```

但该公式不适用于 FP16 subnormal。因此实际设计不是直接用 FP16 exponent，而是：

```text
fp16 value -> bf16 value -> bf16_biased -> 使用 BF16 公式
```

这能让 FP16 subnormal 的真实指数进入 scale 计算。

## reciprocal scale 计算

E8M0 encoded scale 表示：

```text
block_scale = 2^(e8m0_biased - 127)
```

量化阶段需要 reciprocal：

```text
reciprocal_scale = 2^(-(e8m0_biased - 127))
                 = 2^(127 - e8m0_biased)
```

reciprocal scale 在 scratch buffer 中统一按 BF16 bit pattern 保存。BF16 normal 的
biased exponent 为：

```text
bf16_recip_biased_exp = (127 - e8m0_biased) + 127
                      = 254 - e8m0_biased
```

因此统一公式是：

```text
reciprocal_scale_bf16_exp = 254 - e8m0_biased
```

边界：

- `e8m0_biased = 0` 时，reciprocal scale 是 `2^127`，BF16 bits 为 `0x7F00`。
- `e8m0_biased = 254` 时，reciprocal scale 是 `2^-127`，这是 BF16 subnormal，
  需要用 BF16 bits `0x0040` 表示。
- `e8m0_biased = 255` 是 E8M0 NaN，不是 finite scale。

FP16 也使用 BF16 reciprocal scale。原因是 FP16 无法表达很多大的 reciprocal scale。
例如 `e8m0_biased = 106` 时：

```text
reciprocal_scale = 2^(127 - 106) = 2^21
```

这已经超过 FP16 最大 exponent 15，但 BF16 可以表达。因此 scale scratch 不能按 FP16
解释，必须按 BF16 bit pattern 解释。

## 量化阶段

BF16 路径：

```text
input BF16
reciprocal scale BF16
BF16 multiply -> BF16 result
BF16 result -> FP32
FP32 -> FP8 E4M3
```

FP16 路径：

```text
input FP16 -> FP32
reciprocal scale BF16 -> FP32
FP32 multiply -> FP32 result
FP32 -> FP8 E4M3
```

FP16 之所以在 FP32 中乘，是因为 scale 存的是 BF16 bit pattern，不能按 FP16 解释；
同时 FP32 乘法也能避免 FP16 reciprocal scale 表达范围不足导致的溢出或精度损失。

## 特殊值处理

### zero

zero 的 exponent 为 0，走最小 E8M0 scale：

```text
e8m0_biased = 0
reciprocal_scale = 2^127
scaled_value = 0
```

因此 scale/encode 和 subnormal 相同，但数值效果仍然保持 0。

### BF16 subnormal

BF16 subnormal 的 exponent 为 0，也走：

```text
e8m0_biased = 0
reciprocal_scale = 2^127
```

这会尽量把 BF16 subnormal 放大到可量化范围。

### FP16 subnormal

FP16 subnormal 不直接用 FP16 exponent field。先转 BF16，再取 BF16 exponent。这样能
保留 FP16 subnormal 的真实数量级，不会把所有 FP16 subnormal 都当成同一个 exponent。

### Inf 和 NaN

OCP E8M0 没有 infinity 编码，`0xFF` 是 NaN。理论上如果要区分 BF16/FP16 `Inf` 和
`NaN`，需要在最大值阶段保留 mantissa 信息：

```text
exp all ones, mantissa == 0 -> Inf
exp all ones, mantissa != 0 -> NaN
```

当前设计已经让 `maxPtr` 保留 BF16 abs raw bits，所以可以区分：

```text
Inf:
  exp bits = 0x7F80
  mantissa = 0
  E8M0 encoded exp = 0xFE
  reciprocal scale = BF16 2^-127 = 0x0040

NaN:
  exp bits = 0x7F80
  mantissa != 0
  E8M0 encoded exp = 0xFF
  reciprocal scale = BF16 NaN customization, 当前为 0x7F81
```

这和 NN 仓
`/home/bynshard/ops-nn/quant/dynamic_mx_quant/op_kernel/arch35/dynamic_mx_quant_tail_axis_fp8.h`
不同。NN 仓的 OCP 路径只保存 BF16 exponent bits，到 `ComputeScaleOCP` 时，
`Inf` 和 `NaN` 都只剩：

```text
maxExp = 0x7F80
```

因此 NN 仓无法区分 `Inf` 和 `NaN`。本实现为了正确处理 `Inf`，有意把最大值阶段从
“保存 exponent bits”改为“保存 BF16 abs raw bits”。

## 旧设计问题总结

旧文档/旧实现里主要有几个问题：

1. FP16 直接用 `fp16_biased + 104` 会错误处理 FP16 subnormal。
2. FP16 scale 如果按 FP16 保存，会无法表达 `2^21` 这类大的 reciprocal scale。
3. BF16 `biased <= 8` 时不能继续做 `biased - 8`，必须 clamp 到 E8M0 encoded 0。
4. E8M0 encoded 0 不是数值 0，而是 `scale = 2^-127`。
5. `zero` 和 `subnormal` 选择相同 scale，但数值效果不同：zero 仍为 zero，subnormal
   会被 `2^127` 尽量放大。
6. `Inf/NaN` 是否区分取决于最大值阶段是否保留 mantissa。当前实现通过保存 BF16
   abs raw bits 区分两者；NN 仓因为只保存 exponent bits，无法区分。

## 参考资料

- OCP Microscaling Formats (MX) Specification v1.0:
  https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- Microsoft microxcaling:
  https://github.com/microsoft/microxcaling
- NVIDIA Transformer Engine MXFP8 documentation:
  https://nvidia.github.io/TransformerEngine/features/low_precision_training/mxfp8/mxfp8.html
- NN 仓实现：
  `/home/bynshard/ops-nn/quant/dynamic_mx_quant/op_kernel/arch35/dynamic_mx_quant_tail_axis_fp8.h`
