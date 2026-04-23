/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

# MXFP8(E4M3) 量化算法形式化说明

本文整理 `include/pto/npu/a5/TQuant.hpp` 中 `TQuant_MXFP8_F32` 与
`TQuant_MXFP8_B16` 的算法设计，并对照 OCP MX 标准与当前工程实现进行说明。

本文目标：

- 形式化描述 `MXFP8(E4M3)` 的 block scale 计算方法
- 解释 `bf16 / fp16 / fp32` 三条路径的指数与 scale 推导
- 明确 zero、极小值、`Inf/NaN` 的当前约定
- 指出原始推导中容易混淆或不正确的地方

## 1. 标准背景

### 1.1 OCP MX 的基本定义

根据 OCP MX 规范，MX block 由两部分组成：

- block scale `X`
- block 内元素 `P_i`

解码关系为：

- 若 `X = NaN`，则整块值全为 `NaN`
- 若 `X != NaN`，则每个元素满足 `v_i = X * P_i`

对从标量向量 `V = [V_i]` 转为 MX-compliant format 的最低支持语义，OCP 规定：

1. `X` 取为 `max(|V_i|)` 的不超过它的最大 2 次幂，再除以元素格式可表示的最大 2 次幂。
2. `P_i = quantize(V_i / X)`。

对 `MXFP8 E4M3`：

- block 大小为 32
- scale 类型为 `E8M0`
- `E8M0` bias 为 `127`
- `E8M0` 支持指数范围 `[-127, 127]`
- `E8M0` 没有 zero encoding，只有一个 `NaN = 0xFF`

### 1.2 为什么本文里目标指数是 8

对 `E4M3`，最大可表示值不是 2 的整数次幂，而是 `448`。  
但 OCP 在 conversion 最低语义里要求的是：

- “除以元素格式**可表示的最大 2 次幂**”

对 `E4M3`，最大可表示的 2 次幂是：

- `2^8`

因此，如果采用“先求 `floor(log2(absmax))` 再构造 E8M0”的实现方式，那么目标就是：

- 让量化后 block 最大值的指数落到 `8`

这正是本文中的推导基础。

补充说明：

- 一些实现（例如 NVIDIA Transformer Engine）会用 `amax / 448` 再转 `E8M0`
- 这相当于把 mantissa 也纳入考虑，数值更细
- 而当前 PTO 路径采用的是“只基于指数”的构造方式，因此核心目标是指数对齐到 `8`

## 2. 记号定义

本文统一使用以下符号：

- `realexp`
  - 某一浮点数的真实指数
- `biasedexp`
  - 某一浮点格式里的 biased exponent
- `e8m0_biasedexp`
  - 输出到 `expPtr` 的 E8M0 指数编码
- `shared_scale`
  - block scale `X`
- `recip_scale`
  - 量化乘法内部使用的倒数 scale，即 `1 / X`

格式 bias：

- `fp32` bias = `127`
- `bf16` bias = `127`
- `fp16` bias = `15`
- `e8m0` bias = `127`

## 3. 算法目标

对于一个 block 的绝对值最大值 `absmax`，我们希望：

- `absmax / shared_scale` 的指数大致落在 `8`

等价地：

- `shared_scale` 约等于 `absmax / 2^8`

而量化时实际做的是：

- `q_i = value_i * recip_scale`
- 其中 `recip_scale = 1 / shared_scale`

因此：

- 先求 `shared_scale`
- 再构造 `recip_scale`
- 最后做 `value * recip_scale -> E4M3`

## 4. AbsMax 计算

### 4.1 现有实现

当前实现中：

- `fp32` 走 `AbsReduceMax_*`
- `bf16/fp16` 走 `AbsReduceMax_b16_ND*`

流程本质是：

1. 对输入做 `abs`
2. 做 block 内 max
3. 每 32 个元素归约出一个 `absmax`

### 4.2 关于 `bf16 / fp16` 共用 16-bit 路径

这一点需要表述准确。

可以说：

- `bf16/fp16` 目前共用一套 16-bit `absmax` 归约框架

但不建议直接写成：

- “把 `fp16` 和 `bf16` 都统一当做 `bf16` 来求 absmax”

因为当前实际代码里：

- `fp16` 输入最终仍通过 `(vector_f16&)` 路径参与 `abs/max/reduce`
- 它不是简单按 `bf16` 数值解释后做整数比较

更严谨的说法是：

- 它们共用同一套 16-bit 向量归约结构
- 对 `Inf` 能正确保留
- 对 `NaN` 的正确性依赖 `vmax / vcgmax` 的 NaN 传播语义

### 4.3 `0 / Inf / NaN`

在当前约定下：

- `0`
  - `absmax = 0`
  - 后续 special 逻辑会单独处理
- `Inf`
  - `abs(Inf) = Inf`
  - 一般会成为 block max
- `NaN`
  - 若 `vmax / vcgmax` 传播 `NaN`，则 block max 会保留 special
  - 若底层实现忽略 `NaN`，则 mixed-NaN block 的结果会依赖底层指令语义

本文后续 special 推导都基于以下工程前提：

- `vmax / vcgmax` 传播 `NaN`

## 5. `bf16 / fp32` 的指数与 scale 推导

`bf16` 与 `fp32` 的 bias 都是 `127`，所以指数推导完全同型。

### 5.1 `e8m0` 指数推导

我们希望：

- `quantized_realexp = 8`

于是：

- `source_realexp - e8m0_realexp = 8`

因此：

- `e8m0_realexp = source_realexp - 8`

转成 E8M0 biased exponent：

- `e8m0_biasedexp = e8m0_realexp + 127`
- `= (source_realexp - 8) + 127`

对 `bf16/fp32`，因为：

- `source_realexp = source_biasedexp - 127`

所以：

- `e8m0_biasedexp = (source_biasedexp - 127 - 8) + 127`
- `= source_biasedexp - 8`

即：

```text
e8m0_biasedexp = source_biasedexp - 8
```

### 5.2 小指数 underflow 问题

这里正是你原始推导抓到的核心问题。

因为：

- `E8M0` 支持的指数范围是 `[-127, 127]`
- `bf16/fp32` 正常数范围是 `[-126, 127]`

若 `source_realexp < -119`，则：

- `e8m0_realexp = source_realexp - 8 < -127`

这超出了 `E8M0` 可表示范围。

例如：

- `bf16 2^-120`
- `biasedexp = 7`
- 原始公式给出 `e8m0_biasedexp = 7 - 8 = -1`

这是非法的。

因此对 `bf16/fp32` 必须加下限钳位：

```text
e8m0_biasedexp = source_biasedexp <= 8 ? 0 : source_biasedexp - 8
```

这表示：

- 当值已经小到需要的 shared scale 低于 `2^-127` 时
- 直接把 block scale 钳在 `E8M0` 最小正常值 `2^-127`

### 5.3 reciprocal scale 推导

量化时用的是：

- `recip_scale = 1 / shared_scale`

若当前 block 的 E8M0 编码是 `e8m0_biasedexp`，则：

- `e8m0_realexp = e8m0_biasedexp - 127`
- `shared_scale = 2^(e8m0_realexp)`

所以：

- `recip_scale = 2^(-e8m0_realexp)`

如果要把这个 reciprocal scale 构造成 `bf16/fp32` 的 biased exponent，公式是：

- `scale_biased_exp = -e8m0_realexp + bias`

对 `bf16/fp32`，bias 是 `127`，所以：

- `scale_biased_exp = -(e8m0_biasedexp - 127) + 127`
- `= 254 - e8m0_biasedexp`

因此：

```text
scale_biased_exp = 254 - e8m0_biasedexp
```

### 5.4 这里原始推导里最容易错的地方

你原始草稿里有一句：

- `scale_bf16_biased_exp = bf16_biased <= 8 ? 0 : ...`

这对**非零极小值**是不对的。

正确逻辑不是：

- “小于等于 8 就不做 scale”

而是：

- 先把 `e8m0_biasedexp` 钳到 `0`
- 再根据钳后的 `e8m0_biasedexp` 去构造 reciprocal scale

也就是：

```text
e8m0_biasedexp = source_biasedexp <= 8 ? 0 : source_biasedexp - 8
scale_biased_exp = 254 - e8m0_biasedexp
```

于是当 `source_biasedexp <= 8` 且输入非零时：

- `e8m0_biasedexp = 0`
- `scale_biased_exp = 254`

这对应：

- reciprocal scale = `2^127`

这正是当前 `bf16/fp32` 修正后的实现语义。

### 5.5 exact zero 必须单独处理

`E8M0` 没有 zero encoding。  
因此 exact zero block 需要单独约定。

当前工程约定是：

- `exp = 0`
- 内部 reciprocal scale scratch = `0`

这与“极小非零值”的行为不同：

- 极小非零值：`exp = 0`, `scale = 2^127`
- exact zero：`exp = 0`, `scale = 0`

所以不能把：

- subnormal
- 极小 normal
- exact zero

混成同一个分支。

## 6. `fp16` 的指数与 scale 推导

### 6.1 `e8m0` 指数推导

仍然从目标关系开始：

- `fp16_realexp - e8m0_realexp = 8`

因此：

- `e8m0_realexp = fp16_realexp - 8`

转成 E8M0 biased exponent：

- `e8m0_biasedexp = e8m0_realexp + 127`
- `= (fp16_realexp - 8) + 127`

而：

- `fp16_realexp = fp16_biasedexp - 15`

所以：

- `e8m0_biasedexp = (fp16_biasedexp - 15 - 8) + 127`
- `= fp16_biasedexp + 104`

即：

```text
e8m0_biasedexp = fp16_biasedexp + 104
```

由于：

- `fp16_biasedexp ∈ [0, 31]`

所以：

- `e8m0_biasedexp ∈ [104, 135]`

因此对 `fp16`：

- `shared_exp` 本身不会像 `bf16/fp32` 那样掉到 0 以下
- 不需要 `bf16` 那种 `<=8` 下限钳位

### 6.2 为什么 `fp16` 的 reciprocal scale 不能按 `fp16` 构造

如果直接按 `fp16` 自己的 biased exponent 来构造 reciprocal scale：

- `scale_fp16_biased_exp = -e8m0_realexp + 15`

代入：

- `e8m0_realexp = fp16_realexp - 8`

得到：

- `scale_fp16_biased_exp = -(fp16_realexp - 8) + 15`
- `= -(fp16_biasedexp - 15 - 8) + 15`
- `= 38 - fp16_biasedexp`

而：

- `fp16_biasedexp ∈ [0, 31]`
- `38 - fp16_biasedexp ∈ [7, 38]`

问题在于：

- `fp16` 最大正常 biased exponent 只有 `30`

因此当输入较小时，正确 reciprocal scale 会超出 `fp16` 可表示范围。

例如：

- `fp16 max = 2^-13`
- `fp16_biasedexp = 2`
- `e8m0_biasedexp = 106`
- `shared_scale = 2^(106 - 127) = 2^-21`
- 正确 reciprocal scale 应为 `2^21`

但：

- `2^21` 不可能作为 IEEE fp16 正常数表示

### 6.3 正确做法：用 `bf16` 或 `fp32` 表达 reciprocal scale

因此 `fp16` 路径应当：

- `expPtr` 继续输出正确的 E8M0
- 但内部 reciprocal scale 不再用 IEEE fp16 编码

当前工程采用的是：

- reciprocal scale 以 **BF16 编码** 的 16-bit scratch 保存
- 在量化乘法时：
  - `half -> float`
  - `bf16-encoded scale -> float`
  - 再在 `float` 中相乘

### 6.4 `fp16` 的 BF16-encoded reciprocal scale 推导

因为 reciprocal scale 最终按 `bf16` 编码保存，所以它的 biased exponent 应是：

- `scale_bf16_biased_exp = -e8m0_realexp + 127`

继续展开：

- `e8m0_realexp = e8m0_biasedexp - 127`

所以：

- `scale_bf16_biased_exp = -(e8m0_biasedexp - 127) + 127`
- `= 254 - e8m0_biasedexp`

再代入：

- `e8m0_biasedexp = fp16_biasedexp + 104`

得到：

- `scale_bf16_biased_exp = 254 - (fp16_biasedexp + 104)`
- `= 150 - fp16_biasedexp`

即：

```text
scale_bf16_biased_exp = 150 - fp16_biasedexp
```

或等价地：

```text
scale_bf16_biased_exp = 254 - e8m0_biasedexp
```

这就是当前代码中：

- `recip_scale_base = 0xFE`

的来源。

它不是在构造 `fp16` scale，  
而是在构造 **BF16-encoded reciprocal scale**。

### 6.5 例子：`fp16 max = 2^-13`

有：

- `fp16_realexp = -13`
- `fp16_biasedexp = 2`

则：

- `e8m0_biasedexp = 2 + 104 = 106`
- `shared_scale = 2^(106 - 127) = 2^-21`
- 正确 reciprocal scale = `2^21`

按 BF16 编码构造：

- `scale_bf16_biased_exp = 254 - 106 = 148`

所以 scale bits 为：

- `148 << 7 = 0x4A00`

`0x4A00` 按 BF16 解释，正是：

- `2^21`

量化乘法：

- `2^-13 * 2^21 = 2^8`

这正好把 block max 映射到目标指数 `8`。

## 7. `Inf / NaN / Zero` 的实现约定

### 7.1 exact zero

当前落地约定：

- block max 精确等于 0
- 则：
  - `exp = 0`
  - internal reciprocal scale = `0`

这不是 OCP 唯一指定的语义，而是实现约定。

### 7.2 极小非零值

对 `bf16/fp32`：

- 若需要的 `e8m0_realexp < -127`
- 则钳位到：
  - `e8m0_biasedexp = 0`
  - reciprocal scale = `2^127`

对 `fp16`：

- `shared_exp` 不会 underflow 到 0 以下
- 但 reciprocal scale 可能大于 `fp16` 可表示范围
- 因此使用 BF16-encoded scale

### 7.3 `Inf / NaN`

当前工程为与 `ops-nn` 路径对齐，采用以下约定：

- 若 block max 为 special
  - `exp = 0xFF`
  - internal reciprocal scale = NaN

这使得量化数据路径也能稳定进入 special 结果。

需要注意：

- mixed `NaN` block 的正确性依赖 block max 归约阶段能保留 `NaN`
- 这在当前实现里依赖底层 `vmax / vcgmax` 的 NaN 传播语义

## 8. 对原始设计的结论

你的整体设计方向是对的，关键正确点包括：

1. 目标确实是让量化后 block max 的指数落到 `8`
2. `bf16/fp32` 确实存在 `biasedexp <= 8` 的 underflow 问题
3. `fp16` 的 `shared_exp` 不会出现同类 underflow
4. `fp16` 的真正问题是 reciprocal scale 装不进 `fp16`
5. `fp16` 应统一用 `bf16` 或 `fp32` 表达 reciprocal scale，并在乘法时提升精度

但有两点必须修正：

### 8.1 错误点一：不能把极小非零值当成 “不做 scale”

原始想法中：

- `bf16_biased <= 8 ? scale = 0`

这是不对的。

正确做法是：

- exact zero 才设 `scale = 0`
- 极小非零值应设：
  - `exp = 0`
  - `scale = 2^127`

### 8.2 错误点二：`fp16/bf16` 共用 absmax 路径的表述要更严谨

不建议简单写成：

- “统一当做 bf16 来求 absmax”

更准确的说法是：

- 当前实现共用一套 16-bit 向量归约框架
- `Inf` 可正确保留
- `NaN` 的正确性依赖底层 max 指令的传播语义

## 9. 推荐的最终公式

### 9.1 `bf16 / fp32`

```text
source_biasedexp = extract_exp(source)
e8m0_biasedexp = source_biasedexp <= 8 ? 0 : source_biasedexp - 8
scale_biased_exp = 254 - e8m0_biasedexp
```

额外 special 约定：

```text
if exact_zero:
    e8m0_biasedexp = 0
    reciprocal_scale = 0
if special:
    e8m0_biasedexp = 0xFF
    reciprocal_scale = NaN
```

### 9.2 `fp16`

```text
fp16_biasedexp = extract_exp(source)
e8m0_biasedexp = fp16_biasedexp + 104
scale_bf16_biased_exp = 254 - e8m0_biasedexp
```

等价写法：

```text
scale_bf16_biased_exp = 150 - fp16_biasedexp
```

乘法路径：

```text
value_fp16 -> float
scale_bf16 -> float
mul in float
cast to e4m3
```

## 10. 参考资料

### OCP MX 标准

- Open Compute Project, *OCP Microscaling Formats (MX) Specification*, Version 1.0, Sep 2023.
  - E8M0 定义：Section 5.4.1
  - 标量向量到 MX 的最低支持转换语义：Section 6.3
  - 链接：
    - https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

### 一个常见实现说明

- NVIDIA Transformer Engine, *MXFP8 documentation*
  - 描述了 `amax_block / max_fp8` 的工程实现思路
  - 其中 `max_fp8 = 448`，更偏向 mantissa-aware 的实现
  - 链接：
    - https://nvidia.github.io/TransformerEngine/features/low_precision_training/mxfp8/mxfp8.html
