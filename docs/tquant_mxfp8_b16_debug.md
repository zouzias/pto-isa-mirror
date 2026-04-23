/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

# TQuant MXFP8 B16 调试说明

这份文档说明当前 `include/pto/npu/a5/TQuant.hpp` 中
`bf16/fp16 -> MXFP8 E4M3` 路径的算法行为。

范围：

- 只关注 `TQuant_MXFP8_B16`
- 只关注算法语义
- 不覆盖偏移、尾块、UB 对齐这类实现问题

相关函数：

- `AbsReduceMax_b16_ND*`
- `ExtractB8ExponentAndScaling<T>`
- `CalcQuantizedFP8Values<T>`

## 前提假设

本文基于以下前提：

1. `vabs` 对 `NaN/Inf` 仍保持 special 语义。
2. `vmax / vcgmax` 对 `NaN` 采用传播语义。
3. 对一个 32 元素 block，只要其中有一个元素是 `NaN`，最终传到
   `ExtractB8ExponentAndScaling` 的 block max 也会是 special。

在这些前提下，当前算法在下面这些 case 上是自洽的。

## 总体流程

对一个 32 元素 block，整体流程如下：

1. `AbsReduceMax_b16_ND*`
   - 读入 16-bit 输入
   - 做 `abs`
   - 做 `max`
   - 每 32 个元素归约出一个 max
   - 写到 `maxPtr`

2. `ExtractB8ExponentAndScaling<T>`
   - 从 `maxPtr` 提取 exponent
   - 计算 `shared_exp`
   - 构造内部 reciprocal scale scratch
   - 处理 zero / special
   - 写出：
     - `expPtr`：E8M0 shared exponent
     - `scalingPtr`：内部 reciprocal scale scratch

3. `CalcQuantizedFP8Values<T>`
   - 原始输入乘 reciprocal scale
   - 再转成 FP8 E4M3

## 关键常量

这些常量来自当前的 `ExtractB8ExponentAndScaling<T>`。

对 `bf16`：

- exponent mask：`0x7F80`
- exponent shift：`7`
- special exponent check：`0xFF`
- `shared_exp = biasedexp - 8`
- 如果 `biasedexp <= 8`，则钳成 `shared_exp = 0`
- reciprocal 最小 scale：`0x7F00`，也就是 `bf16 2^127`

对 `fp16`：

- exponent mask：`0x7C00`
- exponent shift：`10`
- special exponent check：`0x1F`
- `shared_exp = biasedexp - (8 - 112) = biasedexp + 104`
- reciprocal scale scratch 仍然存在 16-bit buffer 里，但这里存的不是
  IEEE fp16，而是 `bf16` 编码，后面按 `bf16 -> float` 解释

`bf16/fp16` 共用的 special 规则：

- 精确零 block：
  - `exp = 0`
  - reciprocal scale scratch = `0`
- special block（`Inf` 或 `NaN` max）：
  - `exp = 0xFF`
  - reciprocal scale scratch = `0x7F81`（`bf16 NaN`）

## 调试时先分清三件事

调试时建议先把问题拆成这三步：

1. `AbsReduceMax_b16_ND*` 之后，`maxPtr` 里到底是什么？
2. `ExtractB8ExponentAndScaling<T>` 从这个 max 里提取出了什么 exponent？
3. `CalcQuantizedFP8Values<T>` 最终用了什么 reciprocal scale？

三步中只要有一步错了，最终 FP8 结果就会错。

## Case 1：BF16 全 0 Block

输入例子：

- 32 个元素
- 每个元素都是 `0x0000`（`bf16 +0`）

bit 字段：

- sign = `0`
- exponent = `0`
- mantissa = `0`

### 第一步：`AbsReduceMax_b16_ND*`

- `abs(+0) = +0`
- block max 仍然是 `0x0000`
- `maxPtr[group] = 0x0000`

### 第二步：`ExtractB8ExponentAndScaling<bfloat16_t>`

- `vb16_max = 0x0000`
- `vb16_exponent = (0x0000 & 0x7F80) >> 7 = 0`
- 中间值 `shared_exp = 0 - 8 = -8`
- `preg_no_scale = true`，因为 `0 <= 8`
- 如果只看 BF16 小指数分支，会先变成：
  - `shared_exp = 0`
  - `scale = 0x7F00`
- 但后面 exact zero 分支会覆盖掉它：
  - `preg_zero = true`
  - `vb16_shared_exp = 0`
  - `vb16_scaling = 0`

最终 block 输出：

- `expPtr[group] = 0x00`
- `scalingPtr[group] = 0x0000`

### 第三步：`CalcQuantizedFP8Values<bfloat16_t>`

- `value * scale = 0 * 0 = 0`
- 最终 FP8 数据全 0

## Case 2：BF16 极小非零值 `5.51e-39`

输入例子：

- 32 个元素
- 每个元素都是 `5.51e-39`
- `bf16` bit pattern：`0x003C`

核对过的值：

- `0x003C -> 5.5101297694794727e-39`

bit 字段：

- sign = `0`
- exponent = `0`
- mantissa = `0x3C`

这是一个 `bf16` subnormal，不是 0。

### 第一步：`AbsReduceMax_b16_ND*`

- `abs(x) = x`
- block max 仍然是 `0x003C`
- `maxPtr[group] = 0x003C`

### 第二步：`ExtractB8ExponentAndScaling<bfloat16_t>`

- `vb16_max = 0x003C`
- `vb16_exponent = (0x003C & 0x7F80) >> 7 = 0`
- 原始公式得到 `shared_exp = 0 - 8 = -8`
- `preg_no_scale = true`，因为 `0 <= 8`
- BF16 小指数规则生效：
  - `shared_exp = 0`
  - reciprocal scale scratch = `0x7F00`
- exact zero 不会命中，因为 raw bits 不是 0
- special 也不会命中，因为 exponent 不是全 1

最终 block 输出：

- `expPtr[group] = 0x00`
- `scalingPtr[group] = 0x7F00`

`0x7F00` 按 `bf16` 解释就是：

- `2^127 = 1.7014118346046923e+38`

### 第三步：`CalcQuantizedFP8Values<bfloat16_t>`

量化乘法实际做的是：

- `q = 5.5101297694794727e-39 * 2^127`
- `q ≈ 0.93749999999999989`

再转 E4M3 后，会得到一个非零有限 FP8 值，接近 `0.9375`。

这就是之前出错的那个 case。  
之前错在 reciprocal scale 被错误写成了 `1.0`。

## Case 3：BF16 小 normal 值 `2^-120`

输入例子：

- 32 个元素
- 每个元素都是 `2^-120`
- `bf16` bit pattern：`0x0380`

核对过的值：

- `0x0380 -> 7.5231638452626401e-37`

bit 字段：

- sign = `0`
- exponent = `7`
- mantissa = `0`

### 第一步：`AbsReduceMax_b16_ND*`

- block max 仍然是 `0x0380`

### 第二步：`ExtractB8ExponentAndScaling<bfloat16_t>`

- `vb16_exponent = 7`
- 原始公式得到 `shared_exp = 7 - 8 = -1`
- `preg_no_scale = true`，因为 `7 <= 8`
- BF16 小指数规则覆盖：
  - `shared_exp = 0`
  - reciprocal scale scratch = `0x7F00 = 2^127`

最终 block 输出：

- `expPtr[group] = 0x00`
- `scalingPtr[group] = 0x7F00`

### 第三步：`CalcQuantizedFP8Values<bfloat16_t>`

- `q = 2^-120 * 2^127 = 2^7 = 128`
- 最终 FP8 数据是有限且非零的

这个 case 最早暴露出来的是：

- `shared_exp` 算成负数
- reciprocal scale 也跟着给错

## Case 4：FP16 小值 `2^-8`

输入例子：

- 32 个元素
- 每个元素都是 `2^-8`
- `fp16` bit pattern：`0x1C00`

核对过的值：

- `0x1C00 -> 0.00390625`

bit 字段：

- sign = `0`
- exponent = `7`
- mantissa = `0`

### 第一步：`AbsReduceMax_b16_ND*`

- `abs(x) = x`
- block max 保持不变

### 第二步：`ExtractB8ExponentAndScaling<half>`

- `vb16_exponent = (0x1C00 & 0x7C00) >> 10 = 7`
- FP16 要先映射到 E8M0 bias：
  - `shared_exp = 7 - (8 - 112) = 111`
- 这里不会走 BF16 那种 `shared_exp = 0` clamp
- reciprocal scale scratch 按 BF16 编码构造：
  - `scale_bits = (0xFE - 111) << 7 = 0x4780`

最终 block 输出：

- `expPtr[group] = 111`
- `scalingPtr[group] = 0x4780`

`0x4780` 按 `bf16` 解释就是：

- `65536 = 2^16`

### 第三步：`CalcQuantizedFP8Values<half>`

重点在这里：  
FP16 路径现在不是在 FP16 里做乘法。

当前实际路径是：

1. `half -> float`
2. `bf16-encoded scale -> float`
3. 在 `float` 中乘

所以真正做的是：

- `q = 2^-8 * 2^16 = 256`

这就绕开了旧问题：

- 如果把 `2^16` 当 IEEE fp16 存，会直接溢成 `Inf`

## Case 5：BF16 / FP16 常见 normal 值

这两个 case 很适合做 sanity check，因为它们不是极值，是常见 finite block。

### BF16 值 `1.0`

输入：

- `bf16` bits：`0x3F80`
- exponent = `127`

则有：

- `shared_exp = 127 - 8 = 119`
- reciprocal scale exponent = `254 - 119 = 135`
- reciprocal scale scratch = `135 << 7 = 0x4380`
- `0x4380` 按 `bf16` 解释是 `256`
- 量化乘法：
  - `q = 1.0 * 256 = 256`

### FP16 值 `0.5`

输入：

- `fp16` bits：`0x3800`
- exponent = `14`

则有：

- `shared_exp = 14 + 104 = 118`
- reciprocal scale scratch = `(254 - 118) << 7 = 0x4400`
- `0x4400` 按 `bf16` 解释是 `512`
- 量化乘法：
  - `q = 0.5 * 512 = 256`

这两个普通 case 的意义是：

- 虽然源格式不同
- 但 block max 接近 E4M3 动态边界时，最终都会被映射到相近的量化区间

## Case 6：BF16 全 `Inf`

输入例子：

- 32 个元素
- 每个元素都是 `bf16 +Inf`
- bits：`0x7F80`

### 第一步：`AbsReduceMax_b16_ND*`

- `abs(+Inf) = +Inf`
- block max 仍然是 special

### 第二步：`ExtractB8ExponentAndScaling<bfloat16_t>`

- 提取出的 exponent 是全 1：`0xFF`
- exact zero 不命中
- special 命中：
  - `shared_exp = 0xFF`
  - reciprocal scale scratch = `0x7F81`

最终 block 输出：

- `expPtr[group] = 0xFF`
- `scalingPtr[group] = 0x7F81`

`0x7F81` 按 `bf16` 解释是一个 quiet NaN payload。

### 第三步：`CalcQuantizedFP8Values<bfloat16_t>`

- 数据会乘上 `bf16 NaN`
- 整个 block 走 special / NaN 路径

## Case 7：FP16 全 `Inf`

输入例子：

- 32 个元素
- 每个元素都是 `fp16 +Inf`
- bits：`0x7C00`

### 第一步：`AbsReduceMax_b16_ND*`

- `abs(+Inf) = +Inf`
- block max 仍然是 special

### 第二步：`ExtractB8ExponentAndScaling<half>`

- 提取出的 exponent 是全 1：`0x1F`
- special 命中：
  - `shared_exp = 0xFF`
  - reciprocal scale scratch = `0x7F81`

最终 block 输出：

- `expPtr[group] = 0xFF`
- `scalingPtr[group] = 0x7F81`

### 第三步：`CalcQuantizedFP8Values<half>`

- scale scratch 会按 `bf16 NaN` 解释
- 输入值会先转成 `float`
- `float(value) * float(NaN) = NaN`
- 整个 block 走 special / NaN 路径

## Case 8：BF16 / FP16 任意包含 `NaN` 的 Block

这一节依赖文档最开头的前提：

- `vmax / vcgmax` 对 `NaN` 采用传播语义

如果这个前提成立，那么无论 `bf16` 还是 `fp16`：

1. 32 元素 block 里有一个元素是 `NaN`
2. `AbsReduceMax_b16_ND*` 会把这个 `NaN` 传播下来
3. `maxPtr[group]` 变成 special
4. `ExtractB8ExponentAndScaling<T>` 看到 exponent 全 1
5. 最终 block 输出变成：
   - `expPtr[group] = 0xFF`
   - `scalingPtr[group] = 0x7F81`

所以在这个前提下，当前算法会把下面几类 block 都当成 special：

- 全 `NaN`
- finite + `NaN` 混合
- `Inf` + `NaN` 混合

## 调试时优先看什么

如果某个 block 结果不对，建议按这个顺序看：

1. raw `maxPtr[group]`
2. 提取出来的 exponent
3. 最终写出的 `expPtr[group]`
4. 最终写出的 `scalingPtr[group]`
5. FP8 cast 前的乘法中间值

常见故障模式：

- block 意外量化成 0
  - 先看 `scalingPtr[group]` 是否错误地变成了 `0` 或 `1.0`
- FP16 小指数 block 炸成 `Inf`
  - 先看 scale scratch 是否被错误当成了 IEEE fp16，而不是 BF16-encoded scale
- block 没有进 special
  - 先看 `maxPtr[group]` 是否真的保留了 `NaN/Inf`

## Python 复算片段

下面这段足够把上面的例子重新算一遍。

```python
import struct
import math

def bits_to_f32(u):
    return struct.unpack(">f", struct.pack(">I", u))[0]

def bf16_bits_to_f32(u16):
    return bits_to_f32(u16 << 16)

def fp16_bits_to_f32(u16):
    s = (u16 >> 15) & 1
    e = (u16 >> 10) & 0x1F
    m = u16 & 0x3FF
    if e == 0:
        if m == 0:
            v = 0.0
        else:
            v = (m / 1024.0) * (2 ** -14)
    elif e == 0x1F:
        v = math.inf if m == 0 else math.nan
    else:
        v = (1.0 + m / 1024.0) * (2 ** (e - 15))
    return -v if s and not math.isnan(v) else v

print(hex(0x003C), bf16_bits_to_f32(0x003C))   # bf16 5.51e-39
print(hex(0x0380), bf16_bits_to_f32(0x0380))   # bf16 2^-120
print(hex(0x1C00), fp16_bits_to_f32(0x1C00))   # fp16 2^-8
print(hex(0x7F00), bf16_bits_to_f32(0x7F00))   # bf16 2^127
print(hex(0x4780), bf16_bits_to_f32(0x4780))   # bf16 2^16
print(5.5101297694794727e-39 * (2 ** 127))
print((2 ** -8) * (2 ** 16))
```
