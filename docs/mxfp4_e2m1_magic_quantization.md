# MXFP4 E2M1 量化说明

本文记录 PTO `QuantType::MXFP4_E2M1` 针对 FP16 输入的实现。当前实现只覆盖
ND 输出；A5 路径目前按 `validCols == srcCols` 的连续输入场景进行设计和验证。

## 范围

- 输入类型：FP16。
- 元素格式：OCP 风格 FP4 E2M1。
- 尺度格式：E8M0，每 32 个源元素对应 1 个尺度值。
- 输出布局：打包 ND，每个字节存两个 E2M1 半字节，低半字节在前。
- FP32 和 BF16 输入路径当前不在本文实现范围内。

## 分组尺度计算

每个 32 元素分组的尺度计算如下：

```text
M = max(abs(x_i))
scale = 2^(floor(log2(M)) - 2)
reciprocal_scale = 1 / scale = 2^(2 - floor(log2(M)))
e8m0 = floor(log2(M)) + 127 - 2
```

这里的常数 `2` 来自 E2M1 可表达的最大 2 的幂指数。E2M1 的最大 2 的幂值
是 `4`，所以对应指数是 `2`。这和 MXFP8 E4M3 不同，E4M3 对应的常数是
`8`。

A5 FP16 路径复用现有 MXFP8 的规约结构：FP16 输入先转换成 BF16 参与
分组最大值规约，然后尺度提取阶段把 MXFP8 E4M3 的最大指数常数 `8`
改为 E2M1 的 `2`。

倒数尺度以 BF16 bit 形式保存在临时 `scaling` tile 中。这样做很重要，因为
E2M1 的倒数尺度可能比 FP16 可表达范围更大，尤其是全零分组或很小的
分组。元素转换阶段，A5 路径会把 FP16 输入和 BF16 倒数尺度都转换到 FP32，
在 FP32 中做乘法，然后执行本文后面的
magic-add E2M1 编码。当前 A5 路径不使用硬件 BF16 到 FP4 的转换，因为那会
先把 FP16 输入截断到 BF16，再做 FP4 舍入，不能满足精确模拟要求。

分组绝对值最大值使用与 FP16 MXFP8 路径一致的 1D 连续规约选择逻辑。

完整分组窗口的 A5 元素转换也沿用 FP16 MXFP8 的窗口形状：一次
`DINTLV_B16` 加载消费 256 个 FP16 元素，一次 `E2B_B16` 加载广播这 8 个
32 元素 group 的 BF16 倒数尺度。FP32 转换后，四路数据分别对应源下标
`mod 4` 的四条流。E2M1 编码值会先分别打包 `(0,1)` 和 `(2,3)` 两组，再交错
成打包 ND 字节顺序。不足 8 个 group 的尾部走 32 元素 group 路径。

## A5 窗口路径数据排布

本节说明 A5 FP16 完整窗口路径中，输入、尺度、寄存器和最终搬出的对应
关系。这里讨论的是 `CalcQuantizedFP4E2M1Values_Half_Window`：一次处理
256 个 FP16，对应 8 个 MX group，最终输出 128 个打包 FP4 字节。

源数据在 UB 中是连续 ND 排布：

```text
src:
x0, x1, x2, x3, x4, x5, x6, x7, ... , x255
```

### DINTLV_B16 双加载

`DINTLV_B16` 一次加载两个 b16 VL：

```cpp
vlds(v_input_0, v_input_1, srcPtr, window * 256, DINTLV_B16);
```

加载后并不是简单的前 128 个元素进入 `v_input_0`、后 128 个元素进入
`v_input_1`，而是按偶/奇位置拆开：

```text
src:       x0  x1  x2  x3  x4  x5  x6  x7  ...
           |   |   |   |   |   |   |   |
input_0:   x0      x2      x4      x6      ...
input_1:       x1      x3      x5      x7  ...
```

也就是：

```text
v_input_0 = x0, x2, x4, x6, x8,  ...
v_input_1 = x1, x3, x5, x7, x9,  ...
```

### FP16 到 FP32 后的四路顺序

b16 向量有 128 个通道，FP32 向量只有 64 个通道，所以 b16 到 f32
需要分 `PART_EVEN` 和 `PART_ODD`。经过转换后，数据按源下标 `mod 4`
分成四路：

```text
v0 = vcvt(v_input_0, PART_EVEN) = x0, x4, x8,  x12, ...
v1 = vcvt(v_input_1, PART_EVEN) = x1, x5, x9,  x13, ...
v2 = vcvt(v_input_0, PART_ODD)  = x2, x6, x10, x14, ...
v3 = vcvt(v_input_1, PART_ODD)  = x3, x7, x11, x15, ...
```

图示：

```text
连续源数据:
x0  x1  x2  x3 | x4  x5  x6  x7 | x8  x9  x10 x11 | ...

FP32 四路:
v0: x0          | x4          | x8           | ...
v1:     x1      |     x5      |     x9       | ...
v2:         x2  |         x6  |         x10  | ...
v3:             x3 |         x7 |          x11 | ...
```

所以双加载之后，FP32 寄存器中的数据不是 `x0,x1,x2,x3,...` 的连续顺序，
而是四条 `index % 4` 流。

### 尺度广播和输入的对应

MXFP4 每 32 个元素共用一个倒数尺度。一个完整窗口有 8 个
group：

```text
group0: x0   .. x31    scale0
group1: x32  .. x63    scale1
group2: x64  .. x95    scale2
...
group7: x224 .. x255   scale7
```

尺度通过 `E2B_B16` 加载：

```cpp
vlds((vector_u16 &)v_scaling_bf16,
     (__ubuf__ uint16_t *)scalingPtr,
     8 * window,
     E2B_B16);
vcvt(v_scaling_f32, v_scaling_bf16, preg_all_b16, PART_EVEN);
```

`E2B_B16` 会把 8 个 BF16 尺度按 group 展开/广播到和 FP32 通道对齐
的布局。逻辑上可以理解为：

```text
v_scaling_f32:
s0, s0, s0, s0, s0, s0, s0, s0,
s1, s1, s1, s1, s1, s1, s1, s1,
s2, s2, ...
```

虽然输入被拆成了四路，但每一路在一个 32 元素 group 内刚好有 8 个
通道，所以同一个 `v_scaling_f32` 可以直接乘四路输入：

```text
group0:
v0 通道: x0, x4,  x8,  x12, x16, x20, x24, x28
v1 通道: x1, x5,  x9,  x13, x17, x21, x25, x29
v2 通道: x2, x6,  x10, x14, x18, x22, x26, x30
v3 通道: x3, x7,  x11, x15, x19, x23, x27, x31
尺度:    s0, s0,  s0,  s0,  s0,  s0,  s0,  s0
```

因此：

```cpp
vmul(v0, v0, v_scaling_f32, preg_f32, MODE_ZEROING);
vmul(v1, v1, v_scaling_f32, preg_f32, MODE_ZEROING);
vmul(v2, v2, v_scaling_f32, preg_f32, MODE_ZEROING);
vmul(v3, v3, v_scaling_f32, preg_f32, MODE_ZEROING);
```

对应关系是正确的，不需要手写尺度广播循环。

### FP32 Magic-Add 到有符号编码

四路 FP32 缩放后数值进入 `CalcE2M1SignedCodeI32` 后，执行本文中的
FP32 magic-add E2M1 模拟。输出是 signed int4 编码值的 int32 表示：

```text
code0_reg = c0, c4, c8,  c12, ...
code1_reg = c1, c5, c9,  c13, ...
code2_reg = c2, c6, c10, c14, ...
code3_reg = c3, c7, c11, c15, ...
```

其中 `ci = e2m1_code(xi * scale_group)`。负数编码值使用 signed-int4 形式：

```text
原始半字节 0x8..0xF  <=>  signed 值 -8..-1
```

这样后续无论是手动取低 4 bit，还是改成 `s32 -> s16 -> int4` 打包，
语义都是一致的。

### 成对打包和最终 ND 顺序

FP4 输出是两个半字节一个字节，低半字节放偶数元素，高半字节放奇数
元素：

```text
字节0 = c0 | (c1 << 4)
字节1 = c2 | (c3 << 4)
字节2 = c4 | (c5 << 4)
字节3 = c6 | (c7 << 4)
...
```

因为四路编码值不是连续顺序，所以先分别打包两组：

```text
pair01 = pack(code0_reg, code1_reg)
       = [c0,c1], [c4,c5], [c8,c9], ...
       = 字节0, 字节2, 字节4, ...

pair23 = pack(code2_reg, code3_reg)
       = [c2,c3], [c6,c7], [c10,c11], ...
       = 字节1, 字节3, 字节5, ...
```

此时两个寄存器的含义正好是打包字节的偶数流和奇数流：

```text
pair01: 字节0, 字节2, 字节4, 字节6, ...
pair23: 字节1, 字节3, 字节5, 字节7, ...
```

当前实现使用 `vintlv` 在寄存器中先交错：

```cpp
vintlv(v_output, v_scratch, v_pair01, v_pair23);
```

得到：

```text
v_output:
字节0, 字节1, 字节2, 字节3, 字节4, 字节5, ...
```

然后用普通单搬出 `NORM_B8` 写出：

```cpp
vsts(v_output, dstPtr, window * 128, NORM_B8, preg_b8);
```

因此最终 dst 中的数据是连续 ND 打包 FP4 顺序：

```text
dst:
[c0,c1], [c2,c3], [c4,c5], [c6,c7], ...
```

### 为什么当前不是直接双搬出

A5 的 b8 双搬出 `INTLV_B8` 语义与 `pair01/pair23` 的数据形态一致：

```text
src0: 字节0, 字节2, 字节4, ...
src1: 字节1, 字节3, 字节5, ...
dst:  字节0, 字节1, 字节2, 字节3, ...
```

但是 `INTLV_B8` 会忽略 mask，且一次写出长度是 `VL * 2`。当前完整窗口
只产生：

```text
输入: 256 个 FP16 元素
输出: 128 个打包 FP4 字节
```

而 b8 的 `VL` 是 256 字节，双搬出会写 512 字节：

```text
src0: 256 字节
src1: 256 字节
dst:  512 字节
```

这和当前窗口的 128 字节输出粒度不匹配，会覆盖后续 UB 数据。因此当前
实现选择：

```text
pair01/pair23 -> vintlv 寄存器内交错 -> NORM_B8 单搬出 128 字节
```

如果将来要使用 `INTLV_B8` 双搬出，需要把处理粒度扩大到能提供完整
`src0/src1` 两个 b8 VL 的输出，例如至少聚合 4 个当前窗口：

```text
4 * 256 个 FP16 输入 = 1024 个 FP16
4 * 128 字节输出     = 512 字节打包 FP4
```

但这样还需要把 4 个窗口的偶数字节流拼成完整 `src0`，把奇数字节
流拼成完整 `src1`，复杂度和寄存器/调度压力都需要重新评估。

## E2M1 元素编码

正数幅值编码如下：

```text
幅值编码  数值
0         0
1         0.5
2         1
3         1.5
4         2
5         3
6         4
7         6
```

对于有限的缩放后数值：

```text
a = abs(x * reciprocal_scale)
E = clamp(floor(log2(a)), 0, 2)
step = 2^(E - 1)
q = RN_even(a / step)
mag_code = clamp(q + (E << 1), 0, 7)
code = sign | mag_code
```

CPU 参考实现使用 FP32 magic-add 完成舍入：

```text
biased_exp = clamp(exponent_bits(a), 127, 129)
magic_bits = (biased_exp + 22) << 23
q = bits(float32(a + magic)) - magic_bits
base_code = (biased_exp - 127) << 1
mag_code = clamp(q + base_code, 0, 7)
```

这种写法避免了原始公式中显式的 `-127` 再 `+127`：

```text
E = clamp(exponent_bits(a) - 127, 0, 2)
magic_bits = (E + 127 + 22) << 23
```

## 有符号 Int4 解释

原始 E2M1 半字节是 `sign | mag_code`，其中 `sign` 是 `0x0` 或 `0x8`。
如果使用 signed-int4 打包转换，负数原始半字节 `0x8..0xF` 必须以
signed 值 `-8..-1` 的形式传入：

```text
signed_code = sign ? (mag_code - 8) : mag_code
```

这个映射本质上就是二进制补码重解释：

```text
mag_code = 3, sign = 1
原始半字节 = 0xB
signed int4 值 = -5
```

CPU 参考实现会直接写打包后的原始半字节。A5 ND 路径在向量寄存器中生成
等价的 signed-code 形式，然后取低 4 bit，并按低半字节在前的约定把两个
编码值打包到一个字节。

### 符号处理方案比较

FP4 E2M1 的符号位可以有两种处理方式。

方案一是在打包前生成 signed int4 编码值：

```text
mag_code = encode(abs(x * reciprocal_scale))  // 0..7
signed_code = sign ? (mag_code - 8) : mag_code
```

这个方案中，传给 `int16 -> int4` 的值范围是 `[-8, 7]`，正好落在 signed
int4 可表达范围内，不存在溢出问题。硬件把这些 signed int4 值打包后，其低
4 bit 就是最终需要的 FP4 半字节：

```text
mag_code = 3, sign = 1
signed_code = 3 - 8 = -5
signed int4 低 4 bit = 0xB = 0x8 | 0x3
```

方案二是先只打包正的 magnitude，再额外 OR 符号位：

```text
mag_code = encode(abs(x * reciprocal_scale))  // 0..7
先把 mag_code 转成 int16，再打包成 int4
再生成 sign mask，与 packed byte 做 OR
```

这个方案中，`int16 -> int4` 的输入全是 `0..7`，也不会溢出。但是符号并没有
消失，只是被推迟到打包后处理。打包后两个元素共用一个字节，因此符号 mask
必须区分低半字节和高半字节：

```text
byte = mag_even | (mag_odd << 4)
sign_mask = (sign_even ? 0x08 : 0) | (sign_odd ? 0x80 : 0)
final_byte = byte | sign_mask
```

在 A5 当前窗口路径中，数据已经拆成 `(0,1)` 和 `(2,3)` 两组分别打包：

```text
pair01 = [c0,c1], [c4,c5], ...
pair23 = [c2,c3], [c6,c7], ...
```

如果采用方案二，就需要为 `pair01` 和 `pair23` 分别生成对应的 sign mask，
再各自做一次 OR。生成 sign mask 本身也需要额外的常量、选择或移位操作。

因此当前推荐方案一：在打包前直接生成 signed code。它的优点是：

- `int16 -> int4` 输入范围正好是 signed int4 的合法范围 `[-8, 7]`。
- 不需要在打包后额外生成 `0x08/0x80` 符号 mask。
- 不需要对 `pair01/pair23` 分别增加 OR。
- 寄存器压力和指令数通常更低。

方案二只有在硬件的 `int16 -> int4` 转换不能正确接受 signed int4 输入，或者
后续需要复用纯 magnitude packed byte 时才值得考虑。当前 A5 路径没有这个
需求。

## 特殊值

- 全零分组产生 `e8m0 = 0`；零元素仍编码为零。
- `Inf` max 产生 `e8m0 = 0xFF`。
- 元素 `Inf` 饱和到幅值编码 `7`。
- 元素 `NaN` 当前与 CPU 参考实现一致，映射为正数幅值编码 `7`。
