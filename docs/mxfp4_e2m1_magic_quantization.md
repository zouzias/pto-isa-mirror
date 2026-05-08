# MXFP4 E2M1 Quantization

This note documents the PTO `QuantType::MXFP4_E2M1` implementation for FP16 input.
The current implementation targets ND output only. The A5 path is currently
validated for the contiguous case where `validCols == srcCols`.

## Scope

- Source: FP16.
- Element format: OCP-style FP4 E2M1.
- Scale format: E8M0, one scale per 32 source elements.
- Output layout: packed ND, two E2M1 nibbles per byte, low nibble first.
- FP32 and BF16 source paths are intentionally not implemented here.

## Block Scale

For each 32-element block:

```text
M = max(abs(x_i))
scale = 2^(floor(log2(M)) - 2)
reciprocal_scale = 1 / scale = 2^(2 - floor(log2(M)))
e8m0 = floor(log2(M)) + 127 - 2
```

The constant `2` is the largest power-of-two exponent carried by E2M1 because
the largest power-of-two value is `4`. This differs from MXFP8 E4M3, whose
corresponding constant is `8`.

The A5 FP16 path reuses the existing MXFP8 reduction structure: FP16 input is
first converted to BF16 for the block max reduction, then the scale extraction
changes the max element exponent from E4M3's `8` to E2M1's `2`.

The reciprocal scale is kept as BF16 bits in the scratch `scaling` tile. This
matters because the E2M1 reciprocal scale can be much larger than FP16 can
represent, especially for zero or tiny blocks. During element conversion the A5
path converts FP16 input to BF16, multiplies by this BF16 reciprocal scale, and
uses the hardware BF16-to-FP4 E2M1 conversion to produce packed bytes.
For the first A5 implementation, block absmax uses the same 1D contiguous reducer
selection as the FP16 MXFP8 path.

## E2M1 Element Encoding

Positive magnitude codes are:

```text
mag_code  value
0         0
1         0.5
2         1
3         1.5
4         2
5         3
6         4
7         6
```

For a finite scaled value:

```text
a = abs(x * reciprocal_scale)
E = clamp(floor(log2(a)), 0, 2)
step = 2^(E - 1)
q = RN_even(a / step)
mag_code = clamp(q + (E << 1), 0, 7)
code = sign | mag_code
```

The CPU reference implements this with FP32 magic-add rounding:

```text
biased_exp = clamp(exponent_bits(a), 127, 129)
magic_bits = (biased_exp + 22) << 23
q = bits(float32(a + magic)) - magic_bits
base_code = (biased_exp - 127) << 1
mag_code = clamp(q + base_code, 0, 7)
```

This avoids the explicit `-127` then `+127` sequence in the original formula:

```text
E = clamp(exponent_bits(a) - 127, 0, 2)
magic_bits = (E + 127 + 22) << 23
```

## Signed Int4 Interpretation

The raw E2M1 nibble is `sign | mag_code`, where `sign` is either `0x0` or `0x8`.
When using a signed-int4 packing conversion, the negative raw nibbles `0x8..0xF`
must be presented as signed values `-8..-1`:

```text
signed_code = sign ? (mag_code - 8) : mag_code
```

That mapping is just two's-complement reinterpretation:

```text
mag_code = 3, sign = 1
raw nibble = 0xB
signed int4 value = -5
```

The CPU reference writes packed raw nibbles directly. A signed-int4 conversion
path can use the `signed_code` form above because the resulting two's-complement
nibble is byte-identical to `sign | mag_code`. The current A5 ND path writes the
packed FP4 output bytes with the same low-nibble-first convention through the
hardware FP4 conversion.

## Special Values

- Zero blocks produce `e8m0 = 0`; zero elements still encode as zero.
- `Inf` maxima produce `e8m0 = 0xFF`.
- Element `Inf` saturates to magnitude code `7`.
- Element `NaN` currently maps to positive magnitude code `7` in the CPU
  reference.
