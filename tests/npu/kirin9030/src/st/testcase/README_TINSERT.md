# Kirin9030 TINSERT Test Cases

## Overview
This directory contains test cases for TINSERT instruction on Kirin9030 architecture, designed based on A5 TINSERT tests but adapted to Kirin9030's data type constraints.

## Key Differences from A5

### Data Type Constraints
Kirin9030 has stricter type constraints than A5:
- **Non-quantized path**: Source and destination types must match exactly
  - Supported: `half → half`, `int32_t → int32_t`
- **Quantized path**: Supports limited type conversions
  - `int32_t → half/int8_t/uint8_t/int16_t`
  - `half → int8_t/uint8_t/int16_t`

### Unsupported Features
Kirin9030 does NOT support:
- `bfloat16_t` type
- `hifloat8_t`, `float8_e4m3_t`, `float8_e5m2_t` types
- `float8_e8m0_t`, `float4_e2m1x2_t`, `float4_e1m2x2_t` types
- NZ Split modes (`SPLIT2`, `SPLIT4`) - A5-only feature
- Acc→Vec DualMode (`SplitM`, `SplitN`) - may not be supported

## Test Structure

### tinsert_acc2mat/
Tests for Acc → Mat insertion path:
- **Non-quantized**: `half → half`, `int32_t → int32_t`
- **Scalar-quant**: `int32_t → half/int8_t`, `half → int8_t`
- **Vector-quant**: Same as scalar-quant with per-column scaling
- **ReLU mode**: Tests with ReLU activation

**Test cases**:
1. `half, half, half, 16×16×16` (ND layout)
2. `half, half, half, 32×32×32` (ND layout)
3. `int32_t, half, half, 16×16×16` (ND layout)
4. `half, half, half, 16×16×16` (ReLU, ND)
5. `half, half, half, 32×32×32` (ReLU, ND)
6. `half, half, half, 60×127×120` (NZ layout)
7. `int8_t, int8_t, int8_t, 30×48×64` (FB quant)
8. `half, int8_t, int8_t, 60×128×32` (FB quant)
9. `int8_t, half, half, 30×48×64` (FB quant)

### tinsert_acc2vec/
Tests for Acc → Vec insertion path:
- **Non-quantized**: `half → half`, `int32_t → int32_t`
- **Scalar-quant**: `int32_t → half/int8_t`, `half → int8_t`
- **Vector-quant**: Same as scalar-quant
- Supports NZ→ND, NZ→DN, NZ→NZ layout paths

**Test cases**:
1. `half, half, half, 60×127×120` (NZ→ND)
2. `int32_t, half, half, 6×7×8` (NZ→ND)
3. `half, half, half, 60×127×120` (NZ→NZ)
4. `int8_t, int8_t, int8_t, 30×48×64` (FB quant)
5. `half, int8_t, int8_t, 60×128×32` (FB quant)
6. `int8_t, half, half, 30×48×64` (FB quant)

### tinsert_vec/
Tests for Vec → Vec and Vec → Mat insertion:
- **Vec→Vec ND**: Various sizes and types
- **Vec→Vec NZ**: NZ layout insertion
- **Vec→Mat**: UB→L1 transfer
- **Scalar insertion**: Single element insertion
- **Partial valid region**: Valid shape tests

**Test cases** (54 total):
- **ND Vec→Vec** (15 cases): `half/int32_t/int8_t/int16_t/uint8_t/uint16_t/uint32_t`
- **ND scalar** (9 cases): Single element insertion
- **NZ Vec→Vec** (14 cases): NZ layout insertion
- **NZ scalar** (11 cases): NZ scalar insertion
- **Vec→Mat** (4 cases): `half/int8_t/int32_t`

## Supported Data Types Summary

| Path | Supported Types |
|------|----------------|
| **Acc→Mat (no quant)** | `half`, `int32_t` |
| **Acc→Mat (quant)** | `int32_t→half/int8_t/uint8_t/int16_t`, `half→int8_t/uint8_t/int16_t` |
| **Acc→Vec** | Same as Acc→Mat |
| **Vec→Vec** | `half`, `int32_t`, `int8_t`, `int16_t`, `uint8_t`, `uint16_t`, `uint32_t` |
| **Vec→Mat** | `half`, `int8_t`, `int32_t` |

## Running Tests

### Build
```bash
python3 tests/script/build_st.py -v kirin9030 -t tinsert_acc2mat
python3 tests/script/build_st.py -v kirin9030 -t tinsert_acc2vec
python3 tests/script/build_st.py -v kirin9030 -t tinsert_vec
```

### Run
```bash
python3 tests/script/run_st.py -v kirin9030 -t tinsert_acc2mat -g *
python3 tests/script/run_st.py -v kirin9030 -t tinsert_acc2vec -g *
python3 tests/script/run_st.py -v kirin9030 -t tinsert_vec -g *
```

## Test Coverage

Total test cases: **69**
- Acc→Mat: 9 cases
- Acc→Vec: 6 cases
- Vec→Vec ND: 15 cases
- Vec→Vec ND scalar: 9 cases
- Vec→Vec NZ: 14 cases
- Vec→Vec NZ scalar: 11 cases
- Vec→Mat: 4 cases

This covers all major TINSERT use cases on Kirin9030 while respecting its type constraints.