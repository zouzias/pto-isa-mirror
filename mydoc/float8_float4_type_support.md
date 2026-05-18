# A5 指令中 float8/float4 数据类型支持情况

## 背景

统计 a5 已实现的指令中，分别包含了以下低精度数据类型的支持情况：

- `float8_e4m3_t`（MXFP8 E4M3，8位浮点）
- `float8_e5m2_t`（MXFP8 E5M2，8位浮点）
- `float8_e8m0_t`（MXFP8 E8M0，8位指数-only）
- `float4_e2m1x2_t`（MXFP4 E2M1，4位浮点，每字节2元素）
- `float4_e1m2x2_t`（MXFP4 E1M2，4位浮点，每字节2元素）

（另 `hifloat8_t` 作为华为自定义 8-bit 浮点类型在部分指令中与上述类型并列出现）

---

## 一、各指令对低精度类型的支持清单

### 1. float8_e4m3_t

| 指令 | 文件 | 使用位置 |
|------|------|---------|
| **TLoad** | `a5/TLoad.hpp` | 在类型断言中与 int8_t/int16_t 等并列 |
| **TStore** | `a5/TStore.hpp` | GM 输出类型断言；Tile 类型断言 line 171 |
| **TMov** | `a5/TMov.hpp` | Acc→Vec/Vec→Mat 类型断言 |
| **TCvt** | `a5/TCvt.hpp` | `castData<float8_e4m3_t>`：float→e4m3 和 e4m3→float 双向转换 |
| **TMatmul** | `a5/TMatmul.hpp` | `isSupportedFp8Combo`：支持 e4m3×e4m3、e4m3×e5m2 等组合 |
| **TGather** | `a5/TGather.hpp` | 源数据类型限制中列出 |
| **TExtract** | `a5/TExtract.hpp` | 支持的类型列表中列出 |
| **TInsert** | `a5/TInsert.hpp` | Vec/Vec、Vec/Mat 路径类型断言 |
| **MGather** | `a5/MGather.hpp` | 类型限制中列出 |
| **MScatter** | `a5/MScatter.hpp` | 类型限制中列出 |

**共 10 个指令**

### 2. float8_e5m2_t

| 指令 | 文件 | 使用位置 |
|------|------|---------|
| **TLoad** | `a5/TLoad.hpp` | 类型断言 line 1222-1223 |
| **TStore** | `a5/TStore.hpp` | Tile 类型断言 line 172 |
| **TMov** | `a5/TMov.hpp` | Acc→Vec/Vec→Mat 类型断言 |
| **TCvt** | `a5/TCvt.hpp` | `castData<float8_e5m2_t>`：float↔e5m2 转换 |
| **TMatmul** | `a5/TMatmul.hpp` | `isSupportedFp8Combo`：e5m2×e4m3、e5m2×e5m2 |
| **TGather** | `a5/TGather.hpp` | 源数据类型限制中列出 |
| **TExtract** | `a5/TExtract.hpp` | 支持的类型列表中列出 |
| **TInsert** | `a5/TInsert.hpp` | Vec/Vec、Vec/Mat 路径类型断言 |
| **MGather** | `a5/MGather.hpp` | 类型限制中列出 |
| **MScatter** | `a5/MScatter.hpp` | 类型限制中列出 |

**共 10 个指令**

### 3. float8_e8m0_t

| 指令 | 文件 | 使用位置 |
|------|------|---------|
| **TLoad** | `a5/TLoad.hpp` | MX_A_ZZ 和 MX_B_NN 布局的 DType 断言（仅限这种场景） |
| **TStore** | `a5/TStore.hpp` | Tile 类型断言 line 174 |
| **TMov** | `a5/TMov.hpp` | 从 Vec→ZZ 布局移动时：必须为 uint8_t/hifloat8_t/float8_e8m0_t |
| **TExtract** | `a5/TExtract.hpp` | 支持的类型列表中列出 |
| **TInsert** | `a5/TInsert.hpp` | Vec/Vec、Vec/Mat 路径类型断言 |

**共 5 个指令**

### 4. float4_e2m1x2_t

| 指令 | 文件 | 使用位置 |
|------|------|---------|
| **TLoad** | `a5/TLoad.hpp` | 多处 fp4 类型分支判断（line 55-56 等） |
| **TStore** | `a5/TStore.hpp` | Tile 类型断言 line 176；fp4 路径 line 488-489 |
| **TMov** | `a5/TMov.hpp` | fp4 类型判定 |
| **TCvt** | `a5/TCvt.hpp` | `castData<float4_e2m1x2_t>`：bf16↔fp4 双向转换 |
| **TMatmul** | `a5/TMatmul.hpp` | `isSupportedFp4Combo`：支持 e2m1×e2m1、e2m1×e1m2 等 |
| **TExtract** | `a5/TExtract.hpp` | fp4 类型特殊路径处理 |
| **TInsert** | `a5/TInsert.hpp` | fp4 类型特殊路径处理 |
| **TQuant** | `a5/TQuant.hpp` | MXFP4_E2M1 量化输出断言 |

**共 8 个指令**

### 5. float4_e1m2x2_t

| 指令 | 文件 | 使用位置 |
|------|------|---------|
| **TLoad** | `a5/TLoad.hpp` | 多处 fp4 类型分支判断 |
| **TStore** | `a5/TStore.hpp` | Tile 类型断言 line 175；fp4 路径 |
| **TMov** | `a5/TMov.hpp` | fp4 类型判定 |
| **TCvt** | `a5/TCvt.hpp` | `castData<float4_e1m2x2_t>`：bf16↔fp4 双向转换 |
| **TMatmul** | `a5/TMatmul.hpp` | `isSupportedFp4Combo`：e1m2×e1m2、e1m2×e2m1 |
| **TExtract** | `a5/TExtract.hpp` | fp4 类型特殊路径处理 |
| **TInsert** | `a5/TInsert.hpp` | fp4 类型特殊路径处理 |

**共 7 个指令**

---

## 二、汇总表

| 指令 | f8_e4m3 | f8_e5m2 | f8_e8m0 | f4_e2m1x2 | f4_e1m2x2 | 说明 |
|------|:-------:|:-------:|:-------:|:----------:|:----------:|------|
| **TLoad** | ✅ | ✅ | ✅ | ✅ | ✅ | 全部支持；e8m0 仅限 MX 布局 |
| **TStore** | ✅ | ✅ | ✅ | ✅ | ✅ | 全部支持 |
| **TMov** | ✅ | ✅ | ✅ | ✅ | ✅ | 全部支持 |
| **TCvt** | ✅ | ✅ | ❌ | ✅ | ✅ | 不支持 e8m0（非数值类型无需转换） |
| **TMatmul** | ✅ | ✅ | ❌ | ✅ | ✅ | fp8 combo + fp4 combo；e8m0 不参与 |
| **TExtract** | ✅ | ✅ | ✅ | ✅ | ✅ | 全部支持 |
| **TInsert** | ✅ | ✅ | ✅ | ✅ | ✅ | 全部支持 |
| **TGather** | ✅ | ✅ | ❌ | ❌ | ❌ | 仅 fp8 数值类型 |
| **TQuant** | ❌ | ❌ | ❌ | ✅ | ❌ | 仅 MXFP4_E2M1 输出 |
| **MGather** | ✅ | ✅ | ❌ | ❌ | ❌ | 仅 fp8 数值类型 |
| **MScatter** | ✅ | ✅ | ❌ | ❌ | ❌ | 仅 fp8 数值类型 |

### 统计

| 数据类型 | 支持的指令数 | 指令列表 |
|----------|:-----------:|---------|
| **float8_e4m3_t** | 10 | TLoad, TStore, TMov, TCvt, TMatmul, TGather, TExtract, TInsert, MGather, MScatter |
| **float8_e5m2_t** | 10 | 同上 |
| **float8_e8m0_t** | 5 | TLoad, TStore, TMov, TExtract, TInsert |
| **float4_e2m1x2_t** | 8 | TLoad, TStore, TMov, TCvt, TMatmul, TExtract, TInsert, TQuant |
| **float4_e1m2x2_t** | 7 | TLoad, TStore, TMov, TCvt, TMatmul, TExtract, TInsert |

### 全支持（5/5 类型）的指令

TLoad、TStore、TMov、TExtract、TInsert — 这 5 个指令覆盖了所有 5 种低精度类型。
