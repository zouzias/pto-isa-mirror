# 轻量级 Costmodel 接口设计

## 1. 背景

当前 costmodel 基于原生 NPU 代码路径，通过实例化完整的 Tile 结构体、执行 PTO IMPL、采集 CCE trace 来估算 cycle。优点是计算精准，缺点是依赖完整的 PTO ISA 类型信息（Tile 模板参数、Layout、Fractal 等），只能在 PTO 实现完成之后使用。

编译器调度场景需要一个**轻量级 costmodel**：

- **输入简单**：仅 PTO 指令名、数据类型、形状（行、列），以及个别指令的少量额外参数
- **计算快速**：无需实例化 Tile，无需 trace 采集，直接查表/公式计算
- **不依赖 PTO 实现细节**：编译器侧不需要知道 Tile 模板、BLayout、SLayout 等

## 2. 接口设计

### 2.1 枚举定义

```cpp
// PTO 指令枚举
enum class PtoOpcode {
    TADD,   TSUB,   TMUL,   TDIV,
    TADDS,  TSUBS,  TMULS,  TDIVS,
    TMINS,  TMAXS,
    TABS,   TNEG,
    TEXP,   TSQRT,  TRSQRT, TLOG,
    TRELU,  TLRELU, TNOT,
    TROWSUM, TROWMAX, TROWMIN, TROWPROD,
    TCOLSUM, TCOLMAX, TCOLMIN, TCOLPROD,
    TMATMUL, TGEMV,
    TCVT,
    TMOV,   TLOAD,  TSTORE,
    TTRANS,
    TSORT32, TMRGSORT,
    TSEL,   TSCATTER,
    TEXTRACT, TINSERT,
    TROWEXPAND, TCOLEXPAND,
    TLOADCONV,
};

// 数据类型枚举
enum class DType : uint8_t {
    Float,      // fp32
    Half,       // fp16
    Int8,
    Int16,
    Int32,
    Uint8,
    Uint16,
    Uint32,
    BFloat16,
};

// 布局枚举（仅 TLOAD/TSTORE 等需要）
enum class MemLayout : uint8_t {
    ND,         // 行优先
    DN,         // 列优先
    NZ,         // fractal NZ
};

// 舍入模式（仅 TCVT 需要）
enum class RoundMode : uint8_t {
    CAST_NONE,
    CAST_ROUND,
    CAST_FLOOR,
    CAST_CEIL,
    CAST_RINT,
    CAST_ODD,
};
```

### 2.2 统一输入结构

```cpp
struct CostModelInput {
    // ---- 必填字段 ----
    PtoOpcode   op;             // PTO 指令
    DType       dtype;          // 主数据类型
    int64_t     rows;           // Tile 行数
    int64_t     cols;           // Tile 列数

    // ---- 条件选填字段（按指令类别使用）----

    // TMATMUL / TGEMV：三矩阵维度
    DType       dtype2      = DType::Float;  // 右矩阵/累加器数据类型
    int64_t     k           = 0;             // 内积维度 K（M×K × K×N -> M×N）

    // TCVT：目标数据类型
    DType       dst_dtype   = DType::Float;  // 转换目标类型
    RoundMode   round_mode  = RoundMode::CAST_NONE;

    // TLOAD / TSTORE：内存布局
    MemLayout   layout      = MemLayout::ND;

    // TEXTRACT / TINSERT：子块位置
    int64_t     dst_rows    = 0;             // 提取目标行数
    int64_t     dst_cols    = 0;             // 提取目标列数

    // TMRGSORT：排序参数
    int64_t     block_len   = 0;             // 块长度
    int64_t     src_count   = 1;             // 归并源数量（1,2,4）

    // TLOADCONV：卷积参数
    int64_t     channels    = 0;             // 通道数
    int64_t     height      = 0;             // 特征图高
    int64_t     width       = 0;             // 特征图宽
};
```

### 2.3 输出结构

```cpp
struct CostModelResult {
    double  cycles;         // 估算的总 cycle 数
    // 可扩展：拆分为 startup_cycles + compute_cycles 等
};

// 核心接口
CostModelResult EstimateCycles(const CostModelInput &input);
```

## 3. 指令分类与输入规格

根据输入模式，将已实现 ST 的 27 条 PTO 指令分为以下几类：

### 3.1 二元元素级运算

**指令**：`TADD`, `TSUB`, `TMUL`

**语义**：`dst[i,j] = src0[i,j] OP src1[i,j]`

**输入**：

| 字段 | 说明 |
|------|------|
| `op` | TADD / TSUB / TMUL |
| `dtype` | 操作数数据类型 |
| `rows` | Tile 行数 |
| `cols` | Tile 列数 |

**输出**：`cycles`

**不使用的字段**：无

---

### 3.2 一元元素级运算

**指令**：`TABS`, `TEXP`, `TSQRT`

**语义**：`dst[i,j] = OP(src[i,j])`

**输入**：

| 字段 | 说明 |
|------|------|
| `op` | TABS / TEXP / TSQRT |
| `dtype` | 数据类型 |
| `rows` | Tile 行数 |
| `cols` | Tile 列数 |

**输出**：`cycles`

---

### 3.3 标量二元运算

**指令**：`TADDS`, `TSUBS`, `TMULS`, `TDIVS`, `TMINS`, `TMAXS`

**语义**：`dst[i,j] = src[i,j] OP scalar`（标量不参与 cycle 估算，不影响结果）

**输入**：

| 字段 | 说明 |
|------|------|
| `op` | TADDS / TSUBS / TMULS / TDIVS / TMINS / TMAXS |
| `dtype` | 数据类型 |
| `rows` | Tile 行数 |
| `cols` | Tile 列数 |

**输出**：`cycles`

**注意**：标量值本身不影响 cycle 估算，因此无需作为输入。

---

### 3.4 行归约运算

**指令**：`TROWSUM`, `TROWMAX`, `TROWMIN`

**语义**：沿行方向归约，输出形状为 `(1, cols)`

**输入**：

| 字段 | 说明 |
|------|------|
| `op` | TROWSUM / TROWMAX / TROWMIN |
| `dtype` | 数据类型 |
| `rows` | 输入 Tile 行数 |
| `cols` | 输入 Tile 列数 |

**输出**：`cycles`

---

### 3.5 列归约运算

**指令**：`TCOLSUM`, `TCOLMAX`, `TCOLMIN`

**语义**：沿列方向归约，输出形状为 `(rows, 1)`

**输入**：

| 字段 | 说明 |
|------|------|
| `op` | TCOLSUM / TCOLMAX / TCOLMIN |
| `dtype` | 数据类型 |
| `rows` | 输入 Tile 行数 |
| `cols` | 输入 Tile 列数 |

**输出**：`cycles`

---

### 3.6 矩阵乘法

**指令**：`TMATMUL`

**语义**：`C(M×N) = A(M×K) × B(K×N)`

**输入**：

| 字段 | 说明 |
|------|------|
| `op` | TMATMUL |
| `dtype` | 左矩阵（A）数据类型 |
| `dtype2` | 右矩阵（B）/累加器数据类型（默认与 dtype 相同） |
| `rows` | M（输出行数 = A 的行数） |
| `cols` | N（输出列数 = B 的列数） |
| `k` | K（内积维度 = A 的列数 = B 的行数） |

**输出**：`cycles`

**说明**：这是唯一需要三个维度（M, K, N）的指令。`rows` 对应 M，`cols` 对应 N，`k` 为内积维度。TMATMUL 的 cycle 估算与 M×K×N 乘加量直接相关。

---

### 3.7 类型转换

**指令**：`TCVT`

**语义**：`dst[i,j] = cast<src_type -> dst_type>(src[i,j])`

**输入**：

| 字段 | 说明 |
|------|------|
| `op` | TCVT |
| `dtype` | 源数据类型 |
| `dst_dtype` | 目标数据类型 |
| `rows` | Tile 行数 |
| `cols` | Tile 列数 |
| `round_mode` | 舍入模式（默认 CAST_NONE） |

**输出**：`cycles`

**说明**：cycle 取决于源类型和目标类型的组合（如 fp32→fp16 vs fp16→int32 可能有不同的硬件开销）。

---

### 3.8 数据搬运

**指令**：`TMOV`

**语义**：`dst = src`（Tile 间拷贝）

**输入**：

| 字段 | 说明 |
|------|------|
| `op` | TMOV |
| `dtype` | 数据类型 |
| `rows` | Tile 行数 |
| `cols` | Tile 列数 |

**输出**：`cycles`

---

### 3.9 数据加载

**指令**：`TLOAD`

**语义**：从 Global Tensor 加载到 Tile

**输入**：

| 字段 | 说明 |
|------|------|
| `op` | TLOAD |
| `dtype` | 数据类型 |
| `rows` | Tile 行数 |
| `cols` | Tile 列数 |
| `layout` | 内存布局（ND / DN） |

**输出**：`cycles`

**说明**：不同 layout（行优先 ND vs 列优先 DN）可能影响搬运效率和 cycle 数。

---

### 3.10 转置

**指令**：`TTRANS`

**语义**：`dst[j,i] = src[i,j]`，输出形状 `(cols, rows)`

**输入**：

| 字段 | 说明 |
|------|------|
| `op` | TTRANS |
| `dtype` | 数据类型 |
| `rows` | 输入行数 |
| `cols` | 输入列数 |

**输出**：`cycles`

---

### 3.11 排序

#### TSORT32

**语义**：每 32 个元素为一组进行排序

**输入**：

| 字段 | 说明 |
|------|------|
| `op` | TSORT32 |
| `dtype` | 数据类型 |
| `rows` | Tile 行数 |
| `cols` | Tile 列数 |

**输出**：`cycles`

**说明**：排序粒度固定为 32 个元素。cycle 估算与 `(rows × cols) / 32` 个排序单元数量相关。

#### TMRGSORT

**语义**：多路归并排序

**输入**：

| 字段 | 说明 |
|------|------|
| `op` | TMRGSORT |
| `dtype` | 数据类型 |
| `rows` | Tile 行数 |
| `cols` | Tile 列数 |
| `block_len` | 块长度（单路排序时使用） |
| `src_count` | 归并源数量（1, 2, 4） |

**输出**：`cycles`

---

### 3.12 选择

**指令**：`TSEL`

**语义**：按 mask 选择 `src0` 或 `src1` 的元素

**输入**：

| 字段 | 说明 |
|------|------|
| `op` | TSEL |
| `dtype` | 数据类型 |
| `rows` | Tile 行数 |
| `cols` | Tile 列数 |

**输出**：`cycles`

---

### 3.13 散射写入

**指令**：`TSCATTER`

**语义**：按索引将元素散射到目标位置

**输入**：

| 字段 | 说明 |
|------|------|
| `op` | TSCATTER |
| `dtype` | 数据类型 |
| `rows` | Tile 行数 |
| `cols` | Tile 列数 |

**输出**：`cycles`

---

### 3.14 子块提取

**指令**：`TEXTRACT`

**语义**：从大 Tile 中提取 `(dst_rows, dst_cols)` 大小的子块

**输入**：

| 字段 | 说明 |
|------|------|
| `op` | TEXTRACT |
| `dtype` | 数据类型 |
| `rows` | 源 Tile 行数 |
| `cols` | 源 Tile 列数 |
| `dst_rows` | 目标子块行数 |
| `dst_cols` | 目标子块列数 |

**输出**：`cycles`

---

### 3.15 行广播

**指令**：`TROWEXPAND`

**语义**：将第一行广播到所有行

**输入**：

| 字段 | 说明 |
|------|------|
| `op` | TROWEXPAND |
| `dtype` | 数据类型 |
| `rows` | 目标行数 |
| `cols` | 列数 |

**输出**：`cycles`

---

### 3.16 卷积加载

**指令**：`TLOADCONV`

**语义**：将卷积特征图加载到 ConvTile

**输入**：

| 字段 | 说明 |
|------|------|
| `op` | TLOADCONV |
| `dtype` | 数据类型 |
| `channels` | 通道数 |
| `height` | 特征图高 |
| `width` | 特征图宽 |

**输出**：`cycles`

**说明**：该指令的输入形状为 4D `(N, C1, H, W)` 格式，其中 N 通常为 1。cycle 取决于 `(C1 × H × W × sizeof(dtype))` 的数据量。

## 4. 汇总表

| 指令 | 类别 | 必填输入 | 可选输入 |
|------|------|----------|----------|
| TADD | 二元元素级 | op, dtype, rows, cols | - |
| TSUB | 二元元素级 | op, dtype, rows, cols | - |
| TMUL | 二元元素级 | op, dtype, rows, cols | - |
| TABS | 一元元素级 | op, dtype, rows, cols | - |
| TEXP | 一元元素级 | op, dtype, rows, cols | - |
| TSQRT | 一元元素级 | op, dtype, rows, cols | - |
| TADDS | 标量二元 | op, dtype, rows, cols | - |
| TSUBS | 标量二元 | op, dtype, rows, cols | - |
| TMULS | 标量二元 | op, dtype, rows, cols | - |
| TDIVS | 标量二元 | op, dtype, rows, cols | - |
| TMINS | 标量二元 | op, dtype, rows, cols | - |
| TMAXS | 标量二元 | op, dtype, rows, cols | - |
| TROWSUM | 行归约 | op, dtype, rows, cols | - |
| TROWMAX | 行归约 | op, dtype, rows, cols | - |
| TROWMIN | 行归约 | op, dtype, rows, cols | - |
| TCOLSUM | 列归约 | op, dtype, rows, cols | - |
| TCOLMAX | 列归约 | op, dtype, rows, cols | - |
| TMATMUL | 矩阵乘法 | op, dtype, rows, cols | **dtype2**, **k** |
| TCVT | 类型转换 | op, dtype, dst_dtype, rows, cols | round_mode |
| TMOV | 数据搬运 | op, dtype, rows, cols | - |
| TLOAD | 数据加载 | op, dtype, rows, cols | layout |
| TTRANS | 转置 | op, dtype, rows, cols | - |
| TSORT32 | 排序 | op, dtype, rows, cols | - |
| TMRGSORT | 归并排序 | op, dtype, rows, cols | block_len, src_count |
| TSEL | 条件选择 | op, dtype, rows, cols | - |
| TSCATTER | 散射写入 | op, dtype, rows, cols | - |
| TEXTRACT | 子块提取 | op, dtype, rows, cols | dst_rows, dst_cols |
| TROWEXPAND | 行广播 | op, dtype, rows, cols | - |
| TLOADCONV | 卷积加载 | op, dtype, channels, height, width | - |

## 5. 使用示例

```cpp
#include "pto/costmodel/lightweight_costmodel.hpp"

// 示例 1：估算 TADD 的 cycle
CostModelInput input;
input.op    = PtoOpcode::TADD;
input.dtype = DType::Float;
input.rows  = 64;
input.cols  = 64;
auto result = EstimateCycles(input);
// result.cycles -> 估算的 cycle 数

// 示例 2：估算 TMATMUL 的 cycle
CostModelInput input;
input.op    = PtoOpcode::TMATMUL;
input.dtype = DType::Half;
input.rows  = 64;   // M
input.cols  = 64;   // N
input.k     = 128;  // K
auto result = EstimateCycles(input);

// 示例 3：估算 TCVT 的 cycle
CostModelInput input;
input.op        = PtoOpcode::TCVT;
input.dtype     = DType::Float;
input.dst_dtype = DType::Half;
input.rows      = 64;
input.cols      = 64;
input.round_mode = RoundMode::CAST_NONE;
auto result = EstimateCycles(input);
```

## 6. 实现策略建议

轻量级 costmodel 的实现**不依赖**当前 trace-based 后端，而是直接基于公式/查表：

1. **查表法**：为每条指令维护一个 `(dtype, arch) -> base_cost` 的查找表
2. **公式法**：根据 `rows × cols`（即元素量）计算缩放系数
3. **特殊处理**：
   - TMATMUL：cycle 与 `M × N × K` 相关
   - TCVT：cycle 取决于 `src_dtype × dst_dtype` 组合
   - TLOAD：cycle 取决于数据量和 layout

核心估算公式（以元素级指令为例）：

```text
cycles = startup_overhead + ceil(element_count / throughput_per_cycle) * pipeline_latency
```

其中 `element_count = rows × cols`，`throughput_per_cycle` 和 `pipeline_latency` 由架构和 dtype 决定。
