# Int64 架构门控方案

> 状态：**已实施（方案 B + 复用已有宏）**，已验证通过
> 范围：`include/pto/npu/a5`、`include/pto/npu/kirin9030`、`include/pto/npu/kirinX90`
> 关联文件：`include/pto/common/arch_macro.hpp`、`include/pto/common/arch_capability.hpp`

## 1. 背景与问题

`include/pto/npu/kirin9030/header.hpp` 与 `include/pto/npu/kirinX90/header.hpp` 复用了大量
`include/pto/npu/a5/*.hpp` 指令头（见 `kirin9030/header.hpp:37-119`、`kirinX90/header.hpp:42-122`）。

a5 近期新增了 64 位整数运算支持，引入 5 个 Int64 头文件（`Int64Common/Int64Binary/Int64Div/
Int64Reduce/Int64Rearrange`），其中使用了 kirin 硬件不支持的 intrinsic，例如：

- `vaddcs` / `vsubcs`（带进位的加减，`Int64Common.hpp:26,35,82`、`Int64Binary.hpp:255,259,296,300`）
- `vmull` / `vmula`（64 位乘法，`Int64Common.hpp:42-44`、`Int64Binary.hpp:261-263`）
- `vintlv` / `vdintlv`（64 位交错，`Int64Common.hpp:97,101,108,112`）
- `vlds` / `vsts` 配合 `DINTLV_B32`（64 位 load/store 交错，`Int64Binary.hpp:108-109,249-250`）

这些 intrinsic 的函数声明在 kirin 编译环境中不存在，导致编译失败。

kirin9030 / kirinX90 不需要实际运行 int64/uint64 算子，只需保证代码能编译通过、在 kirin 上禁用
int64 即可（编译期拒绝）。

## 2. 已复现的编译错误

环境：CANN 9.1.0（bisheng/CCE，Clang 15.0.5 分支）。

| 架构 | 验证命令 | 结果 |
|---|---|---|
| kirin9030 | `python3 tests/script/build_st.py -r sim -v kirin9030 -t tadd` | 失败：`Int64Binary.hpp:259` 等 `vsubcs`/`vmull`/`vaddc`/`vaddcs` 未声明 |
| kirinX90 | `python3 tests/script/run_st.py -r sim -v kirinX90 -t tadd` | 失败：`Int64Common.hpp:25-42` + `Int64Binary.hpp:254-299` 同一批 intrinsic 未声明 |

注意：kirinX90 必须用 `run_st.py`（走 `kirinX90/src/st`，CMakeLists 设 `CCE_AICORE_ARCH dav-l300` +
`__NPU_ARCH__=3003`）；`build_st.py` 对 kirinX90 会共用 kirin9030 目录导致 `Unsupported CCE architecture`。

## 3. 根因分析

根因有两条：

### 3.1 非模板函数被立即编译

`Int64Common.hpp` 中存在 13 个**非模板** `PTO_INTERNAL void` 函数（`Int64Common.hpp:20-137`），
函数体直接调用 `vaddcs`/`vsubcs`/`vmull`/`vmula`/`vintlv`/`vdintlv` 等 intrinsic。只要该头文件
被 include，这些非模板函数就会被编译 → kirin 上符号不存在 → 编译失败。

### 3.2 无条件 include + 无条件 dispatch

约 30 个被 kirin 复用的 a5 指令头**无条件** `#include` 了 Int64 头文件，并通过
`if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>)` 或
`if constexpr (sizeof(T) == 8)` 分支调用 Int64 实现。虽然调用在 `if constexpr` 丢弃分支内，
但 Int64 头文件本身被无条件展开，触发 3.1 的非模板函数编译。

### 3.3 CCE 编译器对丢弃分支的两阶段查找行为（关键实测）

通过 stdin 方式实测 bisheng（CCE）编译器对 `if constexpr` 丢弃分支中"未声明名字"的行为：

| # | 场景 | CCE 行为 |
|---|---|---|
| A | 丢弃分支 + 枚举已声明 + 函数模板名未声明（C++17） | **warning** `-Wc++20-extensions` |
| B | 丢弃分支 + 枚举未声明（`Int64Op`）（C++17） | **error** `use of undeclared identifier 'Int64Op'` |
| C | 分支被实例化（T 真为 int64 且函数未声明） | **error** `use of undeclared identifier 'Int64Binary'` |
| D | 丢弃分支 + 函数模板名未声明（**C++20**） | **零 warning 零 error**（C++20 特性 P0846） |
| E | 常量 false 条件（`if constexpr (0)`）+ 未声明名字 | 分支体未声明名仍 **error** |
| F | caps 风格架构感知谓词（恒 false 但依赖 T）+ 未声明名字 | `Int64Op` 未声明仍 **error** |

**结论**：谓词（caps）或常量条件都不能避免"丢弃分支体被 phase-1 名字查找"。唯一干净的
路径是 **预处理层删除**（`#if` 门控）或 **名字可见**（桩声明）。

## 4. 影响面

经完整核查，被 kirin9030/kirinX90 `header.hpp` 复用、且引入了 Int64 头的 a5 指令文件共
**30 个**，分布如下：

| 依赖的 Int64 文件 | 受影响的 a5 指令头（kirin 复用） | 数量 |
|---|---|---|
| `Int64Binary.hpp` | TAdd, TSub, TMul, TMin, TMax, TShl, TShr, TAddS, TMulS, TMins, TMaxs, TShlS, TShrS, TCmp, TSel, TSels | 16 |
| `Int64Reduce.hpp` | TColSum, TColMax, TColMin, TPartAdd, TPartMax, TPartMin, TRowReduce | 7 |
| `Int64Div.hpp` | TDiv, TDivS | 2 |
| `Int64Rearrange.hpp` | TColExpand, TRowExpand, TExpandS, TScatter, TTri | 5 |

**不含**的 4 个 a5 文件（TGather, TRem, TRemS, TSubS）：由 `pto/common/pto_instr_impl.hpp:197-329`
的 `#ifdef PTO_NPU_ARCH_A5` 块引入，kirin 编译不进入 → 无需门控。

多调用点文件：TDivS（2 处）、TRowReduce（3 处）、TScatter（2 处），其余各 1 处，合计 **34 个调用点**。

## 5. 方案演进与对比

### 5.1 已实施过：方案 B（集中门控 5 个 Int64 文件 + 桩声明）

把门控集中到 5 个 Int64 文件，加 `#else` 桩声明块；30 个指令头不动。

- 改动文件数：**6**（5 个 Int64 文件 + `arch_macro.hpp`）
- 桩声明：17 个模板前向声明（`Int64Binary.hpp:5`、`Int64Div.hpp:2`、`Int64Reduce.hpp:4`、
  `Int64Rearrange.hpp:6`）
- 验证结果：kirin9030/kirinX90 26+13 个用例 kernel 编译通过；a5 回归通过
- 局限：桩声明数量多（17 个）；kirin 误用 int64 时错误信息为链接期/实例化期"未定义"而非清晰
  `static_assert`

**当前工作区即此状态**（已实施、已验证、未提交），门控条件复用已有宏
`defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)`，未创建新宏。

### 5.2 方案 A'：门控 30 个指令头的 int64 分支（零桩）

在每个指令头的 int64 分支体加 `#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)` 门控，
`#else` 放 `static_assert`；5 个 Int64 文件整体门控（连 enum），删除全部桩。

- 改动文件数：**35**（5 Int64 + 30 指令头）；`arch_macro.hpp` 不变
- 编辑点：34 处 `#if/#else/#endif`（每处 3 行）
- 编译器兼容性：100%（预处理层删除，不依赖 CCE 对丢弃分支的行为）
- 误用 int64 报错：`static_assert` 清晰错误（已实测，见 5.3 测试 17/18）
- 维护成本：新增 Int64 指令头需加门控（分散，易漏）

### 5.3 方案 A' 的关键代码模式（已实测验证）

**模式 1（`is_same_v` 条件，占多数）**：

```cpp
// 原来:
if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
    Int64Binary<Int64Op::Add, T, ...>(...);
} else {
    ...
}
// 改为:
if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
    Int64Binary<Int64Op::Add, T, ...>(...);
#else
    static_assert(sizeof(T) != 8, "int64/uint64 is not supported on this architecture");
#endif
} else {
    ...
}
```

**模式 2（`sizeof(T) == 8` 条件，TCmp/TSels/TSel/TScatter）**：同样处理。

实测结果：

| 场景 | 结果 |
|---|---|
| 正常编译（T=float/int32，kirin） | 零 warning 零 error ✓ |
| 误用 int64（T=int64_t，kirin） | `static_assert` 清晰报错 ✓ |

`static_assert(sizeof(T) != 8, ...)` 安全性：`sizeof(T)` 依赖 T → 丢弃分支不求值（T 非 8 字节
时不报错）；T 为 8 字节时分支保留 → 断言触发。

### 5.4 caps::/ArchTraits 谓词体系的定位（实测澄清）

用户提议使用项目既有的 `caps::` 谓词体系（`arch_capability.hpp`，与 `IsBF16/IsFP8` 同构）：
新增 `SupportsInt64` 能力标志（A5/A6=true，Kirin=false），`caps::IsInt64<T>()` 架构感知化。

**实测结论（测试 F）**：caps 谓词**不能独立解决**名字可见性问题——即使谓词在 kirin 恒
false，丢弃分支体里的 `Int64Op`（非依赖名）仍会被 phase-1 名字查找并报 error。

caps 的真实价值是**语义层**：把"条件判断"从 `is_same_v` 硬编码升级为架构能力声明（与
FP8/BF16 同构），必须与桩或门控之一配合使用：

| 组合 | caps + 桩 | caps + 门控（A'） |
|---|---|---|
| 名字可见性兜底 | 桩（17 个） | `#if` 预处理删除 |
| 条件语义化（与 FP8 同构） | ✓ | ✓ |
| kirin 误用 int64 报错 | 链接期（晦涩） | `static_assert`（清晰） |
| 改动文件 | 35 + arch_capability | 35 + arch_capability |

## 6. 最终决策与已实施状态

**最终采用方案 B，且复用项目已有宏组合，暂不加 caps。**

### 6.1 门控条件：复用已有宏（不使用新宏）

`PTO_INT64_ARCH_SUPPORTED` 宏**未创建**。门控条件直接使用项目既有写法：

```cpp
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
```

这是项目惯例（`arch_capability.hpp:59`、`TPow.hpp:585,675`、`TFMod.hpp:29`、
`TFModS.hpp:28` 等均为此写法）。

不可复用的宏：
- `PTO_COMM_NOT_SUPPORTED`：反向用会包含 A2A3（不支持 Int64）→ 语义不符
- `PTO_URMA_SUPPORTED`：仅 A5 的 3510 变体，不含 A6 → 不匹配

### 6.2 已实施的改动（6 个文件）

| # | 文件 | 改动 |
|---|---|---|
| 1 | `include/pto/npu/a5/Int64Common.hpp` | `enum Int64Op` 保留；13 个非模板函数用 `#if defined(PTO_NPU_ARCH_A5) \|\| defined(PTO_NPU_ARCH_A6)` 门控 |
| 2 | `include/pto/npu/a5/Int64Binary.hpp` | 实现包进 `#if`；`#else` 加 5 个桩声明 |
| 3 | `include/pto/npu/a5/Int64Div.hpp` | 实现包进 `#if`；`#else` 加 2 个桩声明 |
| 4 | `include/pto/npu/a5/Int64Reduce.hpp` | 实现包进 `#if`；`#else` 加 4 个桩声明 |
| 5 | `include/pto/npu/a5/Int64Rearrange.hpp` | 实现包进 `#if`；`#else` 加 6 个桩声明 |
| 6 | `include/pto/common/arch_macro.hpp` | **无改动**（未新增宏） |

**不改动**：30 个 a5 指令头、kirin 两个 `header.hpp`、`pto_instr_impl.hpp`。

### 6.3 验证结果（已全部通过）

| 架构 | 结果 |
|---|---|
| kirin9030 | ✅ 12 个 Int64 消费用例 kernel 编译通过（tadd/tsub/tdiv/tdivs/tcmp/tsel/tscatter/ttri/tcolsum/tpartadd/trowexpand/texpands） |
| kirinX90 | ✅ 8 个 Int64 消费用例 kernel 编译通过 |
| a5 回归 | ✅ `build success` |

### 6.4 说明

- caps `SupportsInt64` 语义化暂缓：caps 不能替代桩（实测），仅作语义增强，后续需要时再引入
- 方案 A'（门控 30 指令头零桩）作为备选保留：若未来不想维护桩，可切换

## 7. 备选方案 A'（未采用，仅存档）

若未来放弃桩方案，可切换为门控 30 个指令头的 int64 分支（零桩）：

- 改动文件数：**35**（5 Int64 + 30 指令头）
- 编辑点：34 处 `#if/#else/#endif`（TDivS 2、TRowReduce 3、TScatter 2，其余各 1）
- 模式：

```cpp
if constexpr (std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t>) {
#if defined(PTO_NPU_ARCH_A5) || defined(PTO_NPU_ARCH_A6)
    Int64Binary<Int64Op::Add, T, ...>(...);
#else
    static_assert(sizeof(T) != 8, "int64/uint64 is not supported on this architecture");
#endif
} else {
    ...
}
```

- 已实测：正常编译零 warning 零 error；误用 int64 时 static_assert 清晰报错
- 代价：新增 Int64 指令头需加门控（分散，易漏）

## 附录 A：30 个受影响 a5 指令头清单

```
Int64Binary.hpp (16):  TAdd TSub TMul TMin TMax TShl TShr
                       TAddS TMulS TMins TMaxs TShlS TShrS
                       TCmp TSel TSels
Int64Reduce.hpp (7):   TColSum TColMax TColMin
                       TPartAdd TPartMax TPartMin
                       TRowReduce
Int64Div.hpp (2):      TDiv TDivS
Int64Rearrange.hpp (5):TColExpand TRowExpand TExpandS TScatter TTri
```

## 附录 B：验证命令

```bash
# kirin9030（build_st.py，走 kirin9030/src/st，dav-l311）
python3 tests/script/build_st.py -r sim -v kirin9030 -t tadd

# kirinX90（必须用 run_st.py，走 kirinX90/src/st，dav-l300 + __NPU_ARCH__=3003）
python3 tests/script/run_st.py -r sim -v kirinX90 -t tadd

# a5 回归
python3 tests/script/build_st.py -r sim -v a5 -t tadd

# 单 kernel 目标（跳过链接，避免 runtime_camodel 库缺失干扰）
(cd tests/npu/kirin9030/src/st/build && make tadd_kernel)
(cd tests/npu/kirinX90/src/st/build && make tadd_kernel)
```

注：链接阶段 `-lruntime_camodel` 缺失是 simulator 库路径的环境问题，与 Int64 改动无关；
kernel 编译目标通过即证明 Int64 问题已解决。
