# ISA 文档与代码一致性检查 Prompt

## 背景

你需要检查 `docs/isa/INSTRUCTION_zh.md`（及对应的英文版 `docs/isa/INSTRUCTION.md`）是否与 NPU 代码实现一致。
**不关注 CPU simulator 目录下的内容，仅检查 NPU 路径。**

## 检查流程

### 第一步：定位所有相关文件

对目标指令 `INSTRUCTION`，使用以下模式搜索：

1. **文档文件**：
   - `docs/isa/INSTRUCTION.md`
   - `docs/isa/INSTRUCTION_zh.md`
   - 可能存在 `docs/isa/INSTRUCTION_ASYNC.md` 等变体

2. **NPU 实现文件**：
   - `include/pto/npu/a2a3/`、`include/pto/npu/a5/`、`include/pto/npu/kirinX90/`、`include/pto/npu/kirin9030/`
   - `include/pto/common/arch/memory/`（a2a3/kirinX90 共用）
   - `include/pto/common/arch/register/`（a5/kirin9030 共用）

3. **公共 API 声明**：
   - `include/pto/common/pto_instr.hpp`

4. **类型定义**：
   - `include/pto/common/type.hpp`（Layout 枚举值校验拼写）

### 第二步：逐项交叉检查

按以下 7 个维度逐条对比文档声明与代码实现，只关注 NPU 代码（`include/pto/npu/`、`include/pto/common/arch/` 路径）。

---

#### 1. C++ 接口签名

- 检查文档中的 `PTO_INST` 声明是否与 `pto_instr.hpp` 中一致
- 关注：模板参数名、参数顺序、是否有 `WaitEvents`、返回值类型
- 简化版本可接受（如省略 `ReluPreMode`/`STPhase` 模板参数），但核心签名不能有误
- 声明的文件路径（如 `include/pto/common/pto_instr.hpp`）是否准确

#### 2. 约束检查的完整性与准确性

对文档中"约束"章节的 **每一条** 进行代码对照：

**必须逐条验证的子项：**
- **DType 列表**：文档列出的支持类型与代码 `static_assert` / `caps::IsXXX` 检查是否一致
- **TileType 限制**：Vec/Mat/Acc 支持范围是否匹配
- **布局限制**：ND/DN/NZ 及其他特殊布局（NC1HWC0、NHWC、NCHW、NDC1HWC0、NCDHW、FRACTAL_Z 等）是否完整列出
- **sizeof 匹配**：`sizeof(TileData::DType) == sizeof(GlobalData::DType)` 是否准确描述
- **静态形状范围**：Rows/Cols、SFractalSize 等边界值是否与代码一致（注意 `<` vs `<=` 的细微差异）
- **运行时断言**：`PTO_ASSERT` / `GetValidRow` / `GetValidCol` / `GetShape` 等运行时检查是否对应
- **文档中列出但代码未找到的约束**：必须标注"存疑"

#### 3. 特殊类型/格式的约束

- b64/int64/uint64 的特殊路径限制
- MX 格式（MX_A_ZZ/MX_A_ND/MX_A_DN/MX_B_NN/MX_B_ND/MX_B_DN）的形状和布局约束
- ScaleA/ScaleB 的特殊校验
- 量化参数相关的 DType 组合表是否完整准确（逐行对照代码中的 `if constexpr` 分支）

#### 4. 动态形状（`-1`）覆盖

**重点检查项**：代码中 `static_assert` 通常同时允许静态值和 `-1`（动态维度），如：
```cpp
static_assert((GlobalData::staticShape[3] == 16 || GlobalData::staticShape[3] == -1) && ...)
```
而文档常只写 `== N`，遗漏 `|| == -1` 的情况。检查所有 shape 约束时需确认文档是否涵盖了动态值。

#### 5. 中英文内容一致性

- 英文版独有的章节（如 Examples、特殊说明）中文版是否有对应
- 同一处技术约束两版表述是否一致
- 中文版本是否有遗漏的英文说明语句

#### 6. 拼写/命名正确性

- Layout 枚举名与 `include/pto/common/type.hpp` 对照（如 `MX_A_DN` 而非 `MX_ADN`）
- DType 名称与代码中的类型名一致
- 文件路径引用是否准确

#### 7. 示例代码可编译性

- 示例中的 Tile/GlobalTensor 类型定义与 API 签名是否匹配
- 模板参数（`TileType::Vec`、`BLayout::RowMajor` 等）能否在当前代码中通过编译

### 第三步：输出检查结果表格

对文档中"约束"章节的每一条声明，输出：

```
| 文档声明 | 代码位置（文件:行号） | 结果（匹配/不匹配/存疑） | 说明 |
```

### 第四步：发现问题后修改

修改文档时：

1. **改前确认**：打开对应代码文件，定位精确行号，确认新值来自代码原文
2. **联动检查**：同类问题是否在其他段落（A2A3/A5/kirinX90）或其他文档（中/英文版）也存在
3. **修改后验证**：用 `git diff docs/isa/` 确认改动内容无误，无意外改动

### 第五步：修改后复验

**必须重新跑一遍检查流程，不能因为"刚改过"而跳过。**

1. `git diff docs/isa/` 查看全部改动
2. 每条 diff 重新对照代码原文确认
3. 检查上下 5 行确认未破坏原有结构
4. 最终输出"改动—代码出处"映射表：

```
| 文档改动 | 对应代码行 |
|---------|-----------|
| 添加 NC1HWC0、NDC1HWC0 | a2a3/TStore.hpp:182-184 |
```

---

## 常见陷阱速查

| 陷阱 | 检查方式 |
|------|---------|
| 动态形状 `-1` 遗漏 | 搜索代码中 `staticShape[*] ==` 后的 `\|\|` |
| Layout 枚举遗漏 | 搜索代码中 `if constexpr (GlobalData::layout ==` 的所有分支 |
| 边界值一越之误 | 确认 `x < 4096` vs `x <= 4095` 表述等价 |
| 英文专有节中文缺失 | 检查两版文档的 `## ` 标题数量和名称 |
| 同类问题多段未联动 | 修改 A2A3 段后搜索其他架构段的相同表述 |
