# Skill: PTO 芯片架构命名空间抽取

## 概述

本 Skill 用于将 PTO 指令库中特定芯片架构的实现代码抽取到独立的命名空间中，实现架构间的代码隔离。通过 `MAP_INSTR_IMPL` 宏统一处理命名空间解析，确保公共 API 层无需修改。

## 适用场景

- 为新芯片架构（如 A5、Kirin9030 等）创建独立命名空间
- 将现有架构的实现代码迁移到命名空间中
- 解决多架构共存时的符号冲突问题

## 核心修改方案

### 1. 架构宏定义 (`include/pto/common/arch_macro.hpp`)

**目的**：定义 `MAP_INSTR_IMPL` 宏和 `arch` 命名空间别名

```cpp
#ifdef PTO_NPU_ARCH_A5
namespace pto { namespace a5 {} }  // 前向声明
namespace arch = ::pto::a5;        // 全局命名空间别名
#define MAP_INSTR_IMPL(API) arch::API##_IMPL
#else
#define MAP_INSTR_IMPL(API) API##_IMPL
#endif
```

**关键点**：
- `arch` 命名空间别名定义在全局作用域，可在任何位置使用
- `MAP_INSTR_IMPL` 宏将 `API` 转换为 `arch::API_IMPL`（A5）或 `API_IMPL`（其他架构）
- 前向声明 `namespace pto { namespace a5 {} }` 确保命名空间在使用前已定义

### 2. 实现文件命名空间包装

**目的**：将所有 `_IMPL` 函数包装到 `namespace pto { namespace <arch> { ... } }` 中

**标准模式**：
```cpp
namespace pto {
namespace a5 {

// 所有辅助结构体、检查函数、实现函数
template <typename T>
struct AddOp { ... };

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TAddCheck(...) { ... }

template <typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
PTO_INTERNAL void TADD_IMPL(TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1) { ... }

} // namespace a5
} // namespace pto
```

**批量处理脚本**：
```python
# wrap_namespace.py
import re
from pathlib import Path

def wrap_file(filepath, arch_name):
    content = Path(filepath).read_text()
    
    # 查找 namespace pto { 位置
    pto_match = re.search(r'^namespace pto \{', content, re.MULTILINE)
    if not pto_match:
        return False
    
    # 在 namespace pto { 后插入 namespace <arch> {
    insert_pos = pto_match.end()
    content = content[:insert_pos] + f'\nnamespace {arch_name} {{' + content[insert_pos:]
    
    # 查找 } // namespace pto 位置
    close_match = re.search(r'^\} // namespace pto', content, re.MULTILINE)
    if close_match:
        insert_pos = close_match.start()
        content = content[:insert_pos] + f'}} // namespace {arch_name}\n' + content[insert_pos:]
    
    Path(filepath).write_text(content)
    return True

# 使用示例
arch_name = 'a5'
arch_dir = f'include/pto/npu/{arch_name}'
for filepath in Path(arch_dir).rglob('*.hpp'):
    wrap_file(filepath, arch_name)
```

### 3. 前向声明 (`include/pto/common/pto_instr_impl.hpp`)

**目的**：在包含架构头文件前声明命名空间

```cpp
#ifdef PTO_NPU_ARCH_A5
#include "pto/npu/a5/TAssign.hpp"
#include "pto/npu/a5/TSync.hpp"
// ... 其他 a5 头文件
#endif
```

**注意**：由于 `arch_macro.hpp` 已在全局作用域定义了 `namespace pto { namespace a5 {} }`，此处无需重复声明。

### 4. 公共类型提取 (`include/pto/common/type.hpp`)

**目的**：将跨架构使用的类型从架构特定文件中提取到公共位置

**已提取的类型**：
```cpp
// 从 a5/MGather.hpp 提取
enum class GatherExec : uint8_t {
    Scalar = 0,
    Simt = 1
};

// 从 a5/TInsert.hpp 提取
enum class TInsertMode : uint8_t {
    SPLIT2 = 2,
    SPLIT4 = 3,
};

// 从 a2a3/TMrgSort.hpp 和 a5/TMrgSort.hpp 提取
struct MrgSortExecutedNumList {
    uint16_t mrgSortList0;
    uint16_t mrgSortList1;
    uint16_t mrgSortList2;
    uint16_t mrgSortList3;
};
```

**原文件修改**：
```cpp
// a5/MGather.hpp - 删除本地定义，改为包含公共头文件
#include <pto/common/type.hpp>

namespace pto {
namespace a5 {
// 直接使用 GatherExec，无需重新定义
} // namespace a5
} // namespace pto
```

### 5. Friend 声明更新 (`include/pto/common/pto_tile.hpp`)

**目的**：更新 `Tile` 和 `GlobalTensor` 类中的 friend 声明，使用 `MAP_INSTR_IMPL` 宏

**修改前**：
```cpp
template <typename T, typename AddrType>
#ifdef PTO_NPU_ARCH_A5
friend AICORE void a5::TASSIGN_IMPL(T &src, AddrType addr);
#else
friend AICORE void TASSIGN_IMPL(T &src, AddrType addr);
#endif
```

**修改后**：
```cpp
template <typename T, typename AddrType>
friend AICORE void MAP_INSTR_IMPL(TASSIGN)(T &src, AddrType addr);
```

**关键点**：
- 保留前向声明块（第 25-30 行），因为 friend 声明需要函数已被声明
- 使用 `MAP_INSTR_IMPL(TASSIGN)` 而非 `MAP_INSTR_IMPL(TASSIGN_IMPL)`，因为宏会自动添加 `_IMPL` 后缀

### 6. 直调 `_IMPL` 替换

**目的**：将所有直接调用 `XXX_IMPL` 的代码替换为 `MAP_INSTR_IMPL(XXX)` 宏调用

**批量替换脚本**：
```python
# replace_impl_calls.py
import re
from pathlib import Path

def replace_impl_calls(filepath):
    content = Path(filepath).read_text()
    original = content
    
    # 匹配模式：XXX_IMPL( 或 XXX_IMPL<...>(
    # 排除：PTO_INTERNAL void、friend、namespace、#ifndef、#define、//、using
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        # 跳过声明行
        if any(keyword in line for keyword in [
            'PTO_INTERNAL void', 'friend', 'namespace', 
            '#ifndef', '#define', '//', 'using'
        ]):
            new_lines.append(line)
            continue
        
        # 替换 XXX_IMPL( 为 MAP_INSTR_IMPL(XXX)(
        line = re.sub(r'([A-Z][A-Z0-9_]*)_IMPL\(', r'MAP_INSTR_IMPL(\1)(', line)
        new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    if content != original:
        Path(filepath).write_text(content)
        return True
    return False

# 使用示例
files_to_process = [
    'include/pto/npu/a2a3/TAlloc.hpp',
    'include/pto/npu/a2a3/TPush.hpp',
    'include/pto/npu/a2a3/TPop.hpp',
    'include/pto/npu/a2a3/TSubView.hpp',
    'include/pto/npu/a2a3/TQuant.hpp',
    'include/pto/npu/a2a3/TReshape.hpp',
    'include/pto/comm/a2a3/TPut.hpp',
    'include/pto/comm/a2a3/TGet.hpp',
    'include/pto/comm/async/sdma/TPrefetchAsyncImpl.hpp',
]

for filepath in files_to_process:
    if replace_impl_calls(filepath):
        print(f"Updated: {filepath}")
```

### 7. 测试文件更新

**目的**：为测试文件添加 `using namespace pto::<arch>;` 以访问内部函数

**修改示例**：
```cpp
// tests/npu/a5/src/st/testcase/tcolexpand/tcolexpand_kernel.cpp
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include "acl/acl.h"

using namespace pto;
namespace pto { namespace a5 {} }  // 前向声明
using namespace pto::a5;           // 访问内部函数

namespace TColExpandTest {
// 测试代码
}
```

**需要更新的测试文件**：
- 直接调用 `_IMPL` 函数的测试（如 `tcolexpand`、`tconcat`、`tconcatidx`、`tconcatdstidx`）
- 使用架构内部辅助函数的测试（如 `tinsert`、`tmov_zz`、`tpushpop_*`、`tquant_dn`）

### 8. 特殊文件处理

#### TAssign.hpp（A5 独立实现）

**原因**：A5 的 `TASSIGN_IMPL` 需要独立实现，不复用 A3

```cpp
// include/pto/npu/a5/TAssign.hpp
#ifndef TASSIGN_A5_HPP
#define TASSIGN_A5_HPP
#include <cstdint>
#include <pto/common/pto_tile.hpp>

namespace pto {
namespace a5 {
template <typename T, typename AddrType>
PTO_INTERNAL void TASSIGN_IMPL(T &obj, AddrType addr)
{
    // A5 特定实现
    if constexpr (is_tile_data_v<T> || is_conv_tile_v<T>) {
        obj.assignData(reinterpret_cast<typename T::TileDType>(static_cast<std::uintptr_t>(addr)));
    } else {
        obj.SetAddr(addr);
    }
}
} // namespace a5
} // namespace pto
#endif
```

#### TSync.hpp（部分包装）

**原因**：`Event` 结构体需要保留在 `namespace pto` 中，仅 `TSYNC_IMPL` 包装到 `namespace a5`

```cpp
// include/pto/npu/a5/TSync.hpp
namespace pto {

namespace a5 {
template <Op OpCode>
PTO_INTERNAL static constexpr pipe_t GetPipeByOpForA5() { ... }

template <Op OpCode>
PTO_INTERNAL void TSYNC_IMPL() { ... }
} // namespace a5

// Event 结构体保留在 namespace pto
template <Op SrcOp, Op DstOp, bool AutoToken = true, event_t EventID = EVENT_ID0>
struct Event : EventBase<...> { ... };

} // namespace pto
```

#### TCvt.hpp（条件包装）

**原因**：根据 `__DAV_VEC__` 和 `__DAV_CUBE__` 宏条件编译

```cpp
// include/pto/npu/a5/TCvt.hpp
#ifdef __DAV_VEC__
#include "pto/common/arch/register/tcvt_common.hpp"
namespace pto {
namespace a5 {
using ::pto::TCVT_IMPL;  // 使用公共实现
} // namespace a5
} // namespace pto

#elif defined(__DAV_CUBE__)
namespace pto {
namespace a5 {
// Cube 特定实现（空函数）
template <typename TileDataD, typename TileDataS>
PTO_INTERNAL void TCVT_IMPL(...) {}
} // namespace a5
} // namespace pto
#endif
```

#### TSubView.hpp、TReshape.hpp、TPrefetchAsync.hpp（转发包装）

**原因**：这些文件复用其他架构的实现，需要创建转发包装

```cpp
// include/pto/npu/a5/TSubView.hpp
#ifndef TSUBVIEW_A5_HPP
#define TSUBVIEW_A5_HPP
#include "pto/npu/a2a3/TSubView.hpp"

namespace pto {
namespace a5 {
template <typename TileDataDst, typename TileDataSrc>
PTO_INTERNAL void TSUBVIEW_IMPL(TileDataDst &dst, TileDataSrc &src, uint16_t rowIdx, uint16_t colIdx)
{
    ::TSUBVIEW_IMPL(dst, src, rowIdx, colIdx);  // 调用全局实现
}
} // namespace a5
} // namespace pto
#endif
```

#### TPartBinOps.hpp（循环依赖处理）

**原因**：`TPartBinOps.hpp` 调用 `TMOV_IMPL`，但 `TMov.hpp` 包含 `TPartBinOps.hpp`，形成循环依赖

**解决方案**：添加前向声明

```cpp
// include/pto/npu/a5/TPartBinOps.hpp
namespace pto {
namespace a5 {

// 前向声明解决循环依赖
template <typename DstTileData, typename SrcTileData>
PTO_INTERNAL void TMOV_IMPL(DstTileData &dst, SrcTileData &src);

// 使用直接调用而非 MAP_INSTR_IMPL（避免限定名查找问题）
template <typename Op, ...>
PTO_INTERNAL void TPartBinOp(...) {
    if (src0ValidRow == 0 || src0ValidCol == 0) {
        TMOV_IMPL(dst, src1);  // 直接调用，依赖 ADL
    }
}

} // namespace a5
} // namespace pto
```

## 验证流程

### 1. 单测试验证

```bash
# A5 架构
python3 tests/script/build_st.py -r sim -v a5 -t tadd

# A3 架构
python3 tests/script/build_st.py -r npu -v a3 -t tadd
```

### 2. 完整测试套件

```bash
# A5 完整测试
python3 tests/script/build_st.py -r sim -v a5 -t all

# A3 完整测试
python3 tests/script/build_st.py -r npu -v a3 -t all
```

### 3. 检查标准

- ✅ 构建成功（无错误）
- ✅ 无新增 warning（预存 warning 可忽略）
- ✅ 所有测试用例通过

## 常见问题与解决方案

### 问题 1：`arch` 命名空间未定义

**症状**：
```
error: use of undeclared identifier 'arch'
```

**原因**：`arch_macro.hpp` 未在使用 `arch` 的文件之前包含

**解决方案**：确保 `arch_macro.hpp` 通过 include 链被包含，或在 `pto_instr_impl.hpp` 中添加前向声明

### 问题 2：循环依赖导致编译失败

**症状**：
```
error: call to function 'TMOV_IMPL' that is neither visible in the template definition nor found by argument-dependent lookup
```

**原因**：文件 A 包含文件 B，文件 B 调用文件 A 中定义的函数

**解决方案**：
1. 添加前向声明
2. 使用直接调用而非 `MAP_INSTR_IMPL`（依赖 ADL）

### 问题 3：Friend 声明找不到函数

**症状**：
```
error: 'a5::TASSIGN_IMPL' is not a member of 'pto::a5'
```

**原因**：Friend 声明使用了限定名，但函数尚未声明

**解决方案**：保留前向声明块，或使用 `MAP_INSTR_IMPL` 宏

### 问题 4：测试文件无法访问内部函数

**症状**：
```
error: use of undeclared identifier 'TCOLEXPAND_IMPL'
```

**原因**：测试文件使用 `using namespace pto;` 但内部函数在 `pto::a5` 中

**解决方案**：添加 `using namespace pto::a5;`

### 问题 5：公共类型未定义

**症状**：
```
error: 'GatherExec' was not declared in this scope
```

**原因**：类型定义在架构特定文件中，未被提取到公共位置

**解决方案**：将类型提取到 `include/pto/common/type.hpp`

## 检查清单

- [ ] `arch_macro.hpp` 定义了 `MAP_INSTR_IMPL` 宏和 `arch` 命名空间别名
- [ ] 所有架构实现文件包装在 `namespace pto { namespace <arch> { ... } }` 中
- [ ] 公共类型提取到 `include/pto/common/type.hpp`
- [ ] `pto_tile.hpp` 中的 friend 声明使用 `MAP_INSTR_IMPL` 宏
- [ ] 所有直调 `_IMPL` 的代码替换为 `MAP_INSTR_IMPL` 宏调用
- [ ] 测试文件添加 `using namespace pto::<arch>;`
- [ ] 特殊文件（TAssign、TSync、TCvt 等）正确处理
- [ ] 循环依赖通过前向声明解决
- [ ] A5 完整测试套件构建成功
- [ ] A3 完整测试套件构建成功

## 参考资料

- A5 命名空间抽取实现：`git log --oneline --all --grep="namespace"`
- MAP_INSTR_IMPL 宏定义：`include/pto/common/arch_macro.hpp`
- 公共 API 层：`include/pto/common/pto_instr.hpp`
- 实现层：`include/pto/common/pto_instr_impl.hpp`

## 版本历史

- **v1.0** (2026-01-23)：初始版本，基于 A5 架构实现
- 修改文件数：165
- 新增行数：832
- 删除行数：471
