<p align="center">
  <img src="docs/figures/pto_logo.svg" alt="PTO Tile Lib" width="220" />
</p>

# PTO Tile Library（中文说明）

本文件补充“下游项目如何用 CMake 集成 pto-isa”。其他内容请先参考英文版 `README.md`。

## CMake 集成

下游项目可以用现代 CMake（target-based）方式集成 pto-isa，不需要手写 `-I...` / `-D...`。

pto-isa 提供两个 header-only 的 INTERFACE 目标：

- `pto-isa::includes`：只带 include 目录（建议 host / CPU 仿真场景使用）
- `pto-isa::headers`：include + 必需的编译宏（自动检测：`MEMORY_BASE` / `REGISTER_BASE` / `__CPU_SIM`）

### 方式 1：FetchContent / add_subdirectory

```cmake
include(FetchContent)
FetchContent_Declare(
  pto-isa
  GIT_REPOSITORY <url>
  GIT_TAG <tag>
)

# pto-isa 默认自动检测：
# - `npu-smi info` 包含 910B*/910C* -> MEMORY_BASE
# - `npu-smi info` 包含 950*       -> REGISTER_BASE
# - 没有 `npu-smi`                  -> __CPU_SIM
# 交叉编译时建议显式覆盖：
# set(PTO_ISA_BACKEND cpu CACHE STRING "" FORCE)
# set(PTO_ISA_BACKEND npu CACHE STRING "" FORCE)
# set(PTO_ISA_NPU_ADDRESSING memory CACHE STRING "" FORCE)   # 910B/910C
# set(PTO_ISA_NPU_ADDRESSING register CACHE STRING "" FORCE) # 950

FetchContent_MakeAvailable(pto-isa)

target_link_libraries(my_kernel PRIVATE pto-isa::headers)
```

### 方式 2：find_package（安装后使用）

先安装 CMake package config：

```bash
cmake -S . -B build \
  -DPTO_ISA_ENABLE_RUN_PACKAGE=OFF \
  -DPTO_ISA_INSTALL_CMAKE_PACKAGE=ON \
  -DPTO_ISA_BACKEND=auto \
  -DCMAKE_INSTALL_PREFIX=/path/to/prefix
cmake --install build
```

下游项目中使用：

```cmake
find_package(pto-isa CONFIG REQUIRED)
target_link_libraries(my_kernel PRIVATE pto-isa::headers)
```

### Host / CPU 仿真

如果机器上没有 `npu-smi`，`pto-isa::headers` 会默认走 `__CPU_SIM`。在纯 host 场景下也可以只 link `pto-isa::includes`，并由下游自行定义 `__CPU_SIM`。

```cmake
target_link_libraries(my_host_app PRIVATE pto-isa::includes)
target_compile_definitions(my_host_app PRIVATE __CPU_SIM)
```
