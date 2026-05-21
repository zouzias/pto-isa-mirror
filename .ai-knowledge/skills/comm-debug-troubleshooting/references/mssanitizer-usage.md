# mssanitizer 内存检测使用指南

## 概述

mssanitizer 是昇腾平台的内存检测工具，可以检测 Kernel 中的内存越界问题。SHMEM 已适配 mssanitizer。

## 使用步骤

### 1. 编译时启用

```bash
# SHMEM 示例
bash scripts/build.sh -examples -mssanitizer

# 自定义项目：添加编译选项
-g --cce-enable-sanitizer
```

### 2. 运行时检测

```bash
# 基本用法
mssanitizer -- ./my_program arg1 arg2

# 完整格式
mssanitizer <options> -- <user_program> <user_options>
```

### 3. SHMEM 示例

```bash
# 编译
bash scripts/build.sh -examples -mssanitizer

# 运行 AllGather 示例
cd examples/allgather
bash run.sh -ranks 2 -tool mssanitizer
```

## 输出解读

### 内存越界报告

mssanitizer 检测到越界时，输出包含：

1. **越界地址**：发生越界的内存地址
2. **越界大小**：越界访问的字节数
3. **所属 Kernel**：哪个 Kernel 发生越界
4. **核号**：哪个 AI Core
5. **卡号**：哪个 NPU 设备
6. **调用栈**：越界代码的调用链

### 注意事项

**`aclshmem_malloc` 的特殊行为**：

`aclshmem_malloc` 是在已映射的连续虚拟内存上做划分，不涉及实际物理内存分配。因此：
- 如果超出 `aclshmem_malloc` 分配范围，但该虚拟地址已被映射到物理内存，mssanitizer **不会报错**
- 这种情况下数据虽然不会段错误，但是**逻辑上仍然是越界**
- 需要配合 debug 构建来更完整地检测 SHMEM 内存问题

## 常见检测场景

### 场景 1：通信 Buffer 越界

```cpp
// 错误：偏移计算错误导致写出 buffer 范围
aclshmem_mte_put_nbi(symBuf + wrongOffset, src, nbytes, destPe);
```

### 场景 2：Tiling 计算错误

```cpp
// 错误：最后一个 chunk 超出实际数据范围
for (int i = 0; i < commTurn; i++) {
    // 最后一次 i 时 offset + chunkSize 可能超出 buffer
    process(buf + i * chunkSize, chunkSize);
}
```

### 场景 3：Padding 未考虑

```cpp
// 错误：原始 buffer 大小不含 padding，但通信量含 padding
aclshmem_put(symBuf, localBuf, paddedSize, destPe);
// localBuf 实际大小 < paddedSize → 越界
```
