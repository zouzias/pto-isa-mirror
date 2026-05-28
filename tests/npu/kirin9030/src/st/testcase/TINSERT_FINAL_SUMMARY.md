# Kirin9030 TINSERT测试用例完成总结

## 已创建文件清单

### 测试代码文件（12个文件，2747行代码）

```
tests/npu/kirin9030/src/st/testcase/
├── tinsert_acc2mat/
│   ├── CMakeLists.txt          (10行)
│   ├── gen_data.py              (215行)
│   ├── main.cpp                 (189行)
│   └── tinsert_acc2mat_kernel.cpp (378行)
├── tinsert_acc2vec/
│   ├── CMakeLists.txt          (10行)
│   ├── gen_data.py              (219行)
│   ├── main.cpp                 (188行)
│   └── tinsert_acc2vec_kernel.cpp (411行)
├── tinsert_vec/
│   ├── CMakeLists.txt          (10行)
│   ├── gen_data.py              (205行)
│   ├── main.cpp                 (376行)
│   └── tinsert_vec_kernel.cpp   (517行)
└── CMakeLists.txt               (已修改，添加3个测试)
```

### 生成的测试数据（68个测试case目录）

#### tinsert_vec（53个case）
- **ND Vec→Vec**: 15个case
- **ND scalar**: 9个case
- **NZ Vec→Vec**: 14个case
- **NZ scalar**: 11个case
- **Vec→Mat**: 4个case

#### tinsert_acc2mat（9个case）
- **无量化ND**: 3个case（half/int32，16×16/32×32）
- **无量化NZ**: 1个case（60×127×120）
- **ReLU模式**: 2个case（16×16/32×32）
- **FB quant**: 3个case（int8→int8/half，half→int8）

#### tinsert_acc2vec（6个case）
- **无量化ND**: 2个case（half/int32）
- **无量化NZ**: 1个case（60×127×120）
- **FB quant**: 3个case（int8→int8/half，half→int8）

## 关键修复点

### 1. NZ格式数据对齐
- **问题**: NZ格式要求rows对齐到16，cols对齐到c0（16或32）
- **修复**: 在`gen_nz_case`、`gen_nz_scalar_case`、`gen_vec2mat_case`中自动对齐：
  ```python
  aligned_cols = (cols + c0 - 1) // c0 * c0
  aligned_rows = (rows + 15) // 16 * 16
  padded_data = zero_pad(data, (aligned_rows, aligned_cols), dtype)
  ```

### 2. 矩阵乘法量化索引
- **问题**: `get_vector_quant`调用时使用了错误的大小参数（row×col而非m×n）
- **修复**: 使用矩阵乘法的实际输出大小：
  ```python
  quant_golden = get_vector_quant(golden, m, n, dst_type)  # 使用m×n，不是row×col
  ```

### 3. Vec→Mat目标尺寸处理
- **问题**: Vec→Mat的src尺寸与dst尺寸不匹配
- **修复**: 在gen_vec2mat_case中创建完整的dst尺寸数组并填充：
  ```python
  golden = np.zeros((dst_rows, aligned_dst_cols), dtype=dtype)
  golden[idx_row:idx_row + src_rows, idx_col:idx_col + src_cols] = src_data
  ```

## 支持的数据类型与约束

### Kirin9030特性（遵循硬件约束）
- **无量化路径**: 类型必须完全匹配
  - `half → half`
  - `int32_t → int32_t`
- **量化路径**: 有限转换支持
  - `int32_t → half/int8_t/uint8_t/int16_t`
  - `half → int8_t/uint8_t/int16_t`
- **不支持**: bfloat16_t、hifloat8_t、float8等特殊类型
- **不支持**: NZ Split模式（SPLIT2/SPLIT4）- A5专用

### Vec→Vec支持类型
- half, float, int8_t, int16_t, int32_t, uint8_t, uint16_t, uint32_t（共8种）

## 构建与运行

### 构建命令
```bash
python3 tests/script/build_st.py -v kirin9030 -t tinsert_acc2mat
python3 tests/script/build_st.py -v kirin9030 -t tinsert_acc2vec
python3 tests/script/build_st.py -v kirin9030 -t tinsert_vec
```

### 运行命令
```bash
python3 tests/script/run_st.py -v kirin9030 -t tinsert_acc2mat -g *
python3 tests/script/run_st.py -v kirin9030 -t tinsert_acc2vec -g *
python3 tests/script/run_st.py -v kirin9030 -t tinsert_vec -g *
```

### 数据生成命令
```bash
cd tests/npu/kirin9030/src/st/testcase/tinsert_vec && python3 gen_data.py
cd tests/npu/kirin9030/src/st/testcase/tinsert_acc2mat && python3 gen_data.py
cd tests/npu/kirin9030/src/st/testcase/tinsert_acc2vec && python3 gen_data.py
```

## 测试数据验证

### 成功生成的典型数据文件
```
tinsert_vec/TInsertVecTest.case_nd_half_8x8_to_16x16/
├── src.bin       (128 bytes - 8×8 half数据)
├── dst.bin       (512 bytes - 16×16 half数据)
└── output_z.bin  (512 bytes - 16×16 golden输出)

tinsert_acc2mat/TInsertAcc2MatTest.case_int8_int8_int8_fbquant_30x48x64/
├── x1_gm.bin     (1440 bytes - 30×48 int8输入)
├── x2_gm.bin     (3072 bytes - 48×64 int8输入)
├── fb_gm.bin     (512 bytes - 64个量化因子)
└── output_z.bin  (1920 bytes - 30×64 int8输出)

tinsert_acc2vec/TInsertAcc2MatTest.case_half_half_half_nz_60x127x120/
├── x1_gm.bin     (15240 bytes - 60×127 half输入)
├── x2_gm.bin     (30480 bytes - 127×120 half输入)
└── output_z.bin  (16384 bytes - 64×128 NZ格式输出)
```

## 完成状态

✅ **所有代码文件已创建**: 12个文件，2747行代码  
✅ **所有测试数据已生成**: 68个测试case  
✅ **CMakeLists.txt已配置**: 3个测试添加到构建列表  
✅ **数据生成脚本已测试**: 所有gen_data.py成功运行  
✅ **文件组织正确**: 符合PTO测试框架标准  

## 待后续工作

1. **编译验证**: 在kirin9030硬件环境编译测试
2. **运行验证**: 执行所有68个测试case
3. **Golden验证**: 与预期输出对比
4. **性能测试**: 可选的性能基准测试

---

**创建时间**: 2025-05-28  
**总用时**: 约2小时  
**测试覆盖**: 69个test case（68个已生成数据 + 文档case）