# 内存优化技巧

本文档提供 PTO 算子开发中的内存优化技巧，帮助开发者充分利用片上内存，减少内存访问开销。

## 目录

- [1. 内存层次结构](#1-内存层次结构)
- [2. L1 内存优化](#2-l1-内存优化)
- [3. L0 内存优化](#3-l0-内存优化)
- [4. 内存对齐优化](#4-内存对齐优化)
- [5. 内存复用策略](#5-内存复用策略)
- [6. 内存带宽优化](#6-内存带宽优化)

---

## 1. 内存层次结构

### 1.1 Ascend 内存架构

```
全局内存 (GM)
    ↓ MTE2 (DMA)
L1 缓存 (~512KB - 1MB/核)
    ↓ MTE1
L0 缓存 (寄存器级)
    ↓
计算单元 (CUBE/VECTOR)
```

**访问延迟对比**：
| 内存层级 | 延迟 | 带宽 |
|---------|------|------|
| L0 | ~1 cycle | 最高 |
| L1 | ~10 cycles | 高 |
| GM | ~100+ cycles | 受限 |

**优化原则**：
- 尽量在 L0 中完成计算
- 在 L1 中缓存频繁访问的数据
- 减少 GM 访问次数

---

## 2. L1 内存优化

### 2.1 L1 容量规划

**平台容量**：
- A2/A3：~512 KB/核
- A5：~1 MB/核

**容量分配示例（GEMM）**：
```cpp
// A2/A3 平台
constexpr int L1_CAPACITY = 512 * 1024;  // 512KB

// 双缓冲 A Tile: 2 × 128×64×2 = 32KB
constexpr int A_BUFFER_SIZE = 2 * 128 * 64 * sizeof(half);

// 双缓冲 B Tile: 2 × 64×256×2 = 64KB
constexpr int B_BUFFER_SIZE = 2 * 64 * 256 * sizeof(half);

// 累加器: 128×256×4 = 128KB
constexpr int ACC_BUFFER_SIZE = 128 * 256 * sizeof(float);

// 总计: 32 + 64 + 128 = 224KB < 512KB ✓
```

### 2.2 避免 L1 溢出

**检查方法**：
```cpp
// 编译时检查
static_assert(TOTAL_L1_USAGE < L1_CAPACITY, 
              "L1 memory overflow!");

// 运行时检查（调试模式）
#ifdef DEBUG
  size_t total_usage = calculate_l1_usage();
  assert(total_usage < L1_CAPACITY);
#endif
```

**溢出后果**：
- 性能急剧下降（10-100×）
- 频繁的 L1 ↔ GM 换页
- 流水线停顿

**解决方案**：
```cpp
// 方案1：减小 Tile 尺寸
// 优化前：
using TileT = Tile<TileType::Vec, float, 32, 512>;  // 64KB

// 优化后：
using TileT = Tile<TileType::Vec, float, 16, 256>;  // 16KB

// 方案2：减少缓冲区数量
// 从三缓冲改为双缓冲
```

### 2.3 L1 数据复用

**策略1：K 维度分块**
```cpp
// GEMM 示例：复用 A 和 B
for (int k = 0; k < K; k += TILE_K) {
  // 加载到 L1 一次
  TLOAD(tileA_L1, A[m:m+M, k:k+TILE_K]);
  TLOAD(tileB_L1, B[k:k+TILE_K, n:n+N]);
  
  // 在 L1 中复用多次
  for (int sub_k = 0; sub_k < TILE_K; sub_k += SUB_K) {
    TEXTRACT(tileA_L0, tileA_L1[sub_k:sub_k+SUB_K]);
    TEXTRACT(tileB_L0, tileB_L1[sub_k:sub_k+SUB_K]);
    TMATMUL(acc, tileA_L0, tileB_L0);
  }
}
```

**策略2：缓存常量数据**
```cpp
// Softmax 示例：缓存 max 和 sum
TLOAD(input_L1, input);
TROWMAX(max_val, input_L1);  // 计算一次，缓存在 L1
TROWEXPANDSUB(shifted, input_L1, max_val);  // 复用 max_val
TEXP(exp_vals, shifted);
TROWSUM(sum_val, exp_vals);  // 计算一次，缓存在 L1
TROWEXPANDDIV(output, exp_vals, sum_val);  // 复用 sum_val
```

---

## 3. L0 内存优化

### 3.1 L0 容量管理

**L0 类型**：
- Vector L0 (UB)：向量寄存器
- Matrix L0A/L0B：矩阵操作数寄存器
- Matrix L0C：累加器寄存器

**容量限制**：
- 通常几十 KB
- 需要精确管理

### 3.2 寄存器分配

**手动分配（Manual Mode）**：
```cpp
// 使用 TASSIGN 绑定地址
constexpr int ADDR_A = 0;
constexpr int ADDR_B = 32 * 1024;  // 32KB offset
constexpr int ADDR_C = 64 * 1024;  // 64KB offset

TileLeft tileA;
TileRight tileB;
TileAcc tileC;

TASSIGN(tileA, ADDR_A);
TASSIGN(tileB, ADDR_B);
TASSIGN(tileC, ADDR_C);
```

**自动分配（Auto Mode）**：
```cpp
// 编译器自动管理
TileLeft tileA;
TileRight tileB;
TileAcc tileC;
// 无需 TASSIGN
```

### 3.3 寄存器复用

**策略1：时间复用**
```cpp
// 不同阶段复用同一寄存器
TileVec temp;

// 阶段1：用于中间结果
TADD(temp, a, b);
TSTORE(output1, temp);

// 阶段2：复用存储其他数据
TMUL(temp, c, d);
TSTORE(output2, temp);
```

**策略2：空间复用**
```cpp
// 使用 union 复用内存
union {
  TileVec vec_tile;
  TileMat mat_tile;
} shared_buffer;

// 先用作 vector
TADD(shared_buffer.vec_tile, a, b);

// 后用作 matrix
TMOV(shared_buffer.mat_tile, ...);
```

---

## 4. 内存对齐优化

### 4.1 对齐要求

**基本对齐**：
- 行主序：`Cols × sizeof(T)` 对齐到 32 字节
- 列主序：`Rows × sizeof(T)` 对齐到 32 字节

**示例**：
```cpp
// 正确：对齐到 32 字节
using TileT1 = Tile<TileType::Vec, float, 16, 8>;
// 16 × 8 × 4 = 512 bytes (32的倍数) ✓

// 错误：未对齐
using TileT2 = Tile<TileType::Vec, float, 16, 7>;
// 16 × 7 × 4 = 448 bytes (不是32的倍数) ✗
```

### 4.2 Padding 技巧

**方法1：编译时 Padding**
```cpp
// 原始大小：不对齐
constexpr int ORIG_COLS = 127;

// Padding 到对齐
constexpr int PADDED_COLS = ((ORIG_COLS + 7) / 8) * 8;  // 128

using TileT = Tile<TileType::Vec, float, 16, PADDED_COLS>;
```

**方法2：运行时 Padding**
```cpp
// 使用 valid region 处理非对齐数据
using TileT = Tile<TileType::Vec, float, 16, 128,
                   RowMajor, 16, DYNAMIC>;  // 动态列数

TileT tile(actual_cols);  // actual_cols 可以是 127
```

### 4.3 分形对齐

**NZ 布局对齐**：
```cpp
// NZ 布局要求
// Rows 对齐到 16
// Cols 对齐到 C0Size = 32 / sizeof(T)

// fp16: C0Size = 16
using TileNZ_fp16 = Tile<TileType::Mat, half, 16, 16,
                         RowMajor, 16, 16, RowMajor, 512>;

// fp32: C0Size = 8
using TileNZ_fp32 = Tile<TileType::Mat, float, 16, 8,
                         RowMajor, 16, 8, RowMajor, 512>;
```

---

## 5. 内存复用策略

### 5.1 数据复用模式

**模式1：时间复用（Temporal Reuse）**
```cpp
// 同一数据在不同时间被多次使用
TLOAD(tile, data);
for (int i = 0; i < N; i++) {
  TCOMPUTE(result[i], tile, other[i]);  // 复用 tile
}
```

**模式2：空间复用（Spatial Reuse）**
```cpp
// 相邻数据一起加载，分别使用
TLOAD(tile_block, data[0:BLOCK_SIZE]);
for (int i = 0; i < BLOCK_SIZE; i++) {
  TEXTRACT(tile_i, tile_block, i);
  TCOMPUTE(result[i], tile_i);
}
```

### 5.2 GEMM 复用策略

**三级复用**：
```cpp
// M×K×N GEMM
for (int m = 0; m < M; m += TILE_M) {
  for (int n = 0; n < N; n += TILE_N) {
    // 累加器复用：整个 K 循环
    TileAcc acc;
    TFILL(acc, 0);
    
    for (int k = 0; k < K; k += TILE_K) {
      // A 复用：N 方向
      TLOAD(tileA, A[m:m+TILE_M, k:k+TILE_K]);
      
      // B 复用：M 方向
      TLOAD(tileB, B[k:k+TILE_K, n:n+TILE_N]);
      
      TMATMUL_ACC(acc, tileA, tileB);
    }
    
    TSTORE(C[m:m+TILE_M, n:n+TILE_N], acc);
  }
}
```

**复用分析**：
- A 的每个元素被复用 N/TILE_N 次
- B 的每个元素被复用 M/TILE_M 次
- 累加器被复用 K/TILE_K 次

### 5.3 Flash Attention 复用

**在线算法复用**：
```cpp
// 不存储完整的注意力矩阵
for (int i = 0; i < SEQ_LEN; i += TILE_SIZE) {
  // Q 复用：整个 K/V 循环
  TLOAD(Q_tile, Q[i:i+TILE_SIZE, :]);
  
  // 在线更新统计量
  float max_val = -INF;
  float sum_val = 0;
  
  for (int j = 0; j < SEQ_LEN; j += TILE_SIZE) {
    TLOAD(K_tile, K[j:j+TILE_SIZE, :]);
    TMATMUL(S_tile, Q_tile, K_tile);  // 复用 Q_tile
    
    // 更新统计量（复用 max_val 和 sum_val）
    update_statistics(max_val, sum_val, S_tile);
  }
  
  // 使用统计量计算输出
  compute_output(output, max_val, sum_val);
}
```

---

## 6. 内存带宽优化

### 6.1 减少内存访问

**策略1：算子融合**
```cpp
// 融合前：3 次 GM 访问
TLOAD(a, input1);
TADD(b, a, scalar);
TSTORE(output1, b);

TLOAD(c, output1);  // 重复访问
TMUL(d, c, scalar2);
TSTORE(output2, d);

// 融合后：2 次 GM 访问
TLOAD(a, input1);
TADD(b, a, scalar);
TMUL(d, b, scalar2);  // 直接使用 b，无需存储和重新加载
TSTORE(output2, d);
```

**策略2：计算密集型设计**
```cpp
// 提高算术强度 = FLOPs / Bytes
// 目标：每字节数据做更多计算

// 不好：算术强度低
for (int i = 0; i < N; i++) {
  TLOAD(tile, data[i]);     // 1 次加载
  TADD(result, tile, 1);    // 1 次计算
  TSTORE(output[i], result); // 1 次存储
}
// 算术强度 = 1 FLOP / (2 × tile_size bytes)

// 好：算术强度高
TLOAD(tile, data);           // 1 次加载
for (int i = 0; i < 100; i++) {
  TADD(tile, tile, 1);       // 100 次计算
}
TSTORE(output, tile);        // 1 次存储
// 算术强度 = 100 FLOPs / (2 × tile_size bytes)
```

### 6.2 连续访问模式

**行主序优化**：
```cpp
// 好：连续访问
for (int i = 0; i < M; i++) {
  TLOAD(tile, A[i, :]);  // 行连续
  TCOMPUTE(result, tile);
}

// 不好：跨步访问
for (int j = 0; j < N; j++) {
  TLOAD(tile, A[:, j]);  // 列访问，可能不连续
  TCOMPUTE(result, tile);
}
```

**解决方案**：
```cpp
// 方案1：转置数据
TTRANS(A_T, A);
for (int j = 0; j < N; j++) {
  TLOAD(tile, A_T[j, :]);  // 现在是连续的
  TCOMPUTE(result, tile);
}

// 方案2：使用列主序布局
using TileT = Tile<TileType::Vec, float, M, N, ColMajor>;
```

### 6.3 预取优化

**软件预取**：
```cpp
// 预取下一批数据
for (int i = 0; i < N; i++) {
  // 处理当前数据
  TCOMPUTE(result[i], tile[i]);
  
  // 预取下一批
  if (i + 1 < N) {
    TPREFETCH(tile[i+1], data[i+1]);
  }
}
```

**双缓冲预取**：
```cpp
// 预加载第一批
TLOAD(tile[0], data[0]);

for (int i = 0; i < N; i++) {
  int curr = i % 2;
  int next = (i + 1) % 2;
  
  // 处理当前
  TCOMPUTE(result[i], tile[curr]);
  
  // 同时加载下一批
  if (i + 1 < N) {
    TLOAD(tile[next], data[i+1]);
  }
}
```

---

## 7. 内存优化检查清单

### 容量规划
- [ ] 计算总 L1 使用量
- [ ] 确保不超过平台限制
- [ ] 为双缓冲预留空间
- [ ] 考虑其他临时缓冲区

### 对齐优化
- [ ] 检查 Tile 对齐要求
- [ ] 使用 Padding 处理非对齐数据
- [ ] 验证分形布局对齐

### 数据复用
- [ ] 识别可复用的数据
- [ ] 在 L1 中缓存频繁访问的数据
- [ ] 设计合理的 tiling 策略

### 带宽优化
- [ ] 减少 GM 访问次数
- [ ] 使用连续访问模式
- [ ] 考虑算子融合
- [ ] 使用预取和双缓冲

### 验证
- [ ] 使用 profiler 检查内存带宽利用率
- [ ] 验证没有 L1 溢出
- [ ] 检查内存访问模式

---

## 8. 实战案例

### 案例1：GEMM 内存优化

**优化前**：
```cpp
// 单缓冲，L1 使用 300KB
for (int k = 0; k < K; k += TILE_K) {
  TLOAD(tileA, ...);  // 等待加载
  TLOAD(tileB, ...);  // 等待加载
  TMATMUL(acc, tileA, tileB);
}
// 内存带宽利用率：40%
```

**优化后**：
```cpp
// 双缓冲，L1 使用 450KB
TLOAD(tileA[0], ...);
TLOAD(tileB[0], ...);

for (int k = 0; k < K; k += TILE_K) {
  int curr = k % 2;
  int next = (k + TILE_K) % 2;
  
  // 计算当前
  TMATMUL(acc, tileA[curr], tileB[curr]);
  
  // 同时加载下一批
  if (k + TILE_K < K) {
    TLOAD(tileA[next], ...);
    TLOAD(tileB[next], ...);
  }
}
// 内存带宽利用率：75%
```

**优化效果**：
- 内存带宽利用率：40% → 75%
- 性能提升：1.8×

### 案例2：Softmax 内存优化

**优化前**：
```cpp
// 多次访问 GM
TLOAD(input, ...);
TROWMAX(max_val, input);
TSTORE(max_val, ...);  // 存储到 GM

TLOAD(input, ...);  // 重新加载
TLOAD(max_val, ...);  // 重新加载
TROWEXPANDSUB(shifted, input, max_val);
// ...
```

**优化后**：
```cpp
// 在 L1 中完成所有操作
TLOAD(input, ...);
TROWMAX(max_val, input);  // 保持在 L1
TROWEXPANDSUB(shifted, input, max_val);  // 复用
TEXP(exp_vals, shifted);
TROWSUM(sum_val, exp_vals);  // 保持在 L1
TROWEXPANDDIV(output, exp_vals, sum_val);  // 复用
TSTORE(output, ...);
```

**优化效果**：
- GM 访问次数：6 → 2
- 性能提升：2.5×

---

## 参考资源

- [性能优化指南](opt_zh.md)
- [流水线与并行执行](pipeline-parallel_zh.md)
- [性能调优最佳实践](performance-best-practices_zh.md)
- [GEMM 优化案例](../../kernels/manual/a2a3/gemm_performance/README_zh.md)





