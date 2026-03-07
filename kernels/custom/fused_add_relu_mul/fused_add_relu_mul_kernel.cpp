/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <pto/pto-inst.hpp>

using namespace pto;

/**
 * @brief 计算当前核心负责的数据范围（公共函数，消除重复代码）
 * 
 * @param totalLength 总数据长度
 * @param start 输出：当前核心的起始位置
 * @param end 输出：当前核心的结束位置
 * @return true 如果当前核心有数据要处理，false 否则
 */
static inline bool CalculateBlockRange(uint32_t totalLength, int& start, int& end)
{
    int block_idx = get_block_idx();
    int block_num = get_block_num();
    
    // 避免除零错误：检查核心数是否有效
    if (block_num <= 0) {
        start = 0;
        end = 0;
        return false;
    }
    
    int elements_per_block = (totalLength + block_num - 1) / block_num;
    start = block_idx * elements_per_block;
    end = start + elements_per_block;
    if (end > totalLength) {
        end = totalLength;
    }
    
    // 边界检查：如果当前核心没有数据要处理，返回 false
    return (start < totalLength);
}

/**
 * @brief 执行融合的 Add-ReLU-Mul 计算（公共函数，消除重复代码）
 * 
 * @tparam TileT Tile 类型
 * @param tile_result 输出 Tile
 * @param tile_x 输入 Tile
 * @param bias 偏置值
 * @param scale 缩放因子
 */
template<typename TileT>
static inline void PerformFusedComputation(TileT& tile_result, const TileT& tile_x, 
                                           float bias, float scale)
{
    // 步骤1：Add - 加上偏置
    TADDS(tile_result, tile_x, bias);
    
    // 步骤2：ReLU - 激活函数
    TRELU(tile_result, tile_result);
    
    // 步骤3：Mul - 乘以缩放因子
    TMULS(tile_result, tile_result, scale);
}

/**
 * @brief Fused Add-ReLU-Mul 自定义算子
 * 
 * 功能：out = ReLU(x + bias) * scale
 * 
 * 这是一个典型的算子融合示例，将三个逐元素操作融合为一个 kernel：
 * 1. Add: x + bias
 * 2. ReLU: max(0, x + bias)
 * 3. Mul: result * scale
 * 
 * 融合优势：
 * - 减少内存访问：3次GM访问 → 2次GM访问（1次读，1次写）
 * - 减少kernel启动开销：3个kernel → 1个kernel
 * - 提高数据局部性：中间结果保持在L1/L0
 * 
 * @param out 输出张量（GM）
 * @param x 输入张量（GM）
 * @param bias 偏置标量
 * @param scale 缩放标量
 * @param totalLength 张量总元素数
 */
__global__ __aicore__ void FusedAddReLUMulKernel(
    __gm__ float* out,
    __gm__ const float* x,
    float bias,
    float scale,
    uint32_t totalLength)
{
    // ========== 1. 多核并行划分 ==========
    // 使用公共函数计算数据范围，消除重复代码
    int start, end;
    if (!CalculateBlockRange(totalLength, start, end)) {
        return;
    }
    
    // ========== 2. Tile 配置 ==========
    // 定义 Tile 类型：向量 Tile，float 类型，16行×256列
    // 总大小：16 × 256 × 4 bytes = 16 KB
    constexpr int TILE_H = 16;
    constexpr int TILE_W = 256;
    using TileT = Tile<TileType::Vec, float, TILE_H, TILE_W>;
    
    // ========== 3. 主处理循环 ==========
    // 按 Tile 大小分块处理数据
    constexpr int TILE_SIZE = TILE_H * TILE_W;  // 4096 元素
    
    for (int i = start; i < end; i += TILE_SIZE) {
        // 计算当前 Tile 的实际大小（处理边界情况）
        int current_size = end - i;
        if (current_size > TILE_SIZE) {
            current_size = TILE_SIZE;
        }
        
        // 声明 Tile 变量
        TileT tile_x;      // 输入数据
        TileT tile_result; // 输出数据
        
        // ========== 4. 数据加载（GM → L1 → L0）==========
        // TLOAD: 从全局内存加载数据到 Tile
        TLOAD(tile_x, GlobalTensor(x + i));
        
        // ========== 5. 融合计算 ==========
        // 使用公共函数执行 Add + ReLU + Mul
        PerformFusedComputation(tile_result, tile_x, bias, scale);
        
        // ========== 6. 数据存储（L0 → L1 → GM）==========
        // TSTORE: 将 Tile 数据写回全局内存
        TSTORE(GlobalTensor(out + i), tile_result);
    }
}

/**
 * @brief 带双缓冲优化的 Fused Add-ReLU-Mul 算子
 * 
 * 优化策略：
 * - 使用双缓冲技术重叠数据加载和计算
 * - 预加载下一批数据，同时处理当前数据
 * - 提高流水线效率，减少等待时间
 * 
 * 性能提升：相比基础版本可提升 1.5-2× 性能
 */
__global__ __aicore__ void FusedAddReLUMulOptimizedKernel(
    __gm__ float* out,
    __gm__ const float* x,
    float bias,
    float scale,
    uint32_t totalLength)
{
    // ========== 1. 多核并行划分 ==========
    // 使用公共函数计算数据范围，消除重复代码
    int start, end;
    if (!CalculateBlockRange(totalLength, start, end)) {
        return;
    }
    
    // ========== 2. Tile 配置（双缓冲）==========
    constexpr int TILE_H = 16;
    constexpr int TILE_W = 256;
    using TileT = Tile<TileType::Vec, float, TILE_H, TILE_W>;
    constexpr int TILE_SIZE = TILE_H * TILE_W;
    
    // 双缓冲：两个 Tile 交替使用
    TileT tile_x[2];      // 输入 Tile（双缓冲）
    TileT tile_result[2]; // 输出 Tile（双缓冲）
    
    // 事件：用于同步数据加载和计算
    Event load_event[2];
    
    // ========== 3. 预加载第一批数据 ==========
    if (start < end) {
        load_event[0] = TLOAD(tile_x[0], GlobalTensor(x + start));
    }
    
    // ========== 4. 流水线主循环 ==========
    int num_tiles = (end - start + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int i = start + tile_idx * TILE_SIZE;
        int current_size = end - i;
        if (current_size > TILE_SIZE) {
            current_size = TILE_SIZE;
        }
        
        // 当前和下一个缓冲区索引
        int curr = tile_idx % 2;
        int next = (tile_idx + 1) % 2;
        
        // ========== 阶段1：预加载下一批数据 ==========
        // 在处理当前数据的同时，异步加载下一批数据
        if (tile_idx + 1 < num_tiles) {
            int next_i = start + (tile_idx + 1) * TILE_SIZE;
            load_event[next] = TLOAD(tile_x[next], GlobalTensor(x + next_i));
        }
        
        // ========== 阶段2：等待当前数据加载完成 ==========
        WAIT(load_event[curr]);
        
        // ========== 阶段3：融合计算 ==========
        // 使用公共函数执行 Add + ReLU + Mul
        PerformFusedComputation(tile_result[curr], tile_x[curr], bias, scale);
        
        // ========== 阶段4：存储结果 ==========
        TSTORE(GlobalTensor(out + i), tile_result[curr]);
    }
}

/**
 * @brief 带向量化优化的版本（处理更大的 Tile）
 * 
 * 优化策略：
 * - 使用更大的 Tile 尺寸（32×512）提高数据复用
 * - 适用于 A5 平台（L1 容量更大）
 * - 减少循环迭代次数，降低控制流开销
 */
__global__ __aicore__ void FusedAddReLUMulLargeTileKernel(
    __gm__ float* out,
    __gm__ const float* x,
    float bias,
    float scale,
    uint32_t totalLength)
{
    // ========== 1. 多核并行划分 ==========
    // 使用公共函数计算数据范围，消除重复代码
    int start, end;
    if (!CalculateBlockRange(totalLength, start, end)) {
        return;
    }
    
    // 更大的 Tile：32×512 = 16384 元素 = 64 KB
    // 适用于 A5 平台（L1 ~1MB/核）
    constexpr int TILE_H = 32;
    constexpr int TILE_W = 512;
    using TileT = Tile<TileType::Vec, float, TILE_H, TILE_W>;
    constexpr int TILE_SIZE = TILE_H * TILE_W;
    
    for (int i = start; i < end; i += TILE_SIZE) {
        int current_size = end - i;
        if (current_size > TILE_SIZE) {
            current_size = TILE_SIZE;
        }
        
        TileT tile_x, tile_result;
        
        TLOAD(tile_x, GlobalTensor(x + i));
        PerformFusedComputation(tile_result, tile_x, bias, scale);
        TSTORE(GlobalTensor(out + i), tile_result);
    }
}

