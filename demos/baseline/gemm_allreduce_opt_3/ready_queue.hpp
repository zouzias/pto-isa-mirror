/**
 * Ready Queue - Per-block queue for tile scheduling
 *
 * Multi-block safe design:
 *   - Each compute block has its own queue (no contention on enqueue)
 *   - Comm kernel polls all queues (one cache invalidation per queue)
 *   - Supports arbitrary completion order from parallel blocks
 *
 * Benefits over shared queue:
 *   - No atomic operations needed for enqueue (single producer per queue)
 *   - O(B) poll instead of O(N) flag polling, where B = number of blocks
 *   - Better cache locality per block
 */

#pragma once

#include <cstdint>

#ifndef __CCE_KT_TEST__
#include "pto/comm/pto_comm_inst.hpp"
#endif

// Maximum number of compute blocks supported
constexpr int MAX_COMPUTE_BLOCKS = 32;

// ============================================================================
// PerBlockQueue - Single-producer queue for one compute block
//
// Memory layout (cache-line aligned):
//   [0-3]   tail       - producer write position (compute kernel)
//   [4-7]   count      - number of items in queue
//   [8-11]  capacity   - max queue size (= tiles per block)
//   [12-15] block_id   - which block owns this queue
//   [16-31] padding    - cache line alignment
//   [32+]   data[]     - tile indices buffer
//
// Single producer (compute block), single consumer (comm kernel)
// ============================================================================
struct alignas(64) PerBlockQueue {
    volatile int32_t tail;       // Producer write position
    volatile int32_t count;      // Current number of items in queue
    int32_t capacity;            // Queue capacity
    int32_t block_id;            // Block ID that owns this queue
    int32_t padding[4];          // Pad to 32 bytes
    int32_t data[1];             // Flexible array (actual size = capacity)
};

// ============================================================================
// MultiBlockQueueSet - Array of per-block queues
//
// Layout:
//   [0-3]     num_blocks     - total number of compute blocks
//   [4-7]     total_tiles    - total number of tiles across all blocks
//   [8-11]    tiles_per_blk  - tiles per block (for even distribution)
//   [12-15]   padding
//   [16-31]   padding
//   [32+]     offsets[]      - byte offset to each PerBlockQueue
//   [after]   queues data    - actual PerBlockQueue structures
// ============================================================================
struct alignas(64) MultiBlockQueueSet {
    int32_t num_blocks;          // Number of compute blocks
    int32_t total_tiles;         // Total tiles across all blocks
    int32_t tiles_per_block;     // Tiles per block (ceiling division)
    int32_t consumed_count;      // Total tiles consumed by comm kernel
    int32_t padding[4];          // Pad to 32 bytes
    int32_t queue_offsets[MAX_COMPUTE_BLOCKS];  // Byte offsets to each queue
    // PerBlockQueue structures follow after offsets
};

// ============================================================================
// Size calculations
// ============================================================================
constexpr size_t PerBlockQueueSize(int capacity)
{
    // Base struct size (without flexible array) + data array
    return sizeof(PerBlockQueue) - sizeof(int32_t) + capacity * sizeof(int32_t);
}

inline size_t MultiBlockQueueSetSize(int num_blocks, int tiles_per_block)
{
    // Header + offsets array + all per-block queues
    size_t header_size = sizeof(MultiBlockQueueSet);
    size_t per_queue_size = PerBlockQueueSize(tiles_per_block);
    // Align each queue to 64 bytes
    per_queue_size = ((per_queue_size + 63) / 64) * 64;
    return header_size + num_blocks * per_queue_size;
}

// ============================================================================
// Host-side: Initialize a single per-block queue
// ============================================================================
inline void PerBlockQueueInit(PerBlockQueue* queue, int capacity, int block_id)
{
    queue->tail = 0;
    queue->count = 0;
    queue->capacity = capacity;
    queue->block_id = block_id;
    for (int i = 0; i < 4; i++) queue->padding[i] = 0;
    for (int i = 0; i < capacity; i++) {
        queue->data[i] = -1;
    }
}

// ============================================================================
// Host-side: Initialize the multi-block queue set
// ============================================================================
inline void MultiBlockQueueSetInit(MultiBlockQueueSet* qset, int num_blocks, int total_tiles)
{
    qset->num_blocks = num_blocks;
    qset->total_tiles = total_tiles;
    qset->tiles_per_block = (total_tiles + num_blocks - 1) / num_blocks;  // Ceiling
    qset->consumed_count = 0;
    for (int i = 0; i < 4; i++) qset->padding[i] = 0;
    
    // Calculate offsets for each per-block queue
    size_t per_queue_size = PerBlockQueueSize(qset->tiles_per_block);
    per_queue_size = ((per_queue_size + 63) / 64) * 64;  // Align to 64 bytes
    
    size_t base_offset = sizeof(MultiBlockQueueSet);
    
    for (int b = 0; b < num_blocks; b++) {
        qset->queue_offsets[b] = static_cast<int32_t>(base_offset + b * per_queue_size);
        
        // Initialize the per-block queue
        PerBlockQueue* pq = reinterpret_cast<PerBlockQueue*>(
            reinterpret_cast<uint8_t*>(qset) + qset->queue_offsets[b]);
        PerBlockQueueInit(pq, qset->tiles_per_block, b);
    }
    
    // Zero out unused offsets
    for (int b = num_blocks; b < MAX_COMPUTE_BLOCKS; b++) {
        qset->queue_offsets[b] = 0;
    }
}

// ============================================================================
// Host-side: Reset all queues (for reuse between iterations)
// ============================================================================
inline void MultiBlockQueueSetReset(MultiBlockQueueSet* qset)
{
    qset->consumed_count = 0;
    for (int b = 0; b < qset->num_blocks; b++) {
        PerBlockQueue* pq = reinterpret_cast<PerBlockQueue*>(
            reinterpret_cast<uint8_t*>(qset) + qset->queue_offsets[b]);
        pq->tail = 0;
        pq->count = 0;
    }
}

// ============================================================================
// Device-side: Get pointer to this block's queue
// ============================================================================
#ifndef __CCE_KT_TEST__
AICORE inline volatile __gm__ PerBlockQueue* GetMyBlockQueue(
    volatile __gm__ MultiBlockQueueSet* qset, 
    int block_idx)
{
    // Read offset from queue set
    dcci((__gm__ void*)&qset->queue_offsets[block_idx], SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
    
    int32_t offset = qset->queue_offsets[block_idx];
    return reinterpret_cast<volatile __gm__ PerBlockQueue*>(
        reinterpret_cast<volatile __gm__ uint8_t*>(qset) + offset);
}
#endif

// ============================================================================
// Device-side enqueue (producer - compute kernel)
//
// Single-producer per queue - NO contention, NO atomic operations needed!
// Each compute block only writes to its own queue.
//
// Returns: true if successful, false if queue full
// ============================================================================
#ifdef __CCE_KT_TEST__
// Host test version
inline bool PerBlockQueueEnqueue(PerBlockQueue* queue, int32_t tile_idx)
{
    int32_t cap = queue->capacity;
    
    // Check if queue is full
    if (queue->count >= cap) {
        return false;
    }
    
    // Write tile index to current tail position
    int32_t slot = queue->tail;
    queue->data[slot] = tile_idx;
    queue->tail = slot + 1;  // No wrap needed - single producer
    queue->count++;
    
    return true;
}
#else
// Device version - single producer, no atomics needed
AICORE inline bool PerBlockQueueEnqueue(
    volatile __gm__ PerBlockQueue* queue, 
    int32_t tile_idx)
{
    // Read current state
    dcci((__gm__ void*)&queue->count, SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
    
    int32_t cap = queue->capacity;
    int32_t cnt = queue->count;
    
    if (cnt >= cap) {
        return false;  // Queue full (shouldn't happen with proper sizing)
    }
    
    // Get current tail position
    dcci((__gm__ void*)&queue->tail, SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
    
    int32_t slot = queue->tail;
    
    // Write tile index
    queue->data[slot] = tile_idx;
    dcci((__gm__ void*)&queue->data[slot], SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
    
    // Update tail
    queue->tail = slot + 1;
    dcci((__gm__ void*)&queue->tail, SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
    
    // Increment count (this signals the consumer)
    queue->count = cnt + 1;
    dcci((__gm__ void*)&queue->count, SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
    
    return true;
}
#endif

// ============================================================================
// Device-side: Fast enqueue with caller-tracked state
//
// Optimization: The standard PerBlockQueueEnqueue reads count and tail from GM
// (2 dcci reads) and writes data, tail, count (3 dcci writes) = 5 dcci total.
// Since compute kernel is single-producer, it can track tail and count locally,
// reducing to 2 dcci calls (data write + count signal).
//
// The consumer only reads count (via TTEST) and data[head]. It never reads tail.
// So tail writes don't need dcci for consumer visibility.
//
// Parameters:
//   local_slot: caller-tracked slot position (starts at 0, incremented by caller)
//   local_count: caller-tracked item count (starts at 0, incremented by caller)
// ============================================================================
AICORE inline void PerBlockQueueEnqueueFast(
    volatile __gm__ PerBlockQueue* queue,
    int32_t tile_idx,
    int32_t local_slot)
{
    // Write tile index to data array
    queue->data[local_slot] = tile_idx;
    dcci((__gm__ void*)&queue->data[local_slot], SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
    
    // Update tail (no dcci needed - consumer doesn't read tail)
    queue->tail = local_slot + 1;
    
    // Increment count - this signals the consumer via TTEST
    queue->count = local_slot + 1;
    dcci((__gm__ void*)&queue->count, SINGLE_CACHE_LINE);
    __asm__ __volatile__("");
}

// ============================================================================
// Device-side: Enqueue using MultiBlockQueueSet
// Convenience wrapper that finds the correct queue for this block
// ============================================================================
#ifndef __CCE_KT_TEST__
AICORE inline bool MultiBlockEnqueue(
    volatile __gm__ MultiBlockQueueSet* qset,
    int block_idx,
    int32_t tile_idx)
{
    volatile __gm__ PerBlockQueue* my_queue = GetMyBlockQueue(qset, block_idx);
    return PerBlockQueueEnqueue(my_queue, tile_idx);
}

// Fast version: uses cached queue pointer and caller-tracked slot position
AICORE inline void MultiBlockEnqueueFast(
    volatile __gm__ PerBlockQueue* cached_queue,
    int32_t tile_idx,
    int32_t local_slot)
{
    PerBlockQueueEnqueueFast(cached_queue, tile_idx, local_slot);
}
#endif

// ============================================================================
// Device-side: Try dequeue from a single per-block queue
//
// Single consumer, so no atomics needed for head tracking.
// We track consumed items using a local head counter.
//
// Returns: tile index, or -1 if queue is empty
// ============================================================================
#ifdef __CCE_KT_TEST__
// Host test version
inline int32_t PerBlockQueueTryDequeue(PerBlockQueue* queue, int32_t* head)
{
    if (*head >= queue->count) {
        return -1;  // Nothing new in this queue
    }
    
    int32_t tile_idx = queue->data[*head];
    (*head)++;
    return tile_idx;
}
#else
// Device version: use PTO TTEST for count check (hardware-friendly sync API)
AICORE inline int32_t PerBlockQueueTryDequeue(
    volatile __gm__ PerBlockQueue* queue,
    int32_t local_head)  // Caller tracks head position
{
    // Use TTEST to check if at least one new item (count >= local_head + 1)
    pto::comm::Signal sig(const_cast<__gm__ int32_t*>(&queue->count));
    if (!pto::comm::TTEST(sig, local_head + 1, pto::comm::WaitCmp::GE)) {
        return -1;  // Nothing new in this queue
    }

    // Read tile index at local_head position (dcci for data visibility)
    dcci((__gm__ void*)&queue->data[local_head], SINGLE_CACHE_LINE);
    __asm__ __volatile__("");

    return queue->data[local_head];
}
#endif

// ============================================================================
// Device-side: Poll all queues and dequeue next available tile
//
// Round-robin polling across all per-block queues.
// Each queue is polled with O(1) cost (just check count).
// Total poll cost is O(num_blocks) instead of O(num_tiles).
//
// Parameters:
//   - qset: MultiBlockQueueSet in GM
//   - heads: Array of local head positions (one per queue, in UB or registers)
//   - start_block: Where to start polling (for round-robin fairness)
//
// Returns: tile index, or -1 if all queues empty
// ============================================================================
#ifndef __CCE_KT_TEST__
AICORE inline int32_t MultiBlockTryDequeue(
    volatile __gm__ MultiBlockQueueSet* qset,
    int32_t* heads,      // Local head positions (one per block)
    int32_t num_blocks,
    int32_t* next_block) // In/out: where to start next poll
{
    // Round-robin across all queues
    for (int i = 0; i < num_blocks; i++) {
        int32_t b = (*next_block + i) % num_blocks;
        
        volatile __gm__ PerBlockQueue* pq = GetMyBlockQueue(qset, b);
        int32_t tile = PerBlockQueueTryDequeue(pq, heads[b]);
        
        if (tile >= 0) {
            heads[b]++;  // Advance this queue's head
            *next_block = (b + 1) % num_blocks;  // Next time, start from next queue
            return tile;
        }
    }
    
    return -1;  // All queues empty
}
#endif

// ============================================================================
// Device-side: Blocking poll - waits until a tile is available
//
// Continuously polls all queues until one has a tile ready.
// Used when we know more tiles will be produced.
// ============================================================================
#ifndef __CCE_KT_TEST__
AICORE inline int32_t MultiBlockDequeueBlocking(
    volatile __gm__ MultiBlockQueueSet* qset,
    int32_t* heads,
    int32_t num_blocks,
    int32_t* next_block)
{
    while (true) {
        int32_t tile = MultiBlockTryDequeue(qset, heads, num_blocks, next_block);
        if (tile >= 0) {
            return tile;
        }
        // All queues empty, keep spinning
    }
}
#endif
