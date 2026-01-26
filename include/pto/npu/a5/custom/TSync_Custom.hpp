/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TSYNC_CUSTOM_HPP
#define TSYNC_CUSTOM_HPP

#include <pto/common/type.hpp>
#include <pto/common/utils.hpp>

namespace pto {

// Operation types for TSync - identifies the producer/consumer operation
enum class SyncOpType : uint8_t {
    TSTORE_C2GM,  // Store (Cube core operation via PIPE_FIX)
    TSTORE_V2GM,  // Store (Vector core operation via PIPE_MTE3)
    TMOV_C2UB,    // TMOV from L0C to UB (Cube core operation via PIPE_FIX)
    TINSERT_V2L1, // TINSERT from UB to L1 (Vector core operation via PIPE_MTE2)
    TLOAD         // Load operation (consumer operation)
};

// Compile-time direction inference based on producer/consumer ops
// TSTORE_C2GM (producer) + TLOAD (consumer) = Cube to Vector via PIPE_FIX
// TSTORE_V2GM (producer) + TLOAD (consumer) = Vector to Cube via PIPE_MTE3
// TMOV_C2UB (producer) + TLOAD (consumer) = Cube to Vector via PIPE_FIX
// TINSERT_V2L1 (producer) + TLOAD (consumer) = Vector to Cube via PIPE_MTE2
template <SyncOpType ProducerOp, SyncOpType ConsumerOp>
struct SyncTraits {
    // Direction is inferred from producer operation:
    static constexpr bool is_cube_to_vec_fix = (ProducerOp == SyncOpType::TSTORE_C2GM);
    static constexpr bool is_cube_to_vec_mte3 = (ProducerOp == SyncOpType::TMOV_C2UB);
    static constexpr bool is_cube_to_vec = is_cube_to_vec_fix || is_cube_to_vec_mte3;
    static constexpr bool is_vec_to_cube_mte3 = (ProducerOp == SyncOpType::TSTORE_V2GM);
    static constexpr bool is_vec_to_cube_mte2 = (ProducerOp == SyncOpType::TINSERT_V2L1);
    static constexpr bool is_vec_to_cube = is_vec_to_cube_mte3 || is_vec_to_cube_mte2;
    
    static_assert(ConsumerOp == SyncOpType::TLOAD, "Consumer operation must be TLOAD");
    static_assert(is_cube_to_vec || is_vec_to_cube, 
                  "Producer must be TSTORE_C2GM, TMOV_C2UB (Cube) or TSTORE_V2GM, TINSERT_V2L1 (Vector)");
};

namespace detail {
    template <int N>
    struct FlagIDTag {
        static constexpr int value = N;
    };
    
    // Base counter starts at 0 (user IDs start from 0 to 12)
    constexpr int kUserFlagIDStart = 0;
    constexpr int kMaxFlagID = 12;
    constexpr int kNumUserFlags = kMaxFlagID - kUserFlagIDStart + 1;  // 12 flags
}

/**
 * TSync - Lightweight synchronization primitive for intra-core dependencies
 * 
 * 
 * Usage with manual flag ID:
 *   constexpr TSync<TSTORE_C2GM, TLOAD> sync = {BUF0_QK_READY};
 *   constexpr TSync<TMOV_C2UB, TLOAD> sync_tmov = {BUF0_QK_READY};  // For TMOV path
 * 
 * Forward dependency (producer -> consumer):
 *   Producer: sync.record()  // Signal data ready
 *   Consumer: sync.wait()    // Wait for data
 * 
 * Backward dependency (consumer -> producer):
 *   Producer: sync.allocate() // Wait for buffer space
 *   Consumer: sync.free()     // Signal buffer available
 * 
 * Template Parameters:
 *   ProducerOp: Producer operation (TSTORE_C2GM, TMOV_C2UB, or TSTORE_V2GM)
 *   ConsumerOp: Consumer operation (TLOAD)
 */
template <SyncOpType ProducerOp, SyncOpType ConsumerOp>
struct TSync_Custom {
    using Traits = SyncTraits<ProducerOp, ConsumerOp>;
    static constexpr bool is_c2v = Traits::is_cube_to_vec;
    static constexpr bool is_c2v_fix = Traits::is_cube_to_vec_fix;
    static constexpr bool is_c2v_mte3 = Traits::is_cube_to_vec_mte3;
    static constexpr bool is_v2c = Traits::is_vec_to_cube;
    static constexpr bool is_v2c_mte3 = Traits::is_vec_to_cube_mte3;
    static constexpr bool is_v2c_mte2 = Traits::is_vec_to_cube_mte2;
    
    uint16_t flag_id;  // FFTS flag ID for cross-core synchronization
    
    // Forward dependency: record (producer) and wait (consumer)

    /**
     * record - Producer signals that data is ready
     * Called by the producer after completing the operation (TSTORE_C2GM, TMOV_C2UB, TSTORE_V2GM, or TINSERT_V2L1)
     */
    AICORE inline void record() const {
        if constexpr (is_c2v_fix) {
            // Cube produces via TSTORE (PIPE_FIX), Vector consumes
            set_intra_block(PIPE_FIX, flag_id);
            set_intra_block(PIPE_FIX, flag_id + 16);
        } else if constexpr (is_c2v_mte3) {
            // Cube produces via TMOV (PIPE_FIX for instruction completion), Vector consumes
            // TMOV is issued from Cube and uses PIPE_FIX for completion signaling
            // Note: The actual data transfer may use MTE3 hardware, but the instruction
            // completion is signaled via PIPE_FIX from the Cube core's perspective
            set_intra_block(PIPE_FIX, flag_id);
            set_intra_block(PIPE_FIX, flag_id + 16);
        } else if constexpr (is_v2c_mte3) {
            // Vector produces via TSTORE to GM (PIPE_MTE3), Cube consumes
            set_intra_block(PIPE_MTE3, flag_id);
        } else { // is_v2c_mte2
            // Vector produces via TINSERT to L1 (PIPE_MTE2), Cube consumes
            set_intra_block(PIPE_MTE2, flag_id);
            set_intra_block(PIPE_MTE2, flag_id + 16);
        }
    }
    
    /**
     * wait - Consumer waits for data to be ready
     * Called by the consumer before accessing the data (TLOAD)
     */
    AICORE inline void wait() const {
        if constexpr (is_c2v) {
            // Vector waits for Cube (both PIPE_FIX and PIPE_MTE3 producers signal to PIPE_V consumer)
            wait_intra_block(PIPE_V, flag_id);
        } else { // is_v2c (both mte3 and mte2)
            // Cube waits for Vector
            wait_intra_block(PIPE_MTE2, flag_id);
            wait_intra_block(PIPE_MTE2, flag_id + 16);
        }
    }
    
    // Backward dependency: allocate (producer) and free (consumer)
    
    /**
     * allocate - Producer waits for buffer space to be available
     * Called by the producer before writing new data
     */
    AICORE inline void allocate() const {
        if constexpr (is_c2v_fix) {
            // Cube (TSTORE) waits for Vector to free buffer
            wait_intra_block(PIPE_FIX, flag_id + 1);
            wait_intra_block(PIPE_FIX, flag_id + 1 + 16);
        } else if constexpr (is_c2v_mte3) {
            // Cube (TMOV) waits for Vector to free buffer via PIPE_FIX
            wait_intra_block(PIPE_FIX, flag_id + 1);
            wait_intra_block(PIPE_FIX, flag_id + 1 + 16);
        } else if constexpr (is_v2c_mte3) {
            // Vector (TSTORE to GM) waits for Cube to free buffer via PIPE_MTE3
            wait_intra_block(PIPE_MTE3, flag_id + 1);
        } else { // is_v2c_mte2
            // Vector (TINSERT to L1) waits for Cube to free buffer via PIPE_MTE2
            wait_intra_block(PIPE_MTE2, flag_id + 1);
        }
    }
    
    /**
     * free - Consumer signals that buffer space is available
     * Called by the consumer after consuming data
     */
    AICORE inline void free() const {
        if constexpr (is_c2v) {
            // Vector frees buffer for Cube (signals to both PIPE_FIX and PIPE_MTE3)
            set_intra_block(PIPE_V, flag_id + 1);
        } else { // is_v2c (both mte3 and mte2)
            // Cube frees buffer for Vector
            set_intra_block(PIPE_MTE2, flag_id + 1);
            set_intra_block(PIPE_MTE2, flag_id + 1 + 16);
        }
    }
};


} // namespace pto

#endif // TSYNC_HPP
