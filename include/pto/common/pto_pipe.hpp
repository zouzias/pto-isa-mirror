/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License"). Please refer to the License for details. You may not use this
file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN "AS
IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A
PARTICULAR PURPOSE. See LICENSE in the root of the software repository for the
full text of the License.
*/

#ifndef PTO_PIPE_HPP
#define PTO_PIPE_HPP

#include <cstdint>
#include <pto/common/type.hpp>

namespace pto {
PTO_INTERNAL uint16_t _getCVCoreSyncMsg(uint16_t flag_id,
                                        uint16_t base_const = 0x1) {
  return ((base_const & 0xf) + ((0x2) << 4) + ((flag_id & 0xf) << 8));
}

enum class CoreType {
  CUBE,  // The Matrix Multiplication Unit
  VECTOR // The Vector Processing Unit
};

// -----------------------------------------------------------------------------
// Traits: Map logical CoreType to physical PipeStage
// -----------------------------------------------------------------------------
template <CoreType type> struct PipeStageTraits;

#ifdef MEMORY_BASE
// Mapping for CUBE Role
template <> struct PipeStageTraits<CoreType::CUBE> {
  // Cube produces data (Matmul) via FIX pipeline (usually)
  static constexpr pipe_t ProdSignalPipe = PIPE_FIX;
  // Cube consumes data (Load) via MTE2 pipeline
  static constexpr pipe_t ConsSignalPipe = PIPE_MTE2;
};

// Mapping for VECTOR Role
template <> struct PipeStageTraits<CoreType::VECTOR> {
  // Vector produces data (Store) via MTE3 pipeline
  static constexpr pipe_t ProdSignalPipe = PIPE_MTE3;
  // Vector consumes data (Load) via MTE2 pipeline
  static constexpr pipe_t ConsSignalPipe = PIPE_MTE2;
};
#else

#endif

/**
 * Pipe: Manages Cross-Core FIFO Synchronization
 * @tparam ReadyFlag    Signal from Producer to Consumer (Data Ready)
 * @tparam ConsumedFlag Signal from Consumer to Producer (Space Released)
 * @tparam Depth        FIFO Depth (e.g., 2 for Double Buffering)
 * @tparam Period       Sync Period (Sync once every N tiles)
 * @tparam ProdRole     Logic role of Producer (CUBE/VECTOR) -> Deduce signal pipe
 * @tparam ConsRole     Logic role of Consumer (CUBE/VECTOR) -> Deduce signal pipe
 */
template <unsigned ReadyFlag, unsigned ConsumedFlag, int Depth, int Period,
          CoreType ProdRole, CoreType ConsRole>
struct Pipe {
  // Expose constants for Instruction layer
  static constexpr int PipeDepth = Depth;

  // Auto-deduced hardware pipelines based on CoreType
  static constexpr pipe_t ProdPipe = PipeStageTraits<ProdRole>::ProdSignalPipe;
  static constexpr pipe_t ConsPipe = PipeStageTraits<ConsRole>::ConsSignalPipe;

  // -------------------------------------------------------------------------
  // Producer Interface
  // -------------------------------------------------------------------------
  struct Producer {
    int iter = 0; // Private iterator for Producer

    /**
     * alloc: Request space in FIFO
     * Logic:
     * 1. (iter >= Depth): Startup protection. Don't check flags when buffer is
     * empty.
     * 2. (iter % Period == 0): Sparse sync. Only check flag periodically.
     */
    PTO_INTERNAL void alloc() {
      if (iter >= Depth && (iter % Period) == 0) {
        wait_flag_dev(ConsumedFlag);
      }
    }

    /**
     * record: Commit data and signal Consumer
     * Uses auto-deduced 'ProdPipe' to ensure data is visible before signaling.
     */
    PTO_INTERNAL void record() {
      ffts_cross_core_sync(ProdPipe, _getCVCoreSyncMsg(ReadyFlag));
      iter++;
    }
  };

  // -------------------------------------------------------------------------
  // Consumer Interface
  // -------------------------------------------------------------------------
  struct Consumer {
    int iter = 0; // Private iterator for Consumer

    /**
     * wait: Block until data is ready
     * Consumers strictly wait for data (no sparse optimization for safety).
     */
    PTO_INTERNAL void wait() { wait_flag_dev(ReadyFlag); }

    /**
     * free: Release space in FIFO
     * Logic:
     * 1. (iter >= Depth - Period): Silence at start. Don't signal if Producer
     * is still enjoying the initial free buffer space.
     * 2. (is_sync_step): Accumulate free slots and signal in batches.
     */
    PTO_INTERNAL void free() {
      bool is_sync_step = ((iter + 1) % Period) == 0;
      if (iter >= Depth - Period && is_sync_step) {
        ffts_cross_core_sync(ConsPipe, _getCVCoreSyncMsg(ConsumedFlag));
      }
      iter++;
    }
  };
};
} // namespace pto

#endif // PTO_PIPE_HPP