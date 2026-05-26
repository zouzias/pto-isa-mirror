#pragma once

#include <cstdint>

// ============================================================================
// MoE Dispatch Operator Configuration
//
// Standalone Dispatch communication operator for MegaMoE PTO-ISA validation.
// Pulls quantized tokens from remote ranks' shmem into local workspace,
// separating token data and per-token scales.
// ============================================================================

// Default shape parameters (overridable via cmake -D)
#ifndef CONFIG_EP
#define CONFIG_EP 2
#endif

#ifndef CONFIG_EXPERT_PER_RANK
#define CONFIG_EXPERT_PER_RANK 1
#endif

#ifndef CONFIG_HIDDEN_SIZE
#define CONFIG_HIDDEN_SIZE 128
#endif

#ifndef CONFIG_MAX_TOKENS_PER_RANK
#define CONFIG_MAX_TOKENS_PER_RANK 64
#endif

#ifndef CONFIG_MAX_OUTPUT_SIZE
#define CONFIG_MAX_OUTPUT_SIZE 512
#endif

#ifndef CONFIG_FIRST_DEVICE_ID
#define CONFIG_FIRST_DEVICE_ID 0
#endif

// Hardware constants (matching MegaMoE reference implementation)
static constexpr int32_t UB_ALIGN = 32;
static constexpr int32_t UB_HALF_SIZE = 96 * 1024;
static constexpr int32_t UB_MOVE_NUM = 2;

// Per-row byte stride in remote shmem: hiddenSize bytes of int8 data + UB_ALIGN padding (containing float scale)
inline constexpr int32_t ShmemRowStride(int32_t hiddenSize)
{
    return hiddenSize + UB_ALIGN;
}

// Dispatch kernel launch parameters
struct MoeDispatchParams {
    int32_t EP;
    int32_t expertPerRank;
    int32_t hiddenSize;
    int32_t maxOutputSize;
    int32_t maxTokensPerRank;
};
