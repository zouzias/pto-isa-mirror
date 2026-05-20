#pragma once

#include <stdint.h>

struct DispatchFFNCombineLaunchArgs {
    void *x = nullptr;
    void *weight1 = nullptr;
    void *weight2 = nullptr;
    void *expert_idx = nullptr;
    void *scale1 = nullptr;
    void *scale2 = nullptr;
    void *probs = nullptr;
    void *x_active_mask = nullptr;
    void *out = nullptr;
    void *expert_token_nums = nullptr;
    void *workspace = nullptr;
#if defined(__CCE_AICORE__)
    uint8_t *tiling = nullptr;
#else
    void *tiling = nullptr;
#endif
    uint32_t block_dim = 1;
};

void launchDispatchFFNCombine(const DispatchFFNCombineLaunchArgs &args, void *stream);
