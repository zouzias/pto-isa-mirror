#pragma once

#include <stddef.h>
#include <stdint.h>

constexpr size_t kMegaMoeProfileEntryBytes = 4096U;
constexpr size_t kMegaMoeProfileEntriesPerBlock = 3U; // cube + two vec subblocks
constexpr size_t kMegaMoeProfileBytesPerBlock = kMegaMoeProfileEntryBytes * kMegaMoeProfileEntriesPerBlock;

constexpr size_t kMegaMoeProfileKernelStart = 0U;
constexpr size_t kMegaMoeProfileKernelEnd = 1U;

struct MegaMoeLaunchArgs {
    void *ffts = nullptr;
    void *x = nullptr;
    void *weight1 = nullptr;
    void *weight2 = nullptr;
    void *expert_idx = nullptr;
    void *scale1 = nullptr;
    void *scale2 = nullptr;
    void *probs = nullptr;
    void *out = nullptr;
    void *expert_token_nums = nullptr;
    void *workspace = nullptr;
    void *tiling = nullptr;
    void *profile_data = nullptr;
    uint32_t block_dim = 1;
    uint32_t start_sync_debug = 0;
};

void launchMegaMoe(const MegaMoeLaunchArgs &args, void *stream);
