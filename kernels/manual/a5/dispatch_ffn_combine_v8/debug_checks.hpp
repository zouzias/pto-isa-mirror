#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "runtime_context.hpp"
#include "tiling_builder.hpp"

struct DebugDeviceBuffer {
    void *ptr = nullptr;
    size_t bytes = 0;
};

struct DebugHostBuffer {
    void *ptr = nullptr;
    size_t bytes = 0;
};

std::string BuildStageProfileReportText(int rank_id, const DebugHostBuffer &profile_host, uint32_t block_dim,
                                        double sys_cnt_multiple_ns, bool start_sync_debug = false,
                                        const CaseConfig *cfg = nullptr,
                                        const DispatchFFNCombineBuildResult *build = nullptr,
                                        const std::vector<int32_t> *expert_token_nums = nullptr);

bool RunStageDebugChecks(int rank_id, int world_size, const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                         const StandaloneRankRuntime &runtime, const DebugDeviceBuffer &workspace_dev,
                         const DebugDeviceBuffer &expert_token_nums_dev, const DebugDeviceBuffer &weight1_dev,
                         const DebugDeviceBuffer &scale1_dev, const DebugDeviceBuffer &weight2_dev,
                         const DebugDeviceBuffer &scale2_dev, const DebugDeviceBuffer &probs_dev,
                         const DebugDeviceBuffer &out_dev, const std::vector<uint8_t> &expert_idx,
                         const std::vector<uint8_t> &x, const std::vector<uint8_t> &weight1,
                         const std::vector<uint8_t> &scale1, const std::vector<uint8_t> &weight2,
                         const std::vector<uint8_t> &scale2, const std::vector<uint8_t> &probs,
                         const std::vector<uint16_t> &actual_out, const std::string &case_dir, bool skip_accuracy);
