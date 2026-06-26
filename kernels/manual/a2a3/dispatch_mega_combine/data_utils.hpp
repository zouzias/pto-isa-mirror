#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "tiling_builder.hpp"

struct RankFileSet {
    std::string x;
    std::string weight1;
    std::string weight2;
    std::string expert_idx;
    std::string scale1;
    std::string scale2;
    std::string probs;
    std::string x_active_mask;
    std::string expected_out;
};

struct AccuracyReport {
    bool pass = false;
    size_t mismatch_count = 0;
    size_t err_threshold = 0;
    size_t nan_or_inf_count = 0;
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
};

CaseConfig LoadCaseConfig(const std::string &case_json_path);
RankFileSet BuildRankFileSet(const std::string &case_dir, int rank);
std::vector<uint8_t> ReadBinaryFile(const std::string &path);
void WriteBinaryFile(const std::string &path, const void *data, size_t bytes);
AccuracyReport CompareFp16File(const std::vector<uint16_t> &expected, const std::vector<uint16_t> &actual, double atol,
                               double rtol);
