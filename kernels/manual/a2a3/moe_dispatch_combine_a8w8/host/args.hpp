/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_DISPATCH_COMBINE_A8W8_HOST_ARGS_HPP_
#define MOE_DISPATCH_COMBINE_A8W8_HOST_ARGS_HPP_

#include "moe_dispatch_combine_a8w8_types.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace moe_dispatch_combine_a8w8 {

struct RuntimeOptions {
    std::string runMode = "npu";
    std::string socVersion = "Ascend910B1";
    std::string caseName = "small";
    std::string dataDir = "./out";
    uint32_t seed = 1234;
    uint32_t dryRun = 1;
    uint32_t skipKernelLaunch = 1;
    uint32_t timeline = 0;
    uint32_t debug = 0;
    uint64_t hcclBuffSizeMb = 0;
    bool rankSet = false;
};

struct ProgramArgs {
    ShapeConfig shape;
    RankConfig rank;
    RuntimeOptions runtime;
};

inline uint32_t ParseU32(const std::string &value, const char *name)
{
    size_t parsed = 0;
    unsigned long out = std::stoul(value, &parsed, 10);
    if (parsed != value.size()) {
        throw std::invalid_argument(std::string("invalid integer for ") + name + ": " + value);
    }
    return static_cast<uint32_t>(out);
}

inline uint64_t ParseU64(const std::string &value, const char *name)
{
    size_t parsed = 0;
    unsigned long long out = std::stoull(value, &parsed, 10);
    if (parsed != value.size()) {
        throw std::invalid_argument(std::string("invalid integer for ") + name + ": " + value);
    }
    return static_cast<uint64_t>(out);
}

inline const char *RequireValue(int argc, char **argv, int *index, const char *name)
{
    if (*index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
    }
    ++(*index);
    return argv[*index];
}

inline ProgramArgs DefaultArgs()
{
    ProgramArgs args;
    args.rank.rankNum = args.shape.rankNum;
    args.rank.ndevices = args.shape.rankNum;
    return args;
}

inline ProgramArgs ParseArgs(int argc, char **argv)
{
    ProgramArgs args = DefaultArgs();
    bool ndevicesSet = false;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto value = [&](const char *name) { return std::string(RequireValue(argc, argv, &i, name)); };
        if (key == "-h" || key == "--help") {
            throw std::invalid_argument("help is handled by run_a3.sh");
        } else if (key == "-r" || key == "--run-mode") {
            args.runtime.runMode = value(key.c_str());
        } else if (key == "-v" || key == "--soc-version") {
            args.runtime.socVersion = value(key.c_str());
        } else if (key == "-pes" || key == "--pes" || key == "--nranks" || key == "--rank-num") {
            args.shape.rankNum = ParseU32(value(key.c_str()), key.c_str());
            args.rank.rankNum = args.shape.rankNum;
        } else if (key == "--rank") {
            args.rank.rankId = ParseU32(value(key.c_str()), key.c_str());
            args.runtime.rankSet = true;
        } else if (key == "--rank-from-mpi") {
            args.rank.rankFromMpi = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "-M" || key == "--tokens") {
            args.shape.m = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "-K" || key == "--hidden" || key == "--hidden-size") {
            args.shape.hiddenSize = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "-N" || key == "--intermediate" || key == "--intermediate-size") {
            args.shape.intermediateSize = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "-topK" || key == "--topk") {
            args.shape.topK = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "-expertPerPe" || key == "--experts-per-rank") {
            args.shape.expertPerRank = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--max-tokens-per-expert") {
            args.shape.maxTokensPerExpert = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "-tileCols" || key == "--payload-tile-cols") {
            args.shape.payloadTileCols = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--gmm-block-m") {
            args.shape.gmmBlockM = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--gmm-block-n") {
            args.shape.gmmBlockN = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--gmm-block-k") {
            args.shape.gmmBlockK = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--dtype-in") {
            args.shape.dtypeIn = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--dtype-out") {
            args.shape.dtypeOut = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "-device-base" || key == "--device-base" || key == "--first-device") {
            args.rank.deviceBase = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--ndevices") {
            args.rank.ndevices = ParseU32(value(key.c_str()), key.c_str());
            ndevicesSet = true;
        } else if (key == "--case-name") {
            args.runtime.caseName = value(key.c_str());
        } else if (key == "--seed") {
            args.runtime.seed = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--data-dir") {
            args.runtime.dataDir = value(key.c_str());
        } else if (key == "--dry-run") {
            args.runtime.dryRun = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--skip-kernel-launch") {
            args.runtime.skipKernelLaunch = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--timeline") {
            args.runtime.timeline = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--debug") {
            args.runtime.debug = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--hccl-buffsize-mb") {
            args.runtime.hcclBuffSizeMb = ParseU64(value(key.c_str()), key.c_str());
        } else if (key == "--warmup" || key == "--measure-iters" || key == "--iters") {
            throw std::invalid_argument("warmup_iters=3 and measure_iters=5 are fixed for this project");
        } else if (key == "--case" || key == "--case-all") {
            throw std::invalid_argument("hidden case presets are unsupported; pass explicit shape parameters");
        } else {
            throw std::invalid_argument("unknown option: " + key);
        }
    }
    if (!ndevicesSet) {
        args.rank.ndevices = args.shape.rankNum;
    }
    args.rank.rankNum = args.shape.rankNum;
    return args;
}

inline void ValidateArgs(const ProgramArgs &args)
{
    const ShapeConfig &shape = args.shape;
    if (shape.rankNum == 0 || shape.expertPerRank == 0 || shape.topK == 0 || shape.m == 0 ||
        shape.hiddenSize == 0 || shape.intermediateSize == 0 || shape.maxTokensPerExpert == 0 ||
        shape.payloadTileCols == 0 || shape.gmmBlockM == 0 || shape.gmmBlockN == 0 || shape.gmmBlockK == 0) {
        throw std::invalid_argument("shape and tiling fields must be nonzero");
    }
    if (shape.hiddenSize % shape.payloadTileCols != 0) {
        throw std::invalid_argument("hiddenSize must be divisible by payloadTileCols in M0");
    }
    if (static_cast<uint64_t>(args.rank.deviceBase) + shape.rankNum > args.rank.ndevices) {
        throw std::invalid_argument("deviceBase + rankNum > ndevices");
    }
    if (args.rank.rankFromMpi == 0 && !args.runtime.rankSet && shape.rankNum > 1) {
        throw std::invalid_argument("--rank must be explicit when --rank-from-mpi 0 and rankNum > 1");
    }
}

inline void PrintArgSummary(std::ostream &os, const ProgramArgs &args)
{
    os << "[RunConfig]\n";
    os << "  case_name=" << args.runtime.caseName << " run_mode=" << args.runtime.runMode
       << " soc_version=" << args.runtime.socVersion << "\n";
    os << "  rankNum=" << args.shape.rankNum << " rankId=" << args.rank.rankId
       << " rank_from_mpi=" << args.rank.rankFromMpi << " deviceBase=" << args.rank.deviceBase
       << " ndevices=" << args.rank.ndevices << "\n";
    os << "  warmup_iters=" << kWarmupIters << " measure_iters=" << kMeasureIters
       << " dry_run=" << args.runtime.dryRun << " timeline=" << args.runtime.timeline << "\n";
}

} // namespace moe_dispatch_combine_a8w8

#endif // MOE_DISPATCH_COMBINE_A8W8_HOST_ARGS_HPP_
