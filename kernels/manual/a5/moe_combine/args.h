/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef MOE_COMBINE_ARGS_H_
#define MOE_COMBINE_ARGS_H_

#include "common.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace moe_combine {

#ifndef CONFIG_MOE_COMBINE_MIX_AIC_BLOCKS
#define CONFIG_MOE_COMBINE_MIX_AIC_BLOCKS 20U
#endif

#ifndef CONFIG_MOE_COMBINE_MIX_AIV_RATIO
#define CONFIG_MOE_COMBINE_MIX_AIV_RATIO 2U
#endif

#ifndef CONFIG_MOE_COMBINE_AIV_NUM
#define CONFIG_MOE_COMBINE_AIV_NUM (CONFIG_MOE_COMBINE_MIX_AIC_BLOCKS * CONFIG_MOE_COMBINE_MIX_AIV_RATIO)
#endif

#ifndef CONFIG_MOE_COMBINE_MAX_AIV_NUM
#define CONFIG_MOE_COMBINE_MAX_AIV_NUM 128U
#endif

struct MoeCombineArgs {
    MoeCombineShape shape;
    MoeCombineRuntimeConfig runtime;
    std::string runMode = "npu";
    std::string socVersion = "Ascend950PR_958b";
    std::string mpiBin;
    std::string dataDir = "./out";
    double rtol = 1e-2;
    double atol = 1e-2;
    bool rankSet = false;
    bool nranksSet = false;
};

struct MoeCombineResourceConfig {
    uint32_t defaultAicBlocks;
    uint32_t defaultAivRatio;
    uint32_t defaultAivBlocks;
    uint32_t maxAivBlocks;
};

inline bool StartsWith(const std::string &value, const char *prefix)
{
    return value.rfind(prefix, 0) == 0;
}

inline bool IsA5Soc(const std::string &socVersion)
{
    return socVersion.empty() || socVersion == "Ascend950" || StartsWith(socVersion, "Ascend950DT_") ||
           StartsWith(socVersion, "Ascend950PR_");
}

inline MoeCombineResourceConfig GetResourceConfig(const std::string &socVersion)
{
    if (!IsA5Soc(socVersion)) {
        throw std::runtime_error("unsupported moe_combine A5 resource config for soc: " + socVersion);
    }
    return MoeCombineResourceConfig{CONFIG_MOE_COMBINE_MIX_AIC_BLOCKS, CONFIG_MOE_COMBINE_MIX_AIV_RATIO,
                                    CONFIG_MOE_COMBINE_AIV_NUM, CONFIG_MOE_COMBINE_MAX_AIV_NUM};
}

inline uint32_t ChooseDefaultAivBlocks(const std::string &socVersion, const MoeCombineShape &)
{
    return GetResourceConfig(socVersion).defaultAivBlocks;
}

inline uint32_t ParseU32(const std::string &value, const char *name)
{
    size_t parsed = 0;
    unsigned long out = std::stoul(value, &parsed, 10);
    if (parsed != value.size()) {
        throw std::invalid_argument(std::string("invalid integer for ") + name + ": " + value);
    }
    return static_cast<uint32_t>(out);
}

inline double ParseDouble(const std::string &value, const char *name)
{
    size_t parsed = 0;
    double out = std::stod(value, &parsed);
    if (parsed != value.size()) {
        throw std::invalid_argument(std::string("invalid float for ") + name + ": " + value);
    }
    return out;
}

inline const char *RequireValue(int argc, char **argv, int *index, const char *name)
{
    if (*index + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
    }
    ++(*index);
    return argv[*index];
}

inline MoeCombineArgs DefaultArgs()
{
    MoeCombineArgs args;
    args.shape.ep = 2;
    args.shape.m = 64;
    args.shape.k = 7168;
    args.shape.topK = 8;
    args.shape.expertPerRank = 2;
    args.shape.expertNum = args.shape.ep * args.shape.expertPerRank;
    args.shape.maxOutputSize = 0;
    args.shape.aivBlocks = 0;
    args.runtime.deviceBase = 0;
    args.runtime.ndevices = args.shape.ep;
    args.runtime.rankFromMpi = 1;
    args.runtime.rank = 0;
    args.runtime.nranks = args.shape.ep;
    args.runtime.debug = 0;
    args.runtime.iters = 1;
    args.runtime.warmup = 1;
    args.runtime.seed = 1234;
    args.runtime.genData = 1;
    args.runtime.verify = 1;
    args.runtime.skipRun = 0;
    args.runtime.skipBuild = 0;
    args.runtime.cleanBuild = 1;
    args.runtime.skipKernels = 0;
    args.runtime.hostGoldenOnly = 0;
    args.runtime.combineReturnOnly = 0;
    args.runtime.keepHcclShm = 0;
    args.runtime.hcclBuffSizeMb = 0;
    return args;
}

inline MoeCombineArgs ParseArgs(int argc, char **argv)
{
    MoeCombineArgs args = DefaultArgs();
    bool ndevicesSet = false;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        auto value = [&](const char *name) { return std::string(RequireValue(argc, argv, &i, name)); };
        if (key == "--case" || key == "--case-all") {
            throw std::invalid_argument("case presets are unsupported; pass explicit shape parameters");
        } else if (key == "-r" || key == "--run-mode") {
            args.runMode = value(key.c_str());
        } else if (key == "-v" || key == "--soc-version") {
            args.socVersion = value(key.c_str());
        } else if (key == "-pes" || key == "--pes") {
            args.shape.ep = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--nranks") {
            args.shape.ep = ParseU32(value(key.c_str()), key.c_str());
            args.runtime.nranks = args.shape.ep;
            args.nranksSet = true;
        } else if (key == "-M" || key == "--tokens") {
            args.shape.m = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "-K" || key == "--hidden") {
            args.shape.k = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "-topK" || key == "--topk") {
            args.shape.topK = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "-expertPerPe" || key == "--experts-per-rank") {
            args.shape.expertPerRank = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--max-output-size") {
            args.shape.maxOutputSize = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "-aivBlocks" || key == "--aiv-blocks") {
            args.shape.aivBlocks = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "-device-base" || key == "--device-base" || key == "--first-device") {
            args.runtime.deviceBase = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--ndevices") {
            args.runtime.ndevices = ParseU32(value(key.c_str()), key.c_str());
            ndevicesSet = true;
        } else if (key == "--mpi-bin") {
            args.mpiBin = value(key.c_str());
        } else if (key == "--hccl-buffsize-mb") {
            args.runtime.hcclBuffSizeMb = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--keep-hccl-shm") {
            args.runtime.keepHcclShm = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--rank-from-mpi") {
            args.runtime.rankFromMpi = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--rank") {
            args.runtime.rank = ParseU32(value(key.c_str()), key.c_str());
            args.rankSet = true;
        } else if (key == "-debug" || key == "--debug") {
            args.runtime.debug = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "-iters" || key == "--iters") {
            args.runtime.iters = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "-warmup" || key == "--warmup") {
            args.runtime.warmup = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--seed") {
            args.runtime.seed = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--data-dir") {
            args.dataDir = value(key.c_str());
        } else if (key == "--gen-data") {
            args.runtime.genData = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--verify") {
            args.runtime.verify = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--rtol") {
            args.rtol = ParseDouble(value(key.c_str()), key.c_str());
        } else if (key == "--atol") {
            args.atol = ParseDouble(value(key.c_str()), key.c_str());
        } else if (key == "--skip-run") {
            args.runtime.skipRun = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--skip-build") {
            args.runtime.skipBuild = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--clean-build") {
            args.runtime.cleanBuild = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--skip-kernels") {
            args.runtime.skipKernels = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--host-golden-only") {
            args.runtime.hostGoldenOnly = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--combine-return-only") {
            args.runtime.combineReturnOnly = ParseU32(value(key.c_str()), key.c_str());
        } else {
            throw std::invalid_argument("unknown option: " + key);
        }
    }
    args.shape.expertNum = args.shape.ep * args.shape.expertPerRank;
    if (args.shape.maxOutputSize == 0) {
        args.shape.maxOutputSize = args.shape.ep * args.shape.m * args.shape.topK;
    }
    if (!ndevicesSet) {
        args.runtime.ndevices = args.shape.ep;
    }
    if (!args.nranksSet) {
        args.runtime.nranks = args.shape.ep;
    }
    if (args.shape.aivBlocks == 0) {
        args.shape.aivBlocks = ChooseDefaultAivBlocks(args.socVersion, args.shape);
    }
    return args;
}

inline void ValidateArgs(const MoeCombineArgs &args)
{
    const MoeCombineShape &shape = args.shape;
    const MoeCombineResourceConfig resource = GetResourceConfig(args.socVersion);
    if (args.runMode != "npu") {
        throw std::invalid_argument("run-mode must be npu for the first version");
    }
    if (shape.ep == 0 || shape.m == 0 || shape.k == 0 || shape.topK == 0 || shape.expertPerRank == 0) {
        throw std::invalid_argument("shape fields must be nonzero");
    }
    if (shape.aivBlocks == 0 || shape.aivBlocks > resource.maxAivBlocks) {
        throw std::invalid_argument("aivBlocks must be in [1, maxAivBlocks]");
    }
    uint64_t requiredRows = static_cast<uint64_t>(shape.ep) * shape.m * shape.topK;
    if (shape.maxOutputSize < requiredRows) {
        throw std::invalid_argument("maxOutputSize is smaller than EP * M * topK; capacity/drop is unsupported");
    }
    if (static_cast<uint64_t>(args.runtime.deviceBase) + shape.ep > args.runtime.ndevices) {
        throw std::invalid_argument("deviceBase + pes > ndevices");
    }
}

inline void PrintRunSummary(const MoeCombineArgs &args)
{
    const MoeCombineShape &shape = args.shape;
    std::cout << "RUN_MODE=" << args.runMode << "\n";
    std::cout << "SOC_VERSION=" << args.socVersion << "\n";
    std::cout << "PES=" << shape.ep << " DEVICE_BASE=" << args.runtime.deviceBase
              << " NDEVICES=" << args.runtime.ndevices << "\n";
    std::cout << "M=" << shape.m << " K=" << shape.k << " TOPK=" << shape.topK
              << " EXPERT_PER_PE=" << shape.expertPerRank << " MAX_OUTPUT_SIZE=" << shape.maxOutputSize << "\n";
    std::cout << "AIV_BLOCKS=" << shape.aivBlocks << " TILE_COLS=" << kMoeCombineTileCols
              << " ROW_CHUNK=" << kMoeCombineRowChunk << " METADATA_PAD=" << kMoeCombineMetadataPad << "\n";
    const MoeCombineResourceConfig resource = GetResourceConfig(args.socVersion);
    std::cout << "RESOURCE defaultAicBlocks=" << resource.defaultAicBlocks
              << " defaultAivRatio=" << resource.defaultAivRatio << " defaultAivBlocks=" << resource.defaultAivBlocks
              << " maxAivBlocks=" << resource.maxAivBlocks << "\n";
    std::cout << "DATA_DIR=" << args.dataDir << " SEED=" << args.runtime.seed << "\n";
    std::cout << "WARMUP=" << args.runtime.warmup << " ITERS=" << args.runtime.iters
              << " VERIFY=" << args.runtime.verify << " DEBUG=" << args.runtime.debug << "\n";
}

} // namespace moe_combine

#endif // MOE_COMBINE_ARGS_H_
