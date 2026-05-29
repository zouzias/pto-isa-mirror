/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_COMBINE_TILE_ARGS_H_
#define DISPATCH_COMBINE_TILE_ARGS_H_

#include "moe_dispatch_combine_a8w8_runtime_types.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace dispatch_combine_tile {

struct DispatchCombineTileArgs {
    DispatchCombineTileShape shape;
    DispatchCombineTileRuntimeConfig runtime;
    std::string runMode = "npu";
    std::string socVersion = "Ascend910B1";
    std::string mpiBin;
    std::string dataDir = "./out";
    std::string caseName = "small";
    std::string backend = "m1-mock";
    uint32_t intermediateSize = 32;
    double rtol = 1e-2;
    double atol = 1e-2;
    bool rankSet = false;
    bool nranksSet = false;
};

inline uint32_t ChooseDefaultAivBlocks(const DispatchCombineTileShape &)
{
    return 8;
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

inline DispatchCombineTileArgs DefaultArgs()
{
    DispatchCombineTileArgs args;
    args.shape.ep = 2;
    args.shape.m = 64;
    args.shape.k = 7168;
    args.shape.topK = 8;
    args.shape.expertPerRank = 2;
    args.shape.expertNum = args.shape.ep * args.shape.expertPerRank;
    args.shape.maxOutputSize = 0;
    args.shape.aivBlocks = 0;
    args.shape.tileCols = 1024;
    args.shape.gmmBlockM = 16;
    args.shape.gmmBlockN = 16;
    args.shape.gmmBlockK = 32;
    args.shape.rowChunk = 0;
    args.shape.metadataPad = 16;
    args.shape.signalValue = 1;
    args.runtime.deviceBase = 4;
    args.runtime.ndevices = 8;
    args.runtime.rankFromMpi = 1;
    args.runtime.rank = 0;
    args.runtime.nranks = args.shape.ep;
    args.runtime.debug = 0;
    args.runtime.iters = 5;
    args.runtime.warmup = 3;
    args.runtime.seed = 1234;
    args.runtime.genData = 1;
    args.runtime.verify = 1;
    args.runtime.skipRun = 0;
    args.runtime.skipBuild = 0;
    args.runtime.cleanBuild = 1;
    args.runtime.skipKernels = 0;
    args.runtime.hostGoldenOnly = 0;
    args.runtime.dispatchMetadataOnly = 0;
    args.runtime.dispatchOnly = 0;
    args.runtime.gmm1Only = 0;
    args.runtime.gmm1EpilogueOnly = 0;
    args.runtime.activationOnly = 0;
    args.runtime.gmm2Only = 0;
    args.runtime.combineReturnOnly = 0;
    args.runtime.keepHcclShm = 0;
    args.runtime.hcclBuffSizeMb = 0;
    return args;
}

inline DispatchCombineTileArgs ParseArgs(int argc, char **argv)
{
    DispatchCombineTileArgs args = DefaultArgs();
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
        } else if (key == "--nranks" || key == "--rank-num") {
            args.shape.ep = ParseU32(value(key.c_str()), key.c_str());
            args.runtime.nranks = args.shape.ep;
            args.nranksSet = true;
        } else if (key == "-M" || key == "--tokens") {
            args.shape.m = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "-K" || key == "--hidden" || key == "--hidden-size") {
            args.shape.k = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "-N" || key == "--intermediate" || key == "--intermediate-size") {
            args.intermediateSize = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "-topK" || key == "--topk") {
            args.shape.topK = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "-expertPerPe" || key == "--experts-per-rank") {
            args.shape.expertPerRank = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--max-output-size") {
            args.shape.maxOutputSize = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--max-tokens-per-expert") {
            args.shape.maxOutputSize = args.shape.expertPerRank * ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "-aivBlocks" || key == "--aiv-blocks") {
            args.shape.aivBlocks = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "-tileCols" || key == "--tile-cols" || key == "--payload-tile-cols") {
            args.shape.tileCols = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--gmm-block-m") {
            args.shape.gmmBlockM = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--gmm-block-n") {
            args.shape.gmmBlockN = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--gmm-block-k") {
            args.shape.gmmBlockK = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--row-chunk") {
            args.shape.rowChunk = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--metadata-pad") {
            args.shape.metadataPad = ParseU32(value(key.c_str()), key.c_str());
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
        } else if (key == "--case-name") {
            args.caseName = value(key.c_str());
        } else if (key == "--backend") {
            args.backend = value(key.c_str());
        } else if (key == "--data-dir") {
            args.dataDir = value(key.c_str());
        } else if (key == "--dry-run") {
            args.runtime.skipKernels = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--skip-kernel-launch") {
            args.runtime.skipKernels = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--timeline") {
            (void)ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--dtype-in" || key == "--dtype-out") {
            (void)ParseU32(value(key.c_str()), key.c_str());
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
        } else if (key == "--dispatch-metadata-only") {
            args.runtime.dispatchMetadataOnly = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--dispatch-only") {
            args.runtime.dispatchOnly = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--gmm1-only") {
            args.runtime.gmm1Only = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--gmm1-epilogue-only") {
            args.runtime.gmm1EpilogueOnly = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--activation-only") {
            args.runtime.activationOnly = ParseU32(value(key.c_str()), key.c_str());
        } else if (key == "--gmm2-only") {
            args.runtime.gmm2Only = ParseU32(value(key.c_str()), key.c_str());
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
    if (!ndevicesSet && args.runtime.ndevices < args.runtime.deviceBase + args.shape.ep) {
        args.runtime.ndevices = args.runtime.deviceBase + args.shape.ep;
    }
    if (!args.nranksSet) {
        args.runtime.nranks = args.shape.ep;
    }
    if (args.shape.aivBlocks == 0) {
        args.shape.aivBlocks = ChooseDefaultAivBlocks(args.shape);
    }
    return args;
}

inline void ValidateArgs(const DispatchCombineTileArgs &args)
{
    const DispatchCombineTileShape &shape = args.shape;
    if (args.runMode != "npu") {
        throw std::invalid_argument("run-mode must be npu for the first version");
    }
    if (shape.ep == 0 || shape.m == 0 || shape.k == 0 || shape.topK == 0 || shape.expertPerRank == 0 ||
        shape.tileCols == 0 || shape.gmmBlockM == 0 || shape.gmmBlockN == 0 || shape.gmmBlockK == 0 ||
        shape.metadataPad == 0) {
        throw std::invalid_argument("shape fields, tileCols, and metadataPad must be nonzero");
    }
    if (args.backend != "m1-mock" && args.backend != "int8") {
        throw std::invalid_argument("backend must be m1-mock or int8");
    }
    if (shape.k % shape.tileCols != 0) {
        throw std::invalid_argument("K % tileCols must be 0");
    }
    uint64_t requiredRows = static_cast<uint64_t>(shape.ep) * shape.m * shape.topK;
    if (shape.maxOutputSize < requiredRows) {
        throw std::invalid_argument("maxOutputSize is smaller than EP * M * topK; capacity/drop is unsupported");
    }
    if (static_cast<uint64_t>(args.runtime.deviceBase) + shape.ep > args.runtime.ndevices) {
        throw std::invalid_argument("deviceBase + pes > ndevices");
    }
}

inline void PrintRunSummary(const DispatchCombineTileArgs &args)
{
    const DispatchCombineTileShape &shape = args.shape;
    std::cout << "RUN_MODE=" << args.runMode << "\n";
    std::cout << "SOC_VERSION=" << args.socVersion << "\n";
    std::cout << "CASE_NAME=" << args.caseName << "\n";
    std::cout << "BACKEND=" << args.backend << "\n";
    std::cout << "PES=" << shape.ep << " DEVICE_BASE=" << args.runtime.deviceBase
              << " NDEVICES=" << args.runtime.ndevices << "\n";
    std::cout << "M=" << shape.m << " K=" << shape.k << " INTERMEDIATE_SIZE=" << args.intermediateSize
              << " TOPK=" << shape.topK << " EXPERT_PER_PE=" << shape.expertPerRank
              << " MAX_OUTPUT_SIZE=" << shape.maxOutputSize << "\n";
    std::cout << "GMM_BLOCK_M=" << shape.gmmBlockM << " GMM_BLOCK_N=" << shape.gmmBlockN
              << " GMM_BLOCK_K=" << shape.gmmBlockK << "\n";
    std::cout << "AIV_BLOCKS=" << shape.aivBlocks << " TILE_COLS=" << shape.tileCols << " ROW_CHUNK=" << shape.rowChunk
              << " METADATA_PAD=" << shape.metadataPad << "\n";
    std::cout << "DATA_DIR=" << args.dataDir << " SEED=" << args.runtime.seed << "\n";
    std::cout << "WARMUP=" << args.runtime.warmup << " ITERS=" << args.runtime.iters
              << " VERIFY=" << args.runtime.verify << " DEBUG=" << args.runtime.debug << "\n";
}

} // namespace dispatch_combine_tile

#endif // DISPATCH_COMBINE_TILE_ARGS_H_
