/**
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DISPATCH_COMBINE_TILE_GOLDEN_H_
#define DISPATCH_COMBINE_TILE_GOLDEN_H_

#include "args.h"
#include "layout.h"

#include <cerrno>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace dispatch_combine_tile {

struct HostInputData {
    std::vector<float> inputA;
    std::vector<int32_t> expertIdx;
    std::vector<float> probs;
};

struct CpuGoldenData {
    std::vector<float> dispatchedA;
    std::vector<float> expertOutput;
    std::vector<float> outputC;
    std::vector<float> ptrD;
    std::vector<int32_t> localTokenPerExpert;
    std::vector<int32_t> peerTokenPerExpert;
    std::vector<int32_t> expandedRowIdx;
    std::vector<int32_t> cumsumPerExpert;
    std::vector<int32_t> dispatchOffset;
    std::vector<int32_t> prevSumBeforeRank;
    std::vector<uint32_t> ownerRows;
    uint64_t totalRoutes = 0;
    uint64_t invalidRoutes = 0;
};

struct CompareResult {
    uint64_t elementCount;
    uint64_t mismatchCount;
    uint64_t firstMismatchIndex;
    float actual;
    float expected;
};

namespace golden_detail {

struct RouteRef {
    uint32_t src = 0;
    uint32_t token = 0;
    uint32_t slot = 0;
    uint32_t expert = 0;
    uint32_t packedRow = 0;
};

inline std::string RankFile(const DispatchCombineTileArgs &args, uint32_t rank, const char *name)
{
    return args.dataDir + "/rank_" + std::to_string(rank) + "_" + name + ".bin";
}

inline void EnsureDataDir(const std::string &path)
{
    if (path.empty()) {
        return;
    }
    if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        throw std::runtime_error("mkdir failed for " + path + ": " + std::strerror(errno));
    }
}

template <typename T>
inline void WriteBinary(const std::string &path, const std::vector<T> &data)
{
    std::ofstream os(path, std::ios::binary);
    if (!os.is_open()) {
        throw std::runtime_error("failed to open output file: " + path);
    }
    os.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
    if (!os.good()) {
        throw std::runtime_error("failed to write output file: " + path);
    }
}

template <typename T>
inline std::vector<T> ReadBinary(const std::string &path, size_t elementCount)
{
    std::vector<T> data(elementCount);
    std::ifstream is(path, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("failed to open input file: " + path);
    }
    is.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(T)));
    if (is.gcount() != static_cast<std::streamsize>(data.size() * sizeof(T))) {
        throw std::runtime_error("input file size mismatch: " + path);
    }
    return data;
}

inline uint16_t FloatToHalf(float value)
{
    union {
        float f;
        uint32_t u;
    } bits{};
    bits.f = value;
    uint32_t sign = (bits.u >> 16) & 0x8000U;
    int32_t exp = static_cast<int32_t>((bits.u >> 23) & 0xFFU) - 127 + 15;
    uint32_t mant = bits.u & 0x007FFFFFU;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

inline float HalfToFloat(uint16_t value)
{
    uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16;
    uint32_t exp = (value >> 10) & 0x1FU;
    uint32_t mant = value & 0x03FFU;
    uint32_t out = 0;
    if (exp == 0) {
        out = sign;
    } else if (exp == 31) {
        out = sign | 0x7F800000U | (mant << 13);
    } else {
        out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    union {
        uint32_t u;
        float f;
    } bits{out};
    return bits.f;
}

inline std::vector<uint16_t> FloatVectorToHalf(const std::vector<float> &src)
{
    std::vector<uint16_t> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = FloatToHalf(src[i]);
    }
    return dst;
}

inline std::vector<float> HalfVectorToFloat(const std::vector<uint16_t> &src)
{
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = HalfToFloat(src[i]);
    }
    return dst;
}

inline HostInputData GenerateDeterministicInputs(const DispatchCombineTileArgs &args, uint32_t rank)
{
    const DispatchCombineTileShape &shape = args.shape;
    HostInputData data;
    data.inputA.resize(static_cast<size_t>(shape.m) * shape.k);
    data.expertIdx.resize(static_cast<size_t>(shape.m) * shape.topK);
    data.probs.resize(static_cast<size_t>(shape.m) * shape.topK);

    for (uint32_t token = 0; token < shape.m; ++token) {
        for (uint32_t col = 0; col < shape.k; ++col) {
            uint32_t mixed = args.runtime.seed + rank * 131U + token * 17U + col * 3U;
            data.inputA[static_cast<size_t>(token) * shape.k + col] = static_cast<float>(mixed % 251U) / 32.0f;
        }
        float probSum = 0.0f;
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t flat = token * shape.topK + slot;
            data.expertIdx[flat] = static_cast<int32_t>((flat + rank) % shape.expertNum);
            float prob = static_cast<float>((args.runtime.seed + rank + flat) % 7U + 1U);
            data.probs[flat] = prob;
            probSum += prob;
        }
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t flat = token * shape.topK + slot;
            data.probs[flat] /= probSum;
        }
    }
    return data;
}

inline HostInputData LoadInputs(const DispatchCombineTileArgs &args, uint32_t rank)
{
    const DispatchCombineTileShape &shape = args.shape;
    size_t inputElems = static_cast<size_t>(shape.m) * shape.k;
    size_t routeElems = static_cast<size_t>(shape.m) * shape.topK;
    HostInputData data;
    data.inputA = HalfVectorToFloat(ReadBinary<uint16_t>(RankFile(args, rank, "inputA"), inputElems));
    data.expertIdx = ReadBinary<int32_t>(RankFile(args, rank, "expertIdx"), routeElems);
    data.probs = ReadBinary<float>(RankFile(args, rank, "probs"), routeElems);
    return data;
}

inline void WriteInputs(const DispatchCombineTileArgs &args, uint32_t rank, const HostInputData &data)
{
    EnsureDataDir(args.dataDir);
    WriteBinary(RankFile(args, rank, "inputA"), FloatVectorToHalf(data.inputA));
    WriteBinary(RankFile(args, rank, "expertIdx"), data.expertIdx);
    WriteBinary(RankFile(args, rank, "probs"), data.probs);
}

inline std::vector<HostInputData> LoadOrGenerateWorldInputs(const DispatchCombineTileArgs &args)
{
    std::vector<HostInputData> world(args.shape.ep);
    for (uint32_t rank = 0; rank < args.shape.ep; ++rank) {
        world[rank] = args.runtime.genData != 0 ? GenerateDeterministicInputs(args, rank) : LoadInputs(args, rank);
    }
    return world;
}

inline void WriteDebugFiles(const DispatchCombineTileArgs &args, uint32_t rank, const CpuGoldenData &golden)
{
    if (args.runtime.debug == 0) {
        return;
    }
    EnsureDataDir(args.dataDir);
    WriteBinary(RankFile(args, rank, "localTokenPerExpert"), golden.localTokenPerExpert);
    WriteBinary(RankFile(args, rank, "peerTokenPerExpert"), golden.peerTokenPerExpert);
    WriteBinary(RankFile(args, rank, "cumsumPerExpert"), golden.cumsumPerExpert);
    WriteBinary(RankFile(args, rank, "expandedRowIdx"), golden.expandedRowIdx);
    WriteBinary(RankFile(args, rank, "dispatchedA_head"), FloatVectorToHalf(golden.dispatchedA));
    WriteBinary(RankFile(args, rank, "ptrD_head"), FloatVectorToHalf(golden.ptrD));
}

inline void BuildRoutes(const DispatchCombineTileArgs &args, const std::vector<HostInputData> &worldInputs,
                        std::vector<std::vector<std::vector<RouteRef>>> *routesBySrcExpert,
                        std::vector<std::vector<float>> *packedBySrc, std::vector<std::vector<int32_t>> *expandedBySrc,
                        std::vector<int32_t> *peerTokenPerExpert, uint64_t *totalRoutes, uint64_t *invalidRoutes)
{
    const DispatchCombineTileShape &shape = args.shape;
    uint32_t expertNumPadded = static_cast<uint32_t>(ExpertNumPadded(shape));
    uint32_t expandedRows = shape.m * shape.topK;
    routesBySrcExpert->assign(shape.ep, std::vector<std::vector<RouteRef>>(shape.expertNum));
    packedBySrc->assign(shape.ep, std::vector<float>(static_cast<size_t>(expandedRows) * shape.k, 0.0f));
    expandedBySrc->assign(shape.ep, std::vector<int32_t>(expandedRows, -1));
    peerTokenPerExpert->assign(static_cast<size_t>(shape.ep) * expertNumPadded, 0);
    *totalRoutes = 0;
    *invalidRoutes = 0;

    for (uint32_t src = 0; src < shape.ep; ++src) {
        std::vector<uint32_t> localTokenPerExpert(shape.expertNum, 0);
        for (uint32_t token = 0; token < shape.m; ++token) {
            for (uint32_t slot = 0; slot < shape.topK; ++slot) {
                ++(*totalRoutes);
                size_t routeIndex = static_cast<size_t>(token) * shape.topK + slot;
                int32_t expert = worldInputs[src].expertIdx[routeIndex];
                if (expert < 0 || static_cast<uint32_t>(expert) >= shape.expertNum) {
                    ++(*invalidRoutes);
                    continue;
                }
                ++localTokenPerExpert[static_cast<uint32_t>(expert)];
            }
        }

        std::vector<uint32_t> expertBase(shape.expertNum, 0);
        uint32_t running = 0;
        for (uint32_t expert = 0; expert < shape.expertNum; ++expert) {
            expertBase[expert] = running;
            running += localTokenPerExpert[expert];
            (*peerTokenPerExpert)[static_cast<size_t>(src) * expertNumPadded + expert] =
                static_cast<int32_t>(localTokenPerExpert[expert]);
        }

        std::vector<uint32_t> cursor(shape.expertNum, 0);
        for (uint32_t token = 0; token < shape.m; ++token) {
            for (uint32_t slot = 0; slot < shape.topK; ++slot) {
                size_t routeIndex = static_cast<size_t>(token) * shape.topK + slot;
                int32_t expert = worldInputs[src].expertIdx[routeIndex];
                if (expert < 0 || static_cast<uint32_t>(expert) >= shape.expertNum) {
                    continue;
                }
                uint32_t expertId = static_cast<uint32_t>(expert);
                uint32_t packedRow = expertBase[expertId] + cursor[expertId]++;
                (*expandedBySrc)[src][routeIndex] = static_cast<int32_t>(packedRow);
                (*routesBySrcExpert)[src][expertId].push_back(RouteRef{src, token, slot, expertId, packedRow});
                for (uint32_t col = 0; col < shape.k; ++col) {
                    (*packedBySrc)[src][static_cast<size_t>(packedRow) * shape.k + col] =
                        worldInputs[src].inputA[static_cast<size_t>(token) * shape.k + col];
                }
            }
        }
    }
}

} // namespace golden_detail

inline HostInputData GenerateOrLoadInputs(const DispatchCombineTileArgs &args, uint32_t myRank)
{
    HostInputData inputs = args.runtime.genData != 0 ? golden_detail::GenerateDeterministicInputs(args, myRank) :
                                                       golden_detail::LoadInputs(args, myRank);
    if (args.runtime.genData != 0) {
        golden_detail::WriteInputs(args, myRank, inputs);
    }
    return inputs;
}

inline void GenerateAllInputFiles(const DispatchCombineTileArgs &args)
{
    for (uint32_t rank = 0; rank < args.shape.ep; ++rank) {
        golden_detail::WriteInputs(args, rank, golden_detail::GenerateDeterministicInputs(args, rank));
    }
}

inline CpuGoldenData ComputeCpuGolden(const DispatchCombineTileArgs &args, const HostInputData &inputs, uint32_t myRank)
{
    (void)inputs;
    const DispatchCombineTileShape &shape = args.shape;
    uint32_t expertNumPadded = static_cast<uint32_t>(ExpertNumPadded(shape));
    uint32_t expandedRows = shape.m * shape.topK;

    std::vector<HostInputData> worldInputs = golden_detail::LoadOrGenerateWorldInputs(args);
    std::vector<std::vector<std::vector<golden_detail::RouteRef>>> routesBySrcExpert;
    std::vector<std::vector<float>> packedBySrc;
    std::vector<std::vector<int32_t>> expandedBySrc;

    CpuGoldenData golden;
    golden_detail::BuildRoutes(args, worldInputs, &routesBySrcExpert, &packedBySrc, &expandedBySrc,
                               &golden.peerTokenPerExpert, &golden.totalRoutes, &golden.invalidRoutes);

    golden.localTokenPerExpert.assign(expertNumPadded, 0);
    for (uint32_t expert = 0; expert < shape.expertNum; ++expert) {
        golden.localTokenPerExpert[expert] =
            golden.peerTokenPerExpert[static_cast<size_t>(myRank) * expertNumPadded + expert];
    }
    golden.expandedRowIdx = expandedBySrc[myRank];
    golden.cumsumPerExpert.assign(static_cast<size_t>(shape.ep) * expertNumPadded, 0);
    for (uint32_t src = 0; src < shape.ep; ++src) {
        int32_t sum = 0;
        for (uint32_t expert = 0; expert < expertNumPadded; ++expert) {
            sum += golden.peerTokenPerExpert[static_cast<size_t>(src) * expertNumPadded + expert];
            golden.cumsumPerExpert[static_cast<size_t>(src) * expertNumPadded + expert] = sum;
        }
    }

    golden.ownerRows.assign(shape.ep, 0);
    for (uint32_t owner = 0; owner < shape.ep; ++owner) {
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            uint32_t globalExpert = owner * shape.expertPerRank + localExpert;
            for (uint32_t src = 0; src < shape.ep; ++src) {
                golden.ownerRows[owner] += routesBySrcExpert[src][globalExpert].size();
            }
        }
    }

    golden.dispatchOffset.assign(shape.expertPerRank, 0);
    golden.prevSumBeforeRank.assign(static_cast<size_t>(shape.ep) * shape.expertPerRank, 0);
    uint32_t dispatchCursor = 0;
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        uint32_t globalExpert = myRank * shape.expertPerRank + localExpert;
        golden.dispatchOffset[localExpert] = static_cast<int32_t>(dispatchCursor);
        uint32_t beforeRank = 0;
        for (uint32_t src = 0; src < shape.ep; ++src) {
            golden.prevSumBeforeRank[static_cast<size_t>(src) * shape.expertPerRank + localExpert] =
                static_cast<int32_t>(beforeRank);
            uint32_t rows = static_cast<uint32_t>(routesBySrcExpert[src][globalExpert].size());
            beforeRank += rows;
            dispatchCursor += rows;
        }
    }
    if (dispatchCursor > shape.maxOutputSize) {
        throw std::runtime_error("CPU golden dispatched rows exceed maxOutputSize");
    }

    golden.dispatchedA.assign(static_cast<size_t>(shape.maxOutputSize) * shape.k, 0.0f);
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        uint32_t globalExpert = myRank * shape.expertPerRank + localExpert;
        for (uint32_t src = 0; src < shape.ep; ++src) {
            uint32_t dstStart =
                static_cast<uint32_t>(golden.dispatchOffset[localExpert]) +
                static_cast<uint32_t>(
                    golden.prevSumBeforeRank[static_cast<size_t>(src) * shape.expertPerRank + localExpert]);
            const auto &routes = routesBySrcExpert[src][globalExpert];
            for (uint32_t row = 0; row < routes.size(); ++row) {
                uint32_t outRow = dstStart + row;
                uint32_t packedRow = routes[row].packedRow;
                for (uint32_t col = 0; col < shape.k; ++col) {
                    golden.dispatchedA[static_cast<size_t>(outRow) * shape.k + col] =
                        packedBySrc[src][static_cast<size_t>(packedRow) * shape.k + col];
                }
            }
        }
    }
    golden.expertOutput = golden.dispatchedA;

    std::vector<std::vector<float>> ptrDBySrc(shape.ep,
                                              std::vector<float>(static_cast<size_t>(expandedRows) * shape.k, 0.0f));
    for (uint32_t dst = 0; dst < shape.ep; ++dst) {
        uint32_t dstDispatchCursor = 0;
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            uint32_t globalExpert = dst * shape.expertPerRank + localExpert;
            for (uint32_t src = 0; src < shape.ep; ++src) {
                const auto &routes = routesBySrcExpert[src][globalExpert];
                for (uint32_t row = 0; row < routes.size(); ++row) {
                    const golden_detail::RouteRef &route = routes[row];
                    int32_t dstPackedRow =
                        expandedBySrc[src][static_cast<size_t>(route.token) * shape.topK + route.slot];
                    if (dstPackedRow < 0) {
                        continue;
                    }
                    uint32_t expertOutputRow = dstDispatchCursor + row;
                    for (uint32_t col = 0; col < shape.k; ++col) {
                        float value = 0.0f;
                        if (dst == myRank) {
                            value = golden.expertOutput[static_cast<size_t>(expertOutputRow) * shape.k + col];
                        } else {
                            value = packedBySrc[src][static_cast<size_t>(route.packedRow) * shape.k + col];
                        }
                        ptrDBySrc[src][static_cast<size_t>(dstPackedRow) * shape.k + col] = value;
                    }
                }
                dstDispatchCursor += routes.size();
            }
        }
    }

    golden.ptrD = ptrDBySrc[myRank];
    golden.outputC.assign(static_cast<size_t>(shape.m) * shape.k, 0.0f);
    for (uint32_t token = 0; token < shape.m; ++token) {
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            size_t routeIndex = static_cast<size_t>(token) * shape.topK + slot;
            int32_t ptrDRow = golden.expandedRowIdx[routeIndex];
            if (ptrDRow < 0) {
                continue;
            }
            float prob = worldInputs[myRank].probs[routeIndex];
            for (uint32_t col = 0; col < shape.k; ++col) {
                golden.outputC[static_cast<size_t>(token) * shape.k + col] +=
                    prob * golden.ptrD[static_cast<size_t>(ptrDRow) * shape.k + col];
            }
        }
    }

    golden_detail::EnsureDataDir(args.dataDir);
    golden_detail::WriteBinary(golden_detail::RankFile(args, myRank, "golden_outputC"),
                               golden_detail::FloatVectorToHalf(golden.outputC));
    golden_detail::WriteDebugFiles(args, myRank, golden);
    return golden;
}

inline CompareResult CompareOutputs(const DispatchCombineTileArgs &args, const CpuGoldenData &golden,
                                    const std::vector<float> &actualOutputC, uint32_t myRank)
{
    (void)myRank;
    CompareResult result{};
    result.elementCount = static_cast<uint64_t>(golden.outputC.size());
    result.firstMismatchIndex = result.elementCount;
    result.actual = 0.0f;
    result.expected = 0.0f;
    if (actualOutputC.size() != golden.outputC.size()) {
        result.mismatchCount = result.elementCount;
        result.firstMismatchIndex = 0;
        return result;
    }
    for (size_t i = 0; i < golden.outputC.size(); ++i) {
        float actual = actualOutputC[i];
        float expected = golden.outputC[i];
        float diff = std::fabs(actual - expected);
        float tol = static_cast<float>(args.atol + args.rtol * std::fabs(expected));
        if (diff > tol) {
            if (result.mismatchCount == 0) {
                result.firstMismatchIndex = i;
                result.actual = actual;
                result.expected = expected;
            }
            ++result.mismatchCount;
        }
    }
    return result;
}

} // namespace dispatch_combine_tile

#endif // DISPATCH_COMBINE_TILE_GOLDEN_H_
