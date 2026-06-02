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

#include "args.hpp"
#include "moe_new_dispatch_combine_a8w8_m1_layout.hpp"

#include <algorithm>
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
    std::vector<uint8_t> xActiveMask;
};

struct CpuGoldenData {
    std::vector<float> dispatchedA;
    std::vector<float> expertOutput;
    std::vector<float> outputC;
    std::vector<float> ptrD;
    std::vector<float> packedA;
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

struct CpuM2ReferenceData {
    std::vector<int8_t> gmm1InputInt8;
    std::vector<float> routingPerTokenScale;
    std::vector<int8_t> weight1Int8;
    std::vector<int8_t> weight2Int8;
    std::vector<uint64_t> scale1Uint64;
    std::vector<uint64_t> scale2Uint64;
    std::vector<int32_t> gmm1AccInt32;
    std::vector<float> gmm1Out;
    std::vector<float> swigluOut;
    std::vector<int8_t> gmm2InputInt8;
    std::vector<float> gmm2PerTokenScale;
    std::vector<int32_t> gmm2AccInt32;
    std::vector<float> gmm2Out;
    std::vector<float> ptrD;
    std::vector<float> outputC;
    uint64_t localRows = 0;
    uint64_t validRows = 0;
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

inline uint32_t FloatToBits32(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline float Bits32ToFloat(uint32_t bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline uint64_t FloatScaleToUint64(float value)
{
    return static_cast<uint64_t>(FloatToBits32(value));
}

inline float Uint64ScaleToFloat(uint64_t value)
{
    return Bits32ToFloat(static_cast<uint32_t>(value & 0xFFFFFFFFULL));
}

inline int8_t DeterministicI8(uint32_t seed, uint64_t index, uint32_t salt)
{
    uint64_t mixed = static_cast<uint64_t>(seed) * 1315423911ULL + index * 2654435761ULL + salt * 97531ULL;
    return static_cast<int8_t>(static_cast<int32_t>(mixed % 255ULL) - 127);
}

inline float DeterministicScale(uint32_t index, uint32_t salt)
{
    return static_cast<float>((index % 7U) + 1U + salt) / 4096.0f;
}

inline int8_t QuantizeToInt8(float value, float scale)
{
    if (scale <= 0.0f || !std::isfinite(scale)) {
        return 0;
    }
    int32_t quant = static_cast<int32_t>(std::round(value / scale));
    quant = std::max(-127, std::min(127, quant));
    return static_cast<int8_t>(quant);
}

inline float Sigmoid(float value)
{
    if (value >= 0.0f) {
        float expNeg = std::exp(-value);
        return 1.0f / (1.0f + expNeg);
    }
    float expPos = std::exp(value);
    return expPos / (1.0f + expPos);
}

inline int32_t SelectExpertForCase(const DispatchCombineTileArgs &args, uint32_t rank, uint32_t token, uint32_t slot)
{
    const DispatchCombineTileShape &shape = args.shape;
    if (args.caseName == "over-capacity") {
        return 0;
    }
    if (args.caseName == "zero-token") {
        uint32_t activeExperts = shape.expertNum / 2;
        activeExperts = activeExperts == 0 ? 1 : activeExperts;
        return static_cast<int32_t>((token + slot + rank) % activeExperts);
    }
    if (args.caseName == "skewed") {
        uint32_t routeIndex = token * shape.topK + slot;
        if ((routeIndex % 4U) != 0U) {
            return 0;
        }
        return static_cast<int32_t>((token + slot + rank) % shape.expertNum);
    }
    return static_cast<int32_t>(((token * shape.topK + slot) + rank) % shape.expertNum);
}

inline bool GeneratedTokenActive(const DispatchCombineTileArgs &args, uint32_t token)
{
    if (args.xActiveMaskMode == "alternate") {
        return (token % 2U) == 0U;
    }
    if (args.xActiveMaskMode == "tail-half") {
        return token < ((args.shape.m + 1U) / 2U);
    }
    return true;
}

inline bool TokenActive(const HostInputData &data, uint32_t token)
{
    return data.xActiveMask.empty() || data.xActiveMask[token] != 0U;
}

inline uint32_t ClampRowsToCapacity(uint32_t cursor, uint32_t rows, uint32_t cap)
{
    if (cursor >= cap || rows == 0U) {
        return 0U;
    }
    uint32_t available = cap - cursor;
    return rows < available ? rows : available;
}

inline HostInputData GenerateDeterministicInputs(const DispatchCombineTileArgs &args, uint32_t rank)
{
    const DispatchCombineTileShape &shape = args.shape;
    HostInputData data;
    data.inputA.resize(static_cast<size_t>(shape.m) * shape.k);
    data.expertIdx.resize(static_cast<size_t>(shape.m) * shape.topK);
    data.probs.resize(static_cast<size_t>(shape.m) * shape.topK);
    if (args.xActiveMaskMode != "none") {
        data.xActiveMask.resize(shape.m, 1U);
    }

    for (uint32_t token = 0; token < shape.m; ++token) {
        if (!data.xActiveMask.empty()) {
            data.xActiveMask[token] = GeneratedTokenActive(args, token) ? 1U : 0U;
        }
        for (uint32_t col = 0; col < shape.k; ++col) {
            uint32_t mixed = args.runtime.seed + rank * 131U + token * 17U + col * 3U;
            data.inputA[static_cast<size_t>(token) * shape.k + col] = static_cast<float>(mixed % 251U) / 32.0f;
        }
        float probSum = 0.0f;
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            uint32_t flat = token * shape.topK + slot;
            data.expertIdx[flat] = SelectExpertForCase(args, rank, token, slot);
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
    if (args.xActiveMaskMode != "none") {
        data.xActiveMask = ReadBinary<uint8_t>(RankFile(args, rank, "xActiveMask"), shape.m);
    }
    return data;
}

inline void WriteInputs(const DispatchCombineTileArgs &args, uint32_t rank, const HostInputData &data)
{
    EnsureDataDir(args.dataDir);
    WriteBinary(RankFile(args, rank, "inputA"), FloatVectorToHalf(data.inputA));
    WriteBinary(RankFile(args, rank, "expertIdx"), data.expertIdx);
    WriteBinary(RankFile(args, rank, "probs"), data.probs);
    if (!data.xActiveMask.empty()) {
        WriteBinary(RankFile(args, rank, "xActiveMask"), data.xActiveMask);
    }
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
    WriteBinary(RankFile(args, rank, "packedA_head"), FloatVectorToHalf(golden.packedA));
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
                if (!TokenActive(worldInputs[src], token)) {
                    ++(*invalidRoutes);
                    continue;
                }
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
                if (!TokenActive(worldInputs[src], token)) {
                    (*expandedBySrc)[src][routeIndex] = static_cast<int32_t>(shape.maxOutputSize);
                    continue;
                }
                int32_t expert = worldInputs[src].expertIdx[routeIndex];
                if (expert < 0 || static_cast<uint32_t>(expert) >= shape.expertNum) {
                    continue;
                }
                uint32_t expertId = static_cast<uint32_t>(expert);
                uint32_t packedRow = expertBase[expertId] + cursor[expertId]++;
                if (packedRow >= shape.maxOutputSize) {
                    (*expandedBySrc)[src][routeIndex] = static_cast<int32_t>(shape.maxOutputSize);
                    continue;
                }
                (*expandedBySrc)[src][routeIndex] = static_cast<int32_t>(packedRow);
                (*routesBySrcExpert)[src][expertId].push_back(RouteRef{src, token, slot, expertId, packedRow});
                for (uint32_t col = 0; col < shape.k; ++col) {
                    (*packedBySrc)[src][static_cast<size_t>(packedRow) * shape.k + col] =
                        worldInputs[src].inputA[static_cast<size_t>(token) * shape.k + col];
                }
            }
        }
    }

    for (uint32_t expertOwner = 0; expertOwner < shape.ep; ++expertOwner) {
        uint32_t dispatchCursor = 0;
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            uint32_t globalExpert = expertOwner * shape.expertPerRank + localExpert;
            for (uint32_t src = 0; src < shape.ep; ++src) {
                auto &routes = (*routesBySrcExpert)[src][globalExpert];
                uint32_t effectiveRows =
                    ClampRowsToCapacity(dispatchCursor, static_cast<uint32_t>(routes.size()), shape.maxOutputSize);
                for (uint32_t row = effectiveRows; row < routes.size(); ++row) {
                    const RouteRef &route = routes[row];
                    (*expandedBySrc)[src][static_cast<size_t>(route.token) * shape.topK + route.slot] =
                        static_cast<int32_t>(shape.maxOutputSize);
                }
                dispatchCursor += effectiveRows;
            }
        }
    }
}

inline int32_t EffectiveRowsForTokenOwner(const DispatchCombineTileShape &shape, const CpuGoldenData &golden,
                                          uint32_t tokenOwner, uint32_t expertOwner, uint32_t localExpert)
{
    uint32_t expertNumPadded = static_cast<uint32_t>(ExpertNumPadded(shape));
    uint32_t globalExpert = expertOwner * shape.expertPerRank + localExpert;
    int32_t cursor = 0;
    int32_t cap = static_cast<int32_t>(shape.maxOutputSize);
    for (uint32_t prevLocalExpert = 0; prevLocalExpert < shape.expertPerRank; ++prevLocalExpert) {
        uint32_t currentGlobalExpert = expertOwner * shape.expertPerRank + prevLocalExpert;
        for (uint32_t src = 0; src < shape.ep; ++src) {
            int32_t rows = golden.peerTokenPerExpert[static_cast<size_t>(src) * expertNumPadded + currentGlobalExpert];
            int32_t effective = 0;
            if (cursor < cap && rows > 0) {
                int32_t available = cap - cursor;
                effective = rows < available ? rows : available;
            }
            if (currentGlobalExpert == globalExpert && src == tokenOwner) {
                return effective;
            }
            cursor += effective;
        }
    }
    return 0;
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

inline std::string RankBinaryFile(const DispatchCombineTileArgs &args, uint32_t rank, const char *name)
{
    return golden_detail::RankFile(args, rank, name);
}

template <typename T>
inline void WriteBinaryFile(const std::string &path, const std::vector<T> &data)
{
    golden_detail::WriteBinary(path, data);
}

inline std::vector<uint16_t> FloatVectorToHalfBits(const std::vector<float> &src)
{
    return golden_detail::FloatVectorToHalf(src);
}

inline std::vector<float> HalfBitsToFloatVector(const std::vector<uint16_t> &src)
{
    return golden_detail::HalfVectorToFloat(src);
}

inline CpuGoldenData ComputeCpuGolden(const DispatchCombineTileArgs &args, const HostInputData &inputs, uint32_t myRank,
                                      bool writeFiles = true)
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
    golden.packedA = packedBySrc[myRank];
    golden.cumsumPerExpert.assign(static_cast<size_t>(shape.ep) * expertNumPadded, 0);
    for (uint32_t src = 0; src < shape.ep; ++src) {
        int32_t sum = 0;
        for (uint32_t expert = 0; expert < expertNumPadded; ++expert) {
            int32_t rows = golden.peerTokenPerExpert[static_cast<size_t>(src) * expertNumPadded + expert];
            if (rows > 0) {
                sum += rows;
            }
            golden.cumsumPerExpert[static_cast<size_t>(src) * expertNumPadded + expert] = sum;
        }
    }

    golden.ownerRows.assign(shape.ep, 0);
    for (uint32_t owner = 0; owner < shape.ep; ++owner) {
        uint32_t dispatchCursor = 0;
        for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
            uint32_t globalExpert = owner * shape.expertPerRank + localExpert;
            for (uint32_t src = 0; src < shape.ep; ++src) {
                uint32_t rows = static_cast<uint32_t>(
                    golden.peerTokenPerExpert[static_cast<size_t>(src) * expertNumPadded + globalExpert]);
                dispatchCursor += golden_detail::ClampRowsToCapacity(dispatchCursor, rows, shape.maxOutputSize);
            }
        }
        golden.ownerRows[owner] = dispatchCursor;
    }

    golden.dispatchOffset.assign(shape.expertPerRank, 0);
    golden.prevSumBeforeRank.assign(static_cast<size_t>(shape.ep) * shape.expertPerRank, 0);
    uint32_t dispatchCursor = 0;
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        uint32_t globalExpert = myRank * shape.expertPerRank + localExpert;
        golden.dispatchOffset[localExpert] = static_cast<int32_t>(dispatchCursor);
        uint32_t expertRows = 0;
        for (uint32_t src = 0; src < shape.ep; ++src) {
            uint32_t sourcePrefix =
                globalExpert == 0 ?
                    0U :
                    static_cast<uint32_t>(
                        golden.cumsumPerExpert[static_cast<size_t>(src) * expertNumPadded + globalExpert - 1U]);
            golden.prevSumBeforeRank[static_cast<size_t>(src) * shape.expertPerRank + localExpert] =
                static_cast<int32_t>(sourcePrefix);
            uint32_t rows = static_cast<uint32_t>(
                golden.peerTokenPerExpert[static_cast<size_t>(src) * expertNumPadded + globalExpert]);
            rows = golden_detail::ClampRowsToCapacity(dispatchCursor + expertRows, rows, shape.maxOutputSize);
            expertRows += rows;
        }
        dispatchCursor += expertRows;
    }
    if (dispatchCursor > shape.maxOutputSize) {
        throw std::runtime_error("CPU golden dispatched rows exceed maxOutputSize");
    }

    golden.dispatchedA.assign(static_cast<size_t>(shape.maxOutputSize) * shape.k, 0.0f);
    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        uint32_t globalExpert = myRank * shape.expertPerRank + localExpert;
        uint32_t expertRows = 0;
        for (uint32_t src = 0; src < shape.ep; ++src) {
            uint32_t dstStart = static_cast<uint32_t>(golden.dispatchOffset[localExpert]) + expertRows;
            uint32_t rawRows = static_cast<uint32_t>(
                golden.peerTokenPerExpert[static_cast<size_t>(src) * expertNumPadded + globalExpert]);
            uint32_t rows = golden_detail::ClampRowsToCapacity(dstStart, rawRows, shape.maxOutputSize);
            uint32_t srcStart = static_cast<uint32_t>(
                golden.prevSumBeforeRank[static_cast<size_t>(src) * shape.expertPerRank + localExpert]);
            for (uint32_t row = 0; row < rows; ++row) {
                uint32_t packedRow = srcStart + row;
                if (packedRow >= shape.maxOutputSize) {
                    continue;
                }
                uint32_t outRow = dstStart + row;
                for (uint32_t col = 0; col < shape.k; ++col) {
                    golden.dispatchedA[static_cast<size_t>(outRow) * shape.k + col] =
                        packedBySrc[src][static_cast<size_t>(packedRow) * shape.k + col];
                }
            }
            expertRows += rows;
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
                uint32_t rawRows = static_cast<uint32_t>(
                    golden.peerTokenPerExpert[static_cast<size_t>(src) * expertNumPadded + globalExpert]);
                uint32_t rows = golden_detail::ClampRowsToCapacity(dstDispatchCursor, rawRows, shape.maxOutputSize);
                for (uint32_t row = 0; row < rows; ++row) {
                    if (row >= routes.size()) {
                        continue;
                    }
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
                dstDispatchCursor += rows;
            }
        }
    }

    golden.ptrD = ptrDBySrc[myRank];
    golden.outputC.assign(static_cast<size_t>(shape.m) * shape.k, 0.0f);
    for (uint32_t token = 0; token < shape.m; ++token) {
        for (uint32_t slot = 0; slot < shape.topK; ++slot) {
            size_t routeIndex = static_cast<size_t>(token) * shape.topK + slot;
            int32_t ptrDRow = golden.expandedRowIdx[routeIndex];
            if (ptrDRow < 0 || static_cast<uint32_t>(ptrDRow) >= shape.maxOutputSize) {
                continue;
            }
            float prob = worldInputs[myRank].probs[routeIndex];
            for (uint32_t col = 0; col < shape.k; ++col) {
                golden.outputC[static_cast<size_t>(token) * shape.k + col] +=
                    prob * golden.ptrD[static_cast<size_t>(ptrDRow) * shape.k + col];
            }
        }
    }

    if (writeFiles) {
        golden_detail::EnsureDataDir(args.dataDir);
        golden_detail::WriteBinary(golden_detail::RankFile(args, myRank, "golden_outputC"),
                                   golden_detail::FloatVectorToHalf(golden.outputC));
        golden_detail::WriteDebugFiles(args, myRank, golden);
    }
    return golden;
}

inline int32_t DispatchOwnerStart(const DispatchCombineTileArgs &args, const CpuGoldenData &golden,
                                  uint32_t expertOwnerRank, uint32_t tokenOwnerRank, uint32_t localExpert)
{
    const DispatchCombineTileShape &shape = args.shape;
    int32_t prefix = 0;
    for (uint32_t prevTokenOwner = 0; prevTokenOwner < tokenOwnerRank; ++prevTokenOwner) {
        prefix +=
            golden_detail::EffectiveRowsForTokenOwner(shape, golden, prevTokenOwner, expertOwnerRank, localExpert);
    }
    return golden.dispatchOffset[localExpert] + prefix;
}

inline CpuM2ReferenceData ComputeCpuM2Reference(const DispatchCombineTileArgs &args, const CpuGoldenData &golden,
                                                uint32_t myRank, bool buildReturnPayload = true)
{
    const DispatchCombineTileShape &shape = args.shape;
    CpuM2ReferenceData ref;
    const uint32_t globalExpertNum = shape.expertNum;
    const uint32_t w1Cols = args.intermediateSize * 2U;
    const uint32_t localRows = shape.maxOutputSize;
    const size_t gmm1InputElems = static_cast<size_t>(localRows) * shape.k;
    const size_t gmm1AccElems = static_cast<size_t>(localRows) * w1Cols;
    const size_t swigluElems = static_cast<size_t>(localRows) * args.intermediateSize;
    const size_t gmm2AccElems = static_cast<size_t>(localRows) * shape.k;

    ref.localRows = localRows;
    ref.gmm1InputInt8.assign(gmm1InputElems, 0);
    ref.routingPerTokenScale.assign(localRows, 1.0f);
    ref.weight1Int8.resize(static_cast<size_t>(globalExpertNum) * shape.k * w1Cols);
    ref.weight2Int8.resize(static_cast<size_t>(globalExpertNum) * args.intermediateSize * shape.k);
    ref.scale1Uint64.resize(w1Cols);
    ref.scale2Uint64.resize(shape.k);
    ref.gmm1AccInt32.assign(gmm1AccElems, 0);
    ref.gmm1Out.assign(gmm1AccElems, 0.0f);
    ref.swigluOut.assign(swigluElems, 0.0f);
    ref.gmm2InputInt8.assign(swigluElems, 0);
    ref.gmm2PerTokenScale.assign(localRows, 1.0f);
    ref.gmm2AccInt32.assign(gmm2AccElems, 0);
    ref.gmm2Out.assign(gmm2AccElems, 0.0f);

    for (size_t i = 0; i < ref.weight1Int8.size(); ++i) {
        ref.weight1Int8[i] = golden_detail::DeterministicI8(args.runtime.seed, i, 1);
    }
    for (size_t i = 0; i < ref.weight2Int8.size(); ++i) {
        ref.weight2Int8[i] = golden_detail::DeterministicI8(args.runtime.seed, i, 2);
    }
    for (uint32_t col = 0; col < w1Cols; ++col) {
        ref.scale1Uint64[col] = golden_detail::FloatScaleToUint64(golden_detail::DeterministicScale(col, 1));
    }
    for (uint32_t col = 0; col < shape.k; ++col) {
        ref.scale2Uint64[col] = golden_detail::FloatScaleToUint64(golden_detail::DeterministicScale(col, 2));
    }

    for (uint32_t row = 0; row < localRows; ++row) {
        float maxAbs = 0.0f;
        for (uint32_t col = 0; col < shape.k; ++col) {
            float value = golden_detail::HalfToFloat(
                golden_detail::FloatToHalf(golden.dispatchedA[static_cast<size_t>(row) * shape.k + col]));
            maxAbs = std::max(maxAbs, std::fabs(value));
        }
        float scale = maxAbs == 0.0f ? 1.0f : maxAbs / 127.0f;
        ref.routingPerTokenScale[row] = scale;
        for (uint32_t col = 0; col < shape.k; ++col) {
            float value = golden_detail::HalfToFloat(
                golden_detail::FloatToHalf(golden.dispatchedA[static_cast<size_t>(row) * shape.k + col]));
            ref.gmm1InputInt8[static_cast<size_t>(row) * shape.k + col] = golden_detail::QuantizeToInt8(value, scale);
        }
    }

    for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
        uint32_t globalExpert = myRank * shape.expertPerRank + localExpert;
        uint32_t rowBegin = static_cast<uint32_t>(golden.dispatchOffset[localExpert]);
        uint32_t rowEnd = localExpert + 1U < shape.expertPerRank ?
                              static_cast<uint32_t>(golden.dispatchOffset[localExpert + 1U]) :
                              (myRank < golden.ownerRows.size() ? golden.ownerRows[myRank] : localRows);
        if (rowEnd > localRows) {
            rowEnd = localRows;
        }
        for (uint32_t row = rowBegin; row < rowEnd; ++row) {
            ++ref.validRows;
            for (uint32_t n = 0; n < w1Cols; ++n) {
                int32_t acc = 0;
                for (uint32_t k = 0; k < shape.k; ++k) {
                    int8_t a = ref.gmm1InputInt8[static_cast<size_t>(row) * shape.k + k];
                    int8_t b = ref.weight1Int8[(static_cast<size_t>(globalExpert) * shape.k + k) * w1Cols + n];
                    acc += static_cast<int32_t>(a) * static_cast<int32_t>(b);
                }
                size_t outIndex = static_cast<size_t>(row) * w1Cols + n;
                ref.gmm1AccInt32[outIndex] = acc;
                float channelScale = golden_detail::Uint64ScaleToFloat(ref.scale1Uint64[n]);
                ref.gmm1Out[outIndex] = static_cast<float>(acc) * channelScale;
            }
            float swigluMaxAbs = 0.0f;
            float rowRoutingScale = ref.routingPerTokenScale[row];
            for (uint32_t n = 0; n < args.intermediateSize; ++n) {
                float gate = ref.gmm1Out[static_cast<size_t>(row) * w1Cols + n] * rowRoutingScale;
                float up = ref.gmm1Out[static_cast<size_t>(row) * w1Cols + args.intermediateSize + n] * rowRoutingScale;
                float value = golden_detail::Sigmoid(gate) * up;
                ref.swigluOut[static_cast<size_t>(row) * args.intermediateSize + n] = value;
                swigluMaxAbs = std::max(swigluMaxAbs, std::fabs(value));
            }
            float gmm2Scale = swigluMaxAbs == 0.0f ? 1.0f : swigluMaxAbs / 127.0f;
            ref.gmm2PerTokenScale[row] = gmm2Scale;
            for (uint32_t n = 0; n < args.intermediateSize; ++n) {
                size_t index = static_cast<size_t>(row) * args.intermediateSize + n;
                ref.gmm2InputInt8[index] = golden_detail::QuantizeToInt8(ref.swigluOut[index], gmm2Scale);
            }
            for (uint32_t n = 0; n < shape.k; ++n) {
                int32_t acc = 0;
                for (uint32_t k = 0; k < args.intermediateSize; ++k) {
                    int8_t a = ref.gmm2InputInt8[static_cast<size_t>(row) * args.intermediateSize + k];
                    int8_t b =
                        ref.weight2Int8[(static_cast<size_t>(globalExpert) * args.intermediateSize + k) * shape.k + n];
                    acc += static_cast<int32_t>(a) * static_cast<int32_t>(b);
                }
                size_t outIndex = static_cast<size_t>(row) * shape.k + n;
                ref.gmm2AccInt32[outIndex] = acc;
                float channelScale = golden_detail::Uint64ScaleToFloat(ref.scale2Uint64[n]);
                ref.gmm2Out[outIndex] = static_cast<float>(acc) * channelScale * ref.gmm2PerTokenScale[row];
            }
        }
    }
    if (buildReturnPayload) {
        uint32_t expertNumPadded = static_cast<uint32_t>(ExpertNumPadded(shape));
        uint32_t expandedRows = shape.m * shape.topK;
        ref.ptrD.assign(static_cast<size_t>(expandedRows) * shape.k, 0.0f);
        for (uint32_t expertOwner = 0; expertOwner < shape.ep; ++expertOwner) {
            CpuGoldenData ownerGolden = golden;
            CpuM2ReferenceData ownerRef;
            if (expertOwner == myRank) {
                ownerRef = ref;
            } else {
                HostInputData unusedInputs;
                ownerGolden = ComputeCpuGolden(args, unusedInputs, expertOwner, false);
                ownerRef = ComputeCpuM2Reference(args, ownerGolden, expertOwner, false);
            }
            for (uint32_t localExpert = 0; localExpert < shape.expertPerRank; ++localExpert) {
                uint32_t globalExpert = expertOwner * shape.expertPerRank + localExpert;
                int32_t rows =
                    golden_detail::EffectiveRowsForTokenOwner(shape, golden, myRank, expertOwner, localExpert);
                if (rows <= 0) {
                    continue;
                }
                int32_t srcStart = DispatchOwnerStart(args, ownerGolden, expertOwner, myRank, localExpert);
                int32_t dstStart =
                    globalExpert == 0 ?
                        0 :
                        golden.cumsumPerExpert[static_cast<size_t>(myRank) * expertNumPadded + globalExpert - 1U];
                for (int32_t row = 0; row < rows; ++row) {
                    for (uint32_t col = 0; col < shape.k; ++col) {
                        float value = ownerRef.gmm2Out[static_cast<size_t>(srcStart + row) * shape.k + col];
                        ref.ptrD[static_cast<size_t>(dstStart + row) * shape.k + col] =
                            golden_detail::HalfToFloat(golden_detail::FloatToHalf(value));
                    }
                }
            }
        }

        std::vector<HostInputData> worldInputs = golden_detail::LoadOrGenerateWorldInputs(args);
        ref.outputC.assign(static_cast<size_t>(shape.m) * shape.k, 0.0f);
        for (uint32_t token = 0; token < shape.m; ++token) {
            for (uint32_t slot = 0; slot < shape.topK; ++slot) {
                size_t routeIndex = static_cast<size_t>(token) * shape.topK + slot;
                int32_t ptrDRow = golden.expandedRowIdx[routeIndex];
                if (ptrDRow < 0 || static_cast<uint32_t>(ptrDRow) >= shape.maxOutputSize) {
                    continue;
                }
                float prob = worldInputs[myRank].probs[routeIndex];
                for (uint32_t col = 0; col < shape.k; ++col) {
                    ref.outputC[static_cast<size_t>(token) * shape.k + col] +=
                        prob * ref.ptrD[static_cast<size_t>(ptrDRow) * shape.k + col];
                }
            }
        }
    }
    return ref;
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
