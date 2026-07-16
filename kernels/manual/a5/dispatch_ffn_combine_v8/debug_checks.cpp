#include "debug_checks.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"

#include "comm_mpi.h"
#include "data_utils.hpp"
#include "kernel_launch.hpp"
#include "op_kernel/utils/const_args.hpp"

namespace {

using DeviceBuffer = DebugDeviceBuffer;

constexpr uint64_t kHostMiB = 1024U * 1024U;
constexpr uint32_t kHostFrontDoneBaseIndex = 4096;
constexpr uint32_t kHostFrontSignalStride = 16;
constexpr int32_t kHostFrontCountMarkerBias = 0x800000;
constexpr uint32_t kHostDispatchInvalidTask = 0xFFFFFFFFU;
constexpr uint32_t kHostDispatchMinRowBlockBytes = 128U * 1024U;
constexpr uint32_t kHostDispatchRowBlockBlocksPerCore = 1U;
constexpr uint32_t kHostGmm1SegmentPolicyReferenceEpilogue = 1U;
constexpr uint32_t kHostPackedWeightKAlign = 16U;
constexpr uint32_t kHostPackedWeightInt8NAlign = 32U;
constexpr uint32_t kHostSwigluMetadataModeSharedSegmentMeta = 1U;
constexpr uint64_t kHostFrontCheckLayoutDebugMagic = 0x5635464e45574c59ULL;
constexpr uint64_t kHostFrontLayoutDebugMagic = 0x56354652324c5954ULL;
constexpr uint64_t kHostDispatchGatherLayoutDebugMagic = 0x56354450324c5954ULL;
constexpr uint32_t kHostFrontHelperExpectedMask = (1U << 6U) - 1U;
constexpr uint32_t kHostFrontPackedScaleBytes = 32U;
constexpr uint32_t kHostFrontSrcToDstAssistElems = 256U;
constexpr uint32_t kHostFrontInt32OneBlockElems = 8U;
constexpr uint32_t kHostFrontSortAlignElement = 32U;
constexpr uint32_t kHostFrontMaxExpertNum = 5120U;
constexpr uint32_t kHostFrontDynamicQuantColsBuffer = 21U;
constexpr uint32_t kHostFrontDynamicQuantScaleBytes = 64U;
constexpr uint32_t kHostFrontFullLoadHLimit = 7168U;
constexpr uint64_t kHostDispatchCopyChunkDebugMagic = 0x5635444350594348ULL;
constexpr uint64_t kHostCombineLayoutDebugMagic = 0x5635434f4d424c59ULL;
constexpr uint64_t kHostCombineReadyDebugMagic = 0x5635434f4d524459ULL;
constexpr uint64_t kHostCombineMetaDebugMagic = 0x5635434f4d4d4554ULL;
constexpr uint64_t kHostCombineLoadDebugMagic = 0x5635434f4d4c4f44ULL;
constexpr uint64_t kHostCombineDequantDebugMagic = 0x5635434f4d445154ULL;
constexpr uint64_t kHostCombineStoreDebugMagic = 0x5635434f4d535452ULL;
constexpr uint64_t kHostCombineFinalizeDebugMagic = 0x5635434f4d46494eULL;
constexpr uint64_t kHostCombineTaskDebugMagic = 0x5635434f4d54534bULL;
constexpr uint64_t kHostCombineDoneDebugMagic = 0x5635434f4d444f4eULL;
constexpr uint32_t kHostCombineLayoutVersion = 1U;
constexpr uint32_t kHostCombineSmallTokenThreshold = 4096U;
constexpr uint32_t kHostCombineVecTileElems = 8192U;
constexpr uint32_t kHostCombineDirectLargeUbStages = 2U;
constexpr uint32_t kHostCombineSmallTokenSubtileRows = 16U;
constexpr uint32_t kHostCombineSmallTokenSubtileCols = 256U;
constexpr uint32_t kHostCombineSmallMaxElems = 8192U;
constexpr uint32_t kHostCombineSmallScaleElems = kHostCombineSmallMaxElems / kHostCombineSmallTokenSubtileCols;
constexpr uint32_t kHostCombineDirectSmallUbStages = 2U;
constexpr uint32_t kHostGmm2CombineCvModeDirect = 1U;
constexpr uint32_t kHostGmm2CombineCvTileM = 128U;
constexpr uint32_t kHostGmm2CombineCvTileN = 256U;
constexpr uint32_t kHostGmm2CombineCvScaleCacheElems = 4096U;
constexpr uint32_t kHostGmm2CombineCvSlotBytes = kHostGmm2CombineCvTileM * kHostGmm2CombineCvTileN * sizeof(uint16_t);
constexpr uint32_t kHostCombineInitialEventMask = 0x00030003U;
constexpr uint32_t kHostCombineScaleReadDirectScalar = 1U;
constexpr uint32_t kHostCombineSmallLoadModeScaleLoadShift = 8U;
constexpr uint32_t kHostCombineSmallLoadModeScaleCacheHitShift = 16U;
constexpr uint32_t kHostCombineSmallLoadModeLoadCountShift = 24U;
constexpr uint32_t kHostCombineSmallDequantOpMulsRowsShift = 24U;
constexpr uint32_t kHostCombineDequantOpCounts = 0x00010101U;
constexpr uint64_t kHostUnpermuteLayoutDebugMagic = 0x5635554e504c5954ULL;
constexpr uint64_t kHostUnpermuteTaskDebugMagic = 0x5635554e5054534bULL;
constexpr uint64_t kHostUnpermuteMetaDebugMagic = 0x5635554e504d4554ULL;
constexpr uint64_t kHostUnpermuteAccumDebugMagic = 0x5635554e50414343ULL;
constexpr uint64_t kHostUnpermuteOutputDebugMagic = 0x5635554e504f5554ULL;
constexpr uint64_t kHostUnpermuteDoneDebugMagic = 0x5635554e50444f4eULL;
constexpr uint32_t kHostUnpermuteLayoutVersion = 1U;

uint32_t HostCeilDiv(uint32_t value, uint32_t divisor)
{
    return divisor == 0U ? 0U : (value + divisor - 1U) / divisor;
}

struct HostFrontSrcToDstTiling {
    uint32_t need_core_num = 0;
    uint32_t per_core_rows = 0;
    uint32_t last_core_rows = 0;
    uint32_t per_core_per_loop_rows = 0;
    uint32_t per_core_last_loop_rows = 0;
    uint32_t last_core_per_loop_rows = 0;
    uint32_t last_core_last_loop_rows = 0;
};

struct HostFrontGatherQuantTiling {
    uint32_t need_core_num = 0;
    uint32_t activate_rows = 0;
    uint32_t per_core_rows = 0;
    uint32_t last_core_rows = 0;
    uint32_t per_core_per_loop_rows = 0;
    uint32_t per_core_last_loop_rows = 0;
    uint32_t last_core_per_loop_rows = 0;
    uint32_t last_core_last_loop_rows = 0;
    uint32_t per_core_loops = 0;
    uint32_t last_core_loops = 0;
    uint32_t per_loop_cols = 0;
    uint32_t last_loop_cols = 0;
    uint32_t col_loops = 0;
    uint32_t smooth_type = 0;
};

uint64_t HostFrontAlignBytes(uint64_t bytes)
{
    return HostCeilDiv(static_cast<uint32_t>(bytes), UB_ALIGN) * static_cast<uint64_t>(UB_ALIGN);
}

uint32_t HostFrontGetPerOrLastValue(uint32_t value, uint32_t divisor)
{
    if (divisor == 0U) {
        return 0U;
    }
    return value <= divisor ? value : value % divisor;
}

uint32_t HostFrontAlignOneBlockByteCeil(uint64_t value)
{
    return static_cast<uint32_t>((value + 31U) / 32U * 32U);
}

uint32_t HostFrontSrcToDstPerLoopMaxRows(uint32_t aiv_num)
{
    const uint64_t reserved_bytes = static_cast<uint64_t>(kHostFrontSrcToDstAssistElems) * sizeof(float) +
                                    static_cast<uint64_t>(aiv_num) * kHostFrontSortAlignElement;
    if (AtlasA5::UB_SIZE <= reserved_bytes) {
        return 0U;
    }
    return static_cast<uint32_t>((AtlasA5::UB_SIZE - reserved_bytes) / (kHostFrontSortAlignElement * 2U) / 2U);
}

HostFrontSrcToDstTiling HostBuildFrontSrcToDstTiling(uint32_t route_elems, uint32_t aiv_num)
{
    HostFrontSrcToDstTiling tiling;
    if (route_elems == 0U || aiv_num == 0U) {
        return tiling;
    }
    const uint32_t per_loop_max_rows = HostFrontSrcToDstPerLoopMaxRows(aiv_num);
    tiling.per_core_rows = HostCeilDiv(route_elems, aiv_num);
    if (tiling.per_core_rows == 0U || per_loop_max_rows == 0U) {
        return tiling;
    }
    tiling.need_core_num = HostCeilDiv(route_elems, tiling.per_core_rows);
    tiling.last_core_rows = route_elems - tiling.per_core_rows * (tiling.need_core_num - 1U);
    if (per_loop_max_rows >= tiling.per_core_rows) {
        tiling.per_core_per_loop_rows = tiling.per_core_rows;
        tiling.per_core_last_loop_rows = tiling.per_core_rows;
    } else {
        tiling.per_core_per_loop_rows = per_loop_max_rows;
        const uint32_t loops = HostCeilDiv(tiling.per_core_rows, per_loop_max_rows);
        tiling.per_core_last_loop_rows = tiling.per_core_rows - (loops - 1U) * per_loop_max_rows;
    }
    if (per_loop_max_rows >= tiling.last_core_rows) {
        tiling.last_core_per_loop_rows = tiling.last_core_rows;
        tiling.last_core_last_loop_rows = tiling.last_core_rows;
    } else {
        tiling.last_core_per_loop_rows = per_loop_max_rows;
        const uint32_t loops = HostCeilDiv(tiling.last_core_rows, per_loop_max_rows);
        tiling.last_core_last_loop_rows = tiling.last_core_rows - (loops - 1U) * per_loop_max_rows;
    }
    return tiling;
}

uint32_t HostFrontExpertTokenOutExpertNumUbAlign(uint32_t expert_num)
{
    const uint32_t aligned_expert_num =
        HostCeilDiv(expert_num, kHostFrontInt32OneBlockElems) * kHostFrontInt32OneBlockElems;
    return std::min<uint32_t>(aligned_expert_num, kHostFrontMaxExpertNum);
}

HostFrontGatherQuantTiling HostBuildFrontGatherQuantTiling(uint32_t route_elems, uint32_t aiv_num, uint32_t k)
{
    HostFrontGatherQuantTiling tiling;
    tiling.activate_rows = route_elems;
    tiling.smooth_type = 0U;
    if (route_elems == 0U || aiv_num == 0U) {
        return tiling;
    }

    const uint32_t per_core_rows = HostCeilDiv(route_elems, aiv_num);
    if (per_core_rows == 0U) {
        return tiling;
    }
    tiling.need_core_num = HostCeilDiv(route_elems, per_core_rows);
    tiling.per_core_rows = per_core_rows;
    tiling.last_core_rows = route_elems - per_core_rows * (tiling.need_core_num - 1U);

    const uint64_t row_size = HostFrontAlignBytes(static_cast<uint64_t>(per_core_rows) * sizeof(int32_t)) * 4U;
    const uint64_t col_size =
        HostFrontAlignBytes(static_cast<uint64_t>(k) * sizeof(int8_t)) * kHostFrontDynamicQuantColsBuffer;
    const uint64_t scale_size = kHostFrontDynamicQuantScaleBytes;
    uint32_t once_row_size = 0U;
    if (AtlasA5::UB_SIZE > col_size + scale_size + 32U * 4U * 3U) {
        once_row_size =
            static_cast<uint32_t>((AtlasA5::UB_SIZE - col_size - scale_size - 32U * 4U * 3U) / (sizeof(int32_t) * 4U));
        once_row_size = once_row_size / kHostFrontInt32OneBlockElems * kHostFrontInt32OneBlockElems;
    }
    const bool if_one_loop = AtlasA5::UB_SIZE > col_size + scale_size + 32U * 4U * 4U && tiling.smooth_type == 0U &&
                             k == kHostFrontFullLoadHLimit;
    const uint32_t per_core_once_rows = if_one_loop ? std::min<uint32_t>(once_row_size, per_core_rows) : per_core_rows;
    const uint32_t last_core_once_rows =
        if_one_loop ? std::min<uint32_t>(once_row_size, tiling.last_core_rows) : tiling.last_core_rows;
    if (row_size + col_size + scale_size < AtlasA5::UB_SIZE || if_one_loop) {
        tiling.per_core_per_loop_rows = per_core_once_rows;
        tiling.per_core_last_loop_rows =
            if_one_loop ? HostFrontGetPerOrLastValue(per_core_rows, per_core_once_rows) : per_core_rows;
        tiling.last_core_per_loop_rows = last_core_once_rows;
        tiling.last_core_last_loop_rows = if_one_loop ?
                                              HostFrontGetPerOrLastValue(tiling.last_core_rows, last_core_once_rows) :
                                              tiling.last_core_rows;
        tiling.per_core_loops = if_one_loop ? HostCeilDiv(per_core_rows, per_core_once_rows) : 1U;
        tiling.last_core_loops = if_one_loop ? HostCeilDiv(tiling.last_core_rows, last_core_once_rows) : 1U;
        tiling.per_loop_cols = k;
        tiling.last_loop_cols = k;
        tiling.col_loops = 1U;
        return tiling;
    }

    uint32_t base_max_cols = 6144U;
    const uint64_t total_col_size =
        HostFrontAlignBytes(static_cast<uint64_t>(base_max_cols) * sizeof(int8_t)) * kHostFrontDynamicQuantColsBuffer;
    uint32_t base_per_loop_max_rows =
        HostFrontAlignOneBlockByteCeil((AtlasA5::UB_SIZE - total_col_size - scale_size) / sizeof(int32_t)) / 4U;
    if (k < 6144U) {
        base_per_loop_max_rows =
            HostFrontAlignOneBlockByteCeil((AtlasA5::UB_SIZE - col_size - scale_size) / sizeof(int32_t)) / 4U;
    } else if (per_core_rows < base_per_loop_max_rows) {
        base_max_cols =
            HostFrontAlignOneBlockByteCeil(AtlasA5::UB_SIZE - row_size - scale_size) / kHostFrontDynamicQuantColsBuffer;
    }
    tiling.per_loop_cols = std::min<uint32_t>(base_max_cols, k);
    tiling.last_loop_cols = HostFrontGetPerOrLastValue(k, base_max_cols);
    tiling.col_loops = base_max_cols == 0U ? 0U : HostCeilDiv(k, base_max_cols);
    tiling.per_core_per_loop_rows = std::min<uint32_t>(per_core_rows, base_per_loop_max_rows);
    tiling.per_core_last_loop_rows = HostFrontGetPerOrLastValue(per_core_rows, base_per_loop_max_rows);
    tiling.per_core_loops = base_per_loop_max_rows == 0U ? 0U : HostCeilDiv(per_core_rows, base_per_loop_max_rows);
    tiling.last_core_per_loop_rows = std::min<uint32_t>(tiling.last_core_rows, base_per_loop_max_rows);
    tiling.last_core_last_loop_rows = HostFrontGetPerOrLastValue(tiling.last_core_rows, base_per_loop_max_rows);
    tiling.last_core_loops =
        base_per_loop_max_rows == 0U ? 0U : HostCeilDiv(tiling.last_core_rows, base_per_loop_max_rows);
    return tiling;
}

uint64_t HostMc2FullLoadDynamicRequiredUbBytes(uint32_t route_elems, uint32_t k, uint32_t expert_num)
{
    constexpr uint64_t kOneCoreSortBuffer = 6U;
    constexpr uint64_t kOtherRouteBuffer = 3U;
    constexpr uint64_t kDynamicQuantFullLoadColsBuffer = 13U;
    constexpr uint64_t kScaleOutBytes = 64U;
    const uint64_t aligned_route_elems = HostFrontAlignBytes(route_elems);
    const uint64_t sort_space = aligned_route_elems * sizeof(int32_t) * kOneCoreSortBuffer;
    const uint64_t other_space = aligned_route_elems * sizeof(int32_t) * kOtherRouteBuffer;
    const uint64_t expert_space = HostFrontAlignBytes(static_cast<uint64_t>(expert_num) * sizeof(int32_t));
    const uint64_t quant_space = HostFrontAlignBytes(k) * kDynamicQuantFullLoadColsBuffer;
    return sort_space + other_space + expert_space + quant_space + kScaleOutBytes;
}

uint32_t HostFrontSortLen(uint32_t elem_count)
{
    return elem_count * 2U;
}

uint32_t HostFrontSortOffset(uint32_t elem_offset)
{
    return elem_offset * 2U;
}

uint32_t HostPow4Ceil(uint32_t value)
{
    uint32_t result = 1U;
    while (result < value && result <= std::numeric_limits<uint32_t>::max() / 4U) {
        result *= 4U;
    }
    return result < value ? value : result;
}

std::vector<uint16_t> BytesToU16(const std::vector<uint8_t> &bytes)
{
    if (bytes.size() % sizeof(uint16_t) != 0) {
        throw std::runtime_error("fp16 file size is not aligned");
    }
    std::vector<uint16_t> out(bytes.size() / sizeof(uint16_t));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

std::vector<int32_t> BytesToI32(const std::vector<uint8_t> &bytes)
{
    if (bytes.size() % sizeof(int32_t) != 0) {
        throw std::runtime_error("int32 file size is not aligned");
    }
    std::vector<int32_t> out(bytes.size() / sizeof(int32_t));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

std::vector<float> BytesToF32(const std::vector<uint8_t> &bytes)
{
    if (bytes.size() % sizeof(float) != 0) {
        throw std::runtime_error("float file size is not aligned");
    }
    std::vector<float> out(bytes.size() / sizeof(float));
    std::memcpy(out.data(), bytes.data(), bytes.size());
    return out;
}

float Bf16ToFloat(uint16_t value)
{
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t FloatToHalfTrunc(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16U) & 0x8000U);
    const uint32_t exponent = (bits >> 23U) & 0xFFU;
    const uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent == 0U) {
        return sign;
    }
    if (exponent == 0xFFU) {
        return static_cast<uint16_t>(sign | (mantissa == 0U ? 0x7C00U : 0x7E00U));
    }

    const int32_t half_exp = static_cast<int32_t>(exponent) - 127 + 15;
    if (half_exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    if (half_exp <= 0) {
        if (half_exp < -10) {
            return sign;
        }
        const uint32_t mantissa_with_hidden = mantissa | 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - half_exp);
        return static_cast<uint16_t>(sign | (mantissa_with_hidden >> shift));
    }
    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(half_exp) << 10U) |
                                 static_cast<uint16_t>(mantissa >> 13U));
}

uint32_t RoundRightShiftToEven(uint32_t value, uint32_t shift)
{
    if (shift == 0U) {
        return value;
    }
    if (shift >= 32U) {
        return 0;
    }
    const uint32_t truncated = value >> shift;
    const uint32_t remainder_mask = (1U << shift) - 1U;
    const uint32_t remainder = value & remainder_mask;
    const uint32_t halfway = 1U << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (truncated & 1U) != 0U)) {
        return truncated + 1U;
    }
    return truncated;
}

uint16_t FloatToHalfRoundToNearestEven(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16U) & 0x8000U);
    const uint32_t exponent = (bits >> 23U) & 0xFFU;
    const uint32_t mantissa = bits & 0x7FFFFFU;
    if (exponent == 0U) {
        return sign;
    }
    if (exponent == 0xFFU) {
        return static_cast<uint16_t>(sign | (mantissa == 0U ? 0x7C00U : 0x7E00U));
    }

    int32_t half_exp = static_cast<int32_t>(exponent) - 127 + 15;
    if (half_exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    if (half_exp <= 0) {
        const uint32_t mantissa_with_hidden = mantissa | 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - half_exp);
        const uint32_t rounded = RoundRightShiftToEven(mantissa_with_hidden, shift);
        return static_cast<uint16_t>(sign | static_cast<uint16_t>(rounded));
    }

    uint32_t rounded_mantissa = RoundRightShiftToEven(mantissa, 13U);
    if (rounded_mantissa == 0x400U) {
        rounded_mantissa = 0;
        ++half_exp;
        if (half_exp >= 31) {
            return static_cast<uint16_t>(sign | 0x7C00U);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint16_t>(half_exp) << 10U) |
                                 static_cast<uint16_t>(rounded_mantissa));
}

int32_t HalfOrderedValue(uint16_t value)
{
    if ((value & 0x8000U) != 0U) {
        return static_cast<int32_t>(0x8000U - static_cast<uint32_t>(value & 0x7FFFU));
    }
    return static_cast<int32_t>(0x8000U + static_cast<uint32_t>(value));
}

uint32_t HalfUlpDistance(uint16_t expected, uint16_t actual)
{
    return static_cast<uint32_t>(std::abs(HalfOrderedValue(expected) - HalfOrderedValue(actual)));
}

int8_t RoundHalfAwayToInt8(float value)
{
    float rounded = value >= 0.0f ? std::floor(value + 0.5f) : std::ceil(value - 0.5f);
    rounded = std::max(-128.0f, std::min(127.0f, rounded));
    return static_cast<int8_t>(rounded);
}

int8_t RoundHalfUpToInt8(float value)
{
    const float clamped = std::max(-128.0f, std::min(127.0f, value));
    return static_cast<int8_t>(std::floor(clamped + 0.5f));
}

int8_t RoundNearestEvenToInt8(float value)
{
    const float clamped = std::max(-128.0f, std::min(127.0f, value));
    const float lower_float = std::floor(clamped);
    const float frac = clamped - lower_float;
    int32_t rounded = static_cast<int32_t>(lower_float);
    if (frac > 0.5f || (frac == 0.5f && (rounded % 2) != 0)) {
        ++rounded;
    }
    return static_cast<int8_t>(std::max(-128, std::min(127, rounded)));
}

int8_t QuantizeFp32ViaHalfCastRint(float value)
{
    const float half_value = Fp16ToFloat(FloatToHalfRoundToNearestEven(value));
    return RoundNearestEvenToInt8(half_value);
}

std::vector<int8_t> QuantizeBf16RowLikeV3(const std::vector<uint16_t> &x, uint32_t token, uint32_t k, float &scale)
{
    std::vector<int8_t> quant(k, 0);
    float max_abs = 0.0f;
    const size_t row_offset = static_cast<size_t>(token) * k;
    for (uint32_t col = 0; col < k; ++col) {
        max_abs = std::max(max_abs, std::fabs(Bf16ToFloat(x[row_offset + col])));
    }

    scale = max_abs / 127.0f;
    if (scale == 0.0f) {
        scale = 1.0e-6f / 127.0f;
    }
    for (uint32_t col = 0; col < k; ++col) {
        const float divided = Bf16ToFloat(x[row_offset + col]) / scale;
        quant[col] = RoundHalfAwayToInt8(Fp16ToFloat(FloatToHalfTrunc(divided)));
    }
    return quant;
}

int ParseEnvInt(const char *name, int default_value)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    try {
        return std::stoi(value);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string("invalid integer in env: ") + name);
    }
}

uint64_t AlignUpU64(uint64_t value, uint64_t align)
{
    return (value + align - 1U) / align * align;
}

double SysCntTicksToUs(uint64_t ticks, double sys_cnt_multiple_ns)
{
    return static_cast<double>(ticks) * sys_cnt_multiple_ns / 1000.0;
}

const char *StageProfileName(size_t stage)
{
    static constexpr const char *kNames[kDispatchFFNCombineProfileStageCount] = {
        "front", "dispatch", "gmm1", "swiglu", "gmm2", "combine", "unpermute"};
    return stage < kDispatchFFNCombineProfileStageCount ? kNames[stage] : "unknown";
}

const char *FrontDoneProfileName(size_t step)
{
    static constexpr const char *kNames[kDispatchFFNCombineProfileFrontDoneStepCount] = {
        "step7_entrySync", "step8_notify", "step9_wait", "step10_exitSync"};
    return step < kDispatchFFNCombineProfileFrontDoneStepCount ? kNames[step] : "unknown";
}

const char *CombineDetailProfileName(size_t detail)
{
    static constexpr const char *kNames[kDispatchFFNCombineProfileCombineDetailCount] = {
        "directSmallTotal",   "smallGroup",           "smallTile",          "smallWaitGmm2",
        "smallDequantTotal",  "smallDequantScaleMul", "smallRankIntersect", "directLargeTotal",
        "largeGroup",         "largeWaitGmm2Flag",    "largeWaitGmm2Sync",  "largeSegment",
        "largeRow",           "largeStore",           "finalBoundary",      "finalSync",
        "finalCrossRankSync", "combineCvPopWait",    "combineCvScaleLoad", "combineCvFree",
    };
    return detail < kDispatchFFNCombineProfileCombineDetailCount ? kNames[detail] : "unknown";
}

const char *FrontDetailProfileName(size_t detail)
{
    static constexpr const char *kNames[kDispatchFFNCombineProfileFrontDetailCount] = {
        "fullLoadTotal",
        "singleTotal",
        "multiTotal",
        "fullLoadSort",
        "fullLoadCopyExpanded",
        "fullLoadCount",
        "producerBoundary",
        "singleSort",
        "sortVbs",
        "sortLocalMerge",
        "sortMiddleMerge",
        "sortOut",
        "metadata",
        "srcToDst",
        "quant",
        "exchangeCounts",
        "exchangePrepare",
        "exchangePublish",
        "exchangeWaitRestore",
        "cumsum",
        "doneNotify",
        "routeSync",
        "exchangePrePublishSync",
        "exchangeRestoreTotal",
        "exchangeWaitMarker",
        "exchangeRestoreCopy",
        "exchangePreSum",
        "exchangeStorePreSum",
        "exchangePostRestoreSync",
        "exchangeFinalDcci",
        "cumsumInit",
        "cumsumRows",
        "cumsumExpertNums",
        "cumsumFinalSync",
    };
    return detail < kDispatchFFNCombineProfileFrontDetailCount ? kNames[detail] : "unknown";
}

#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
const char *Gmm2CvDetailProfileName(size_t detail)
{
    static constexpr const char *kNames[kDispatchFFNCombineProfileGmm2CvDetailCount] = {
        "gmm2CvStore",
    };
    return detail < kDispatchFFNCombineProfileGmm2CvDetailCount ? kNames[detail] : "unknown";
}
#endif

const char *DispatchDetailProfileName(size_t detail)
{
    static constexpr const char *kNames[kDispatchFFNCombineProfileDispatchDetailCount] = {
        "total",      "setupFlags",  "initialReady", "prevSumInit", "group",           "groupMetadata",
        "sourceRank", "fetchTotal",  "fetchBatch",   "fetchLoad",   "fetchStore",      "storeWait",
        "storeRows",  "storeScales", "groupSync",    "groupReady",  "finalWaitEvents", "finalSync",
    };
    return detail < kDispatchFFNCombineProfileDispatchDetailCount ? kNames[detail] : "unknown";
}

const char *UnpermuteDetailProfileName(size_t detail)
{
    static constexpr const char *kNames[kDispatchFFNCombineProfileUnpermuteDetailCount] = {
        "total",        "setFlags",     "metadataPrefetch", "metadataWait", "batch",     "token",     "fillAcc",
        "topkLoop",     "loadD",        "accumulate",       "storeOutput",  "storeWait", "storeCast", "storeTstore",
        "finalizePipe", "idleFinalize", "debugDone",
    };
    return detail < kDispatchFFNCombineProfileUnpermuteDetailCount ? kNames[detail] : "unknown";
}

size_t ReadyStageProfileStage(size_t ready_stage)
{
    switch (ready_stage) {
        case DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_GMM1:
            return DISPATCH_FFN_COMBINE_PROFILE_STAGE_GMM1;
        case DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_SWIGLU:
            return DISPATCH_FFN_COMBINE_PROFILE_STAGE_SWIGLU;
        case DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_GMM2:
            return DISPATCH_FFN_COMBINE_PROFILE_STAGE_GMM2;
        case DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_COMBINE:
            return DISPATCH_FFN_COMBINE_PROFILE_STAGE_COMBINE;
        default:
            return kDispatchFFNCombineProfileStageCount;
    }
}

size_t ReadyStageForProfileStage(size_t stage)
{
    switch (stage) {
        case DISPATCH_FFN_COMBINE_PROFILE_STAGE_GMM1:
            return DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_GMM1;
        case DISPATCH_FFN_COMBINE_PROFILE_STAGE_SWIGLU:
            return DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_SWIGLU;
        case DISPATCH_FFN_COMBINE_PROFILE_STAGE_GMM2:
            return DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_GMM2;
        case DISPATCH_FFN_COMBINE_PROFILE_STAGE_COMBINE:
            return DISPATCH_FFN_COMBINE_PROFILE_READY_STAGE_COMBINE;
        default:
            return kDispatchFFNCombineProfileReadyStageCount;
    }
}

const char *ProfileEntryKind(size_t profile_idx)
{
    return profile_idx == 0U ? "aic" : "aiv";
}

uint32_t ProfileEntrySubblock(size_t profile_idx)
{
    return profile_idx == 0U ? 0U : static_cast<uint32_t>(profile_idx - 1U);
}

bool StageProfileEntryParticipates(size_t stage, size_t profile_idx)
{
    const bool is_aic = profile_idx == 0U;
    switch (stage) {
        case DISPATCH_FFN_COMBINE_PROFILE_STAGE_GMM1:
        case DISPATCH_FFN_COMBINE_PROFILE_STAGE_GMM2:
            return is_aic;
        case DISPATCH_FFN_COMBINE_PROFILE_STAGE_FRONT:
        case DISPATCH_FFN_COMBINE_PROFILE_STAGE_DISPATCH:
        case DISPATCH_FFN_COMBINE_PROFILE_STAGE_SWIGLU:
        case DISPATCH_FFN_COMBINE_PROFILE_STAGE_COMBINE:
        case DISPATCH_FFN_COMBINE_PROFILE_STAGE_UNPERMUTE:
            return !is_aic;
        default:
            return false;
    }
}

struct StageProfileEnvelope {
    bool valid = false;
    uint32_t active_entries = 0;
    uint64_t start_min = 0;
    uint64_t end_max = 0;
    uint64_t max_core_ticks = 0;
};

struct CombineDetailProfileEnvelope {
    bool valid = false;
    uint32_t active_entries = 0;
    uint64_t start_min = 0;
    uint64_t end_max = 0;
    uint64_t max_core_ticks = 0;
    uint64_t sum_core_ticks = 0;
    uint64_t call_count = 0;
    uint64_t max_call_count = 0;
};

struct Gmm1DetailProfileEnvelope {
    bool valid = false;
    uint32_t active_entries = 0;
    uint64_t start_min = 0;
    uint64_t end_max = 0;
    uint64_t max_core_ticks = 0;
    uint64_t sum_core_ticks = 0;
    uint64_t l1_tile_count = 0;
    uint64_t max_l1_tile_count = 0;
};

#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
struct DispatchV2CTraceProfileEnvelope {
    bool valid = false;
    bool mixed_flag = false;
    uint32_t active_entries = 0;
    uint64_t start_min = 0;
    uint64_t end_min = 0;
    uint64_t end_max = 0;
    uint64_t max_core_ticks = 0;
    uint64_t sum_core_ticks = 0;
    uint64_t call_count = 0;
    uint64_t max_call_count = 0;
    uint64_t flag_id = 0;
};

struct ExpertWallProfile {
    bool valid = false;
    uint64_t start = 0;
    uint64_t end = 0;
};
#endif

void UpdateProfileEnvelope(StageProfileEnvelope &env, uint64_t start, uint64_t end)
{
    if (!env.valid) {
        env.valid = true;
        env.start_min = start;
        env.end_max = end;
    } else {
        env.start_min = std::min(env.start_min, start);
        env.end_max = std::max(env.end_max, end);
    }
    env.active_entries += 1U;
    env.max_core_ticks = std::max(env.max_core_ticks, end - start);
}

void UpdateCombineDetailEnvelope(CombineDetailProfileEnvelope &env, uint64_t start, uint64_t end, uint64_t total,
                                 uint64_t count)
{
    if (!env.valid) {
        env.valid = true;
        env.start_min = start;
        env.end_max = end;
    } else {
        env.start_min = std::min(env.start_min, start);
        env.end_max = std::max(env.end_max, end);
    }
    env.active_entries += 1U;
    env.max_core_ticks = std::max(env.max_core_ticks, total);
    env.sum_core_ticks += total;
    env.call_count += count;
    env.max_call_count = std::max(env.max_call_count, count);
}

void UpdateGmm1DetailEnvelope(Gmm1DetailProfileEnvelope &env, uint64_t start, uint64_t end, uint64_t total,
                              uint64_t l1_tile_count)
{
    if (!env.valid) {
        env.valid = true;
        env.start_min = start;
        env.end_max = end;
    } else {
        env.start_min = std::min(env.start_min, start);
        env.end_max = std::max(env.end_max, end);
    }
    env.active_entries += 1U;
    env.max_core_ticks = std::max(env.max_core_ticks, total);
    env.sum_core_ticks += total;
    env.l1_tile_count += l1_tile_count;
    env.max_l1_tile_count = std::max(env.max_l1_tile_count, l1_tile_count);
}

#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
void UpdateDispatchV2CTraceEnvelope(DispatchV2CTraceProfileEnvelope &env, uint64_t start, uint64_t end, uint64_t flagId,
                                    uint64_t count)
{
    if (!env.valid) {
        env.valid = true;
        env.start_min = start;
        env.end_min = end;
        env.end_max = end;
        env.flag_id = flagId;
    } else {
        env.start_min = std::min(env.start_min, start);
        env.end_min = std::min(env.end_min, end);
        env.end_max = std::max(env.end_max, end);
        if (env.flag_id != flagId) {
            env.mixed_flag = true;
        }
    }
    env.active_entries += 1U;
    env.max_core_ticks = std::max(env.max_core_ticks, end - start);
    env.sum_core_ticks += end - start;
    env.call_count += count;
    env.max_call_count = std::max(env.max_call_count, count);
}
#endif

bool ProfileIntervalValid(const uint64_t *entry, size_t stage)
{
    const uint64_t start = entry[DispatchFFNCombineProfileStageStartIndex(stage)];
    const uint64_t end = entry[DispatchFFNCombineProfileStageEndIndex(stage)];
    return start != 0U && end != 0U && end >= start;
}

bool CombineDetailProfileValid(const uint64_t *entry, size_t detail)
{
    const uint64_t start = entry[DispatchFFNCombineProfileCombineDetailStartIndex(detail)];
    const uint64_t end = entry[DispatchFFNCombineProfileCombineDetailEndIndex(detail)];
    const uint64_t count = entry[DispatchFFNCombineProfileCombineDetailCountIndex(detail)];
    return count != 0U && start != 0U && end != 0U && end >= start;
}

bool FrontDetailProfileValid(const uint64_t *entry, size_t detail)
{
    const uint64_t start = entry[DispatchFFNCombineProfileFrontDetailStartIndex(detail)];
    const uint64_t end = entry[DispatchFFNCombineProfileFrontDetailEndIndex(detail)];
    const uint64_t count = entry[DispatchFFNCombineProfileFrontDetailCountIndex(detail)];
    return count != 0U && start != 0U && end != 0U && end >= start;
}

bool DispatchDetailProfileValid(const uint64_t *entry, size_t detail)
{
    const uint64_t start = entry[DispatchFFNCombineProfileDispatchDetailStartIndex(detail)];
    const uint64_t end = entry[DispatchFFNCombineProfileDispatchDetailEndIndex(detail)];
    const uint64_t count = entry[DispatchFFNCombineProfileDispatchDetailCountIndex(detail)];
    return count != 0U && start != 0U && end != 0U && end >= start;
}

bool UnpermuteDetailProfileValid(const uint64_t *entry, size_t detail)
{
    const uint64_t start = entry[DispatchFFNCombineProfileUnpermuteDetailStartIndex(detail)];
    const uint64_t end = entry[DispatchFFNCombineProfileUnpermuteDetailEndIndex(detail)];
    const uint64_t count = entry[DispatchFFNCombineProfileUnpermuteDetailCountIndex(detail)];
    return count != 0U && start != 0U && end != 0U && end >= start;
}

bool Gmm1DetailProfileValid(const uint64_t *entry, size_t expert)
{
    const uint64_t start = entry[DispatchFFNCombineProfileGmm1DetailStartIndex(expert)];
    const uint64_t end = entry[DispatchFFNCombineProfileGmm1DetailEndIndex(expert)];
    const uint64_t count = entry[DispatchFFNCombineProfileGmm1DetailCountIndex(expert)];
    return count != 0U && start != 0U && end != 0U && end >= start;
}

#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
bool DispatchV2CTraceProfileValid(const uint64_t *entry, size_t event)
{
    const uint64_t start = entry[DispatchFFNCombineProfileDispatchV2CTraceStartIndex(event)];
    const uint64_t end = entry[DispatchFFNCombineProfileDispatchV2CTraceEndIndex(event)];
    const uint64_t count = entry[DispatchFFNCombineProfileDispatchV2CTraceCountIndex(event)];
    return count != 0U && start != 0U && end != 0U && end >= start;
}

bool Gmm2ToCombineTraceProfileValid(const uint64_t *entry, size_t event)
{
    const uint64_t start = entry[DispatchFFNCombineProfileGmm2ToCombineTraceStartIndex(event)];
    const uint64_t end = entry[DispatchFFNCombineProfileGmm2ToCombineTraceEndIndex(event)];
    const uint64_t count = entry[DispatchFFNCombineProfileGmm2ToCombineTraceCountIndex(event)];
    return count != 0U && start != 0U && end != 0U && end >= start;
}

bool Gmm2CvDetailProfileValid(const uint64_t *entry, size_t detail)
{
    const uint64_t start = entry[DispatchFFNCombineProfileGmm2CvDetailStartIndex(detail)];
    const uint64_t end = entry[DispatchFFNCombineProfileGmm2CvDetailEndIndex(detail)];
    const uint64_t count = entry[DispatchFFNCombineProfileGmm2CvDetailCountIndex(detail)];
    return count != 0U && start != 0U && end != 0U && end >= start;
}

bool CombineExpertWallProfileValid(const uint64_t *entry, size_t expert)
{
    const uint64_t start = entry[DispatchFFNCombineProfileCombineExpertWallStartIndex(expert)];
    const uint64_t end = entry[DispatchFFNCombineProfileCombineExpertWallEndIndex(expert)];
    return start != 0U && end != 0U && end >= start;
}

bool Gmm2ExpertWallProfileValid(const uint64_t *entry, size_t expert)
{
    const uint64_t start = entry[DispatchFFNCombineProfileGmm2ExpertWallStartIndex(expert)];
    const uint64_t end = entry[DispatchFFNCombineProfileGmm2ExpertWallEndIndex(expert)];
    return start != 0U && end != 0U && end >= start;
}
#endif

bool FrontDoneProfileIntervalValid(const uint64_t *entry, size_t step)
{
    const uint64_t start = entry[DispatchFFNCombineProfileFrontDoneStartIndex(step)];
    const uint64_t end = entry[DispatchFFNCombineProfileFrontDoneEndIndex(step)];
    return start != 0U && end != 0U && end >= start;
}

bool FrontDoneProfileTotalValid(const uint64_t *entry)
{
    const uint64_t start =
        entry[DispatchFFNCombineProfileFrontDoneStartIndex(DISPATCH_FFN_COMBINE_PROFILE_FRONT_DONE_STEP7_ENTRY_SYNC)];
    const uint64_t end =
        entry[DispatchFFNCombineProfileFrontDoneEndIndex(DISPATCH_FFN_COMBINE_PROFILE_FRONT_DONE_STEP10_EXIT_SYNC)];
    return start != 0U && end != 0U && end >= start;
}

bool ReadyStageProfileIntervalValid(const uint64_t *entry, size_t ready_stage)
{
    const size_t stage = ReadyStageProfileStage(ready_stage);
    if (stage >= kDispatchFFNCombineProfileStageCount) {
        return false;
    }
    const uint64_t start = entry[DispatchFFNCombineProfileReadyStageStartIndex(ready_stage)];
    const uint64_t end = entry[DispatchFFNCombineProfileStageEndIndex(stage)];
    return start != 0U && end != 0U && end >= start;
}

void PrintOrderedByRank(int rank_id, int world_size, const std::string &text)
{
    for (int turn = 0; turn < world_size; ++turn) {
        CommMpiBarrier();
        if (turn == rank_id) {
            std::cout << text << std::endl;
        }
    }
    CommMpiBarrier();
}

struct CoreCountReport {
    bool pass = true;
    size_t mismatch_count = 0;
    size_t first_mismatch = 0;
    size_t actual_nonzero_count = 0;
    size_t first_actual_nonzero = 0;
    int64_t actual_sum = 0;
    int64_t expected_sum = 0;
    int32_t expected = 0;
    int32_t actual = 0;
};

struct ExpectedFrontRoute {
    std::vector<int32_t> expanded_row_idx;
    std::vector<int32_t> dst_to_src_route;
    std::vector<int32_t> local_token_per_expert;
    std::vector<int32_t> expert_base;
    size_t routed_rows = 0;
};

struct ExpectedFrontRouteEntry {
    uint32_t expert = 0;
    uint32_t src_route = 0;
};

ExpectedFrontRoute BuildExpectedFrontRoute(const std::vector<uint8_t> &expert_idx, size_t route_elems,
                                           uint32_t expert_num, uint32_t expert_num_aligned)
{
    const auto expert_idx_i32 = BytesToI32(expert_idx);
    if (expert_idx_i32.size() < route_elems) {
        throw std::runtime_error("expert_idx is smaller than small route elems");
    }

    ExpectedFrontRoute expected;
    expected.expanded_row_idx.assign(route_elems, -1);
    expected.dst_to_src_route.assign(route_elems, -1);
    expected.local_token_per_expert.assign(expert_num_aligned, 0);
    expected.expert_base.assign(expert_num_aligned, 0);

    std::vector<ExpectedFrontRouteEntry> valid_routes;
    valid_routes.reserve(route_elems);
    for (size_t src = 0; src < route_elems; ++src) {
        const int32_t expert_value = expert_idx_i32[src];
        if (expert_value < 0 || static_cast<uint32_t>(expert_value) >= expert_num) {
            continue;
        }
        const uint32_t expert = static_cast<uint32_t>(expert_value);
        if (expert >= expert_num_aligned) {
            throw std::runtime_error("expertNumAligned is smaller than expertNum");
        }
        ++expected.local_token_per_expert[expert];
        valid_routes.push_back(ExpectedFrontRouteEntry{
            expert,
            static_cast<uint32_t>(src),
        });
    }

    std::stable_sort(
        valid_routes.begin(), valid_routes.end(),
        [](const ExpectedFrontRouteEntry &lhs, const ExpectedFrontRouteEntry &rhs) { return lhs.expert < rhs.expert; });

    for (size_t dst = 0; dst < valid_routes.size(); ++dst) {
        const uint32_t src = valid_routes[dst].src_route;
        expected.dst_to_src_route[dst] = static_cast<int32_t>(src);
        expected.expanded_row_idx[src] = static_cast<int32_t>(dst);
    }
    expected.routed_rows = valid_routes.size();

    int32_t expert_base = 0;
    for (uint32_t expert = 0; expert < expert_num_aligned; ++expert) {
        expected.expert_base[expert] = expert_base;
        if (expert < expert_num) {
            expert_base += expected.local_token_per_expert[expert];
        }
    }
    return expected;
}

ExpectedFrontRoute BuildExpectedFrontRouteRun(const std::vector<uint8_t> &expert_idx, size_t route_start,
                                              size_t route_elems, uint32_t expert_num)
{
    const auto expert_idx_i32 = BytesToI32(expert_idx);
    if (route_start + route_elems > expert_idx_i32.size()) {
        throw std::runtime_error("expert_idx is smaller than route run range");
    }

    ExpectedFrontRoute expected;
    expected.dst_to_src_route.assign(route_elems, -1);
    expected.local_token_per_expert.assign(expert_num + 1U, 0);

    std::vector<ExpectedFrontRouteEntry> routes;
    routes.reserve(route_elems);
    for (size_t idx = 0; idx < route_elems; ++idx) {
        const size_t src = route_start + idx;
        const int32_t expert_value = expert_idx_i32[src];
        const uint32_t expert = (expert_value < 0 || static_cast<uint32_t>(expert_value) >= expert_num) ?
                                    expert_num :
                                    static_cast<uint32_t>(expert_value);
        ++expected.local_token_per_expert[expert];
        routes.push_back(ExpectedFrontRouteEntry{
            expert,
            static_cast<uint32_t>(src),
        });
    }

    std::stable_sort(
        routes.begin(), routes.end(),
        [](const ExpectedFrontRouteEntry &lhs, const ExpectedFrontRouteEntry &rhs) { return lhs.expert < rhs.expert; });

    expected.expanded_row_idx.assign(route_elems, 0);
    for (size_t idx = 0; idx < routes.size(); ++idx) {
        expected.expanded_row_idx[idx] = static_cast<int32_t>(routes[idx].expert);
        expected.dst_to_src_route[idx] = static_cast<int32_t>(routes[idx].src_route);
    }
    expected.routed_rows = routes.size();
    return expected;
}

std::vector<int32_t> BuildExpectedCoreCount(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                            const std::vector<uint8_t> &expert_idx)
{
    const auto expert_idx_i32 = BytesToI32(expert_idx);
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const uint32_t aiv_num = cfg.aiv_num;
    const uint32_t expert_num = front.expertNum;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const size_t elem_num = static_cast<size_t>(aiv_num) * expert_num_aligned;
    std::vector<int32_t> expected(elem_num, 0);
    for (uint32_t core = 0; core < aiv_num; ++core) {
        const uint32_t base = cfg.m / aiv_num;
        const uint32_t rem = cfg.m % aiv_num;
        const uint32_t token_start = core * base + (core < rem ? core : rem);
        const uint32_t token_len = base + (core < rem ? 1U : 0U);
        const uint32_t token_end = token_start + token_len;
        for (uint32_t token = token_start; token < token_end; ++token) {
            for (uint32_t k_idx = 0; k_idx < cfg.topk; ++k_idx) {
                const int32_t expert = expert_idx_i32[static_cast<size_t>(token) * cfg.topk + k_idx];
                if (expert >= 0 && static_cast<uint32_t>(expert) < expert_num) {
                    ++expected[static_cast<size_t>(core) * expert_num_aligned + static_cast<uint32_t>(expert)];
                }
            }
        }
    }
    return expected;
}

std::vector<int32_t> BuildExpectedFrontSortedCoreCount(const CaseConfig &cfg,
                                                       const DispatchFFNCombineBuildResult &build,
                                                       const std::vector<uint8_t> &expert_idx,
                                                       const ExpectedFrontRoute &expected_route)
{
    const auto expert_idx_i32 = BytesToI32(expert_idx);
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const uint32_t aiv_num = cfg.aiv_num;
    const uint32_t expert_num = front.expertNum;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const size_t elem_num = static_cast<size_t>(aiv_num) * expert_num_aligned;
    std::vector<int32_t> expected(elem_num, 0);
    if (aiv_num == 0U) {
        return expected;
    }

    for (uint32_t core = 0; core < aiv_num; ++core) {
        const uint32_t base = front.routeElems / aiv_num;
        const uint32_t rem = front.routeElems - base * aiv_num;
        const uint32_t row_start = core * base + (core < rem ? core : rem);
        uint32_t row_end = row_start + base + (core < rem ? 1U : 0U);
        if (row_end > front.routeElems) {
            row_end = front.routeElems;
        }
        for (uint32_t dst = row_start; dst < row_end; ++dst) {
            const int32_t src_route = expected_route.dst_to_src_route[dst];
            if (src_route < 0 || static_cast<size_t>(src_route) >= expert_idx_i32.size()) {
                continue;
            }
            const int32_t expert = expert_idx_i32[static_cast<size_t>(src_route)];
            if (expert >= 0 && static_cast<uint32_t>(expert) < expert_num) {
                ++expected[static_cast<size_t>(core) * expert_num_aligned + static_cast<uint32_t>(expert)];
            }
        }
    }
    return expected;
}

CoreCountReport CheckCoreCount(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                               const DeviceBuffer &workspace_dev, const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    if (front.frontPath == 1U) {
        return CoreCountReport{};
    }
    const size_t elem_num = static_cast<size_t>(cfg.aiv_num) * front.expertNumAligned;
    const size_t bytes = elem_num * sizeof(int32_t);

    std::vector<int32_t> actual(elem_num);
    const auto *device_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual.data(), bytes, device_base + front.coreCountOffset, bytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS) {
        throw std::runtime_error("device->host coreCount copy failed");
    }

    const std::vector<int32_t> expected = BuildExpectedCoreCount(cfg, build, expert_idx);

    CoreCountReport report;
    for (size_t idx = 0; idx < elem_num; ++idx) {
        report.actual_sum += actual[idx];
        report.expected_sum += expected[idx];
        if (actual[idx] != 0) {
            if (report.actual_nonzero_count == 0) {
                report.first_actual_nonzero = idx;
            }
            ++report.actual_nonzero_count;
        }
        if (actual[idx] != expected[idx]) {
            if (report.mismatch_count == 0) {
                report.first_mismatch = idx;
                report.expected = expected[idx];
                report.actual = actual[idx];
            }
            ++report.mismatch_count;
        }
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

struct BaseCursorReport {
    bool pass = true;
    size_t mismatch_count = 0;
    std::string first_table;
    size_t first_idx = 0;
    int32_t expected = 0;
    int32_t actual = 0;
};

void UpdateBaseCursorReport(BaseCursorReport &report, const std::string &table, size_t idx, int32_t expected,
                            int32_t actual)
{
    if (actual == expected) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_table = table;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

BaseCursorReport CheckBaseCursor(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                 const DeviceBuffer &workspace_dev, const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const uint32_t aiv_num = cfg.aiv_num;
    const uint32_t expert_num = front.expertNum;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const size_t table_elems = expert_num_aligned;
    const size_t table_bytes = table_elems * sizeof(int32_t);
    const size_t core_table_elems = static_cast<size_t>(aiv_num) * expert_num_aligned;
    const size_t core_table_bytes = core_table_elems * sizeof(int32_t);
    const auto *device_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    const bool small_front = front.frontPath == 1U;

    std::vector<int32_t> actual_local(table_elems);
    std::vector<int32_t> actual_expert_base(table_elems);
    std::vector<int32_t> actual_core_base(core_table_elems);
    if (aclrtMemcpy(actual_local.data(), table_bytes, device_base + front.localTokenPerExpertOffset, table_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(actual_expert_base.data(), table_bytes, device_base + front.expertBaseOffset, table_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host base/cursor tables copy failed");
    }
    if (!small_front && aclrtMemcpy(actual_core_base.data(), core_table_bytes, device_base + front.coreBaseOffset,
                                    core_table_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host coreBase table copy failed");
    }

    std::vector<int32_t> expected_local(table_elems, 0);
    std::vector<int32_t> expected_expert_base(table_elems, 0);
    std::vector<int32_t> expected_core_base(core_table_elems, 0);

    if (small_front) {
        const ExpectedFrontRoute expected_route =
            BuildExpectedFrontRoute(expert_idx, static_cast<size_t>(cfg.m) * cfg.topk, expert_num, expert_num_aligned);
        expected_local = expected_route.local_token_per_expert;
        expected_expert_base = expected_route.expert_base;
    } else {
        const std::vector<int32_t> expected_core_count = BuildExpectedCoreCount(cfg, build, expert_idx);
        int32_t expert_base = 0;
        for (uint32_t expert = 0; expert < expert_num_aligned; ++expert) {
            int32_t local_token = 0;
            for (uint32_t core = 0; core < aiv_num; ++core) {
                if (expert < expert_num) {
                    local_token += expected_core_count[static_cast<size_t>(core) * expert_num_aligned + expert];
                }
            }
            expected_local[expert] = local_token;
            expected_expert_base[expert] = expert_base;

            int32_t prefix_before_core = 0;
            for (uint32_t core = 0; core < aiv_num; ++core) {
                expected_core_base[static_cast<size_t>(core) * expert_num_aligned + expert] =
                    expert_base + prefix_before_core;
                if (expert < expert_num) {
                    prefix_before_core += expected_core_count[static_cast<size_t>(core) * expert_num_aligned + expert];
                }
            }
            expert_base += local_token;
        }
    }

    BaseCursorReport report;
    for (size_t idx = 0; idx < table_elems; ++idx) {
        UpdateBaseCursorReport(report, "localTokenPerExpert", idx, expected_local[idx], actual_local[idx]);
        UpdateBaseCursorReport(report, "expertBase", idx, expected_expert_base[idx], actual_expert_base[idx]);
    }
    if (!small_front) {
        for (size_t idx = 0; idx < core_table_elems; ++idx) {
            UpdateBaseCursorReport(report, "coreBase", idx, expected_core_base[idx], actual_core_base[idx]);
        }
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

struct FrontSortReport {
    bool pass = true;
    uint32_t active_workers = 0;
    uint32_t subblock0_workers = 0;
    uint32_t subblock1_workers = 0;
    uint32_t route_elems = 0;
    uint32_t aligned_route_elems = 0;
    size_t sorted_key_mismatch_count = 0;
    size_t dst_to_src_mismatch_count = 0;
    size_t expanded_mismatch_count = 0;
    size_t count_mismatch_count = 0;
    size_t first_worker = 0;
    size_t first_idx = 0;
    std::string first_table;
    int32_t expected = 0;
    int32_t actual = 0;
};

void UpdateFrontSortReport(FrontSortReport &report, const std::string &table, size_t worker, size_t idx,
                           int32_t expected, int32_t actual, size_t &mismatch_count)
{
    if (actual == expected) {
        return;
    }
    if (report.first_table.empty()) {
        report.first_table = table;
        report.first_worker = worker;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++mismatch_count;
}

FrontSortReport CheckSmallFrontSort(const DispatchFFNCombineBuildResult &build, const DeviceBuffer &workspace_dev,
                                    const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    FrontSortReport report;
    report.active_workers = frontDebug.smallFrontNeedCoreNum;
    report.subblock0_workers = std::min(static_cast<uint32_t>(frontDebug.smallFrontNeedCoreNum), build.block_dim);
    report.subblock1_workers = frontDebug.smallFrontNeedCoreNum - report.subblock0_workers;
    report.route_elems = front.smallFrontRouteElems;
    report.aligned_route_elems = front.smallFrontAlignedRouteElems;
    if (front.frontPath != 1U || frontDebug.smallFrontMode == 0U || frontDebug.smallFrontDebugMode == 0U ||
        front.stageNum < 3U || frontDebug.smallFrontDebugBytes == 0U ||
        frontDebug.smallFrontDebugBytesPerWorker == 0U || front.smallFrontRouteElems == 0U ||
        front.smallFrontAlignedRouteElems == 0U) {
        return report;
    }

    const size_t route_elems = front.smallFrontRouteElems;
    const size_t aligned_route_elems = front.smallFrontAlignedRouteElems;
    const size_t active_workers = frontDebug.smallFrontNeedCoreNum;
    const size_t worker_debug_bytes = static_cast<size_t>(frontDebug.smallFrontDebugBytesPerWorker);
    const size_t debug_bytes = active_workers * worker_debug_bytes;
    if (debug_bytes > frontDebug.smallFrontDebugBytes ||
        frontDebug.smallFrontDebugOffset + debug_bytes > static_cast<uint64_t>(workspace_dev.bytes)) {
        throw std::runtime_error("frontSort debug workspace is out of range");
    }
    const size_t worker_debug_elems = worker_debug_bytes / sizeof(int32_t);
    const size_t required_elems = 3U * aligned_route_elems + 2U * front.expertNumAligned;
    if (worker_debug_elems < required_elems) {
        throw std::runtime_error("frontSort debug workspace per worker is too small");
    }

    std::vector<int32_t> actual(debug_bytes / sizeof(int32_t), 0);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual.data(), debug_bytes, workspace_base + frontDebug.smallFrontDebugOffset, debug_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host frontSort copy failed");
    }

    const ExpectedFrontRoute expected_route =
        BuildExpectedFrontRoute(expert_idx, route_elems, front.expertNum, front.expertNumAligned);
    const std::vector<int32_t> expert_idx_i32 = BytesToI32(expert_idx);

    for (size_t worker = 0; worker < active_workers; ++worker) {
        const int32_t *actual_sorted_key = actual.data() + worker * worker_debug_elems;
        const int32_t *actual_dst_to_src = actual_sorted_key + aligned_route_elems;
        const int32_t *actual_expanded = actual_dst_to_src + aligned_route_elems;
        const int32_t *actual_local = actual_expanded + aligned_route_elems;
        const int32_t *actual_base = actual_local + front.expertNumAligned;

        for (size_t idx = 0; idx < route_elems; ++idx) {
            int32_t expected_key = -1;
            const int32_t src = expected_route.dst_to_src_route[idx];
            if (src >= 0) {
                expected_key = expert_idx_i32[static_cast<size_t>(src)];
            }
            UpdateFrontSortReport(report, "sortedKey", worker, idx, expected_key, actual_sorted_key[idx],
                                  report.sorted_key_mismatch_count);
            UpdateFrontSortReport(report, "dstToSrcRoute", worker, idx, expected_route.dst_to_src_route[idx],
                                  actual_dst_to_src[idx], report.dst_to_src_mismatch_count);
            UpdateFrontSortReport(report, "expandedRowIdx", worker, idx, expected_route.expanded_row_idx[idx],
                                  actual_expanded[idx], report.expanded_mismatch_count);
        }
        for (size_t idx = 0; idx < front.expertNumAligned; ++idx) {
            UpdateFrontSortReport(report, "localTokenPerExpert", worker, idx,
                                  expected_route.local_token_per_expert[idx], actual_local[idx],
                                  report.count_mismatch_count);
            UpdateFrontSortReport(report, "expertBase", worker, idx, expected_route.expert_base[idx], actual_base[idx],
                                  report.count_mismatch_count);
        }
    }

    report.pass = report.sorted_key_mismatch_count == 0 && report.dst_to_src_mismatch_count == 0 &&
                  report.expanded_mismatch_count == 0 && report.count_mismatch_count == 0;
    return report;
}

struct FrontSortCheckReport {
    bool pass = true;
    uint32_t route_elems = 0;
    uint32_t aligned_route_elems = 0;
    size_t sorted_expert_mismatch_count = 0;
    size_t sorted_payload_mismatch_count = 0;
    size_t permutation_mismatch_count = 0;
    size_t process_mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_table;
    int32_t expected = 0;
    int32_t actual = 0;
};

void UpdateFrontSortCheckReport(FrontSortCheckReport &report, const std::string &table, size_t idx, int32_t expected,
                                int32_t actual, size_t &mismatch_count)
{
    if (actual == expected) {
        return;
    }
    if (report.first_table.empty()) {
        report.first_table = table;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++mismatch_count;
}

FrontSortCheckReport CheckFrontSort(const DispatchFFNCombineBuildResult &build, const DeviceBuffer &workspace_dev,
                                    const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    FrontSortCheckReport report;
    report.route_elems = front.routeElems;
    report.aligned_route_elems = front.alignedRouteElems;
    const bool check_core0_sort = front.frontCase == 21000U || front.frontCase == 11000U;
    const bool check_merge_out = front.frontCase == 11010U && frontDebug.frontSortCheckDebugStep == 4U;
    if (frontDebug.frontCheckMode == 0U || frontDebug.frontCheckDebugMode == 0U ||
        frontDebug.frontCheckStopStep != 2U || (!check_core0_sort && !check_merge_out) || front.routeElems == 0U ||
        front.alignedRouteElems == 0U) {
        return report;
    }

    const size_t route_elems = front.routeElems;
    const size_t aligned_route_elems = front.alignedRouteElems;
    const size_t debug_bytes = 2U * aligned_route_elems * sizeof(int32_t);
    if (frontDebug.frontSortCheckBytes < debug_bytes ||
        frontDebug.frontSortCheckOffset + debug_bytes > static_cast<uint64_t>(workspace_dev.bytes)) {
        throw std::runtime_error("frontSortCheck debug workspace is out of range");
    }

    std::vector<int32_t> actual(debug_bytes / sizeof(int32_t), 0);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual.data(), debug_bytes, workspace_base + frontDebug.frontSortCheckOffset, debug_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host frontSortCheck copy failed");
    }

    const ExpectedFrontRoute expected_route =
        BuildExpectedFrontRoute(expert_idx, route_elems, front.expertNum, front.expertNumAligned);
    const std::vector<int32_t> expert_idx_i32 = BytesToI32(expert_idx);
    const int32_t *actual_sorted_expert = actual.data();
    const int32_t *actual_sorted_payload = actual_sorted_expert + aligned_route_elems;

    for (size_t idx = 0; idx < route_elems; ++idx) {
        const int32_t expected_payload = expected_route.dst_to_src_route[idx];
        int32_t expected_expert = -1;
        if (expected_payload >= 0) {
            expected_expert = expert_idx_i32[static_cast<size_t>(expected_payload)];
        }
        UpdateFrontSortCheckReport(report, "sortedExpert", idx, expected_expert, actual_sorted_expert[idx],
                                   report.sorted_expert_mismatch_count);
        UpdateFrontSortCheckReport(report, "sortedPayload", idx, expected_payload, actual_sorted_payload[idx],
                                   report.sorted_payload_mismatch_count);
    }

    report.pass = report.sorted_expert_mismatch_count == 0 && report.sorted_payload_mismatch_count == 0;
    return report;
}

struct FrontMetadataReport {
    bool pass = true;
    uint32_t route_elems = 0;
    uint32_t expert_num_aligned = 0;
    uint32_t metadata_core_count = 0;
    uint32_t metadata_core_nonzero_count = 0;
    size_t expanded_mismatch_count = 0;
    size_t count_mismatch_count = 0;
    size_t base_mismatch_count = 0;
    size_t core_count_mismatch_count = 0;
    size_t core_base_mismatch_count = 0;
    size_t permutation_mismatch_count = 0;
    size_t process_mismatch_count = 0;
    int64_t count_sum = 0;
    int64_t core_count_sum = 0;
    size_t first_idx = 0;
    std::string first_table;
    int32_t expected = 0;
    int32_t actual = 0;
};

struct FrontRunSortReport {
    bool pass = true;
    uint32_t route_elems = 0;
    uint32_t sort_need_core_num = 0;
    uint32_t vbs_loop_count = 0;
    uint32_t tail_padding_elems = 0;
    uint32_t first_loop_owner = 0;
    uint32_t first_loop_start = 0;
    uint32_t first_loop_elems = 0;
    uint32_t first_loop_sort_num = 0;
    uint32_t first_loop_packed_offset = 0;
    uint32_t first_loop_packed_len = 0;
    uint32_t last_loop_owner = 0;
    uint32_t last_loop_start = 0;
    uint32_t last_loop_elems = 0;
    uint32_t last_loop_sort_num = 0;
    uint32_t last_loop_packed_offset = 0;
    uint32_t last_loop_packed_len = 0;
    size_t sorted_expert_mismatch_count = 0;
    size_t sorted_payload_mismatch_count = 0;
    size_t first_worker = 0;
    size_t first_idx = 0;
    std::string first_table;
    int32_t expected = 0;
    int32_t actual = 0;
};

void UpdateFrontRunSortReport(FrontRunSortReport &report, const std::string &table, size_t worker, size_t idx,
                              int32_t expected, int32_t actual, size_t &mismatch_count)
{
    if (actual == expected) {
        return;
    }
    if (report.first_table.empty()) {
        report.first_table = table;
        report.first_worker = worker;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++mismatch_count;
}

FrontRunSortReport CheckFrontRunSort(const DispatchFFNCombineBuildResult &build, const DeviceBuffer &workspace_dev,
                                     const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    FrontRunSortReport report;
    report.route_elems = front.routeElems;
    report.sort_need_core_num = front.sortNeedCoreNum;
    if (frontDebug.frontCheckMode == 0U || frontDebug.frontCheckDebugMode == 0U ||
        frontDebug.frontCheckStopStep != 2U || front.frontCase != 11010U ||
        (frontDebug.frontSortCheckDebugStep != 1U && frontDebug.frontSortCheckDebugStep != 2U) ||
        front.routeElems == 0U || front.alignedRouteElems == 0U || front.sortNeedCoreNum == 0U ||
        front.sortPerCoreElems == 0U) {
        return report;
    }

    const size_t aligned_route_elems = front.alignedRouteElems;
    const size_t debug_bytes = 2U * aligned_route_elems * sizeof(int32_t);
    const bool check_local_merge = frontDebug.frontSortCheckDebugStep == 2U;
    const uint64_t debug_offset =
        check_local_merge ? frontDebug.frontSortCheckOffset : frontDebug.frontMergeCheckOffset;
    const uint64_t debug_workspace_bytes =
        check_local_merge ? frontDebug.frontSortCheckBytes : frontDebug.frontMergeCheckBytes;
    if (debug_workspace_bytes < debug_bytes ||
        debug_offset + debug_bytes > static_cast<uint64_t>(workspace_dev.bytes)) {
        throw std::runtime_error("front run-sort debug workspace is out of range");
    }

    std::vector<int32_t> actual(debug_bytes / sizeof(int32_t), 0);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual.data(), debug_bytes, workspace_base + debug_offset, debug_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host front run-sort copy failed");
    }
    const int32_t *actual_sorted_expert = actual.data();
    const int32_t *actual_sorted_payload = actual_sorted_expert + aligned_route_elems;

    const bool check_vbs_loop_runs = frontDebug.frontSortCheckDebugStep == 1U;
    for (uint32_t worker = 0; worker < front.sortNeedCoreNum; ++worker) {
        const size_t route_start = static_cast<size_t>(worker) * front.sortPerCoreElems;
        if (route_start >= front.routeElems) {
            continue;
        }
        size_t route_elems = worker == front.sortNeedCoreNum - 1U ? front.sortLastCoreElems : front.sortPerCoreElems;
        if (route_start + route_elems > front.routeElems) {
            route_elems = front.routeElems - route_start;
        }
        if (!check_vbs_loop_runs) {
            const ExpectedFrontRoute expected_run =
                BuildExpectedFrontRouteRun(expert_idx, route_start, route_elems, front.expertNum);
            for (size_t idx = 0; idx < route_elems; ++idx) {
                const size_t actual_idx = route_start + idx;
                UpdateFrontRunSortReport(report, "sortedExpert", worker, idx, expected_run.expanded_row_idx[idx],
                                         actual_sorted_expert[actual_idx], report.sorted_expert_mismatch_count);
                UpdateFrontRunSortReport(report, "sortedPayload", worker, idx, expected_run.dst_to_src_route[idx],
                                         actual_sorted_payload[actual_idx], report.sorted_payload_mismatch_count);
            }
            continue;
        }

        const bool is_last_worker = worker == front.sortNeedCoreNum - 1U;
        const uint32_t loop_count = is_last_worker ? front.sortLastCoreLoops : front.sortPerCoreLoops;
        const uint32_t per_loop_elems = is_last_worker ? front.sortLastCorePerLoopElems : front.sortPerCorePerLoopElems;
        const uint32_t last_loop_elems =
            is_last_worker ? front.sortLastCoreLastLoopElems : front.sortPerCoreLastLoopElems;
        for (uint32_t loop = 0; loop < loop_count; ++loop) {
            const size_t loop_start = route_start + static_cast<size_t>(loop) * per_loop_elems;
            if (loop_start >= route_start + route_elems || loop_start >= front.routeElems) {
                continue;
            }
            size_t loop_elems = loop == loop_count - 1U ? last_loop_elems : per_loop_elems;
            if (loop_start + loop_elems > route_start + route_elems) {
                loop_elems = route_start + route_elems - loop_start;
            }
            if (loop_start + loop_elems > front.routeElems) {
                loop_elems = front.routeElems - loop_start;
            }
            const uint32_t loop_start_u32 = static_cast<uint32_t>(loop_start);
            const uint32_t loop_elems_u32 = static_cast<uint32_t>(loop_elems);
            const uint32_t sort_num = HostCeilDiv(loop_elems_u32, 32U) * 32U;
            const uint32_t packed_offset = HostFrontSortOffset(loop_start_u32);
            const uint32_t packed_len = HostFrontSortLen(loop_elems_u32);
            if (report.vbs_loop_count == 0U) {
                report.first_loop_owner = worker;
                report.first_loop_start = loop_start_u32;
                report.first_loop_elems = loop_elems_u32;
                report.first_loop_sort_num = sort_num;
                report.first_loop_packed_offset = packed_offset;
                report.first_loop_packed_len = packed_len;
            }
            report.last_loop_owner = worker;
            report.last_loop_start = loop_start_u32;
            report.last_loop_elems = loop_elems_u32;
            report.last_loop_sort_num = sort_num;
            report.last_loop_packed_offset = packed_offset;
            report.last_loop_packed_len = packed_len;
            report.tail_padding_elems += sort_num - loop_elems_u32;
            ++report.vbs_loop_count;
            const ExpectedFrontRoute expected_run =
                BuildExpectedFrontRouteRun(expert_idx, loop_start, loop_elems, front.expertNum);
            for (size_t idx = 0; idx < loop_elems; ++idx) {
                const size_t actual_idx = loop_start + idx;
                UpdateFrontRunSortReport(report, "sortedExpert", worker, idx, expected_run.expanded_row_idx[idx],
                                         actual_sorted_expert[actual_idx], report.sorted_expert_mismatch_count);
                UpdateFrontRunSortReport(report, "sortedPayload", worker, idx, expected_run.dst_to_src_route[idx],
                                         actual_sorted_payload[actual_idx], report.sorted_payload_mismatch_count);
            }
        }
    }

    report.pass = report.sorted_expert_mismatch_count == 0 && report.sorted_payload_mismatch_count == 0;
    return report;
}

struct FrontMiddleMergeShape {
    uint32_t list_num = 0;
    uint32_t per_list_elements = 0;
    uint32_t last_list_elements = 0;
    uint32_t output_workspace_index = 1;
    uint64_t debug_offset = 0;
    uint64_t debug_workspace_bytes = 0;
};

FrontMiddleMergeShape BuildFrontMiddleMergeShape(const DispatchFFNCombineFrontReorderTiling &front,
                                                 const DispatchFFNCombineFrontDebugTiling &frontDebug)
{
    FrontMiddleMergeShape shape;
    shape.list_num = front.sortNeedCoreNum;
    shape.per_list_elements = front.sortPerCoreElems;
    shape.last_list_elements = front.sortLastCoreElems;
    shape.output_workspace_index = 1U;
    while (shape.list_num > 4U) {
        const uint32_t group_num = HostCeilDiv(shape.list_num, 4U);
        const uint32_t remain_list_num = shape.list_num - (group_num - 1U) * 4U;
        shape.last_list_elements = shape.per_list_elements * (remain_list_num - 1U) + shape.last_list_elements;
        shape.per_list_elements *= 4U;
        shape.list_num = group_num;
        shape.output_workspace_index = 1U - shape.output_workspace_index;
    }
    shape.output_workspace_index = 1U;
    const uint32_t debug_workspace_index = 1U - shape.output_workspace_index;
    shape.debug_offset =
        debug_workspace_index == 0U ? frontDebug.frontSortCheckOffset : frontDebug.frontMergeCheckOffset;
    shape.debug_workspace_bytes =
        debug_workspace_index == 0U ? frontDebug.frontSortCheckBytes : frontDebug.frontMergeCheckBytes;
    return shape;
}

FrontRunSortReport CheckFrontMiddleMerge(const DispatchFFNCombineBuildResult &build, const DeviceBuffer &workspace_dev,
                                         const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    FrontRunSortReport report;
    report.route_elems = front.routeElems;
    if (frontDebug.frontCheckMode == 0U || frontDebug.frontCheckDebugMode == 0U ||
        frontDebug.frontCheckStopStep != 2U || front.frontCase != 11010U || frontDebug.frontSortCheckDebugStep != 3U ||
        front.sortNeedCoreNum <= 4U || front.routeElems == 0U || front.alignedRouteElems == 0U ||
        front.sortPerCoreElems == 0U) {
        return report;
    }

    const FrontMiddleMergeShape shape = BuildFrontMiddleMergeShape(front, frontDebug);
    report.sort_need_core_num = shape.list_num;
    const size_t aligned_route_elems = front.alignedRouteElems;
    const size_t debug_bytes = 2U * aligned_route_elems * sizeof(int32_t);
    if (shape.debug_workspace_bytes < debug_bytes ||
        shape.debug_offset + debug_bytes > static_cast<uint64_t>(workspace_dev.bytes)) {
        throw std::runtime_error("front middle-merge debug workspace is out of range");
    }

    std::vector<int32_t> actual(debug_bytes / sizeof(int32_t), 0);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual.data(), debug_bytes, workspace_base + shape.debug_offset, debug_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host front middle-merge copy failed");
    }
    const int32_t *actual_sorted_expert = actual.data();
    const int32_t *actual_sorted_payload = actual_sorted_expert + aligned_route_elems;

    size_t covered = 0;
    for (uint32_t list = 0; list < shape.list_num; ++list) {
        const size_t route_start = static_cast<size_t>(list) * shape.per_list_elements;
        if (route_start >= front.routeElems) {
            continue;
        }
        size_t route_elems = list == shape.list_num - 1U ? shape.last_list_elements : shape.per_list_elements;
        if (route_start + route_elems > front.routeElems) {
            route_elems = front.routeElems - route_start;
        }
        covered += route_elems;
        const ExpectedFrontRoute expected_run =
            BuildExpectedFrontRouteRun(expert_idx, route_start, route_elems, front.expertNum);
        for (size_t idx = 0; idx < route_elems; ++idx) {
            const size_t actual_idx = route_start + idx;
            UpdateFrontRunSortReport(report, "sortedExpert", list, idx, expected_run.expanded_row_idx[idx],
                                     actual_sorted_expert[actual_idx], report.sorted_expert_mismatch_count);
            UpdateFrontRunSortReport(report, "sortedPayload", list, idx, expected_run.dst_to_src_route[idx],
                                     actual_sorted_payload[actual_idx], report.sorted_payload_mismatch_count);
        }
    }
    if (covered != front.routeElems) {
        UpdateFrontRunSortReport(report, "coverage", 0, covered, static_cast<int32_t>(front.routeElems),
                                 static_cast<int32_t>(covered), report.sorted_payload_mismatch_count);
    }

    report.pass = report.sorted_expert_mismatch_count == 0 && report.sorted_payload_mismatch_count == 0;
    return report;
}

void UpdateFrontMetadataReport(FrontMetadataReport &report, const std::string &table, size_t idx, int32_t expected,
                               int32_t actual, size_t &mismatch_count)
{
    if (actual == expected) {
        return;
    }
    if (report.first_table.empty()) {
        report.first_table = table;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++mismatch_count;
}

FrontMetadataReport CheckFrontMetadata(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                       const DeviceBuffer &workspace_dev, const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    FrontMetadataReport report;
    report.route_elems = front.routeElems;
    report.expert_num_aligned = front.expertNumAligned;
    const bool check_core0_metadata = front.frontCase == 21000U || front.frontCase == 11000U;
    const bool check_multi_core_metadata = front.frontCase == 11010U;
    report.metadata_core_count = check_multi_core_metadata ? cfg.aiv_num : 0U;
    if (frontDebug.frontCheckMode == 0U || frontDebug.frontCheckDebugMode == 0U || frontDebug.frontCheckStopStep < 3U ||
        (!check_core0_metadata && !check_multi_core_metadata) || front.routeElems == 0U ||
        front.expertNumAligned == 0U) {
        return report;
    }

    const size_t route_elems = front.routeElems;
    const size_t table_elems = front.expertNumAligned;
    const size_t expanded_bytes = route_elems * sizeof(int32_t);
    const size_t table_bytes = table_elems * sizeof(int32_t);
    const size_t core_table_elems = static_cast<size_t>(cfg.aiv_num) * table_elems;
    const size_t core_table_bytes = core_table_elems * sizeof(int32_t);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (front.expandedRowIdxOffset + expanded_bytes > static_cast<uint64_t>(workspace_dev.bytes) ||
        front.localTokenPerExpertOffset + table_bytes > static_cast<uint64_t>(workspace_dev.bytes) ||
        front.expertBaseOffset + table_bytes > static_cast<uint64_t>(workspace_dev.bytes)) {
        throw std::runtime_error("front metadata workspace is out of range");
    }
    if (check_multi_core_metadata &&
        front.coreCountOffset + core_table_bytes > static_cast<uint64_t>(workspace_dev.bytes)) {
        throw std::runtime_error("front metadata core count workspace is out of range");
    }

    std::vector<int32_t> actual_expanded(route_elems, 0);
    std::vector<int32_t> actual_count(table_elems, 0);
    std::vector<int32_t> actual_base(table_elems, 0);
    std::vector<int32_t> actual_core_count(core_table_elems, 0);
    if (aclrtMemcpy(actual_expanded.data(), expanded_bytes, workspace_base + front.expandedRowIdxOffset, expanded_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(actual_count.data(), table_bytes, workspace_base + front.localTokenPerExpertOffset, table_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(actual_base.data(), table_bytes, workspace_base + front.expertBaseOffset, table_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host front metadata copy failed");
    }
    if (check_multi_core_metadata &&
        aclrtMemcpy(actual_core_count.data(), core_table_bytes, workspace_base + front.coreCountOffset,
                    core_table_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host front metadata core count table copy failed");
    }

    const ExpectedFrontRoute expected_route =
        BuildExpectedFrontRoute(expert_idx, route_elems, front.expertNum, front.expertNumAligned);
    std::vector<uint8_t> seen(route_elems, 0);
    for (size_t idx = 0; idx < route_elems; ++idx) {
        UpdateFrontMetadataReport(report, "expandedRowIdx", idx, expected_route.expanded_row_idx[idx],
                                  actual_expanded[idx], report.expanded_mismatch_count);
        if (actual_expanded[idx] < 0 || static_cast<size_t>(actual_expanded[idx]) >= route_elems) {
            ++report.permutation_mismatch_count;
        } else {
            ++seen[static_cast<size_t>(actual_expanded[idx])];
        }
    }
    for (size_t idx = 0; idx < route_elems; ++idx) {
        if (seen[idx] != 1U) {
            ++report.permutation_mismatch_count;
        }
    }
    for (size_t idx = 0; idx < table_elems; ++idx) {
        report.count_sum += actual_count[idx];
        UpdateFrontMetadataReport(report, "localTokenPerExpert", idx, expected_route.local_token_per_expert[idx],
                                  actual_count[idx], report.count_mismatch_count);
        UpdateFrontMetadataReport(report, "expertBase", idx, expected_route.expert_base[idx], actual_base[idx],
                                  report.base_mismatch_count);
    }
    if (report.count_sum != static_cast<int64_t>(route_elems)) {
        ++report.count_mismatch_count;
        if (report.first_table.empty()) {
            report.first_table = "localTokenPerExpertSum";
            report.expected = static_cast<int32_t>(route_elems);
            report.actual = static_cast<int32_t>(report.count_sum);
        }
    }

    if (check_multi_core_metadata) {
        std::vector<int32_t> expected_core_count =
            BuildExpectedFrontSortedCoreCount(cfg, build, expert_idx, expected_route);

        for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
            bool core_nonzero = false;
            for (uint32_t expert = 0; expert < front.expertNumAligned; ++expert) {
                const size_t idx = static_cast<size_t>(core) * table_elems + expert;
                report.core_count_sum += actual_core_count[idx];
                core_nonzero = core_nonzero || actual_core_count[idx] != 0;
                UpdateFrontMetadataReport(report, "coreCount", idx, expected_core_count[idx], actual_core_count[idx],
                                          report.core_count_mismatch_count);
            }
            report.metadata_core_nonzero_count += core_nonzero ? 1U : 0U;
        }
    }

    report.pass = report.expanded_mismatch_count == 0 && report.count_mismatch_count == 0 &&
                  report.base_mismatch_count == 0 && report.core_count_mismatch_count == 0 &&
                  report.permutation_mismatch_count == 0;
    return report;
}

struct ScatterQuantReport {
    bool pass = true;
    uint32_t route_elems = 0;
    size_t expanded_mismatch_count = 0;
    size_t offset_mismatch_count = 0;
    size_t scale_mismatch_count = 0;
    size_t process_mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_table;
    int32_t expected_i32 = 0;
    int32_t actual_i32 = 0;
    float expected_f32 = 0.0f;
    float actual_f32 = 0.0f;
};

struct CountExchangeReport {
    bool pass = true;
    size_t token_mismatch_count = 0;
    size_t presum_mismatch_count = 0;
    size_t process_mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_table;
    int32_t expected = 0;
    int32_t actual = 0;
    uint64_t expected_u64 = 0;
    uint64_t actual_u64 = 0;
};

struct CumsumReport {
    bool pass = true;
    size_t cumsum_mismatch_count = 0;
    size_t expert_token_nums_mismatch_count = 0;
    size_t process_mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_table;
    int32_t expected = 0;
    int32_t actual = 0;
    uint64_t expected_u64 = 0;
    uint64_t actual_u64 = 0;
};

struct FrontDispatchContractReport {
    bool pass = true;
    size_t checked_count = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct FrontDoneReport {
    bool pass = true;
    size_t mismatch_count = 0;
    int first_rank = -1;
    int32_t expected = 0;
    int32_t actual = 0;
};

struct DispatchLayoutReport {
    bool pass = true;
    size_t mismatch_count = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct FrontCheckLayoutReport {
    bool pass = true;
    size_t mismatch_count = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
    uint64_t helper_debug_mask = 0;
    uint64_t helper_debug_failures = 0;
    int32_t quant_expected_rows = 0;
    int32_t quant_actual_rows = 0;
    int32_t quant_expected_stores = 0;
    int32_t quant_actual_stores = 0;
    int32_t quant_active_cores = 0;
    int32_t quant_full_row_mode = 0;
    int32_t quant_column_tiled_mode = 0;
    int32_t quant_marker = 0;
};

struct Gmm1LayoutReport {
    bool pass = true;
    size_t mismatch_count = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct Gmm2LayoutReport {
    bool pass = true;
    size_t mismatch_count = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct Gmm2V2CReport {
    bool pass = true;
    size_t segment_count = 0;
    size_t checked_count = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct Gmm2SegmentReport {
    bool pass = true;
    size_t segment_count = 0;
    uint32_t epilogue_granularity = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct Gmm2TaskReport {
    bool pass = true;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct Gmm2DoneReport {
    bool pass = true;
    size_t checked_count = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct Gmm2FinalSyncReport {
    bool pass = true;
    size_t checked_count = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct SwigluLayoutReport {
    bool pass = true;
    size_t mismatch_count = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct SwigluC2VReport {
    bool pass = true;
    size_t segment_count = 0;
    size_t checked_count = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct SwigluSegmentReport {
    bool pass = true;
    size_t segment_count = 0;
    uint32_t epilogue_granularity = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct SwigluMetadataModeReport {
    bool pass = true;
    uint32_t expected_mode = 0;
    uint32_t actual_mode = 0;
    size_t mismatch_count = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct SwigluSegmentMetaReport {
    bool pass = true;
    uint32_t metadata_mode = 0;
    size_t segment_count = 0;
    size_t checked_count = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct SwigluTaskReport {
    bool pass = true;
    size_t segment_count = 0;
    size_t checked_count = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct SwigluOutputReport {
    bool pass = true;
    size_t segment_count = 0;
    size_t row_count = 0;
    size_t value_count = 0;
    size_t scale_count = 0;
    size_t quant_mismatch_count = 0;
    size_t scale_mismatch_count = 0;
    std::string first_table;
    uint32_t first_segment = 0;
    uint32_t first_row = 0;
    uint32_t first_col = 0;
    int32_t expected_i32 = 0;
    int32_t actual_i32 = 0;
    float expected_f32 = 0.0f;
    float actual_f32 = 0.0f;
    double max_scale_abs_err = 0.0;
    double max_scale_rel_err = 0.0;
    uint64_t ub_bytes = 0;
};

struct SwigluDoneReport {
    bool pass = true;
    size_t segment_count = 0;
    size_t checked_count = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct SwigluFinalSyncReport {
    bool pass = true;
    size_t checked_count = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct Gmm1SyncReport {
    bool pass = true;
    size_t checked_count = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct Gmm1TaskReport {
    bool pass = true;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct Gmm1OutputReport {
    bool pass = true;
    size_t checked_count = 0;
    size_t mismatch_count = 0;
    uint32_t first_group = 0;
    uint32_t first_row = 0;
    uint32_t first_col = 0;
    float expected = 0.0f;
    float actual = 0.0f;
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
};

struct Gmm2OutputReport {
    bool pass = true;
    bool skipped = false;
    std::string skip_reason;
    uint32_t segment_count = 0;
    uint32_t group_count = 0;
    uint32_t col_count = 0;
    size_t row_count = 0;
    size_t checked_count = 0;
    size_t expected_checked_count = 0;
    size_t mismatch_count = 0;
    uint32_t first_group = 0;
    uint32_t first_row = 0;
    uint32_t first_col = 0;
    int32_t expected_acc = 0;
    uint16_t expected_half = 0;
    uint16_t actual_half = 0;
    float expected_f32 = 0.0f;
    float expected_fp16 = 0.0f;
    float actual_f32 = 0.0f;
    uint32_t max_ulp_err = 0;
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
};

struct Gmm1DoneReport {
    bool pass = true;
    size_t segment_count = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct Gmm1SegmentPlanReport {
    bool pass = true;
    size_t segment_count = 0;
    uint32_t epilogue_granularity = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct Gmm1DetailReport {
    bool pass = true;
    uint32_t group_count = 0;
    uint32_t col_count = 0;
    size_t row_count = 0;
    size_t checked_count = 0;
    size_t cumsum_mismatch_count = 0;
    size_t gm_c_mismatch_count = 0;
    std::string first_table;
    size_t first_idx = 0;
    uint32_t first_group = 0;
    uint32_t first_row = 0;
    uint32_t first_col = 0;
    int32_t expected_i32 = 0;
    int32_t actual_i32 = 0;
    int32_t expected_acc = 0;
    uint16_t expected_half = 0;
    uint16_t actual_half = 0;
    float expected_f32 = 0.0f;
    float expected_fp16 = 0.0f;
    float actual_f32 = 0.0f;
    uint32_t max_ulp_err = 0;
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
};

struct DispatchMetadataReport {
    bool pass = true;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct DispatchTaskReport {
    bool pass = true;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
    uint64_t group_count = 0;
    uint64_t total_blocks = 0;
    uint64_t active_slots = 0;
    uint64_t total_rows = 0;
    uint64_t max_rows_per_slot = 0;
    uint64_t min_row_block_rows = std::numeric_limits<uint64_t>::max();
    uint64_t max_row_block_rows = 0;
    uint64_t max_blocks_per_group = 0;
};

struct DispatchGatherReport {
    bool pass = true;
    size_t gm_a_mismatch_count = 0;
    size_t scale_mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_table;
    int32_t expected_i32 = 0;
    int32_t actual_i32 = 0;
    float expected_f32 = 0.0f;
    float actual_f32 = 0.0f;
};

struct DispatchGatherDetailReport {
    bool pass = true;
    size_t segment_count = 0;
    size_t non_empty_segment_count = 0;
    size_t row_count = 0;
    size_t byte_count = 0;
    size_t scale_count = 0;
    size_t gm_a_mismatch_count = 0;
    size_t scale_mismatch_count = 0;
    size_t coverage_mismatch_count = 0;
    uint32_t first_group = 0;
    uint32_t first_src_rank = 0;
    uint32_t first_local_row = 0;
    uint32_t first_dst_row = 0;
    uint32_t first_col = 0;
    std::string first_table;
    int32_t expected_i32 = 0;
    int32_t actual_i32 = 0;
    float expected_f32 = 0.0f;
    float actual_f32 = 0.0f;
    uint64_t expected_u64 = 0;
    uint64_t actual_u64 = 0;
};

struct DispatchCopyChunkReport {
    bool pass = true;
    size_t metadata_mismatch_count = 0;
    size_t gm_a_mismatch_count = 0;
    size_t scale_mismatch_count = 0;
    size_t zero_mismatch_count = 0;
    size_t checked_bytes = 0;
    size_t checked_scales = 0;
    size_t zero_probe_count = 0;
    uint32_t group = 0;
    uint32_t src_rank = 0;
    uint32_t rows = 0;
    uint32_t chunk_rows = 0;
    uint32_t src_row_base = 0;
    uint32_t dst_row_base = 0;
    std::string first_table;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected_u64 = 0;
    uint64_t actual_u64 = 0;
    int32_t expected_i32 = 0;
    int32_t actual_i32 = 0;
    float expected_f32 = 0.0f;
    float actual_f32 = 0.0f;
};

struct DispatchGroupDoneReport {
    bool pass = true;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct CombineLayoutReport {
    bool pass = true;
    size_t mismatch_count = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct CombineReadyReport {
    bool pass = true;
    size_t checked_count = 0;
    size_t marked_count = 0;
    size_t marked_coreidx_mismatch_count = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    uint64_t group0_magic_mask = 0;
    uint32_t first_marked_core_slot = 0;
    uint32_t first_marked_core_value = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct CombineMetaReport {
    bool pass = true;
    size_t segment_count = 0;
    size_t owner_checked_count = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct CombineLoadReport {
    bool pass = true;
    size_t segment_count = 0;
    size_t loaded_count = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct CombineDequantReport {
    bool pass = true;
    size_t segment_count = 0;
    size_t dequant_count = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct CombineStoreReport {
    bool pass = true;
    size_t segment_count = 0;
    size_t stored_count = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct CombineFinalizeReport {
    bool pass = true;
    size_t checked_count = 0;
    size_t reset_zero_count = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct CombineTaskReport {
    bool pass = true;
    size_t checked_count = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct CombineDoneReport {
    bool pass = true;
    size_t checked_count = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct CombineDetailReport {
    bool pass = true;
    bool skipped = false;
    std::string skip_reason;
    size_t segment_count = 0;
    size_t non_empty_segment_count = 0;
    size_t row_count = 0;
    size_t value_count = 0;
    size_t byte_count = 0;
    size_t mismatch_count = 0;
    size_t coverage_mismatch_count = 0;
    uint32_t first_executor_rank = 0;
    uint32_t first_src_rank = 0;
    uint32_t first_group = 0;
    uint32_t first_src_row = 0;
    uint32_t first_dst_row = 0;
    uint32_t first_col = 0;
    uint16_t first_c2_half = 0;
    uint16_t expected_half = 0;
    uint16_t actual_half = 0;
    float first_scale = 0.0f;
    float expected_f32 = 0.0f;
    float actual_f32 = 0.0f;
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    uint32_t max_ulp_err = 0;
};

struct UnpermuteLayoutReport {
    bool pass = true;
    size_t mismatch_count = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct UnpermuteIndexedReport {
    bool pass = true;
    size_t checked_count = 0;
    size_t mismatch_count = 0;
    size_t first_idx = 0;
    std::string first_field;
    uint64_t expected = 0;
    uint64_t actual = 0;
};

struct UnpermuteDetailReport {
    bool pass = true;
    size_t row_count = 0;
    size_t value_count = 0;
    size_t mismatch_count = 0;
    size_t invalid_row_count = 0;
    uint32_t first_token = 0;
    uint32_t first_col = 0;
    uint32_t first_topk = 0;
    int32_t first_expanded_row = 0;
    float first_prob = 0.0f;
    uint16_t expected_half = 0;
    uint16_t actual_half = 0;
    float expected_f32 = 0.0f;
    float actual_f32 = 0.0f;
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;
    uint32_t max_ulp_err = 0;
};

void UpdateScatterIntReport(ScatterQuantReport &report, const std::string &table, size_t idx, int32_t expected,
                            int32_t actual, size_t &mismatch_count)
{
    if (actual == expected) {
        return;
    }
    if (report.first_table.empty()) {
        report.first_table = table;
        report.first_idx = idx;
        report.expected_i32 = expected;
        report.actual_i32 = actual;
    }
    ++mismatch_count;
}

void UpdateScatterScaleReport(ScatterQuantReport &report, size_t idx, float expected, float actual)
{
    const float tolerance = 1.0e-5f + 1.0e-3f * std::fabs(expected);
    if (std::isfinite(expected) && std::isfinite(actual) && std::fabs(actual - expected) <= tolerance) {
        return;
    }
    if (report.first_table.empty()) {
        report.first_table = "scale";
        report.first_idx = idx;
        report.expected_f32 = expected;
        report.actual_f32 = actual;
    }
    ++report.scale_mismatch_count;
}

void GetExpectedTokenRange(const CaseConfig &cfg, uint32_t core, uint32_t &token_start, uint32_t &token_end)
{
    const uint32_t base = cfg.m / cfg.aiv_num;
    const uint32_t rem = cfg.m % cfg.aiv_num;
    token_start = core * base + (core < rem ? core : rem);
    const uint32_t token_len = base + (core < rem ? 1U : 0U);
    token_end = token_start + token_len;
}

ScatterQuantReport CheckScatterQuant(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                     const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                     const std::vector<uint8_t> &x_bytes, const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const uint32_t aiv_num = cfg.aiv_num;
    const uint32_t expert_num = front.expertNum;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const size_t expanded_elems = static_cast<size_t>(cfg.m) * cfg.topk;
    const size_t expanded_bytes = expanded_elems * sizeof(int32_t);
    const auto expert_idx_i32 = BytesToI32(expert_idx);
    const auto x_bf16 = BytesToU16(x_bytes);
    const bool use_front_route = front.frontPath == 1U ||
                                 (frontDebug.frontCheckMode != 0U && frontDebug.frontCheckStopStep != 0U) ||
                                 (frontDebug.frontMode != 0U && frontDebug.frontStopStep != 0U);

    std::vector<int32_t> expected_expanded(expanded_elems, -1);
    size_t routed_rows = 0;
    if (use_front_route) {
        const ExpectedFrontRoute expected_route =
            BuildExpectedFrontRoute(expert_idx, expanded_elems, expert_num, expert_num_aligned);
        expected_expanded = expected_route.expanded_row_idx;
        routed_rows = expected_route.routed_rows;
    } else {
        const std::vector<int32_t> expected_core_count = BuildExpectedCoreCount(cfg, build, expert_idx);
        std::vector<int32_t> expected_core_base(static_cast<size_t>(aiv_num) * expert_num_aligned, 0);
        int32_t expert_base = 0;
        for (uint32_t expert = 0; expert < expert_num_aligned; ++expert) {
            int32_t prefix_before_core = 0;
            int32_t local_token = 0;
            for (uint32_t core = 0; core < aiv_num; ++core) {
                expected_core_base[static_cast<size_t>(core) * expert_num_aligned + expert] =
                    expert_base + prefix_before_core;
                int32_t count = 0;
                if (expert < expert_num) {
                    count = expected_core_count[static_cast<size_t>(core) * expert_num_aligned + expert];
                }
                prefix_before_core += count;
                local_token += count;
            }
            expert_base += local_token;
            routed_rows += static_cast<size_t>(local_token);
        }
        for (uint32_t core = 0; core < aiv_num; ++core) {
            std::vector<int32_t> cursor(expert_num_aligned, 0);
            uint32_t token_start = 0;
            uint32_t token_end = 0;
            GetExpectedTokenRange(cfg, core, token_start, token_end);
            for (uint32_t token = token_start; token < token_end; ++token) {
                for (uint32_t topk_idx = 0; topk_idx < cfg.topk; ++topk_idx) {
                    const size_t src = static_cast<size_t>(token) * cfg.topk + topk_idx;
                    const int32_t expert_value = expert_idx_i32[src];
                    if (expert_value < 0 || static_cast<uint32_t>(expert_value) >= expert_num) {
                        continue;
                    }
                    const uint32_t expert = static_cast<uint32_t>(expert_value);
                    const int32_t dst =
                        expected_core_base[static_cast<size_t>(core) * expert_num_aligned + expert] + cursor[expert]++;
                    expected_expanded[src] = dst;
                }
            }
        }
    }

    std::vector<int8_t> expected_offset_a(routed_rows * cfg.k, 0);
    std::vector<float> expected_scale(routed_rows, 0.0f);
    for (uint32_t token = 0; token < cfg.m; ++token) {
        float row_scale = 0.0f;
        const std::vector<int8_t> quant = QuantizeBf16RowLikeV3(x_bf16, token, cfg.k, row_scale);
        for (uint32_t topk_idx = 0; topk_idx < cfg.topk; ++topk_idx) {
            const size_t src = static_cast<size_t>(token) * cfg.topk + topk_idx;
            const int32_t dst = expected_expanded[src];
            if (dst >= 0 && static_cast<size_t>(dst) < routed_rows) {
                expected_scale[dst] = row_scale;
                std::copy(quant.begin(), quant.end(), expected_offset_a.begin() + static_cast<size_t>(dst) * cfg.k);
            }
        }
    }

    ScatterQuantReport report;
    report.route_elems = front.routeElems;

    std::vector<int32_t> actual_expanded(expanded_elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual_expanded.data(), expanded_bytes, workspace_base + front.expandedRowIdxOffset, expanded_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host expandedRowIdx copy failed");
    }

    constexpr size_t packed_scale_bytes = sizeof(float);
    const size_t packed_stride = static_cast<size_t>(cfg.k) + 32U;
    std::vector<int8_t> actual_offset_a(expected_offset_a.size(), 0);
    std::vector<float> actual_scale(expected_scale.size(), 0.0f);
    std::vector<uint8_t> actual_packed(routed_rows * packed_stride, 0);
    const auto *window_base = reinterpret_cast<const uint8_t *>(runtime.hccl.WindowIn(runtime.hccl.rank_id));
    if (!actual_packed.empty() && aclrtMemcpy(actual_packed.data(), actual_packed.size(), window_base,
                                              actual_packed.size(), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host offsetA copy failed");
    }
    for (size_t row = 0; row < routed_rows; ++row) {
        const size_t packed_offset = row * packed_stride;
        std::copy_n(reinterpret_cast<const int8_t *>(actual_packed.data() + packed_offset), cfg.k,
                    actual_offset_a.begin() + row * cfg.k);
        if (cfg.k + packed_scale_bytes <= packed_stride) {
            std::memcpy(&actual_scale[row], actual_packed.data() + packed_offset + cfg.k, packed_scale_bytes);
        }
    }

    for (size_t idx = 0; idx < expected_expanded.size(); ++idx) {
        UpdateScatterIntReport(report, "expandedRowIdx", idx, expected_expanded[idx], actual_expanded[idx],
                               report.expanded_mismatch_count);
    }
    for (size_t idx = 0; idx < expected_offset_a.size(); ++idx) {
        UpdateScatterIntReport(report, "offsetA", idx, static_cast<int32_t>(expected_offset_a[idx]),
                               static_cast<int32_t>(actual_offset_a[idx]), report.offset_mismatch_count);
    }
    for (size_t idx = 0; idx < expected_scale.size(); ++idx) {
        UpdateScatterScaleReport(report, idx, expected_scale[idx], actual_scale[idx]);
    }
    report.pass = report.expanded_mismatch_count == 0 && report.offset_mismatch_count == 0 &&
                  report.scale_mismatch_count == 0 && report.process_mismatch_count == 0;
    return report;
}

std::vector<int32_t> BuildExpectedLocalTokenPerExpert(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                                      const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto expert_idx_i32 = BytesToI32(expert_idx);
    std::vector<int32_t> expected(front.expertNumAligned, 0);
    for (uint32_t token = 0; token < cfg.m; ++token) {
        for (uint32_t topk_idx = 0; topk_idx < cfg.topk; ++topk_idx) {
            const int32_t expert = expert_idx_i32[static_cast<size_t>(token) * cfg.topk + topk_idx];
            if (expert >= 0 && static_cast<uint32_t>(expert) < front.expertNum) {
                ++expected[static_cast<uint32_t>(expert)];
            }
        }
    }
    return expected;
}

std::vector<int32_t> GatherExpectedTokenRows(int rank_id, int world_size, const std::vector<int32_t> &local_row)
{
    const int row_bytes = static_cast<int>(local_row.size() * sizeof(int32_t));
    std::vector<int32_t> all_rows(local_row.size() * static_cast<size_t>(world_size), 0);
    CommMpiGather(local_row.data(), row_bytes, COMM_MPI_CHAR, rank_id == 0 ? all_rows.data() : nullptr, row_bytes,
                  COMM_MPI_CHAR, 0);
    CommMpiBcast(all_rows.data(), static_cast<int>(all_rows.size() * sizeof(int32_t)), COMM_MPI_CHAR, 0);
    return all_rows;
}

std::vector<uint8_t> GatherAllRankBytes(int rank_id, int world_size, const std::vector<uint8_t> &local,
                                        const char *name)
{
    if (local.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + " is too large for MPI gather");
    }
    const int local_bytes = static_cast<int>(local.size());
    std::vector<uint8_t> all(local.size() * static_cast<size_t>(world_size), 0);
    if (all.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(name) + " all-rank buffer is too large for MPI bcast");
    }
    CommMpiGather(local.data(), local_bytes, COMM_MPI_CHAR, rank_id == 0 ? all.data() : nullptr, local_bytes,
                  COMM_MPI_CHAR, 0);
    CommMpiBcast(all.data(), static_cast<int>(all.size()), COMM_MPI_CHAR, 0);
    return all;
}

void UpdateCountExchangeReport(CountExchangeReport &report, const std::string &table, size_t idx, int32_t expected,
                               int32_t actual, size_t &mismatch_count)
{
    if (actual == expected) {
        return;
    }
    if (report.first_table.empty()) {
        report.first_table = table;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
        report.expected_u64 = static_cast<uint32_t>(expected);
        report.actual_u64 = static_cast<uint32_t>(actual);
    }
    ++mismatch_count;
}

CountExchangeReport CheckCountExchange(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                       const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                       const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const size_t token_elems = static_cast<size_t>(cfg.world_size) * expert_num_aligned;
    const size_t token_bytes = token_elems * sizeof(int32_t);
    const size_t presum_elems = static_cast<size_t>(cfg.world_size) * expert_per_rank;
    const size_t presum_bytes = presum_elems * sizeof(int32_t);

    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);

    std::vector<int32_t> expected_presum(presum_elems, 0);
    const uint32_t local_begin = static_cast<uint32_t>(runtime.hccl.rank_id) * expert_per_rank;
    for (uint32_t src_rank = 0; src_rank < cfg.world_size; ++src_rank) {
        int32_t prev_sum = 0;
        uint32_t local_expert = 0;
        for (uint32_t expert = 0; expert < local_begin + expert_per_rank; ++expert) {
            if (expert >= local_begin) {
                expected_presum[static_cast<size_t>(src_rank) * expert_per_rank + local_expert] = prev_sum;
                ++local_expert;
            }
            prev_sum += expected_tokens[static_cast<size_t>(src_rank) * expert_num_aligned + expert];
        }
    }

    std::vector<int32_t> actual_tokens(token_elems, 0);
    const auto *window_base = reinterpret_cast<const uint8_t *>(runtime.hccl.WindowIn(runtime.hccl.rank_id));
    const uint64_t token_offset = runtime.hccl.WindowBytes() - 2U * kHostMiB;
    if (aclrtMemcpy(actual_tokens.data(), token_bytes, window_base + token_offset, token_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host tokenPerExpert copy failed");
    }

    std::vector<int32_t> actual_presum(presum_elems, 0);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual_presum.data(), presum_bytes, workspace_base + front.preSumBeforeRankOffset, presum_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host preSumBeforeRank copy failed");
    }

    CountExchangeReport report;
    for (size_t idx = 0; idx < token_elems; ++idx) {
        UpdateCountExchangeReport(report, "tokenPerExpert", idx, expected_tokens[idx], actual_tokens[idx],
                                  report.token_mismatch_count);
    }
    for (size_t idx = 0; idx < presum_elems; ++idx) {
        UpdateCountExchangeReport(report, "preSumBeforeRank", idx, expected_presum[idx], actual_presum[idx],
                                  report.presum_mismatch_count);
    }
    const bool check_front_process = frontDebug.frontMode != 0U && front.frontCase == 11000U &&
                                     (frontDebug.frontDebugMode != 0U || frontDebug.frontStopStep != 0U);
    if (check_front_process) {
        if (frontDebug.frontDebugBytes < sizeof(DispatchFFNCombineFrontLayoutDebug)) {
            throw std::runtime_error("front debug area is too small for count exchange process check");
        }
        if (frontDebug.frontDebugOffset + sizeof(DispatchFFNCombineFrontLayoutDebug) >
            static_cast<uint64_t>(workspace_dev.bytes)) {
            throw std::runtime_error("front count exchange header is out of workspace range");
        }
        DispatchFFNCombineFrontLayoutDebug actual_debug;
        std::memset(&actual_debug, 0, sizeof(actual_debug));
        if (aclrtMemcpy(&actual_debug, sizeof(actual_debug), workspace_base + frontDebug.frontDebugOffset,
                        sizeof(actual_debug), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            throw std::runtime_error("device->host front count exchange header copy failed");
        }
        const uint32_t expected_peer_ranks = cfg.world_size == 0U ? 0U : cfg.world_size - 1U;
        UpdateCountExchangeReport(report, "countMarkerBias", 0, kHostFrontCountMarkerBias,
                                  static_cast<int32_t>(actual_debug.countMarkerBias), report.process_mismatch_count);
        UpdateCountExchangeReport(
            report, "tokenPerExpertRowBytes", 0, static_cast<int32_t>(expert_num_aligned * sizeof(int32_t)),
            static_cast<int32_t>(actual_debug.tokenPerExpertRowBytes), report.process_mismatch_count);
        UpdateCountExchangeReport(report, "countPublishedRanks", 0, static_cast<int32_t>(expected_peer_ranks),
                                  static_cast<int32_t>(actual_debug.countPublishedRanks),
                                  report.process_mismatch_count);
        UpdateCountExchangeReport(report, "countWaitedRanks", 0, static_cast<int32_t>(expected_peer_ranks),
                                  static_cast<int32_t>(actual_debug.countWaitedRanks), report.process_mismatch_count);
        UpdateCountExchangeReport(report, "countMarkerRestored", 0, 1,
                                  static_cast<int32_t>(actual_debug.countMarkerRestored),
                                  report.process_mismatch_count);
        UpdateCountExchangeReport(report, "countPreSumBuilt", 0, 1, static_cast<int32_t>(actual_debug.countPreSumBuilt),
                                  report.process_mismatch_count);
        UpdateCountExchangeReport(report, "countOldNotifyWaitUsed", 0, 0,
                                  static_cast<int32_t>(actual_debug.countOldNotifyWaitUsed),
                                  report.process_mismatch_count);
    }
    report.pass =
        report.token_mismatch_count == 0 && report.presum_mismatch_count == 0 && report.process_mismatch_count == 0;
    return report;
}

void UpdateCumsumReport(CumsumReport &report, const std::string &table, size_t idx, int32_t expected, int32_t actual,
                        size_t &mismatch_count)
{
    if (actual == expected) {
        return;
    }
    if (report.first_table.empty()) {
        report.first_table = table;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
        report.expected_u64 = static_cast<uint32_t>(expected);
        report.actual_u64 = static_cast<uint32_t>(actual);
    }
    ++mismatch_count;
}

CumsumReport CheckCumsum(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                         const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                         const DeviceBuffer &expert_token_nums_dev, const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const size_t cumsum_elems = static_cast<size_t>(cfg.world_size) * expert_per_rank;
    const size_t cumsum_bytes = cumsum_elems * sizeof(int32_t);
    const size_t expert_token_nums_bytes = static_cast<size_t>(expert_per_rank) * sizeof(int32_t);

    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);

    std::vector<int32_t> expected_cumsum(cumsum_elems, 0);
    std::vector<int32_t> expected_expert_token_nums(expert_per_rank, 0);
    for (uint32_t local_expert = 0; local_expert < expert_per_rank; ++local_expert) {
        int32_t running_sum = 0;
        for (uint32_t src_rank = 0; src_rank < cfg.world_size; ++src_rank) {
            const size_t token_idx = static_cast<size_t>(src_rank) * expert_num_aligned +
                                     static_cast<size_t>(runtime.hccl.rank_id) * expert_per_rank + local_expert;
            running_sum += expected_tokens[token_idx];
            expected_cumsum[static_cast<size_t>(src_rank) * expert_per_rank + local_expert] = running_sum;
        }
        expected_expert_token_nums[local_expert] = running_sum;
    }

    std::vector<int32_t> actual_cumsum(cumsum_elems, 0);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual_cumsum.data(), cumsum_bytes, workspace_base + front.cumsumMMOffset, cumsum_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host cumsumMM copy failed");
    }

    std::vector<int32_t> actual_expert_token_nums(expert_per_rank, 0);
    if (aclrtMemcpy(actual_expert_token_nums.data(), expert_token_nums_bytes, expert_token_nums_dev.ptr,
                    expert_token_nums_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host expertTokenNums copy failed");
    }

    CumsumReport report;
    for (size_t idx = 0; idx < cumsum_elems; ++idx) {
        UpdateCumsumReport(report, "cumsumMM", idx, expected_cumsum[idx], actual_cumsum[idx],
                           report.cumsum_mismatch_count);
    }
    for (size_t idx = 0; idx < expert_per_rank; ++idx) {
        UpdateCumsumReport(report, "expertTokenNums", idx, expected_expert_token_nums[idx],
                           actual_expert_token_nums[idx], report.expert_token_nums_mismatch_count);
    }
    const bool check_front_process = frontDebug.frontMode != 0U && front.frontCase == 11000U &&
                                     (frontDebug.frontDebugMode != 0U || frontDebug.frontStopStep != 0U);
    if (check_front_process) {
        if (frontDebug.frontDebugBytes < sizeof(DispatchFFNCombineFrontLayoutDebug)) {
            throw std::runtime_error("front debug area is too small for cumsum process check");
        }
        if (frontDebug.frontDebugOffset + sizeof(DispatchFFNCombineFrontLayoutDebug) >
            static_cast<uint64_t>(workspace_dev.bytes)) {
            throw std::runtime_error("front cumsum header is out of workspace range");
        }
        DispatchFFNCombineFrontLayoutDebug actual_debug;
        std::memset(&actual_debug, 0, sizeof(actual_debug));
        if (aclrtMemcpy(&actual_debug, sizeof(actual_debug), workspace_base + frontDebug.frontDebugOffset,
                        sizeof(actual_debug), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            throw std::runtime_error("device->host front cumsum header copy failed");
        }
        UpdateCumsumReport(report, "cumsumCore0Only", 0, 1, static_cast<int32_t>(actual_debug.cumsumCore0Only),
                           report.process_mismatch_count);
        UpdateCumsumReport(report, "cumsumRows", 0, static_cast<int32_t>(cfg.world_size),
                           static_cast<int32_t>(actual_debug.cumsumRows), report.process_mismatch_count);
        UpdateCumsumReport(report, "cumsumLocalExpertCount", 0, static_cast<int32_t>(expert_per_rank),
                           static_cast<int32_t>(actual_debug.cumsumLocalExpertCount), report.process_mismatch_count);
        UpdateCumsumReport(report, "cumsumFormulaReference", 0, 1,
                           static_cast<int32_t>(actual_debug.cumsumFormulaReference), report.process_mismatch_count);
        UpdateCumsumReport(report, "cumsumExpertTokenNumsWritten", 0, 1,
                           static_cast<int32_t>(actual_debug.cumsumExpertTokenNumsWritten),
                           report.process_mismatch_count);
    }
    report.pass = report.cumsum_mismatch_count == 0 && report.expert_token_nums_mismatch_count == 0 &&
                  report.process_mismatch_count == 0;
    return report;
}

void UpdateFrontDispatchContractReport(FrontDispatchContractReport &report, const std::string &field, size_t idx,
                                       uint64_t expected, uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

FrontDispatchContractReport CheckFrontDispatchContract(const CaseConfig &cfg,
                                                       const DispatchFFNCombineBuildResult &build,
                                                       const StandaloneRankRuntime &runtime,
                                                       const DeviceBuffer &workspace_dev,
                                                       const DeviceBuffer &expert_token_nums_dev,
                                                       const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t local_begin = rank * expert_per_rank;
    const size_t token_elems = static_cast<size_t>(rank_size) * expert_num_aligned;
    const size_t token_bytes = token_elems * sizeof(int32_t);
    const size_t local_meta_elems = static_cast<size_t>(rank_size) * expert_per_rank;
    const size_t local_meta_bytes = local_meta_elems * sizeof(int32_t);
    const size_t expert_token_nums_bytes = static_cast<size_t>(expert_per_rank) * sizeof(int32_t);

    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);

    std::vector<uint32_t> expected_group_base(expert_per_rank, 0);
    std::vector<uint32_t> expected_current_m(expert_per_rank, 0);
    uint32_t running_group_base = 0;
    for (uint32_t group = 0; group < expert_per_rank; ++group) {
        expected_group_base[group] = running_group_base;
        uint32_t current_m = 0;
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            current_m += static_cast<uint32_t>(
                expected_tokens[static_cast<size_t>(src_rank) * expert_num_aligned + local_begin + group]);
        }
        expected_current_m[group] = current_m;
        running_group_base += current_m;
    }

    std::vector<int32_t> actual_tokens(token_elems, 0);
    const auto *window_base = reinterpret_cast<const uint8_t *>(runtime.hccl.WindowIn(runtime.hccl.rank_id));
    const uint64_t token_offset = runtime.hccl.WindowBytes() - 2U * kHostMiB;
    if (aclrtMemcpy(actual_tokens.data(), token_bytes, window_base + token_offset, token_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host tokenPerExpert contract copy failed");
    }

    std::vector<int32_t> actual_presum(local_meta_elems, 0);
    std::vector<int32_t> actual_cumsum(local_meta_elems, 0);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual_presum.data(), local_meta_bytes, workspace_base + front.preSumBeforeRankOffset,
                    local_meta_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(actual_cumsum.data(), local_meta_bytes, workspace_base + front.cumsumMMOffset, local_meta_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host front_new dispatch contract metadata copy failed");
    }

    std::vector<int32_t> actual_expert_token_nums(expert_per_rank, 0);
    if (aclrtMemcpy(actual_expert_token_nums.data(), expert_token_nums_bytes, expert_token_nums_dev.ptr,
                    expert_token_nums_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host front_new dispatch contract expertTokenNums copy failed");
    }

    std::vector<uint32_t> actual_group_base(expert_per_rank, 0);
    running_group_base = 0;
    for (uint32_t group = 0; group < expert_per_rank; ++group) {
        actual_group_base[group] = running_group_base;
        running_group_base += static_cast<uint32_t>(actual_expert_token_nums[group]);
    }

    FrontDispatchContractReport report;
    report.checked_count = local_meta_elems;
    for (uint32_t group = 0; group < expert_per_rank; ++group) {
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            const size_t idx = static_cast<size_t>(group) * rank_size + src_rank;
            const size_t local_idx = static_cast<size_t>(src_rank) * expert_per_rank + group;
            const size_t token_idx = static_cast<size_t>(src_rank) * expert_num_aligned + local_begin + group;
            const uint32_t expected_rows = static_cast<uint32_t>(expected_tokens[token_idx]);
            const uint32_t actual_rows = static_cast<uint32_t>(actual_tokens[token_idx]);

            uint32_t expected_src_row_base = 0;
            for (uint32_t expert = 0; expert < local_begin + group; ++expert) {
                expected_src_row_base +=
                    static_cast<uint32_t>(expected_tokens[static_cast<size_t>(src_rank) * expert_num_aligned + expert]);
            }
            const uint32_t actual_src_row_base = static_cast<uint32_t>(actual_presum[local_idx]);
            uint32_t expected_cumsum_before_src = 0;
            for (uint32_t rank_idx = 0; rank_idx < src_rank; ++rank_idx) {
                expected_cumsum_before_src += static_cast<uint32_t>(
                    expected_tokens[static_cast<size_t>(rank_idx) * expert_num_aligned + local_begin + group]);
            }
            const uint32_t actual_cumsum_before_src =
                src_rank == 0U ?
                    0U :
                    static_cast<uint32_t>(actual_cumsum[static_cast<size_t>(src_rank - 1U) * expert_per_rank + group]);
            const uint32_t expected_dst_row_base = expected_group_base[group] + expected_cumsum_before_src;
            const uint32_t actual_dst_row_base = actual_group_base[group] + actual_cumsum_before_src;

            UpdateFrontDispatchContractReport(report, "groupBase", idx, expected_group_base[group],
                                              actual_group_base[group]);
            UpdateFrontDispatchContractReport(report, "currentM", idx, expected_current_m[group],
                                              static_cast<uint32_t>(actual_expert_token_nums[group]));
            UpdateFrontDispatchContractReport(report, "rows", idx, expected_rows, actual_rows);
            UpdateFrontDispatchContractReport(report, "srcRowBase", idx, expected_src_row_base, actual_src_row_base);
            UpdateFrontDispatchContractReport(report, "dstRowBase", idx, expected_dst_row_base, actual_dst_row_base);
        }
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

FrontDoneReport CheckFrontDone(const StandaloneRankRuntime &runtime)
{
    const auto *window_base = reinterpret_cast<const uint8_t *>(runtime.hccl.WindowIn(runtime.hccl.rank_id));
    const auto *signal_base = reinterpret_cast<const int32_t *>(window_base + runtime.hccl.WindowBytes() - kHostMiB);
    std::vector<int32_t> actual(static_cast<size_t>(runtime.hccl.world_size), 0);
    for (int src_rank = 0; src_rank < runtime.hccl.world_size; ++src_rank) {
        const size_t index = kHostFrontDoneBaseIndex + static_cast<size_t>(src_rank) * kHostFrontSignalStride;
        if (aclrtMemcpy(&actual[src_rank], sizeof(int32_t), signal_base + index, sizeof(int32_t),
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            throw std::runtime_error("device->host front-done signal copy failed");
        }
    }

    FrontDoneReport report;
    for (int src_rank = 0; src_rank < runtime.hccl.world_size; ++src_rank) {
        const int32_t expected = (src_rank == runtime.hccl.rank_id) ? 0 : 1;
        if (actual[src_rank] == expected) {
            continue;
        }
        if (report.mismatch_count == 0) {
            report.first_rank = src_rank;
            report.expected = expected;
            report.actual = actual[src_rank];
        }
        ++report.mismatch_count;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

FrontDoneReport CheckFrontDoneInactive(const StandaloneRankRuntime &runtime)
{
    const auto *window_base = reinterpret_cast<const uint8_t *>(runtime.hccl.WindowIn(runtime.hccl.rank_id));
    const auto *signal_base = reinterpret_cast<const int32_t *>(window_base + runtime.hccl.WindowBytes() - kHostMiB);
    std::vector<int32_t> actual(static_cast<size_t>(runtime.hccl.world_size), 0);
    for (int src_rank = 0; src_rank < runtime.hccl.world_size; ++src_rank) {
        const size_t index = kHostFrontDoneBaseIndex + static_cast<size_t>(src_rank) * kHostFrontSignalStride;
        if (aclrtMemcpy(&actual[src_rank], sizeof(int32_t), signal_base + index, sizeof(int32_t),
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            throw std::runtime_error("device->host front-done signal copy failed");
        }
    }

    FrontDoneReport report;
    for (int src_rank = 0; src_rank < runtime.hccl.world_size; ++src_rank) {
        constexpr int32_t expected = 0;
        if (actual[src_rank] == expected) {
            continue;
        }
        if (report.mismatch_count == 0) {
            report.first_rank = src_rank;
            report.expected = expected;
            report.actual = actual[src_rank];
        }
        ++report.mismatch_count;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

void UpdateDispatchLayoutReport(DispatchLayoutReport &report, const std::string &field, uint64_t expected,
                                uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

void UpdateFrontCheckLayoutReport(FrontCheckLayoutReport &report, const std::string &field, uint64_t expected,
                                  uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

FrontCheckLayoutReport CheckFrontCheckLayout(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                             const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    DispatchFFNCombineFrontCheckLayoutDebug actual;
    std::memset(&actual, 0, sizeof(actual));

    if (frontDebug.frontCheckDebugBytes < sizeof(actual)) {
        throw std::runtime_error("front_new debug area is too small for layout header");
    }
    if (frontDebug.frontCheckDebugOffset + sizeof(actual) > static_cast<uint64_t>(workspace_dev.bytes)) {
        throw std::runtime_error("front_new layout header is out of workspace range");
    }
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(&actual, sizeof(actual), workspace_base + frontDebug.frontCheckDebugOffset, sizeof(actual),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host front_new layout header copy failed");
    }

    FrontCheckLayoutReport report;
    report.helper_debug_mask = actual.helperDebugMask;
    report.helper_debug_failures = actual.helperDebugFailures;
    report.quant_expected_rows = actual.quantExpectedRows;
    report.quant_actual_rows = actual.quantActualRows;
    report.quant_expected_stores = actual.quantExpectedStores;
    report.quant_actual_stores = actual.quantActualStores;
    report.quant_active_cores = actual.quantActiveCores;
    report.quant_full_row_mode = actual.quantFullRowMode;
    report.quant_column_tiled_mode = actual.quantColumnTiledMode;
    report.quant_marker = actual.quantMarker;
    UpdateFrontCheckLayoutReport(report, "magic", kHostFrontCheckLayoutDebugMagic, actual.magic);
    UpdateFrontCheckLayoutReport(report, "workspaceBase", reinterpret_cast<uint64_t>(workspace_dev.ptr),
                                 actual.workspaceBase);
    UpdateFrontCheckLayoutReport(report, "expandedRowIdxOffset", front.expandedRowIdxOffset,
                                 actual.expandedRowIdxOffset);
    UpdateFrontCheckLayoutReport(report, "coreCountOffset", front.coreCountOffset, actual.coreCountOffset);
    UpdateFrontCheckLayoutReport(report, "localTokenPerExpertOffset", front.localTokenPerExpertOffset,
                                 actual.localTokenPerExpertOffset);
    UpdateFrontCheckLayoutReport(report, "expertBaseOffset", front.expertBaseOffset, actual.expertBaseOffset);
    UpdateFrontCheckLayoutReport(report, "coreBaseOffset", front.coreBaseOffset, actual.coreBaseOffset);
    UpdateFrontCheckLayoutReport(report, "cumsumMMOffset", front.cumsumMMOffset, actual.cumsumMMOffset);
    UpdateFrontCheckLayoutReport(report, "preSumBeforeRankOffset", front.preSumBeforeRankOffset,
                                 actual.preSumBeforeRankOffset);
    UpdateFrontCheckLayoutReport(report, "frontWorkspaceBytes", front.frontWorkspaceBytes, actual.frontWorkspaceBytes);
    UpdateFrontCheckLayoutReport(report, "frontSortCheckOffset", frontDebug.frontSortCheckOffset,
                                 actual.frontSortCheckOffset);
    UpdateFrontCheckLayoutReport(report, "frontSortCheckBytes", frontDebug.frontSortCheckBytes,
                                 actual.frontSortCheckBytes);
    UpdateFrontCheckLayoutReport(report, "frontMergeCheckOffset", frontDebug.frontMergeCheckOffset,
                                 actual.frontMergeCheckOffset);
    UpdateFrontCheckLayoutReport(report, "frontMergeCheckBytes", frontDebug.frontMergeCheckBytes,
                                 actual.frontMergeCheckBytes);
    UpdateFrontCheckLayoutReport(report, "frontCountScratchOffset", front.frontCountScratchOffset,
                                 actual.frontCountScratchOffset);
    UpdateFrontCheckLayoutReport(report, "frontCountScratchBytes", front.frontCountScratchBytes,
                                 actual.frontCountScratchBytes);
    UpdateFrontCheckLayoutReport(report, "frontCheckDebugOffset", frontDebug.frontCheckDebugOffset,
                                 actual.frontCheckDebugOffset);
    UpdateFrontCheckLayoutReport(report, "frontCheckDebugBytes", frontDebug.frontCheckDebugBytes,
                                 actual.frontCheckDebugBytes);
    UpdateFrontCheckLayoutReport(report, "frontCheckDebugBytesPerWorker", frontDebug.frontCheckDebugBytesPerWorker,
                                 actual.frontCheckDebugBytesPerWorker);
    UpdateFrontCheckLayoutReport(report, "peerOffsetA", 0, actual.peerOffsetA);
    UpdateFrontCheckLayoutReport(report, "peerOffsetPeerTokenPerExpert", runtime.hccl.WindowBytes() - 2U * kHostMiB,
                                 actual.peerOffsetPeerTokenPerExpert);
    UpdateFrontCheckLayoutReport(report, "rank", static_cast<uint64_t>(runtime.hccl.rank_id), actual.rank);
    UpdateFrontCheckLayoutReport(report, "rankSize", static_cast<uint64_t>(runtime.hccl.world_size), actual.rankSize);
    UpdateFrontCheckLayoutReport(report, "coreIdx", 0, actual.coreIdx);
    UpdateFrontCheckLayoutReport(report, "coreNum", cfg.aiv_num, actual.coreNum);
    UpdateFrontCheckLayoutReport(report, "stageNum", front.stageNum, actual.stageNum);
    UpdateFrontCheckLayoutReport(report, "problemM", cfg.m, actual.problemM);
    UpdateFrontCheckLayoutReport(report, "problemK", cfg.k, actual.problemK);
    UpdateFrontCheckLayoutReport(report, "topK", cfg.topk, actual.topK);
    UpdateFrontCheckLayoutReport(report, "expertPerRank", cfg.expert_per_rank, actual.expertPerRank);
    UpdateFrontCheckLayoutReport(report, "expertNum", front.expertNum, actual.expertNum);
    UpdateFrontCheckLayoutReport(report, "expertNumAligned", front.expertNumAligned, actual.expertNumAligned);
    UpdateFrontCheckLayoutReport(report, "frontPath", front.frontPath, actual.frontPath);
    UpdateFrontCheckLayoutReport(report, "frontCheckMode", frontDebug.frontCheckMode, actual.frontCheckMode);
    UpdateFrontCheckLayoutReport(report, "frontCase", front.frontCase, actual.frontCase);
    UpdateFrontCheckLayoutReport(report, "frontCheckDebugMode", frontDebug.frontCheckDebugMode,
                                 actual.frontCheckDebugMode);
    UpdateFrontCheckLayoutReport(report, "frontCheckStopStep", frontDebug.frontCheckStopStep,
                                 actual.frontCheckStopStep);
    UpdateFrontCheckLayoutReport(report, "frontSortCheckDebugStep", frontDebug.frontSortCheckDebugStep,
                                 actual.frontSortCheckDebugStep);
    UpdateFrontCheckLayoutReport(report, "frontCheckDebugSync", frontDebug.frontCheckDebugSync,
                                 actual.frontCheckDebugSync);
    UpdateFrontCheckLayoutReport(report, "routeElems", front.routeElems, actual.routeElems);
    UpdateFrontCheckLayoutReport(report, "alignedRouteElems", front.alignedRouteElems, actual.alignedRouteElems);
    UpdateFrontCheckLayoutReport(report, "sortLoopMaxElement", front.sortLoopMaxElement, actual.sortLoopMaxElement);
    UpdateFrontCheckLayoutReport(report, "fullLoadMaxRouteElems", front.fullLoadMaxRouteElems,
                                 actual.fullLoadMaxRouteElems);
    UpdateFrontCheckLayoutReport(report, "sortNeedCoreNum", front.sortNeedCoreNum, actual.sortNeedCoreNum);
    UpdateFrontCheckLayoutReport(report, "sortPerCoreElems", front.sortPerCoreElems, actual.sortPerCoreElems);
    UpdateFrontCheckLayoutReport(report, "sortLastCoreElems", front.sortLastCoreElems, actual.sortLastCoreElems);
    UpdateFrontCheckLayoutReport(report, "sortPerCoreLoops", front.sortPerCoreLoops, actual.sortPerCoreLoops);
    UpdateFrontCheckLayoutReport(report, "sortPerCorePerLoopElems", front.sortPerCorePerLoopElems,
                                 actual.sortPerCorePerLoopElems);
    UpdateFrontCheckLayoutReport(report, "sortPerCoreLastLoopElems", front.sortPerCoreLastLoopElems,
                                 actual.sortPerCoreLastLoopElems);
    UpdateFrontCheckLayoutReport(report, "sortLastCoreLoops", front.sortLastCoreLoops, actual.sortLastCoreLoops);
    UpdateFrontCheckLayoutReport(report, "sortLastCorePerLoopElems", front.sortLastCorePerLoopElems,
                                 actual.sortLastCorePerLoopElems);
    UpdateFrontCheckLayoutReport(report, "sortLastCoreLastLoopElems", front.sortLastCoreLastLoopElems,
                                 actual.sortLastCoreLastLoopElems);
    UpdateFrontCheckLayoutReport(report, "sortVmsMiddleNeedCoreNum", front.sortVmsMiddleNeedCoreNum,
                                 actual.sortVmsMiddleNeedCoreNum);
    UpdateFrontCheckLayoutReport(report, "sortOutLoopMaxElems", front.sortOutLoopMaxElems, actual.sortOutLoopMaxElems);
    UpdateFrontCheckLayoutReport(report, "frontMode", frontDebug.frontMode, actual.frontMode);
    UpdateFrontCheckLayoutReport(report, "helperDebugMask", kHostFrontHelperExpectedMask, actual.helperDebugMask);
    UpdateFrontCheckLayoutReport(report, "helperDebugFailures", 0U, actual.helperDebugFailures);
    UpdateFrontCheckLayoutReport(report, "helperSortLenFloat33", 66U, actual.helperSortLenFloat33);
    UpdateFrontCheckLayoutReport(report, "helperSortOffsetFloat33", 66U, actual.helperSortOffsetFloat33);
    UpdateFrontCheckLayoutReport(report, "helperPackedStride", cfg.k + 32U, actual.helperPackedStride);
    UpdateFrontCheckLayoutReport(report, "helperPackedScaleOffset", cfg.k, actual.helperPackedScaleOffset);
    UpdateFrontCheckLayoutReport(report, "marker", 1U, actual.marker);
    if (frontDebug.frontCheckDebugMode != 0U && frontDebug.frontCheckStopStep >= 4U) {
        UpdateFrontCheckLayoutReport(report, "quantExpectedRows", cfg.m, actual.quantExpectedRows);
        UpdateFrontCheckLayoutReport(report, "quantActualRows", cfg.m, actual.quantActualRows);
        UpdateFrontCheckLayoutReport(report, "quantExpectedStores", front.routeElems, actual.quantExpectedStores);
        UpdateFrontCheckLayoutReport(report, "quantActualStores", front.routeElems, actual.quantActualStores);
        UpdateFrontCheckLayoutReport(report, "quantSingleMode", 1U,
                                     (actual.quantFullRowMode + actual.quantColumnTiledMode) == 1 ? 1U : 0U);
        UpdateFrontCheckLayoutReport(report, "quantMarker", 1U, actual.quantMarker);
    }

    UpdateFrontCheckLayoutReport(
        report, "frontSortCheckAligned", 1U,
        (frontDebug.frontSortCheckOffset % 512U == 0U && frontDebug.frontSortCheckBytes % 512U == 0U) ? 1U : 0U);
    UpdateFrontCheckLayoutReport(
        report, "frontMergeCheckAligned", 1U,
        (frontDebug.frontMergeCheckOffset % 512U == 0U && frontDebug.frontMergeCheckBytes % 512U == 0U) ? 1U : 0U);
    UpdateFrontCheckLayoutReport(
        report, "frontCountScratchAligned", 1U,
        (front.frontCountScratchOffset % 512U == 0U && front.frontCountScratchBytes % 512U == 0U) ? 1U : 0U);
    UpdateFrontCheckLayoutReport(
        report, "frontCheckDebugAligned", 1U,
        (frontDebug.frontCheckDebugOffset % 512U == 0U && frontDebug.frontCheckDebugBytes % 512U == 0U) ? 1U : 0U);
    UpdateFrontCheckLayoutReport(
        report, "frontSortCheckBeforeMerge", 1U,
        frontDebug.frontSortCheckOffset + frontDebug.frontSortCheckBytes <= frontDebug.frontMergeCheckOffset ? 1U : 0U);
    UpdateFrontCheckLayoutReport(
        report, "frontMergeCheckBeforeCountScratch", 1U,
        frontDebug.frontMergeCheckOffset + frontDebug.frontMergeCheckBytes <= front.frontCountScratchOffset ? 1U : 0U);
    UpdateFrontCheckLayoutReport(
        report, "frontCountScratchBeforeCheckDebug", 1U,
        front.frontCountScratchOffset + front.frontCountScratchBytes <= frontDebug.frontCheckDebugOffset ? 1U : 0U);
    UpdateFrontCheckLayoutReport(
        report, "frontCheckDebugInsideFrontWorkspace", 1U,
        frontDebug.frontCheckDebugOffset + frontDebug.frontCheckDebugBytes <= front.frontWorkspaceBytes ? 1U : 0U);
    if (front.frontCase == 11010U) {
        const uint64_t covered_elems =
            front.sortNeedCoreNum == 0U ?
                0U :
                static_cast<uint64_t>(front.sortNeedCoreNum - 1U) * front.sortPerCoreElems + front.sortLastCoreElems;
        UpdateFrontCheckLayoutReport(report, "frontSortCoverage", front.routeElems, covered_elems);
        UpdateFrontCheckLayoutReport(report, "frontSortNeedCoreClamp", 1U,
                                     front.sortNeedCoreNum <= cfg.aiv_num ? 1U : 0U);
        UpdateFrontCheckLayoutReport(report, "frontSortPerCoreAligned", 1U,
                                     front.sortPerCoreElems % 32U == 0U ? 1U : 0U);
        const uint32_t expected_vms_middle = front.sortNeedCoreNum <= 4U ? 0U : (front.sortNeedCoreNum + 3U) / 4U;
        UpdateFrontCheckLayoutReport(report, "frontVmsMiddleNeedCoreNum", expected_vms_middle,
                                     front.sortVmsMiddleNeedCoreNum);
        UpdateFrontCheckLayoutReport(report, "frontSortOutLoopMaxElems", 2040U, front.sortOutLoopMaxElems);
    }

    report.pass = report.mismatch_count == 0;
    return report;
}

DispatchLayoutReport CheckFrontLayout(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                      const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    DispatchFFNCombineFrontLayoutDebug actual;
    std::memset(&actual, 0, sizeof(actual));

    if (frontDebug.frontDebugBytes < sizeof(actual)) {
        throw std::runtime_error("front debug area is too small for layout header");
    }
    if (frontDebug.frontDebugOffset + sizeof(actual) > static_cast<uint64_t>(workspace_dev.bytes)) {
        throw std::runtime_error("front layout header is out of workspace range");
    }
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(&actual, sizeof(actual), workspace_base + frontDebug.frontDebugOffset, sizeof(actual),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host front layout header copy failed");
    }

    DispatchLayoutReport report;
    const uint32_t expected_route_elems = cfg.m * cfg.topk;
    const uint32_t expected_sort_need_core_num = front.sortNeedCoreNum;
    const uint32_t expected_active_copy_cores =
        std::min<uint32_t>(static_cast<uint32_t>(runtime.hccl.world_size), cfg.aiv_num);
    const uint64_t route_table_bytes =
        AlignUpU64(static_cast<uint64_t>(front.alignedRouteElems) * sizeof(int32_t), 512U);
    const uint64_t sort_ws_bytes =
        std::max(AlignUpU64(static_cast<uint64_t>(front.alignedRouteElems) * 2U * sizeof(int32_t), 512U),
                 AlignUpU64(static_cast<uint64_t>(front.alignedRouteElems) * 2U * sizeof(float), 512U));

    UpdateDispatchLayoutReport(report, "magic", kHostFrontLayoutDebugMagic, actual.magic);
    UpdateDispatchLayoutReport(report, "workspaceBase", reinterpret_cast<uint64_t>(workspace_dev.ptr),
                               actual.workspaceBase);
    UpdateDispatchLayoutReport(report, "expandedRowIdxOffset", front.expandedRowIdxOffset, actual.expandedRowIdxOffset);
    UpdateDispatchLayoutReport(report, "frontExpandedExpertOffset", front.frontExpandedExpertOffset,
                               actual.frontExpandedExpertOffset);
    UpdateDispatchLayoutReport(report, "frontExpandDstToSrcOffset", front.frontExpandDstToSrcOffset,
                               actual.frontExpandDstToSrcOffset);
    UpdateDispatchLayoutReport(report, "frontSortWs0Offset", front.frontSortWs0Offset, actual.frontSortWs0Offset);
    UpdateDispatchLayoutReport(report, "frontSortWs1Offset", front.frontSortWs1Offset, actual.frontSortWs1Offset);
    UpdateDispatchLayoutReport(report, "frontQuantTmpOffset", front.frontQuantTmpOffset, actual.frontQuantTmpOffset);
    UpdateDispatchLayoutReport(report, "frontQuantTmpBytes", front.frontQuantTmpBytes, actual.frontQuantTmpBytes);
    UpdateDispatchLayoutReport(report, "frontDebugOffset", frontDebug.frontDebugOffset, actual.frontDebugOffset);
    UpdateDispatchLayoutReport(report, "frontDebugBytes", frontDebug.frontDebugBytes, actual.frontDebugBytes);
    UpdateDispatchLayoutReport(report, "frontWorkspaceBytes", front.frontWorkspaceBytes, actual.frontWorkspaceBytes);
    UpdateDispatchLayoutReport(report, "peerOffsetA", 0U, actual.peerOffsetA);
    UpdateDispatchLayoutReport(report, "peerOffsetPeerTokenPerExpert", runtime.hccl.WindowBytes() - 2U * kHostMiB,
                               actual.peerOffsetPeerTokenPerExpert);
    UpdateDispatchLayoutReport(report, "rank", static_cast<uint64_t>(runtime.hccl.rank_id), actual.rank);
    UpdateDispatchLayoutReport(report, "rankSize", static_cast<uint64_t>(runtime.hccl.world_size), actual.rankSize);
    UpdateDispatchLayoutReport(report, "coreIdx", 0U, actual.coreIdx);
    UpdateDispatchLayoutReport(report, "coreNum", cfg.aiv_num, actual.coreNum);
    UpdateDispatchLayoutReport(report, "stageNum", front.stageNum, actual.stageNum);
    UpdateDispatchLayoutReport(report, "problemM", cfg.m, actual.problemM);
    UpdateDispatchLayoutReport(report, "problemK", cfg.k, actual.problemK);
    UpdateDispatchLayoutReport(report, "topK", cfg.topk, actual.topK);
    UpdateDispatchLayoutReport(report, "expertPerRank", cfg.expert_per_rank, actual.expertPerRank);
    UpdateDispatchLayoutReport(report, "expertNum", front.expertNum, actual.expertNum);
    UpdateDispatchLayoutReport(report, "expertNumAligned", front.expertNumAligned, actual.expertNumAligned);
    UpdateDispatchLayoutReport(report, "frontCase", front.frontCase, actual.frontCase);
    UpdateDispatchLayoutReport(report, "frontCaseTiling", front.frontCase, actual.frontCase);
    UpdateDispatchLayoutReport(report, "frontMode", frontDebug.frontMode, actual.frontMode);
    UpdateDispatchLayoutReport(report, "frontDebugMode", frontDebug.frontDebugMode, actual.frontDebugMode);
    UpdateDispatchLayoutReport(report, "frontStopStep", frontDebug.frontStopStep, actual.frontStopStep);
    UpdateDispatchLayoutReport(report, "frontSortDebugStep", frontDebug.frontSortDebugStep, actual.frontSortDebugStep);
    UpdateDispatchLayoutReport(report, "routeElems", front.routeElems, actual.routeElems);
    UpdateDispatchLayoutReport(report, "routeElemsExpected", expected_route_elems, front.routeElems);
    UpdateDispatchLayoutReport(report, "alignedRouteElems", front.alignedRouteElems, actual.alignedRouteElems);
    UpdateDispatchLayoutReport(report, "sortNeedCoreNum", front.sortNeedCoreNum, actual.sortNeedCoreNum);
    UpdateDispatchLayoutReport(report, "sortNeedCoreNumExpected", expected_sort_need_core_num, front.sortNeedCoreNum);
    UpdateDispatchLayoutReport(report, "sortPerCoreElems", front.sortPerCoreElems, actual.sortPerCoreElems);
    UpdateDispatchLayoutReport(report, "sortLastCoreElems", front.sortLastCoreElems, actual.sortLastCoreElems);
    UpdateDispatchLayoutReport(report, "sortPerCoreLoops", front.sortPerCoreLoops, actual.sortPerCoreLoops);
    UpdateDispatchLayoutReport(report, "sortPerCorePerLoopElems", front.sortPerCorePerLoopElems,
                               actual.sortPerCorePerLoopElems);
    UpdateDispatchLayoutReport(report, "sortPerCoreLastLoopElems", front.sortPerCoreLastLoopElems,
                               actual.sortPerCoreLastLoopElems);
    UpdateDispatchLayoutReport(report, "sortLastCoreLoops", front.sortLastCoreLoops, actual.sortLastCoreLoops);
    UpdateDispatchLayoutReport(report, "sortLastCorePerLoopElems", front.sortLastCorePerLoopElems,
                               actual.sortLastCorePerLoopElems);
    UpdateDispatchLayoutReport(report, "sortLastCoreLastLoopElems", front.sortLastCoreLastLoopElems,
                               actual.sortLastCoreLastLoopElems);
    UpdateDispatchLayoutReport(report, "sortVmsMiddleNeedCoreNum", front.sortVmsMiddleNeedCoreNum,
                               actual.sortVmsMiddleNeedCoreNum);
    UpdateDispatchLayoutReport(report, "sortOutLoopMaxElems", front.sortOutLoopMaxElems, actual.sortOutLoopMaxElems);
    UpdateDispatchLayoutReport(report, "activeCopyCores", expected_active_copy_cores, actual.activeCopyCores);
    UpdateDispatchLayoutReport(report, "noOldNotifyWait", 1U, actual.noOldNotifyWait);
    if (front.frontCase == 21000U) {
        const HostFrontGatherQuantTiling expected_full_load_tiling =
            HostBuildFrontGatherQuantTiling(front.routeElems, cfg.aiv_num, cfg.k);
        const uint32_t expected_tile_length = static_cast<uint32_t>(
            AlignUpU64(front.sortLastCorePerLoopElems == 0U ? front.routeElems : front.sortLastCorePerLoopElems, 4U));
        const uint32_t expected_sort_num = static_cast<uint32_t>(AlignUpU64(expected_tile_length, 32U));
        const uint64_t expected_required_ub =
            HostMc2FullLoadDynamicRequiredUbBytes(front.routeElems, cfg.k, front.expertNum);
        const uint64_t expected_remain_ub =
            expected_required_ub <= AtlasA5::UB_SIZE ? AtlasA5::UB_SIZE - expected_required_ub : 0U;
        const uint32_t expected_full_load_condition =
            expected_route_elems != 0U && expected_route_elems <= front.sortLoopMaxElement && cfg.k <= 8192U &&
                    cfg.k % kHostFrontPackedScaleBytes == 0U && expected_required_ub <= AtlasA5::UB_SIZE ?
                1U :
                0U;
        UpdateDispatchLayoutReport(report, "producerClass", 3U, actual.producerClass);
        UpdateDispatchLayoutReport(report, "activeAivNum", expected_full_load_tiling.need_core_num,
                                   actual.activeAivNum);
        UpdateDispatchLayoutReport(report, "totalLength", expected_route_elems, actual.totalLength);
        UpdateDispatchLayoutReport(report, "sortLoopMaxElement", front.sortLoopMaxElement, actual.sortLoopMaxElement);
        UpdateDispatchLayoutReport(report, "routeElemsAlias", front.routeElems, actual.routeElemsAlias);
        UpdateDispatchLayoutReport(report, "tileLength", expected_tile_length, actual.tileLength);
        UpdateDispatchLayoutReport(report, "sortNum", expected_sort_num, actual.sortNum);
        UpdateDispatchLayoutReport(report, "fullLoadRequiredUbBytes", expected_required_ub,
                                   actual.fullLoadRequiredUbBytes);
        UpdateDispatchLayoutReport(report, "fullLoadRemainUbBytes", expected_remain_ub, actual.fullLoadRemainUbBytes);
        UpdateDispatchLayoutReport(report, "fullLoadCondition", expected_full_load_condition, actual.fullLoadCondition);
        UpdateDispatchLayoutReport(report, "fullLoadNeedCoreNum", expected_full_load_tiling.need_core_num,
                                   actual.fullLoadNeedCoreNum);
        UpdateDispatchLayoutReport(report, "fullLoadPerCoreRows", expected_full_load_tiling.per_core_rows,
                                   actual.fullLoadPerCoreRows);
        UpdateDispatchLayoutReport(report, "fullLoadLastCoreRows", expected_full_load_tiling.last_core_rows,
                                   actual.fullLoadLastCoreRows);
        UpdateDispatchLayoutReport(report, "fullLoadActivateRows", front.routeElems, actual.fullLoadActivateRows);
        UpdateDispatchLayoutReport(report, "fullLoadCoreRows", expected_full_load_tiling.per_core_rows,
                                   actual.fullLoadCoreRows);
        UpdateDispatchLayoutReport(report, "fullLoadColsScale", cfg.k + kHostFrontPackedScaleBytes,
                                   actual.fullLoadColsScale);
        UpdateDispatchLayoutReport(report, "fullLoadRequiredUbFits", expected_required_ub <= AtlasA5::UB_SIZE ? 1U : 0U,
                                   actual.fullLoadRequiredUbFits);
    }
    if (front.frontCase == 11000U) {
        const uint32_t expected_tile_length = static_cast<uint32_t>(
            AlignUpU64(front.sortLastCorePerLoopElems == 0U ? front.routeElems : front.sortLastCorePerLoopElems, 4U));
        const uint32_t expected_sort_num = static_cast<uint32_t>(AlignUpU64(expected_tile_length, 32U));
        const uint64_t expected_required_ub = static_cast<uint64_t>(expected_sort_num) * sizeof(int32_t) * 2U * 4U;
        UpdateDispatchLayoutReport(report, "producerClass", 2U, actual.producerClass);
        UpdateDispatchLayoutReport(report, "sortOwnerCore", 0U, actual.sortOwnerCore);
        UpdateDispatchLayoutReport(report, "activeAivNum", cfg.aiv_num, actual.activeAivNum);
        UpdateDispatchLayoutReport(report, "totalLength", expected_route_elems, actual.totalLength);
        UpdateDispatchLayoutReport(report, "sortLoopMaxElement", front.sortLoopMaxElement, actual.sortLoopMaxElement);
        UpdateDispatchLayoutReport(report, "oneCoreCondition", 1U,
                                   expected_route_elems <= front.sortLoopMaxElement ? 1U : 0U);
        UpdateDispatchLayoutReport(report, "oneCoreConditionActual", 1U, actual.oneCoreCondition);
        UpdateDispatchLayoutReport(report, "routeElemsAlias", front.routeElems, actual.routeElemsAlias);
        UpdateDispatchLayoutReport(report, "tileLength", expected_tile_length, actual.tileLength);
        UpdateDispatchLayoutReport(report, "sortNum", expected_sort_num, actual.sortNum);
        UpdateDispatchLayoutReport(report, "requiredUbBytesSingleSort", expected_required_ub,
                                   actual.requiredUbBytesSingleSort);
        UpdateDispatchLayoutReport(report, "singleSortUbFits", 1U, actual.singleSortUbFits);
    }
    const bool front_full_load = front.frontCase == 21000U;
    const bool front_control_enabled = frontDebug.frontDebugMode != 0U || frontDebug.frontStopStep != 0U;
    const bool expect_front_boundary = frontDebug.frontStopStep >= 8U || (front_full_load && front_control_enabled);
    if (expect_front_boundary) {
        UpdateDispatchLayoutReport(report, "offsetAPublished", 1U, actual.offsetAPublished);
        UpdateDispatchLayoutReport(report, "tokenPerExpertPublished", 1U, actual.tokenPerExpertPublished);
        UpdateDispatchLayoutReport(report, "preSumPublished", 1U, actual.preSumPublished);
        UpdateDispatchLayoutReport(report, "cumsumPublished", 1U, actual.cumsumPublished);
        UpdateDispatchLayoutReport(report, "expertTokenNumsPublished", 1U, actual.expertTokenNumsPublished);
        UpdateDispatchLayoutReport(report, "dispatchContractReady", 1U, actual.dispatchContractReady);
    }
    UpdateDispatchLayoutReport(report, "marker", 1U, actual.marker);

    UpdateDispatchLayoutReport(
        report, "frontExpandedExpertAligned", 1U,
        (front.frontExpandedExpertOffset % 512U == 0U && route_table_bytes % 512U == 0U) ? 1U : 0U);
    UpdateDispatchLayoutReport(
        report, "frontExpandDstToSrcAligned", 1U,
        (front.frontExpandDstToSrcOffset % 512U == 0U && route_table_bytes % 512U == 0U) ? 1U : 0U);
    UpdateDispatchLayoutReport(report, "frontSortWs0Aligned", 1U,
                               (front.frontSortWs0Offset % 512U == 0U && sort_ws_bytes % 512U == 0U) ? 1U : 0U);
    UpdateDispatchLayoutReport(report, "frontSortWs1Aligned", 1U,
                               (front.frontSortWs1Offset % 512U == 0U && sort_ws_bytes % 512U == 0U) ? 1U : 0U);
    UpdateDispatchLayoutReport(
        report, "frontQuantTmpAligned", 1U,
        (front.frontQuantTmpOffset % 512U == 0U && front.frontQuantTmpBytes % 512U == 0U) ? 1U : 0U);
    UpdateDispatchLayoutReport(
        report, "frontDebugAligned", 1U,
        (frontDebug.frontDebugOffset % 512U == 0U && frontDebug.frontDebugBytes % 512U == 0U) ? 1U : 0U);
    UpdateDispatchLayoutReport(
        report, "frontDebugInsideFrontWorkspace", 1U,
        frontDebug.frontDebugOffset + frontDebug.frontDebugBytes <= front.frontWorkspaceBytes ? 1U : 0U);

    report.pass = report.mismatch_count == 0;
    return report;
}

DispatchLayoutReport CheckDispatchLayout(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                         const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev)
{
    constexpr uint64_t kHostDispatchLayoutDebugMagic = 0x5635444953504c59ULL;
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &dispatch = build.tiling.dispatchTiling;
    DispatchFFNCombineDispatchLayoutDebug actual;
    std::memset(&actual, 0, sizeof(actual));

    if (dispatch.dispatchDebugBytes < sizeof(actual)) {
        throw std::runtime_error("dispatch debug area is too small for layout header");
    }
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(&actual, sizeof(actual), workspace_base + dispatch.dispatchDebugOffset, sizeof(actual),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host dispatch layout header copy failed");
    }

    DispatchLayoutReport report;
    UpdateDispatchLayoutReport(report, "magic", kHostDispatchLayoutDebugMagic, actual.magic);
    UpdateDispatchLayoutReport(report, "workspaceBase", reinterpret_cast<uint64_t>(workspace_dev.ptr),
                               actual.workspaceBase);
    UpdateDispatchLayoutReport(report, "gmAOffset", dispatch.gmAOffset, actual.gmAOffset);
    UpdateDispatchLayoutReport(report, "perTokenScaleOffset", dispatch.perTokenScaleOffset, actual.perTokenScaleOffset);
    UpdateDispatchLayoutReport(report, "dispatchScratchOffset", dispatch.dispatchScratchOffset,
                               actual.dispatchScratchOffset);
    UpdateDispatchLayoutReport(report, "dispatchScratchBytes", dispatch.dispatchScratchBytes,
                               actual.dispatchScratchBytes);
    UpdateDispatchLayoutReport(report, "dispatchScratchBytesPerAiv", dispatch.dispatchScratchBytesPerAiv,
                               actual.dispatchScratchBytesPerAiv);
    UpdateDispatchLayoutReport(report, "dispatchScratchCoreOffset", dispatch.dispatchScratchOffset,
                               actual.dispatchScratchCoreOffset);
    UpdateDispatchLayoutReport(report, "dispatchDebugOffset", dispatch.dispatchDebugOffset, actual.dispatchDebugOffset);
    UpdateDispatchLayoutReport(report, "dispatchDebugBytes", dispatch.dispatchDebugBytes, actual.dispatchDebugBytes);
    UpdateDispatchLayoutReport(report, "dispatchTileBytes", dispatch.dispatchTileBytes, actual.dispatchTileBytes);
    UpdateDispatchLayoutReport(report, "cumsumMMOffset", front.cumsumMMOffset, actual.cumsumMMOffset);
    UpdateDispatchLayoutReport(report, "preSumBeforeRankOffset", front.preSumBeforeRankOffset,
                               actual.preSumBeforeRankOffset);
    UpdateDispatchLayoutReport(report, "frontWorkspaceBytes", front.frontWorkspaceBytes, actual.frontWorkspaceBytes);
    UpdateDispatchLayoutReport(report, "peerOffsetA", 0, actual.peerOffsetA);
    UpdateDispatchLayoutReport(report, "peerOffsetPeerTokenPerExpert", runtime.hccl.WindowBytes() - 2U * kHostMiB,
                               actual.peerOffsetPeerTokenPerExpert);
    UpdateDispatchLayoutReport(report, "rank", static_cast<uint64_t>(runtime.hccl.rank_id), actual.rank);
    UpdateDispatchLayoutReport(report, "rankSize", static_cast<uint64_t>(runtime.hccl.world_size), actual.rankSize);
    UpdateDispatchLayoutReport(report, "coreIdx", 0, actual.coreIdx);
    UpdateDispatchLayoutReport(report, "coreNum", cfg.aiv_num, actual.coreNum);
    UpdateDispatchLayoutReport(report, "stageNum", front.stageNum, actual.stageNum);
    UpdateDispatchLayoutReport(report, "problemK", cfg.k, actual.problemK);
    UpdateDispatchLayoutReport(report, "maxOutputSize", cfg.max_output_size, actual.maxOutputSize);
    UpdateDispatchLayoutReport(report, "packedStride", static_cast<uint64_t>(cfg.k) + 32U, actual.packedStride);

    report.pass = report.mismatch_count == 0;
    return report;
}

DispatchLayoutReport CheckDispatchGatherLayout(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                               const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &dispatch = build.tiling.dispatchTiling;
    DispatchFFNCombineDispatchGatherLayoutDebug actual;
    std::memset(&actual, 0, sizeof(actual));

    if (dispatch.dispatchGatherDebugBytes < sizeof(actual)) {
        throw std::runtime_error("dispatch debug area is too small for layout header");
    }
    if (dispatch.dispatchGatherDebugOffset + sizeof(actual) > static_cast<uint64_t>(workspace_dev.bytes)) {
        throw std::runtime_error("dispatch layout header is out of workspace range");
    }
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(&actual, sizeof(actual), workspace_base + dispatch.dispatchGatherDebugOffset, sizeof(actual),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host dispatch layout header copy failed");
    }

    DispatchLayoutReport report;
    const uint32_t expected_active_copy_cores =
        std::min<uint32_t>(static_cast<uint32_t>(runtime.hccl.world_size), cfg.aiv_num);
    const uint64_t expected_scratch_core_offset = dispatch.dispatchGatherScratchOffset;

    UpdateDispatchLayoutReport(report, "magic", kHostDispatchGatherLayoutDebugMagic, actual.magic);
    UpdateDispatchLayoutReport(report, "workspaceBase", reinterpret_cast<uint64_t>(workspace_dev.ptr),
                               actual.workspaceBase);
    UpdateDispatchLayoutReport(report, "gmAOffset", dispatch.gmAOffset, actual.gmAOffset);
    UpdateDispatchLayoutReport(report, "perTokenScaleOffset", dispatch.perTokenScaleOffset, actual.perTokenScaleOffset);
    UpdateDispatchLayoutReport(report, "dispatchGatherScratchOffset", dispatch.dispatchGatherScratchOffset,
                               actual.dispatchGatherScratchOffset);
    UpdateDispatchLayoutReport(report, "dispatchGatherScratchBytes", dispatch.dispatchGatherScratchBytes,
                               actual.dispatchGatherScratchBytes);
    UpdateDispatchLayoutReport(report, "dispatchGatherScratchBytesPerAiv", dispatch.dispatchGatherScratchBytesPerAiv,
                               actual.dispatchGatherScratchBytesPerAiv);
    UpdateDispatchLayoutReport(report, "dispatchGatherScratchCoreOffset", expected_scratch_core_offset,
                               actual.dispatchGatherScratchCoreOffset);
    UpdateDispatchLayoutReport(report, "dispatchGatherDebugOffset", dispatch.dispatchGatherDebugOffset,
                               actual.dispatchGatherDebugOffset);
    UpdateDispatchLayoutReport(report, "dispatchGatherDebugBytes", dispatch.dispatchGatherDebugBytes,
                               actual.dispatchGatherDebugBytes);
    UpdateDispatchLayoutReport(report, "dispatchGatherTileBytes", dispatch.dispatchGatherTileBytes,
                               actual.dispatchGatherTileBytes);
    UpdateDispatchLayoutReport(report, "cumsumMMOffset", front.cumsumMMOffset, actual.cumsumMMOffset);
    UpdateDispatchLayoutReport(report, "preSumBeforeRankOffset", front.preSumBeforeRankOffset,
                               actual.preSumBeforeRankOffset);
    UpdateDispatchLayoutReport(report, "peerOffsetA", 0U, actual.peerOffsetA);
    UpdateDispatchLayoutReport(report, "peerOffsetPeerTokenPerExpert", runtime.hccl.WindowBytes() - 2U * kHostMiB,
                               actual.peerOffsetPeerTokenPerExpert);
    UpdateDispatchLayoutReport(report, "rank", static_cast<uint64_t>(runtime.hccl.rank_id), actual.rank);
    UpdateDispatchLayoutReport(report, "rankSize", static_cast<uint64_t>(runtime.hccl.world_size), actual.rankSize);
    UpdateDispatchLayoutReport(report, "coreIdx", 0U, actual.coreIdx);
    UpdateDispatchLayoutReport(report, "coreNum", cfg.aiv_num, actual.coreNum);
    UpdateDispatchLayoutReport(report, "stageNum", front.stageNum, actual.stageNum);
    UpdateDispatchLayoutReport(report, "problemK", cfg.k, actual.problemK);
    UpdateDispatchLayoutReport(report, "maxOutputSize", cfg.max_output_size, actual.maxOutputSize);
    UpdateDispatchLayoutReport(report, "packedStride", static_cast<uint64_t>(cfg.k) + 32U, actual.packedStride);
    UpdateDispatchLayoutReport(report, "activeCopyCores", expected_active_copy_cores, actual.activeCopyCores);
    UpdateDispatchLayoutReport(report, "dispatchGatherMode", dispatch.dispatchGatherMode, actual.dispatchGatherMode);
    UpdateDispatchLayoutReport(report, "dispatchGatherDebugMode", dispatch.dispatchGatherDebugMode,
                               actual.dispatchGatherDebugMode);
    UpdateDispatchLayoutReport(report, "dispatchGatherStopStep", dispatch.dispatchGatherStopStep,
                               actual.dispatchGatherStopStep);
    UpdateDispatchLayoutReport(report, "marker", 1U, actual.marker);
    UpdateDispatchLayoutReport(
        report, "dispatchGatherScratchAligned", 1U,
        (dispatch.dispatchGatherScratchOffset % 512U == 0U && dispatch.dispatchGatherScratchBytes % 512U == 0U &&
         dispatch.dispatchGatherScratchBytesPerAiv % 512U == 0U) ?
            1U :
            0U);
    UpdateDispatchLayoutReport(
        report, "dispatchGatherDebugAligned", 1U,
        (dispatch.dispatchGatherDebugOffset % 512U == 0U && dispatch.dispatchGatherDebugBytes % 512U == 0U) ? 1U : 0U);
    UpdateDispatchLayoutReport(
        report, "dispatchGatherDebugInsideWorkspace", 1U,
        dispatch.dispatchGatherDebugOffset + dispatch.dispatchGatherDebugBytes <= workspace_dev.bytes ? 1U : 0U);

    report.pass = report.mismatch_count == 0;
    return report;
}

void UpdateGmm1LayoutReport(Gmm1LayoutReport &report, const std::string &field, uint64_t expected, uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

void UpdateGmm2LayoutReport(Gmm2LayoutReport &report, const std::string &field, uint64_t expected, uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

void UpdateGmm2V2CReport(Gmm2V2CReport &report, const std::string &field, size_t idx, uint64_t expected,
                         uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

void UpdateGmm2SegmentReport(Gmm2SegmentReport &report, const std::string &field, size_t idx, uint64_t expected,
                             uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

void UpdateGmm2TaskReport(Gmm2TaskReport &report, const std::string &field, size_t idx, uint64_t expected,
                          uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

void UpdateGmm2DoneReport(Gmm2DoneReport &report, const std::string &field, size_t idx, uint64_t expected,
                          uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

void UpdateGmm2FinalSyncReport(Gmm2FinalSyncReport &report, const std::string &field, size_t idx, uint64_t expected,
                               uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

bool RangesOverlap(uint64_t begin0, uint64_t bytes0, uint64_t begin1, uint64_t bytes1)
{
    if (bytes0 == 0 || bytes1 == 0) {
        return false;
    }
    const uint64_t end0 = begin0 + bytes0;
    const uint64_t end1 = begin1 + bytes1;
    return begin0 < end1 && begin1 < end0;
}

uint32_t SwigluSegmentNum(uint32_t expert_per_rank);
uint32_t SwigluEpilogueGranularity(uint32_t expert_per_rank);

Gmm1LayoutReport CheckGmm1Layout(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                 const DeviceBuffer &workspace_dev, const DeviceBuffer &weight1_dev,
                                 const DeviceBuffer &scale1_dev, const DeviceBuffer &expert_token_nums_dev)
{
    constexpr uint64_t kHostGmm1LayoutDebugMagic = 0x5635474d4d314c59ULL;
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &dispatch = build.tiling.dispatchTiling;
    const auto &gmm1 = build.tiling.gmm1Tiling;
    DispatchFFNCombineGmm1LayoutDebug actual;
    std::memset(&actual, 0, sizeof(actual));

    if (gmm1.gmm1DebugBytes < sizeof(actual)) {
        throw std::runtime_error("gmm1 debug area is too small for layout header");
    }
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(&actual, sizeof(actual), workspace_base + gmm1.gmm1DebugOffset, sizeof(actual),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host gmm1 layout header copy failed");
    }

    Gmm1LayoutReport report;
    UpdateGmm1LayoutReport(report, "magic", kHostGmm1LayoutDebugMagic, actual.magic);
    UpdateGmm1LayoutReport(report, "workspaceBase", reinterpret_cast<uint64_t>(workspace_dev.ptr),
                           actual.workspaceBase);
    UpdateGmm1LayoutReport(report, "gmAOffset", dispatch.gmAOffset, actual.gmAOffset);
    UpdateGmm1LayoutReport(report, "gmCOffset", gmm1.gmCOffset, actual.gmCOffset);
    UpdateGmm1LayoutReport(report, "perTokenScaleOffset", dispatch.perTokenScaleOffset, actual.perTokenScaleOffset);
    UpdateGmm1LayoutReport(report, "cumsumMMOffset", front.cumsumMMOffset, actual.cumsumMMOffset);
    UpdateGmm1LayoutReport(report, "expertTokenNumsBase", reinterpret_cast<uint64_t>(expert_token_nums_dev.ptr),
                           actual.expertTokenNumsBase);
    UpdateGmm1LayoutReport(report, "weight1Base", reinterpret_cast<uint64_t>(weight1_dev.ptr), actual.weight1Base);
    UpdateGmm1LayoutReport(report, "scale1Base", reinterpret_cast<uint64_t>(scale1_dev.ptr), actual.scale1Base);
    UpdateGmm1LayoutReport(report, "gmm1DebugOffset", gmm1.gmm1DebugOffset, actual.gmm1DebugOffset);
    UpdateGmm1LayoutReport(report, "gmm1DebugBytes", gmm1.gmm1DebugBytes, actual.gmm1DebugBytes);
    UpdateGmm1LayoutReport(report, "rank", build.tiling.runtimeInfo.rank, actual.rank);
    UpdateGmm1LayoutReport(report, "rankSize", build.tiling.runtimeInfo.rankSize, actual.rankSize);
    UpdateGmm1LayoutReport(report, "coreIdx", 0, actual.coreIdx);
    UpdateGmm1LayoutReport(report, "coreNum", build.block_dim, actual.coreNum);
    UpdateGmm1LayoutReport(report, "stageNum", front.stageNum, actual.stageNum);
    UpdateGmm1LayoutReport(report, "problemK", cfg.k, actual.problemK);
    UpdateGmm1LayoutReport(report, "problemN", cfg.n, actual.problemN);
    UpdateGmm1LayoutReport(report, "maxOutputSize", cfg.max_output_size, actual.maxOutputSize);
    UpdateGmm1LayoutReport(report, "expertPerRank", cfg.expert_per_rank, actual.expertPerRank);
    UpdateGmm1LayoutReport(report, "l1TileM", gmm1.l1TileM, actual.l1TileM);
    UpdateGmm1LayoutReport(report, "l1TileN", gmm1.l1TileN, actual.l1TileN);
    UpdateGmm1LayoutReport(report, "l1TileK", gmm1.l1TileK, actual.l1TileK);
    UpdateGmm1LayoutReport(report, "l0TileM", gmm1.l0TileM, actual.l0TileM);
    UpdateGmm1LayoutReport(report, "l0TileN", gmm1.l0TileN, actual.l0TileN);
    UpdateGmm1LayoutReport(report, "l0TileK", gmm1.l0TileK, actual.l0TileK);
    UpdateGmm1LayoutReport(report, "debugMode", gmm1.gmm1DebugMode, actual.debugMode);

    const uint64_t gmCBytes = static_cast<uint64_t>(cfg.max_output_size) * cfg.n * sizeof(uint16_t);
    const uint64_t perTokenScaleBytes = static_cast<uint64_t>(cfg.max_output_size) * sizeof(float);
    const uint64_t perTokenScale2Offset = AlignUpU64(dispatch.perTokenScaleOffset + perTokenScaleBytes, 512U);
    const uint64_t gmABytes = static_cast<uint64_t>(cfg.max_output_size) * cfg.k * sizeof(int8_t);
    const uint64_t gmPermutedOffset = AlignUpU64(dispatch.gmAOffset + gmABytes, 512U);
    const uint64_t gmPermutedBytes = static_cast<uint64_t>(cfg.max_output_size) * (cfg.n / 2U) * sizeof(int8_t);
    const uint64_t gmm2OutputOffset = AlignUpU64(gmm1.gmCOffset + gmCBytes, 512U);
    const uint64_t gmm2OutputBytes = static_cast<uint64_t>(cfg.max_output_size) * cfg.k * sizeof(uint16_t);

    uint64_t gmCOverlap = 0;
    auto mark_gm_c_overlap = [&](uint64_t offset, uint64_t bytes) {
        if (RangesOverlap(gmm1.gmCOffset, gmCBytes, offset, bytes)) {
            gmCOverlap = 1;
        }
    };
    mark_gm_c_overlap(dispatch.perTokenScaleOffset, perTokenScaleBytes);
    mark_gm_c_overlap(perTokenScale2Offset, perTokenScaleBytes);
    mark_gm_c_overlap(dispatch.gmAOffset, gmABytes);
    mark_gm_c_overlap(gmPermutedOffset, gmPermutedBytes);
    mark_gm_c_overlap(gmm2OutputOffset, gmm2OutputBytes);
    mark_gm_c_overlap(dispatch.dispatchScratchOffset, dispatch.dispatchScratchBytes);
    mark_gm_c_overlap(dispatch.dispatchDebugOffset, dispatch.dispatchDebugBytes);
    mark_gm_c_overlap(gmm1.gmm1DebugOffset, gmm1.gmm1DebugBytes);
    UpdateGmm1LayoutReport(report, "gmCWorkspaceOverlap", 0, gmCOverlap);

    UpdateGmm1LayoutReport(report, "gmCOffsetAlignment", 0, gmm1.gmCOffset % 32U);
    UpdateGmm1LayoutReport(report, "gmm1DebugOffsetAlignment", 0, gmm1.gmm1DebugOffset % 512U);
    UpdateGmm1LayoutReport(report, "weight1BaseAlignment", 0, reinterpret_cast<uint64_t>(weight1_dev.ptr) % 32U);
    UpdateGmm1LayoutReport(report, "scale1BaseAlignment", 0, reinterpret_cast<uint64_t>(scale1_dev.ptr) % 32U);

    report.pass = report.mismatch_count == 0;
    return report;
}

Gmm2LayoutReport CheckGmm2Layout(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                 const DeviceBuffer &workspace_dev, const DeviceBuffer &weight2_dev,
                                 const DeviceBuffer &scale2_dev, const DeviceBuffer &expert_token_nums_dev)
{
    constexpr uint64_t kHostGmm2LayoutDebugMagic = 0x5635474d4d324c59ULL;
    constexpr uint32_t kHostGmm2LayoutVersion = 1U;
    constexpr uint32_t kHostGmm2CommonReuseMode = 1U;
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &dispatch = build.tiling.dispatchTiling;
    const auto &gmm1 = build.tiling.gmm1Tiling;
    const auto &swiglu = build.tiling.swigluTiling;
    const auto &gmm2 = build.tiling.gmm2Tiling;
    DispatchFFNCombineGmm2LayoutDebug actual;
    std::memset(&actual, 0, sizeof(actual));

    if (gmm2.gmm2DebugBytes < sizeof(actual)) {
        throw std::runtime_error("gmm2 debug area is too small for layout header");
    }
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(&actual, sizeof(actual), workspace_base + gmm2.gmm2DebugOffset, sizeof(actual),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host gmm2 layout header copy failed");
    }

    const uint64_t per_token_scale_bytes = static_cast<uint64_t>(cfg.max_output_size) * sizeof(float);
    const uint64_t gm_a_bytes = static_cast<uint64_t>(cfg.max_output_size) * cfg.k * sizeof(int8_t);
    const uint64_t gm_permuted_bytes = static_cast<uint64_t>(cfg.max_output_size) * (cfg.n / 2U) * sizeof(int8_t);
    const uint64_t gm_c_bytes = static_cast<uint64_t>(cfg.max_output_size) * cfg.n * sizeof(uint16_t);
    const uint64_t gmm2_output_bytes = static_cast<uint64_t>(cfg.max_output_size) * cfg.k * sizeof(uint16_t);
    const uint64_t expected_gmm2_output_offset = AlignUpU64(gmm1.gmCOffset + gm_c_bytes, 512U);
    const uint64_t expected_gmm2_debug_offset = AlignUpU64(swiglu.swigluDebugOffset + swiglu.swigluDebugBytes, 512U);
    const uint32_t segment_count = SwigluSegmentNum(cfg.expert_per_rank);
    const uint64_t gmm2_v2c_debug_bytes =
        static_cast<uint64_t>(segment_count) * build.block_dim * sizeof(DispatchFFNCombineGmm2V2CDebug);
    const uint64_t gmm2_segment_debug_bytes =
        static_cast<uint64_t>(segment_count) * sizeof(DispatchFFNCombineGmm2SegmentDebug);
    const uint64_t gmm2_task_debug_bytes =
        static_cast<uint64_t>(cfg.expert_per_rank) * build.block_dim * sizeof(DispatchFFNCombineGmm2TaskDebug);
    const uint64_t gmm2_done_debug_bytes =
        static_cast<uint64_t>(build.block_dim) * sizeof(DispatchFFNCombineGmm2DoneDebug);
    const uint64_t gmm2_final_sync_debug_bytes =
        static_cast<uint64_t>(build.block_dim) * sizeof(DispatchFFNCombineGmm2FinalSyncDebug);
    const uint64_t expected_gmm2_debug_bytes =
        AlignUpU64(sizeof(DispatchFFNCombineGmm2LayoutDebug) + gmm2_v2c_debug_bytes + gmm2_segment_debug_bytes +
                       gmm2_task_debug_bytes + gmm2_done_debug_bytes + gmm2_final_sync_debug_bytes,
                   512U);

    Gmm2LayoutReport report;
    UpdateGmm2LayoutReport(report, "magic", kHostGmm2LayoutDebugMagic, actual.magic);
    UpdateGmm2LayoutReport(report, "workspaceBase", reinterpret_cast<uint64_t>(workspace_dev.ptr),
                           actual.workspaceBase);
    UpdateGmm2LayoutReport(report, "gmPermutedTokenOffset", swiglu.gmPermutedTokenOffset, actual.gmPermutedTokenOffset);
    UpdateGmm2LayoutReport(report, "perTokenScale2Offset", swiglu.perTokenScale2Offset, actual.perTokenScale2Offset);
    UpdateGmm2LayoutReport(report, "gmm2OutputOffset", gmm2.gmm2OutputOffset, actual.gmm2OutputOffset);
    UpdateGmm2LayoutReport(report, "weight2Base", reinterpret_cast<uint64_t>(weight2_dev.ptr), actual.weight2Base);
    UpdateGmm2LayoutReport(report, "scale2Base", reinterpret_cast<uint64_t>(scale2_dev.ptr), actual.scale2Base);
    UpdateGmm2LayoutReport(report, "cumsumMMOffset", front.cumsumMMOffset, actual.cumsumMMOffset);
    UpdateGmm2LayoutReport(report, "expertTokenNumsBase", reinterpret_cast<uint64_t>(expert_token_nums_dev.ptr),
                           actual.expertTokenNumsBase);
    UpdateGmm2LayoutReport(report, "gmm2DebugOffset", gmm2.gmm2DebugOffset, actual.gmm2DebugOffset);
    UpdateGmm2LayoutReport(report, "gmm2DebugBytes", gmm2.gmm2DebugBytes, actual.gmm2DebugBytes);
    UpdateGmm2LayoutReport(report, "frontWorkspaceBytes", front.frontWorkspaceBytes, actual.frontWorkspaceBytes);
    UpdateGmm2LayoutReport(report, "dispatchScratchOffset", dispatch.dispatchScratchOffset,
                           actual.dispatchScratchOffset);
    UpdateGmm2LayoutReport(report, "gmm1DebugOffset", gmm1.gmm1DebugOffset, actual.gmm1DebugOffset);
    UpdateGmm2LayoutReport(report, "swigluDebugOffset", swiglu.swigluDebugOffset, actual.swigluDebugOffset);
    UpdateGmm2LayoutReport(report, "gmPermutedTokenBytes", gm_permuted_bytes, actual.gmPermutedTokenBytes);
    UpdateGmm2LayoutReport(report, "perTokenScale2Bytes", per_token_scale_bytes, actual.perTokenScale2Bytes);
    UpdateGmm2LayoutReport(report, "gmm2OutputBytes", gmm2_output_bytes, actual.gmm2OutputBytes);
    UpdateGmm2LayoutReport(report, "rank", build.tiling.runtimeInfo.rank, actual.rank);
    UpdateGmm2LayoutReport(report, "rankSize", build.tiling.runtimeInfo.rankSize, actual.rankSize);
    UpdateGmm2LayoutReport(report, "coreIdx", 0, actual.coreIdx);
    UpdateGmm2LayoutReport(report, "coreNum", build.block_dim, actual.coreNum);
    UpdateGmm2LayoutReport(report, "stageNum", front.stageNum, actual.stageNum);
    UpdateGmm2LayoutReport(report, "problemK", cfg.k, actual.problemK);
    UpdateGmm2LayoutReport(report, "problemN", cfg.n, actual.problemN);
    UpdateGmm2LayoutReport(report, "inputK", cfg.n / 2U, actual.inputK);
    UpdateGmm2LayoutReport(report, "outputN", cfg.k, actual.outputN);
    UpdateGmm2LayoutReport(report, "maxOutputSize", cfg.max_output_size, actual.maxOutputSize);
    UpdateGmm2LayoutReport(report, "expertPerRank", cfg.expert_per_rank, actual.expertPerRank);
    UpdateGmm2LayoutReport(report, "segmentNum", SwigluSegmentNum(cfg.expert_per_rank), actual.segmentNum);
    UpdateGmm2LayoutReport(report, "epilogueGranularity", SwigluEpilogueGranularity(cfg.expert_per_rank),
                           actual.epilogueGranularity);
    UpdateGmm2LayoutReport(report, "l1TileM", gmm2.l1TileM, actual.l1TileM);
    UpdateGmm2LayoutReport(report, "l1TileN", gmm2.l1TileN, actual.l1TileN);
    UpdateGmm2LayoutReport(report, "l1TileK", gmm2.l1TileK, actual.l1TileK);
    UpdateGmm2LayoutReport(report, "l0TileM", gmm2.l0TileM, actual.l0TileM);
    UpdateGmm2LayoutReport(report, "l0TileN", gmm2.l0TileN, actual.l0TileN);
    UpdateGmm2LayoutReport(report, "l0TileK", gmm2.l0TileK, actual.l0TileK);
    UpdateGmm2LayoutReport(report, "debugMode", gmm2.gmm2DebugMode, actual.debugMode);
    UpdateGmm2LayoutReport(report, "commonReuseMode", kHostGmm2CommonReuseMode, actual.commonReuseMode);
    UpdateGmm2LayoutReport(report, "gmPermutedTokenRowBytes", (cfg.n / 2U) * sizeof(int8_t),
                           actual.gmPermutedTokenRowBytes);
    UpdateGmm2LayoutReport(report, "gmm2OutputRowBytes", cfg.k * sizeof(uint16_t), actual.gmm2OutputRowBytes);
    UpdateGmm2LayoutReport(report, "perTokenScale2BytesPerRow", sizeof(float), actual.perTokenScale2BytesPerRow);
    UpdateGmm2LayoutReport(report, "layoutVersion", kHostGmm2LayoutVersion, actual.layoutVersion);
    UpdateGmm2LayoutReport(report, "marker", 1U, actual.marker);

    UpdateGmm2LayoutReport(report, "gmm2OutputOffsetExpected", expected_gmm2_output_offset, gmm2.gmm2OutputOffset);
    UpdateGmm2LayoutReport(report, "gmm2DebugOffsetExpected", expected_gmm2_debug_offset, gmm2.gmm2DebugOffset);
    UpdateGmm2LayoutReport(report, "gmm2DebugBytesExpected", expected_gmm2_debug_bytes, gmm2.gmm2DebugBytes);

    uint64_t gmm2_overlap = 0;
    auto mark_gmm2_overlap = [&](uint64_t offset, uint64_t bytes) {
        if (RangesOverlap(gmm2.gmm2OutputOffset, gmm2_output_bytes, offset, bytes) ||
            RangesOverlap(gmm2.gmm2DebugOffset, gmm2.gmm2DebugBytes, offset, bytes)) {
            gmm2_overlap = 1;
        }
    };
    if (RangesOverlap(gmm2.gmm2OutputOffset, gmm2_output_bytes, gmm2.gmm2DebugOffset, gmm2.gmm2DebugBytes)) {
        gmm2_overlap = 1;
    }
    mark_gmm2_overlap(dispatch.perTokenScaleOffset, per_token_scale_bytes);
    mark_gmm2_overlap(swiglu.perTokenScale2Offset, per_token_scale_bytes);
    mark_gmm2_overlap(dispatch.gmAOffset, gm_a_bytes);
    mark_gmm2_overlap(swiglu.gmPermutedTokenOffset, gm_permuted_bytes);
    mark_gmm2_overlap(gmm1.gmCOffset, gm_c_bytes);
    mark_gmm2_overlap(dispatch.dispatchScratchOffset, dispatch.dispatchScratchBytes);
    mark_gmm2_overlap(dispatch.dispatchDebugOffset, dispatch.dispatchDebugBytes);
    mark_gmm2_overlap(gmm1.gmm1DebugOffset, gmm1.gmm1DebugBytes);
    mark_gmm2_overlap(swiglu.swigluDebugOffset, swiglu.swigluDebugBytes);
    UpdateGmm2LayoutReport(report, "gmm2WorkspaceOverlap", 0, gmm2_overlap);

    UpdateGmm2LayoutReport(report, "gmm2OutputOffsetAlignment", 0, gmm2.gmm2OutputOffset % 512U);
    UpdateGmm2LayoutReport(report, "gmm2DebugOffsetAlignment", 0, gmm2.gmm2DebugOffset % 512U);
    UpdateGmm2LayoutReport(report, "gmPermutedTokenRowAlignment", 0, ((cfg.n / 2U) * sizeof(int8_t)) % 32U);
    UpdateGmm2LayoutReport(report, "gmm2OutputRowAlignment", 0, (cfg.k * sizeof(uint16_t)) % 32U);
    UpdateGmm2LayoutReport(report, "weight2BaseAlignment", 0, reinterpret_cast<uint64_t>(weight2_dev.ptr) % 32U);
    UpdateGmm2LayoutReport(report, "scale2BaseAlignment", 0, reinterpret_cast<uint64_t>(scale2_dev.ptr) % 32U);

    report.pass = report.mismatch_count == 0;
    return report;
}

void UpdateSwigluLayoutReport(SwigluLayoutReport &report, const std::string &field, uint64_t expected, uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

void UpdateSwigluC2VReport(SwigluC2VReport &report, const std::string &field, size_t idx, uint64_t expected,
                           uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

void UpdateSwigluSegmentReport(SwigluSegmentReport &report, const std::string &field, size_t idx, uint64_t expected,
                               uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

void UpdateSwigluMetadataModeReport(SwigluMetadataModeReport &report, const std::string &field, uint64_t expected,
                                    uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

void UpdateSwigluSegmentMetaReport(SwigluSegmentMetaReport &report, const std::string &field, size_t idx,
                                   uint64_t expected, uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

void UpdateSwigluTaskReport(SwigluTaskReport &report, const std::string &field, size_t idx, uint64_t expected,
                            uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

void UpdateSwigluDoneReport(SwigluDoneReport &report, const std::string &field, size_t idx, uint64_t expected,
                            uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_idx = idx;
        report.first_field = field;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

void UpdateSwigluFinalSyncReport(SwigluFinalSyncReport &report, const std::string &field, size_t idx, uint64_t expected,
                                 uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_idx = idx;
        report.first_field = field;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

SwigluLayoutReport CheckSwigluLayout(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                     const DeviceBuffer &workspace_dev, const DeviceBuffer &expert_token_nums_dev)
{
    constexpr uint64_t kHostSwigluLayoutDebugMagic = 0x5635535749474c59ULL;
    constexpr uint32_t kHostSwigluLayoutVersion = 1U;
    constexpr uint32_t kHostSwigluPipelineModeInputOutputSplit = 1U;
    constexpr uint32_t kHostSwigluScale2BufferNum = 2U;
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &dispatch = build.tiling.dispatchTiling;
    const auto &gmm1 = build.tiling.gmm1Tiling;
    const auto &swiglu = build.tiling.swigluTiling;
    DispatchFFNCombineSwigluLayoutDebug actual;
    std::memset(&actual, 0, sizeof(actual));

    if (swiglu.swigluDebugBytes < sizeof(actual)) {
        throw std::runtime_error("swiglu debug area is too small for layout header");
    }
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(&actual, sizeof(actual), workspace_base + swiglu.swigluDebugOffset, sizeof(actual),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host swiglu layout header copy failed");
    }

    const uint64_t per_token_scale_bytes = static_cast<uint64_t>(cfg.max_output_size) * sizeof(float);
    const uint64_t per_token_scale2_offset = AlignUpU64(dispatch.perTokenScaleOffset + per_token_scale_bytes, 512U);
    const uint64_t gm_a_bytes = static_cast<uint64_t>(cfg.max_output_size) * cfg.k * sizeof(int8_t);
    const uint64_t gm_permuted_offset = AlignUpU64(dispatch.gmAOffset + gm_a_bytes, 512U);
    const uint64_t gm_permuted_bytes = static_cast<uint64_t>(cfg.max_output_size) * (cfg.n / 2U) * sizeof(int8_t);
    const uint64_t gm_c_bytes = static_cast<uint64_t>(cfg.max_output_size) * cfg.n * sizeof(uint16_t);
    const uint64_t gmm2_output_offset = AlignUpU64(gmm1.gmCOffset + gm_c_bytes, 512U);
    const uint64_t gmm2_output_bytes = static_cast<uint64_t>(cfg.max_output_size) * cfg.k * sizeof(uint16_t);
    const uint32_t segment_count = SwigluSegmentNum(cfg.expert_per_rank);
    const uint64_t expected_swiglu_segment_meta_offset = AlignUpU64(gmm1.gmm1DebugOffset + gmm1.gmm1DebugBytes, 512U);
    const uint64_t expected_swiglu_segment_meta_bytes =
        AlignUpU64(static_cast<uint64_t>(segment_count) * sizeof(DispatchFFNCombineSwigluSegmentRuntimeMeta), 512U);
    const uint64_t expected_swiglu_debug_offset =
        AlignUpU64(expected_swiglu_segment_meta_offset + expected_swiglu_segment_meta_bytes, 512U);
    const uint64_t swiglu_c2v_debug_bytes =
        static_cast<uint64_t>(segment_count) * cfg.aiv_num * sizeof(DispatchFFNCombineSwigluC2VDebug);
    const uint64_t swiglu_segment_debug_bytes =
        static_cast<uint64_t>(segment_count) * sizeof(DispatchFFNCombineSwigluSegmentDebug);
    const uint64_t swiglu_task_debug_bytes =
        static_cast<uint64_t>(segment_count) * cfg.aiv_num * sizeof(DispatchFFNCombineSwigluTaskDebug);
    const uint64_t swiglu_done_debug_bytes =
        static_cast<uint64_t>(segment_count) * cfg.aiv_num * sizeof(DispatchFFNCombineSwigluDoneDebug);
    const uint64_t swiglu_final_sync_debug_bytes =
        static_cast<uint64_t>(cfg.aiv_num) * sizeof(DispatchFFNCombineSwigluFinalSyncDebug);
    const uint64_t expected_swiglu_debug_bytes =
        AlignUpU64(sizeof(DispatchFFNCombineSwigluLayoutDebug) + swiglu_c2v_debug_bytes + swiglu_segment_debug_bytes +
                       swiglu_task_debug_bytes + swiglu_done_debug_bytes + swiglu_final_sync_debug_bytes,
                   512U);

    SwigluLayoutReport report;
    UpdateSwigluLayoutReport(report, "magic", kHostSwigluLayoutDebugMagic, actual.magic);
    UpdateSwigluLayoutReport(report, "workspaceBase", reinterpret_cast<uint64_t>(workspace_dev.ptr),
                             actual.workspaceBase);
    UpdateSwigluLayoutReport(report, "gmCOffset", gmm1.gmCOffset, actual.gmCOffset);
    UpdateSwigluLayoutReport(report, "perTokenScaleOffset", dispatch.perTokenScaleOffset, actual.perTokenScaleOffset);
    UpdateSwigluLayoutReport(report, "gmPermutedTokenOffset", swiglu.gmPermutedTokenOffset,
                             actual.gmPermutedTokenOffset);
    UpdateSwigluLayoutReport(report, "perTokenScale2Offset", swiglu.perTokenScale2Offset, actual.perTokenScale2Offset);
    UpdateSwigluLayoutReport(report, "cumsumMMOffset", front.cumsumMMOffset, actual.cumsumMMOffset);
    UpdateSwigluLayoutReport(report, "expertTokenNumsBase", reinterpret_cast<uint64_t>(expert_token_nums_dev.ptr),
                             actual.expertTokenNumsBase);
    UpdateSwigluLayoutReport(report, "swigluDebugOffset", swiglu.swigluDebugOffset, actual.swigluDebugOffset);
    UpdateSwigluLayoutReport(report, "swigluDebugBytes", swiglu.swigluDebugBytes, actual.swigluDebugBytes);
    UpdateSwigluLayoutReport(report, "swigluSegmentMetaOffset", swiglu.swigluSegmentMetaOffset,
                             actual.swigluSegmentMetaOffset);
    UpdateSwigluLayoutReport(report, "swigluSegmentMetaBytes", swiglu.swigluSegmentMetaBytes,
                             actual.swigluSegmentMetaBytes);
    UpdateSwigluLayoutReport(report, "gmCBytes", gm_c_bytes, actual.gmCBytes);
    UpdateSwigluLayoutReport(report, "gmPermutedTokenBytes", gm_permuted_bytes, actual.gmPermutedTokenBytes);
    UpdateSwigluLayoutReport(report, "perTokenScaleBytes", per_token_scale_bytes, actual.perTokenScaleBytes);
    UpdateSwigluLayoutReport(report, "perTokenScale2Bytes", per_token_scale_bytes, actual.perTokenScale2Bytes);
    UpdateSwigluLayoutReport(report, "frontWorkspaceBytes", front.frontWorkspaceBytes, actual.frontWorkspaceBytes);
    UpdateSwigluLayoutReport(report, "dispatchScratchOffset", dispatch.dispatchScratchOffset,
                             actual.dispatchScratchOffset);
    UpdateSwigluLayoutReport(report, "gmm1DebugOffset", gmm1.gmm1DebugOffset, actual.gmm1DebugOffset);
    UpdateSwigluLayoutReport(report, "rank", build.tiling.runtimeInfo.rank, actual.rank);
    UpdateSwigluLayoutReport(report, "rankSize", build.tiling.runtimeInfo.rankSize, actual.rankSize);
    UpdateSwigluLayoutReport(report, "coreIdx", 0, actual.coreIdx);
    UpdateSwigluLayoutReport(report, "coreNum", cfg.aiv_num, actual.coreNum);
    UpdateSwigluLayoutReport(report, "stageNum", front.stageNum, actual.stageNum);
    UpdateSwigluLayoutReport(report, "problemN", cfg.n, actual.problemN);
    UpdateSwigluLayoutReport(report, "outputN", cfg.n / 2U, actual.outputN);
    UpdateSwigluLayoutReport(report, "maxOutputSize", cfg.max_output_size, actual.maxOutputSize);
    UpdateSwigluLayoutReport(report, "expertPerRank", cfg.expert_per_rank, actual.expertPerRank);
    UpdateSwigluLayoutReport(report, "segmentNum", SwigluSegmentNum(cfg.expert_per_rank), actual.segmentNum);
    UpdateSwigluLayoutReport(report, "epilogueGranularity", SwigluEpilogueGranularity(cfg.expert_per_rank),
                             actual.epilogueGranularity);
    UpdateSwigluLayoutReport(report, "tileElems", swiglu.swigluTileElems, actual.tileElems);
    UpdateSwigluLayoutReport(report, "ubStages", swiglu.swigluUbStages, actual.ubStages);
    UpdateSwigluLayoutReport(report, "debugMode", swiglu.swigluDebugMode, actual.debugMode);
    UpdateSwigluLayoutReport(report, "pipelineMode", kHostSwigluPipelineModeInputOutputSplit, actual.pipelineMode);
    UpdateSwigluLayoutReport(report, "scale2BufferNum", kHostSwigluScale2BufferNum, actual.scale2BufferNum);
    UpdateSwigluLayoutReport(report, "metadataMode", swiglu.swigluMetadataMode, actual.metadataMode);
    UpdateSwigluLayoutReport(report, "gmCRowBytes", cfg.n * sizeof(uint16_t), actual.gmCRowBytes);
    UpdateSwigluLayoutReport(report, "gmPermutedTokenRowBytes", (cfg.n / 2U) * sizeof(int8_t),
                             actual.gmPermutedTokenRowBytes);
    UpdateSwigluLayoutReport(report, "perTokenScaleBytesPerRow", sizeof(float), actual.perTokenScaleBytesPerRow);
    UpdateSwigluLayoutReport(report, "perTokenScale2BytesPerRow", sizeof(float), actual.perTokenScale2BytesPerRow);
    UpdateSwigluLayoutReport(report, "layoutVersion", kHostSwigluLayoutVersion, actual.layoutVersion);
    UpdateSwigluLayoutReport(report, "marker", 1U, actual.marker);

    UpdateSwigluLayoutReport(report, "perTokenScale2OffsetExpected", per_token_scale2_offset,
                             swiglu.perTokenScale2Offset);
    UpdateSwigluLayoutReport(report, "gmPermutedTokenOffsetExpected", gm_permuted_offset, swiglu.gmPermutedTokenOffset);
    UpdateSwigluLayoutReport(report, "swigluSegmentMetaOffsetExpected", expected_swiglu_segment_meta_offset,
                             swiglu.swigluSegmentMetaOffset);
    UpdateSwigluLayoutReport(report, "swigluSegmentMetaBytesExpected", expected_swiglu_segment_meta_bytes,
                             swiglu.swigluSegmentMetaBytes);
    UpdateSwigluLayoutReport(report, "swigluDebugOffsetExpected", expected_swiglu_debug_offset,
                             swiglu.swigluDebugOffset);
    UpdateSwigluLayoutReport(report, "swigluDebugBytesExpected", expected_swiglu_debug_bytes, swiglu.swigluDebugBytes);

    uint64_t swiglu_overlap = 0;
    auto mark_swiglu_overlap = [&](uint64_t offset, uint64_t bytes) {
        if (RangesOverlap(swiglu.perTokenScale2Offset, per_token_scale_bytes, offset, bytes) ||
            RangesOverlap(swiglu.gmPermutedTokenOffset, gm_permuted_bytes, offset, bytes) ||
            RangesOverlap(swiglu.swigluSegmentMetaOffset, swiglu.swigluSegmentMetaBytes, offset, bytes) ||
            RangesOverlap(swiglu.swigluDebugOffset, swiglu.swigluDebugBytes, offset, bytes)) {
            swiglu_overlap = 1;
        }
    };
    if (RangesOverlap(swiglu.perTokenScale2Offset, per_token_scale_bytes, swiglu.gmPermutedTokenOffset,
                      gm_permuted_bytes) ||
        RangesOverlap(swiglu.perTokenScale2Offset, per_token_scale_bytes, swiglu.swigluDebugOffset,
                      swiglu.swigluDebugBytes) ||
        RangesOverlap(swiglu.gmPermutedTokenOffset, gm_permuted_bytes, swiglu.swigluDebugOffset,
                      swiglu.swigluDebugBytes) ||
        RangesOverlap(swiglu.perTokenScale2Offset, per_token_scale_bytes, swiglu.swigluSegmentMetaOffset,
                      swiglu.swigluSegmentMetaBytes) ||
        RangesOverlap(swiglu.gmPermutedTokenOffset, gm_permuted_bytes, swiglu.swigluSegmentMetaOffset,
                      swiglu.swigluSegmentMetaBytes) ||
        RangesOverlap(swiglu.swigluSegmentMetaOffset, swiglu.swigluSegmentMetaBytes, swiglu.swigluDebugOffset,
                      swiglu.swigluDebugBytes)) {
        swiglu_overlap = 1;
    }
    mark_swiglu_overlap(dispatch.perTokenScaleOffset, per_token_scale_bytes);
    mark_swiglu_overlap(dispatch.gmAOffset, gm_a_bytes);
    mark_swiglu_overlap(gmm1.gmCOffset, gm_c_bytes);
    mark_swiglu_overlap(gmm2_output_offset, gmm2_output_bytes);
    mark_swiglu_overlap(dispatch.dispatchScratchOffset, dispatch.dispatchScratchBytes);
    mark_swiglu_overlap(dispatch.dispatchDebugOffset, dispatch.dispatchDebugBytes);
    mark_swiglu_overlap(gmm1.gmm1DebugOffset, gmm1.gmm1DebugBytes);
    UpdateSwigluLayoutReport(report, "swigluWorkspaceOverlap", 0, swiglu_overlap);

    UpdateSwigluLayoutReport(report, "perTokenScale2OffsetAlignment", 0, swiglu.perTokenScale2Offset % 512U);
    UpdateSwigluLayoutReport(report, "gmPermutedTokenOffsetAlignment", 0, swiglu.gmPermutedTokenOffset % 512U);
    UpdateSwigluLayoutReport(report, "swigluSegmentMetaOffsetAlignment", 0, swiglu.swigluSegmentMetaOffset % 512U);
    UpdateSwigluLayoutReport(report, "swigluDebugOffsetAlignment", 0, swiglu.swigluDebugOffset % 512U);
    UpdateSwigluLayoutReport(report, "gmCRowAlignment", 0, (cfg.n * sizeof(uint16_t)) % 32U);
    UpdateSwigluLayoutReport(report, "gmPermutedTokenRowAlignment", 0, ((cfg.n / 2U) * sizeof(int8_t)) % 32U);

    report.pass = report.mismatch_count == 0;
    return report;
}

void UpdateGmm1SyncReport(Gmm1SyncReport &report, const std::string &field, size_t idx, uint64_t expected,
                          uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

Gmm1SyncReport CheckGmm1Sync(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                             const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                             const std::vector<uint8_t> &expert_idx)
{
    constexpr uint64_t kHostGmm1SyncDebugMagic = 0x5635474d4d315359ULL;
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &gmm1 = build.tiling.gmm1Tiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const uint32_t core_num = build.block_dim;
    const uint64_t sync_offset = gmm1.gmm1DebugOffset + sizeof(DispatchFFNCombineGmm1LayoutDebug);
    const size_t sync_count = static_cast<size_t>(expert_per_rank) * core_num;
    const size_t sync_bytes = sync_count * sizeof(DispatchFFNCombineGmm1SyncDebug);
    if (gmm1.gmm1DebugBytes < sizeof(DispatchFFNCombineGmm1LayoutDebug) + sync_bytes) {
        throw std::runtime_error("gmm1 debug area is too small for sync table");
    }

    std::vector<DispatchFFNCombineGmm1SyncDebug> actual(sync_count);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (!actual.empty() && aclrtMemcpy(actual.data(), sync_bytes, workspace_base + sync_offset, sync_bytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host gmm1 sync table copy failed");
    }

    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);

    Gmm1SyncReport report;
    report.checked_count = sync_count;
    uint32_t group_base = 0;
    const uint32_t local_begin = rank * expert_per_rank;
    const bool dispatch_gather_mode = build.tiling.dispatchTiling.dispatchGatherMode != 0U;
    for (uint32_t group = 0; group < expert_per_rank; ++group) {
        uint32_t current_m_raw = 0;
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            const size_t token_idx = static_cast<size_t>(src_rank) * expert_num_aligned + local_begin + group;
            current_m_raw += static_cast<uint32_t>(expected_tokens[token_idx]);
        }
        const uint32_t remaining = group_base >= cfg.max_output_size ? 0U : cfg.max_output_size - group_base;
        const uint32_t current_m = current_m_raw > remaining ? remaining : current_m_raw;
        for (uint32_t core = 0; core < core_num; ++core) {
            const size_t idx = static_cast<size_t>(group) * core_num + core;
            const auto &entry = actual[idx];
            UpdateGmm1SyncReport(report, "magic", idx, kHostGmm1SyncDebugMagic, entry.magic);
            UpdateGmm1SyncReport(report, "groupIdx", idx, group, entry.groupIdx);
            UpdateGmm1SyncReport(report, "coreIdx", idx, core, entry.coreIdx);
            (void)dispatch_gather_mode;
            const uint32_t logical_group_event_idx = group;
            UpdateGmm1SyncReport(
                report, "flagId", idx,
                V5_DISPATCH_V2C_HARD_FLAG_BASE + logical_group_event_idx / CROSS_CORE_FLAG_MAX_SET_COUNT, entry.flagId);
            UpdateGmm1SyncReport(report, "rank", idx, rank, entry.rank);
            UpdateGmm1SyncReport(report, "rankSize", idx, rank_size, entry.rankSize);
            UpdateGmm1SyncReport(report, "groupBase", idx, group_base, entry.groupBase);
            UpdateGmm1SyncReport(report, "currentM", idx, current_m, entry.currentM);
            UpdateGmm1SyncReport(report, "marker", idx, 1U, entry.marker);
        }
        group_base += current_m;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

void UpdateGmm1TaskReport(Gmm1TaskReport &report, const std::string &field, size_t idx, uint64_t expected,
                          uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

uint32_t CeilDivU32(uint32_t value, uint32_t divisor)
{
    return divisor == 0U ? 0U : (value + divisor - 1U) / divisor;
}

uint32_t SwigluSegmentNum(uint32_t expert_per_rank)
{
    return expert_per_rank <= 1U ? expert_per_rank : 2U;
}

uint32_t SwigluEpilogueGranularity(uint32_t expert_per_rank)
{
    if (expert_per_rank <= 1U) {
        return expert_per_rank;
    }
    return expert_per_rank <= 4U ? expert_per_rank - 1U : expert_per_rank - 3U;
}

uint32_t SwigluSegmentStartExpert(uint32_t expert_per_rank, uint32_t segment)
{
    return segment == 0U ? 0U : SwigluEpilogueGranularity(expert_per_rank);
}

uint32_t SwigluSegmentEndExpert(uint32_t expert_per_rank, uint32_t segment)
{
    return segment == 0U && SwigluSegmentNum(expert_per_rank) == 2U ? SwigluEpilogueGranularity(expert_per_rank) :
                                                                      expert_per_rank;
}

uint64_t SwigluFullRowUbBytes(uint32_t n)
{
    const uint32_t output_n = n / 2U;
    constexpr uint32_t k_swiglu_scale_tile_elems = 128U;
    constexpr uint32_t k_swiglu_scale_chunk_buffers = 2U;
    auto align_ub = [](uint64_t value) { return AlignUpU64(value, 32U); };
    uint64_t offset = 0;
    offset = align_ub(offset + static_cast<uint64_t>(n) * sizeof(uint16_t));
    offset = align_ub(offset + static_cast<uint64_t>(output_n) * sizeof(int8_t));
    offset = align_ub(offset + static_cast<uint64_t>(n) * sizeof(float));
    offset = align_ub(offset + static_cast<uint64_t>(output_n) * sizeof(float));
    offset = align_ub(offset + static_cast<uint64_t>(output_n) * sizeof(float));
    const uint64_t max_scratch_bytes =
        std::max<uint64_t>(32U, align_ub(static_cast<uint64_t>(output_n / 2U) * sizeof(float)));
    offset = align_ub(offset + max_scratch_bytes);
    offset = align_ub(offset + 32U);
    const uint64_t scale_chunk_bytes = align_ub(static_cast<uint64_t>(k_swiglu_scale_tile_elems) * sizeof(float));
    return align_ub(offset * 2U) + k_swiglu_scale_chunk_buffers * scale_chunk_bytes;
}

void GetGmm1BlockCoordMN(uint32_t loop_idx, uint32_t tile_m, uint32_t tile_n, uint32_t &block_m, uint32_t &block_n)
{
    constexpr uint32_t kHostGmm1SwizzleOffset = 9U;
    const uint32_t tile_block_loop = CeilDivU32(tile_n, kHostGmm1SwizzleOffset);
    const uint32_t tile_block_idx = loop_idx / (kHostGmm1SwizzleOffset * tile_m);
    const uint32_t in_tile_block_idx = loop_idx % (kHostGmm1SwizzleOffset * tile_m);
    uint32_t n_col = kHostGmm1SwizzleOffset;
    if (tile_block_idx + 1U == tile_block_loop) {
        n_col = tile_n - kHostGmm1SwizzleOffset * tile_block_idx;
    }
    block_m = in_tile_block_idx / n_col;
    block_n = tile_block_idx * kHostGmm1SwizzleOffset + in_tile_block_idx % n_col;
    if ((tile_block_idx & 1U) != 0U) {
        block_m = tile_m - block_m - 1U;
    }
}

void GetGmm1ActualBlockShapeMN(const DispatchFFNCombineGmm1Tiling &gmm1, uint32_t problem_n, uint32_t current_m,
                               uint32_t tile_m, uint32_t tile_n, uint32_t block_m, uint32_t block_n, uint32_t &actual_m,
                               uint32_t &actual_n)
{
    actual_m = (block_m + 1U == tile_m) ? (current_m - block_m * gmm1.l1TileM) : gmm1.l1TileM;
    actual_n = (block_n + 1U == tile_n) ? (problem_n - block_n * gmm1.l1TileN) : gmm1.l1TileN;
}

Gmm1TaskReport CheckGmm1Task(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                             const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                             const std::vector<uint8_t> &expert_idx)
{
    constexpr uint64_t kHostGmm1TaskDebugMagic = 0x5635474d4d315453ULL;
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &gmm1 = build.tiling.gmm1Tiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const uint32_t core_num = build.block_dim;
    const size_t sync_count = static_cast<size_t>(expert_per_rank) * core_num;
    const size_t sync_bytes = sync_count * sizeof(DispatchFFNCombineGmm1SyncDebug);
    const uint64_t task_offset = gmm1.gmm1DebugOffset + sizeof(DispatchFFNCombineGmm1LayoutDebug) + sync_bytes;
    const size_t task_count = static_cast<size_t>(expert_per_rank) * core_num;
    const size_t task_bytes = task_count * sizeof(DispatchFFNCombineGmm1TaskDebug);
    if (gmm1.gmm1DebugBytes < sizeof(DispatchFFNCombineGmm1LayoutDebug) + sync_bytes + task_bytes) {
        throw std::runtime_error("gmm1 debug area is too small for task table");
    }

    std::vector<DispatchFFNCombineGmm1TaskDebug> actual(task_count);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (!actual.empty() && aclrtMemcpy(actual.data(), task_bytes, workspace_base + task_offset, task_bytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host gmm1 task table copy failed");
    }

    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);
    const uint32_t local_begin = rank * expert_per_rank;
    const uint32_t tile_n = CeilDivU32(cfg.n, gmm1.l1TileN);

    Gmm1TaskReport report;
    uint32_t group_base = 0;
    uint32_t start_core_idx = 0;
    for (uint32_t group = 0; group < expert_per_rank; ++group) {
        uint32_t current_m_raw = 0;
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            const size_t token_idx = static_cast<size_t>(src_rank) * expert_num_aligned + local_begin + group;
            current_m_raw += static_cast<uint32_t>(expected_tokens[token_idx]);
        }
        const uint32_t remaining = group_base >= cfg.max_output_size ? 0U : cfg.max_output_size - group_base;
        const uint32_t current_m = current_m_raw > remaining ? remaining : current_m_raw;
        const uint32_t tile_m = CeilDivU32(current_m, gmm1.l1TileM);
        const uint32_t core_loops = tile_m * tile_n;

        for (uint32_t core = 0; core < core_num; ++core) {
            const size_t idx = static_cast<size_t>(group) * core_num + core;
            DispatchFFNCombineGmm1TaskDebug expected;
            expected.magic = kHostGmm1TaskDebugMagic;
            expected.groupIdx = group;
            expected.coreIdx = core;
            expected.rank = rank;
            expected.rankSize = rank_size;
            expected.groupBase = group_base;
            expected.currentMRaw = current_m_raw;
            expected.currentM = current_m;
            expected.expertTokenNums = current_m_raw;
            expected.tileM = tile_m;
            expected.tileN = tile_n;
            expected.coreLoops = core_loops;
            expected.startCoreIdx = start_core_idx;
            expected.startLoopIdx = ((core < start_core_idx) ? (core + core_num) : core) - start_core_idx;
            if (expected.startLoopIdx < core_loops) {
                expected.assignedTileCount = (core_loops - 1U - expected.startLoopIdx) / core_num + 1U;
                expected.firstLoop = expected.startLoopIdx;
                expected.lastLoop = expected.startLoopIdx + (expected.assignedTileCount - 1U) * core_num;
                GetGmm1BlockCoordMN(expected.firstLoop, tile_m, tile_n, expected.firstBlockM, expected.firstBlockN);
                GetGmm1BlockCoordMN(expected.lastLoop, tile_m, tile_n, expected.lastBlockM, expected.lastBlockN);
                GetGmm1ActualBlockShapeMN(gmm1, cfg.n, current_m, tile_m, tile_n, expected.firstBlockM,
                                          expected.firstBlockN, expected.firstActualM, expected.firstActualN);
                GetGmm1ActualBlockShapeMN(gmm1, cfg.n, current_m, tile_m, tile_n, expected.lastBlockM,
                                          expected.lastBlockN, expected.lastActualM, expected.lastActualN);
            } else {
                expected.firstLoop = kHostDispatchInvalidTask;
                expected.lastLoop = kHostDispatchInvalidTask;
                expected.firstBlockM = kHostDispatchInvalidTask;
                expected.firstBlockN = kHostDispatchInvalidTask;
                expected.lastBlockM = kHostDispatchInvalidTask;
                expected.lastBlockN = kHostDispatchInvalidTask;
            }
            expected.marker = 1U;

            const auto &entry = actual[idx];
            UpdateGmm1TaskReport(report, "magic", idx, expected.magic, entry.magic);
            UpdateGmm1TaskReport(report, "groupIdx", idx, expected.groupIdx, entry.groupIdx);
            UpdateGmm1TaskReport(report, "coreIdx", idx, expected.coreIdx, entry.coreIdx);
            UpdateGmm1TaskReport(report, "rank", idx, expected.rank, entry.rank);
            UpdateGmm1TaskReport(report, "rankSize", idx, expected.rankSize, entry.rankSize);
            UpdateGmm1TaskReport(report, "groupBase", idx, expected.groupBase, entry.groupBase);
            UpdateGmm1TaskReport(report, "currentMRaw", idx, expected.currentMRaw, entry.currentMRaw);
            UpdateGmm1TaskReport(report, "currentM", idx, expected.currentM, entry.currentM);
            UpdateGmm1TaskReport(report, "expertTokenNums", idx, expected.expertTokenNums, entry.expertTokenNums);
            UpdateGmm1TaskReport(report, "tileM", idx, expected.tileM, entry.tileM);
            UpdateGmm1TaskReport(report, "tileN", idx, expected.tileN, entry.tileN);
            UpdateGmm1TaskReport(report, "coreLoops", idx, expected.coreLoops, entry.coreLoops);
            UpdateGmm1TaskReport(report, "startCoreIdx", idx, expected.startCoreIdx, entry.startCoreIdx);
            UpdateGmm1TaskReport(report, "startLoopIdx", idx, expected.startLoopIdx, entry.startLoopIdx);
            UpdateGmm1TaskReport(report, "assignedTileCount", idx, expected.assignedTileCount, entry.assignedTileCount);
            UpdateGmm1TaskReport(report, "firstLoop", idx, expected.firstLoop, entry.firstLoop);
            UpdateGmm1TaskReport(report, "lastLoop", idx, expected.lastLoop, entry.lastLoop);
            UpdateGmm1TaskReport(report, "firstBlockM", idx, expected.firstBlockM, entry.firstBlockM);
            UpdateGmm1TaskReport(report, "firstBlockN", idx, expected.firstBlockN, entry.firstBlockN);
            UpdateGmm1TaskReport(report, "lastBlockM", idx, expected.lastBlockM, entry.lastBlockM);
            UpdateGmm1TaskReport(report, "lastBlockN", idx, expected.lastBlockN, entry.lastBlockN);
            UpdateGmm1TaskReport(report, "firstActualM", idx, expected.firstActualM, entry.firstActualM);
            UpdateGmm1TaskReport(report, "firstActualN", idx, expected.firstActualN, entry.firstActualN);
            UpdateGmm1TaskReport(report, "lastActualM", idx, expected.lastActualM, entry.lastActualM);
            UpdateGmm1TaskReport(report, "lastActualN", idx, expected.lastActualN, entry.lastActualN);
            UpdateGmm1TaskReport(report, "marker", idx, expected.marker, entry.marker);
        }
        group_base += current_m;
        start_core_idx = (start_core_idx + core_loops) % core_num;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

void UpdateGmm1DoneReport(Gmm1DoneReport &report, const std::string &field, size_t idx, uint64_t expected,
                          uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

void UpdateGmm1SegmentPlanReport(Gmm1SegmentPlanReport &report, const std::string &field, size_t idx, uint64_t expected,
                                 uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

uint64_t Gmm1DoneDebugOffset(const DispatchFFNCombineBuildResult &build)
{
    const auto &gmm1 = build.tiling.gmm1Tiling;
    const uint32_t expert_per_rank = build.tiling.dispatchFFNCombineInfo.expertPerRank;
    const uint32_t core_num = build.block_dim;
    const size_t sync_count = static_cast<size_t>(expert_per_rank) * core_num;
    const size_t sync_bytes = sync_count * sizeof(DispatchFFNCombineGmm1SyncDebug);
    const size_t task_count = static_cast<size_t>(expert_per_rank) * core_num;
    const size_t task_bytes = task_count * sizeof(DispatchFFNCombineGmm1TaskDebug);
    return gmm1.gmm1DebugOffset + sizeof(DispatchFFNCombineGmm1LayoutDebug) + sync_bytes + task_bytes;
}

std::vector<uint32_t> BuildExpectedGmm1CurrentM(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                                const StandaloneRankRuntime &runtime,
                                                const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);
    const uint32_t local_begin = rank * expert_per_rank;

    std::vector<uint32_t> current_m(expert_per_rank, 0);
    uint32_t group_base = 0;
    for (uint32_t group = 0; group < expert_per_rank; ++group) {
        uint32_t current_m_raw = 0;
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            const size_t token_idx = static_cast<size_t>(src_rank) * expert_num_aligned + local_begin + group;
            current_m_raw += static_cast<uint32_t>(expected_tokens[token_idx]);
        }
        const uint32_t remaining = group_base >= cfg.max_output_size ? 0U : cfg.max_output_size - group_base;
        current_m[group] = current_m_raw > remaining ? remaining : current_m_raw;
        group_base += current_m[group];
    }
    return current_m;
}

std::vector<uint32_t> BuildExpectedGmm1CurrentMRaw(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                                   const StandaloneRankRuntime &runtime,
                                                   const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);
    const uint32_t local_begin = rank * expert_per_rank;

    std::vector<uint32_t> current_m_raw(expert_per_rank, 0);
    for (uint32_t group = 0; group < expert_per_rank; ++group) {
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            const size_t token_idx = static_cast<size_t>(src_rank) * expert_num_aligned + local_begin + group;
            current_m_raw[group] += static_cast<uint32_t>(expected_tokens[token_idx]);
        }
    }
    return current_m_raw;
}

uint64_t SwigluC2VDebugOffset(const DispatchFFNCombineBuildResult &build)
{
    return build.tiling.swigluTiling.swigluDebugOffset + sizeof(DispatchFFNCombineSwigluLayoutDebug);
}

uint64_t SwigluSegmentDebugOffset(const DispatchFFNCombineBuildResult &build, uint32_t aiv_num)
{
    const uint32_t expert_per_rank = build.tiling.dispatchFFNCombineInfo.expertPerRank;
    const uint32_t segment_count = SwigluSegmentNum(expert_per_rank);
    const size_t c2v_count = static_cast<size_t>(segment_count) * aiv_num;
    return SwigluC2VDebugOffset(build) + c2v_count * sizeof(DispatchFFNCombineSwigluC2VDebug);
}

uint64_t SwigluTaskDebugOffset(const DispatchFFNCombineBuildResult &build, uint32_t aiv_num)
{
    const uint32_t expert_per_rank = build.tiling.dispatchFFNCombineInfo.expertPerRank;
    const uint32_t segment_count = SwigluSegmentNum(expert_per_rank);
    return SwigluSegmentDebugOffset(build, aiv_num) +
           static_cast<uint64_t>(segment_count) * sizeof(DispatchFFNCombineSwigluSegmentDebug);
}

uint64_t SwigluDoneDebugOffset(const DispatchFFNCombineBuildResult &build, uint32_t aiv_num)
{
    const uint32_t expert_per_rank = build.tiling.dispatchFFNCombineInfo.expertPerRank;
    const uint32_t segment_count = SwigluSegmentNum(expert_per_rank);
    return SwigluTaskDebugOffset(build, aiv_num) +
           static_cast<uint64_t>(segment_count) * aiv_num * sizeof(DispatchFFNCombineSwigluTaskDebug);
}

uint64_t SwigluFinalSyncDebugOffset(const DispatchFFNCombineBuildResult &build, uint32_t aiv_num)
{
    const uint32_t expert_per_rank = build.tiling.dispatchFFNCombineInfo.expertPerRank;
    const uint32_t segment_count = SwigluSegmentNum(expert_per_rank);
    return SwigluDoneDebugOffset(build, aiv_num) +
           static_cast<uint64_t>(segment_count) * aiv_num * sizeof(DispatchFFNCombineSwigluDoneDebug);
}

std::vector<DispatchFFNCombineSwigluC2VDebug> ReadSwigluC2VDebugTable(const DispatchFFNCombineBuildResult &build,
                                                                      const DeviceBuffer &workspace_dev,
                                                                      uint32_t aiv_num)
{
    const auto &swiglu = build.tiling.swigluTiling;
    const uint32_t expert_per_rank = build.tiling.dispatchFFNCombineInfo.expertPerRank;
    const uint32_t segment_count = SwigluSegmentNum(expert_per_rank);
    const size_t c2v_count = static_cast<size_t>(segment_count) * aiv_num;
    const size_t c2v_bytes = c2v_count * sizeof(DispatchFFNCombineSwigluC2VDebug);
    const uint64_t c2v_offset = SwigluC2VDebugOffset(build);
    const uint64_t required_bytes = c2v_offset - swiglu.swigluDebugOffset + c2v_bytes;
    if (swiglu.swigluDebugBytes < required_bytes) {
        throw std::runtime_error("swiglu debug area is too small for C2V table");
    }

    std::vector<DispatchFFNCombineSwigluC2VDebug> actual(c2v_count);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (!actual.empty() && aclrtMemcpy(actual.data(), c2v_bytes, workspace_base + c2v_offset, c2v_bytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host swiglu C2V table copy failed");
    }
    return actual;
}

std::vector<DispatchFFNCombineSwigluSegmentDebug> ReadSwigluSegmentDebugTable(
    const DispatchFFNCombineBuildResult &build, const DeviceBuffer &workspace_dev, uint32_t aiv_num)
{
    const auto &swiglu = build.tiling.swigluTiling;
    const uint32_t expert_per_rank = build.tiling.dispatchFFNCombineInfo.expertPerRank;
    const uint32_t segment_count = SwigluSegmentNum(expert_per_rank);
    const size_t segment_bytes = static_cast<size_t>(segment_count) * sizeof(DispatchFFNCombineSwigluSegmentDebug);
    const uint64_t segment_offset = SwigluSegmentDebugOffset(build, aiv_num);
    const uint64_t required_bytes = segment_offset - swiglu.swigluDebugOffset + segment_bytes;
    if (swiglu.swigluDebugBytes < required_bytes) {
        throw std::runtime_error("swiglu debug area is too small for segment table");
    }

    std::vector<DispatchFFNCombineSwigluSegmentDebug> actual(segment_count);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (!actual.empty() && aclrtMemcpy(actual.data(), segment_bytes, workspace_base + segment_offset, segment_bytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host swiglu segment table copy failed");
    }
    return actual;
}

std::vector<DispatchFFNCombineSwigluTaskDebug> ReadSwigluTaskDebugTable(const DispatchFFNCombineBuildResult &build,
                                                                        const DeviceBuffer &workspace_dev,
                                                                        uint32_t aiv_num)
{
    const auto &swiglu = build.tiling.swigluTiling;
    const uint32_t expert_per_rank = build.tiling.dispatchFFNCombineInfo.expertPerRank;
    const uint32_t segment_count = SwigluSegmentNum(expert_per_rank);
    const size_t task_count = static_cast<size_t>(segment_count) * aiv_num;
    const size_t task_bytes = task_count * sizeof(DispatchFFNCombineSwigluTaskDebug);
    const uint64_t task_offset = SwigluTaskDebugOffset(build, aiv_num);
    const uint64_t required_bytes = task_offset - swiglu.swigluDebugOffset + task_bytes;
    if (swiglu.swigluDebugBytes < required_bytes) {
        throw std::runtime_error("swiglu debug area is too small for task table");
    }

    std::vector<DispatchFFNCombineSwigluTaskDebug> actual(task_count);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (!actual.empty() && aclrtMemcpy(actual.data(), task_bytes, workspace_base + task_offset, task_bytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host swiglu task table copy failed");
    }
    return actual;
}

std::vector<DispatchFFNCombineSwigluDoneDebug> ReadSwigluDoneDebugTable(const DispatchFFNCombineBuildResult &build,
                                                                        const DeviceBuffer &workspace_dev,
                                                                        uint32_t aiv_num)
{
    const auto &swiglu = build.tiling.swigluTiling;
    const uint32_t expert_per_rank = build.tiling.dispatchFFNCombineInfo.expertPerRank;
    const uint32_t segment_count = SwigluSegmentNum(expert_per_rank);
    const size_t done_count = static_cast<size_t>(segment_count) * aiv_num;
    const size_t done_bytes = done_count * sizeof(DispatchFFNCombineSwigluDoneDebug);
    const uint64_t done_offset = SwigluDoneDebugOffset(build, aiv_num);
    const uint64_t required_bytes = done_offset - swiglu.swigluDebugOffset + done_bytes;
    if (swiglu.swigluDebugBytes < required_bytes) {
        throw std::runtime_error("swiglu debug area is too small for done table");
    }

    std::vector<DispatchFFNCombineSwigluDoneDebug> actual(done_count);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (!actual.empty() && aclrtMemcpy(actual.data(), done_bytes, workspace_base + done_offset, done_bytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host swiglu done table copy failed");
    }
    return actual;
}

std::vector<DispatchFFNCombineSwigluFinalSyncDebug> ReadSwigluFinalSyncDebugTable(
    const DispatchFFNCombineBuildResult &build, const DeviceBuffer &workspace_dev, uint32_t aiv_num)
{
    const auto &swiglu = build.tiling.swigluTiling;
    const size_t final_sync_bytes = static_cast<size_t>(aiv_num) * sizeof(DispatchFFNCombineSwigluFinalSyncDebug);
    const uint64_t final_sync_offset = SwigluFinalSyncDebugOffset(build, aiv_num);
    const uint64_t required_bytes = final_sync_offset - swiglu.swigluDebugOffset + final_sync_bytes;
    if (swiglu.swigluDebugBytes < required_bytes) {
        throw std::runtime_error("swiglu debug area is too small for final sync table");
    }

    std::vector<DispatchFFNCombineSwigluFinalSyncDebug> actual(aiv_num);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (!actual.empty() && aclrtMemcpy(actual.data(), final_sync_bytes, workspace_base + final_sync_offset,
                                       final_sync_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host swiglu final sync table copy failed");
    }
    return actual;
}

std::vector<DispatchFFNCombineSwigluSegmentRuntimeMeta> ReadSwigluSegmentRuntimeMetaTable(
    const DispatchFFNCombineBuildResult &build, const DeviceBuffer &workspace_dev)
{
    const auto &swiglu = build.tiling.swigluTiling;
    const uint32_t expert_per_rank = build.tiling.dispatchFFNCombineInfo.expertPerRank;
    const uint32_t segment_count = SwigluSegmentNum(expert_per_rank);
    const size_t meta_bytes = static_cast<size_t>(segment_count) * sizeof(DispatchFFNCombineSwigluSegmentRuntimeMeta);
    if (swiglu.swigluSegmentMetaBytes < meta_bytes) {
        throw std::runtime_error("swiglu segment metadata area is too small");
    }

    std::vector<DispatchFFNCombineSwigluSegmentRuntimeMeta> actual(segment_count);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (!actual.empty() && aclrtMemcpy(actual.data(), meta_bytes, workspace_base + swiglu.swigluSegmentMetaOffset,
                                       meta_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host swiglu segment metadata copy failed");
    }
    return actual;
}

uint64_t Gmm2V2CDebugOffset(const DispatchFFNCombineBuildResult &build)
{
    return build.tiling.gmm2Tiling.gmm2DebugOffset + sizeof(DispatchFFNCombineGmm2LayoutDebug);
}

uint64_t Gmm2SegmentDebugOffset(const DispatchFFNCombineBuildResult &build)
{
    const uint32_t expert_per_rank = build.tiling.dispatchFFNCombineInfo.expertPerRank;
    const uint32_t segment_count = SwigluSegmentNum(expert_per_rank);
    return Gmm2V2CDebugOffset(build) +
           static_cast<uint64_t>(segment_count) * build.block_dim * sizeof(DispatchFFNCombineGmm2V2CDebug);
}

uint64_t Gmm2TaskDebugOffset(const DispatchFFNCombineBuildResult &build)
{
    const uint32_t expert_per_rank = build.tiling.dispatchFFNCombineInfo.expertPerRank;
    const uint32_t segment_count = SwigluSegmentNum(expert_per_rank);
    return Gmm2SegmentDebugOffset(build) +
           static_cast<uint64_t>(segment_count) * sizeof(DispatchFFNCombineGmm2SegmentDebug);
}

uint64_t Gmm2DoneDebugOffset(const DispatchFFNCombineBuildResult &build)
{
    const uint32_t expert_per_rank = build.tiling.dispatchFFNCombineInfo.expertPerRank;
    return Gmm2TaskDebugOffset(build) +
           static_cast<uint64_t>(expert_per_rank) * build.block_dim * sizeof(DispatchFFNCombineGmm2TaskDebug);
}

uint64_t Gmm2FinalSyncDebugOffset(const DispatchFFNCombineBuildResult &build)
{
    return Gmm2DoneDebugOffset(build) +
           static_cast<uint64_t>(build.block_dim) * sizeof(DispatchFFNCombineGmm2DoneDebug);
}

std::vector<DispatchFFNCombineGmm2V2CDebug> ReadGmm2V2CDebugTable(const DispatchFFNCombineBuildResult &build,
                                                                  const DeviceBuffer &workspace_dev)
{
    const auto &gmm2 = build.tiling.gmm2Tiling;
    const uint32_t expert_per_rank = build.tiling.dispatchFFNCombineInfo.expertPerRank;
    const uint32_t segment_count = SwigluSegmentNum(expert_per_rank);
    const size_t v2c_count = static_cast<size_t>(segment_count) * build.block_dim;
    const size_t v2c_bytes = v2c_count * sizeof(DispatchFFNCombineGmm2V2CDebug);
    const uint64_t v2c_offset = Gmm2V2CDebugOffset(build);
    const uint64_t required_bytes = v2c_offset - gmm2.gmm2DebugOffset + v2c_bytes;
    if (gmm2.gmm2DebugBytes < required_bytes) {
        throw std::runtime_error("gmm2 debug area is too small for V2C table");
    }

    std::vector<DispatchFFNCombineGmm2V2CDebug> actual(v2c_count);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (!actual.empty() && aclrtMemcpy(actual.data(), v2c_bytes, workspace_base + v2c_offset, v2c_bytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host gmm2 V2C table copy failed");
    }
    return actual;
}

std::vector<DispatchFFNCombineGmm2SegmentDebug> ReadGmm2SegmentDebugTable(const DispatchFFNCombineBuildResult &build,
                                                                          const DeviceBuffer &workspace_dev)
{
    const auto &gmm2 = build.tiling.gmm2Tiling;
    const uint32_t expert_per_rank = build.tiling.dispatchFFNCombineInfo.expertPerRank;
    const uint32_t segment_count = SwigluSegmentNum(expert_per_rank);
    const size_t segment_bytes = static_cast<size_t>(segment_count) * sizeof(DispatchFFNCombineGmm2SegmentDebug);
    const uint64_t segment_offset = Gmm2SegmentDebugOffset(build);
    const uint64_t required_bytes = segment_offset - gmm2.gmm2DebugOffset + segment_bytes;
    if (gmm2.gmm2DebugBytes < required_bytes) {
        throw std::runtime_error("gmm2 debug area is too small for segment table");
    }

    std::vector<DispatchFFNCombineGmm2SegmentDebug> actual(segment_count);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (!actual.empty() && aclrtMemcpy(actual.data(), segment_bytes, workspace_base + segment_offset, segment_bytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host gmm2 segment table copy failed");
    }
    return actual;
}

std::vector<DispatchFFNCombineGmm2TaskDebug> ReadGmm2TaskDebugTable(const DispatchFFNCombineBuildResult &build,
                                                                    const DeviceBuffer &workspace_dev)
{
    const auto &gmm2 = build.tiling.gmm2Tiling;
    const uint32_t expert_per_rank = build.tiling.dispatchFFNCombineInfo.expertPerRank;
    const size_t task_count = static_cast<size_t>(expert_per_rank) * build.block_dim;
    const size_t task_bytes = task_count * sizeof(DispatchFFNCombineGmm2TaskDebug);
    const uint64_t task_offset = Gmm2TaskDebugOffset(build);
    const uint64_t required_bytes = task_offset - gmm2.gmm2DebugOffset + task_bytes;
    if (gmm2.gmm2DebugBytes < required_bytes) {
        throw std::runtime_error("gmm2 debug area is too small for task table");
    }

    std::vector<DispatchFFNCombineGmm2TaskDebug> actual(task_count);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (!actual.empty() && aclrtMemcpy(actual.data(), task_bytes, workspace_base + task_offset, task_bytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host gmm2 task table copy failed");
    }
    return actual;
}

std::vector<DispatchFFNCombineGmm2DoneDebug> ReadGmm2DoneDebugTable(const DispatchFFNCombineBuildResult &build,
                                                                    const DeviceBuffer &workspace_dev)
{
    const auto &gmm2 = build.tiling.gmm2Tiling;
    const size_t done_count = build.block_dim;
    const size_t done_bytes = done_count * sizeof(DispatchFFNCombineGmm2DoneDebug);
    const uint64_t done_offset = Gmm2DoneDebugOffset(build);
    const uint64_t required_bytes = done_offset - gmm2.gmm2DebugOffset + done_bytes;
    if (gmm2.gmm2DebugBytes < required_bytes) {
        throw std::runtime_error("gmm2 debug area is too small for done table");
    }

    std::vector<DispatchFFNCombineGmm2DoneDebug> actual(done_count);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (!actual.empty() && aclrtMemcpy(actual.data(), done_bytes, workspace_base + done_offset, done_bytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host gmm2 done table copy failed");
    }
    return actual;
}

std::vector<DispatchFFNCombineGmm2FinalSyncDebug> ReadGmm2FinalSyncDebugTable(
    const DispatchFFNCombineBuildResult &build, const DeviceBuffer &workspace_dev)
{
    const auto &gmm2 = build.tiling.gmm2Tiling;
    const size_t final_sync_count = build.block_dim;
    const size_t final_sync_bytes = final_sync_count * sizeof(DispatchFFNCombineGmm2FinalSyncDebug);
    const uint64_t final_sync_offset = Gmm2FinalSyncDebugOffset(build);
    const uint64_t required_bytes = final_sync_offset - gmm2.gmm2DebugOffset + final_sync_bytes;
    if (gmm2.gmm2DebugBytes < required_bytes) {
        throw std::runtime_error("gmm2 debug area is too small for final sync table");
    }

    std::vector<DispatchFFNCombineGmm2FinalSyncDebug> actual(final_sync_count);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (!actual.empty() && aclrtMemcpy(actual.data(), final_sync_bytes, workspace_base + final_sync_offset,
                                       final_sync_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host gmm2 final sync table copy failed");
    }
    return actual;
}

SwigluC2VReport CheckSwigluC2VOverlap(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                      const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev)
{
    constexpr uint64_t kHostSwigluC2VDebugMagic = 0x5635535743325644ULL;
    constexpr uint32_t kHostSwigluWaitSourceC2VOnly = 1U;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t segment_count = SwigluSegmentNum(cfg.expert_per_rank);
    const std::vector<DispatchFFNCombineSwigluC2VDebug> actual =
        ReadSwigluC2VDebugTable(build, workspace_dev, cfg.aiv_num);

    SwigluC2VReport report;
    report.segment_count = segment_count;
    report.checked_count = actual.size();
    for (uint32_t segment = 0; segment < segment_count; ++segment) {
        for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
            const size_t idx = static_cast<size_t>(segment) * cfg.aiv_num + core;
            const auto &entry = actual[idx];
            UpdateSwigluC2VReport(report, "magic", idx, kHostSwigluC2VDebugMagic, entry.magic);
            UpdateSwigluC2VReport(report, "segmentIdx", idx, segment, entry.segmentIdx);
            UpdateSwigluC2VReport(report, "coreIdx", idx, core, entry.coreIdx);
            UpdateSwigluC2VReport(report, "rank", idx, rank, entry.rank);
            UpdateSwigluC2VReport(report, "rankSize", idx, rank_size, entry.rankSize);
            UpdateSwigluC2VReport(report, "stageNum", idx, build.tiling.frontReorderTiling.stageNum, entry.stageNum);
            UpdateSwigluC2VReport(report, "waitSource", idx, kHostSwigluWaitSourceC2VOnly, entry.waitSource);
            UpdateSwigluC2VReport(report, "c2vFlagId", idx, V5_C2V_HARD_FLAG_BASE, entry.c2vFlagId);
            UpdateSwigluC2VReport(report, "c2vEnabled", idx, 1U, entry.c2vEnabled);
            UpdateSwigluC2VReport(report, "gmm1DoneConsumed", idx, 0U, entry.gmm1DoneConsumed);
            UpdateSwigluC2VReport(report, "aivSyncAfterWait", idx, 1U, entry.aivSyncAfterWait);
            UpdateSwigluC2VReport(report, "marker", idx, 1U, entry.marker);
        }
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

SwigluSegmentReport CheckSwigluSegment(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                       const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                       const std::vector<uint8_t> &expert_idx)
{
    constexpr uint64_t kHostSwigluSegmentDebugMagic = 0x5635535749475347ULL;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t segment_count = SwigluSegmentNum(expert_per_rank);
    const std::vector<DispatchFFNCombineSwigluSegmentDebug> actual =
        ReadSwigluSegmentDebugTable(build, workspace_dev, cfg.aiv_num);
    const std::vector<uint32_t> current_m = BuildExpectedGmm1CurrentM(cfg, build, runtime, expert_idx);
    const std::vector<uint32_t> current_m_raw = BuildExpectedGmm1CurrentMRaw(cfg, build, runtime, expert_idx);

    SwigluSegmentReport report;
    report.segment_count = segment_count;
    report.epilogue_granularity = SwigluEpilogueGranularity(expert_per_rank);
    uint32_t segment_row_base = 0;
    for (uint32_t segment = 0; segment < segment_count; ++segment) {
        const uint32_t segment_start = SwigluSegmentStartExpert(expert_per_rank, segment);
        const uint32_t segment_end = SwigluSegmentEndExpert(expert_per_rank, segment);
        uint32_t segment_rows = 0;
        uint32_t raw_rows = 0;
        for (uint32_t group = segment_start; group < segment_end; ++group) {
            segment_rows += current_m[group];
            raw_rows += current_m_raw[group];
        }

        const auto &entry = actual[segment];
        UpdateSwigluSegmentReport(report, "magic", segment, kHostSwigluSegmentDebugMagic, entry.magic);
        UpdateSwigluSegmentReport(report, "segmentIdx", segment, segment, entry.segmentIdx);
        UpdateSwigluSegmentReport(report, "rank", segment, rank, entry.rank);
        UpdateSwigluSegmentReport(report, "rankSize", segment, rank_size, entry.rankSize);
        UpdateSwigluSegmentReport(report, "segmentStartExpert", segment, segment_start, entry.segmentStartExpert);
        UpdateSwigluSegmentReport(report, "segmentEndExpert", segment, segment_end, entry.segmentEndExpert);
        UpdateSwigluSegmentReport(report, "segmentRowBase", segment, segment_row_base, entry.segmentRowBase);
        UpdateSwigluSegmentReport(report, "segmentRows", segment, segment_rows, entry.segmentRows);
        UpdateSwigluSegmentReport(report, "cumsumRows", segment, raw_rows, entry.cumsumRows);
        UpdateSwigluSegmentReport(report, "expertTokenRows", segment, raw_rows, entry.expertTokenRows);
        UpdateSwigluSegmentReport(report, "maxOutputSize", segment, cfg.max_output_size, entry.maxOutputSize);
        UpdateSwigluSegmentReport(report, "segmentNum", segment, segment_count, entry.segmentNum);
        UpdateSwigluSegmentReport(report, "epilogueGranularity", segment, SwigluEpilogueGranularity(expert_per_rank),
                                  entry.epilogueGranularity);
        UpdateSwigluSegmentReport(report, "marker", segment, 1U, entry.marker);

        segment_row_base += segment_rows;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

SwigluMetadataModeReport CheckSwigluMetadataMode(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                                 const DeviceBuffer &workspace_dev)
{
    const auto &swiglu = build.tiling.swigluTiling;
    const uint32_t expected_mode = kHostSwigluMetadataModeSharedSegmentMeta;
    DispatchFFNCombineSwigluLayoutDebug actual;
    std::memset(&actual, 0, sizeof(actual));
    if (swiglu.swigluDebugBytes < sizeof(actual)) {
        throw std::runtime_error("swiglu debug area is too small for metadata mode header");
    }
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(&actual, sizeof(actual), workspace_base + swiglu.swigluDebugOffset, sizeof(actual),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host swiglu metadata mode copy failed");
    }

    SwigluMetadataModeReport report;
    report.expected_mode = expected_mode;
    report.actual_mode = actual.metadataMode;
    UpdateSwigluMetadataModeReport(report, "tilingMetadataMode", expected_mode, swiglu.swigluMetadataMode);
    UpdateSwigluMetadataModeReport(report, "layoutMetadataMode", expected_mode, actual.metadataMode);
    report.pass = report.mismatch_count == 0;
    return report;
}

SwigluSegmentMetaReport CheckSwigluSegmentMeta(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                               const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                               const std::vector<uint8_t> &expert_idx)
{
    const auto &swiglu = build.tiling.swigluTiling;
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t segment_count = SwigluSegmentNum(expert_per_rank);
    const uint32_t epilogue_granularity = SwigluEpilogueGranularity(expert_per_rank);

    SwigluSegmentMetaReport report;
    report.metadata_mode = swiglu.swigluMetadataMode;
    report.segment_count = segment_count;
    if (swiglu.swigluMetadataMode != kHostSwigluMetadataModeSharedSegmentMeta) {
        report.pass = report.mismatch_count == 0;
        return report;
    }

    const std::vector<DispatchFFNCombineSwigluSegmentRuntimeMeta> actual =
        ReadSwigluSegmentRuntimeMetaTable(build, workspace_dev);
    const std::vector<uint32_t> current_m = BuildExpectedGmm1CurrentM(cfg, build, runtime, expert_idx);
    const std::vector<uint32_t> current_m_raw = BuildExpectedGmm1CurrentMRaw(cfg, build, runtime, expert_idx);
    report.checked_count = actual.size();

    uint32_t segment_row_base = 0;
    for (uint32_t segment = 0; segment < segment_count; ++segment) {
        const uint32_t segment_start = SwigluSegmentStartExpert(expert_per_rank, segment);
        const uint32_t segment_end = SwigluSegmentEndExpert(expert_per_rank, segment);
        uint32_t segment_rows = 0;
        uint32_t raw_rows = 0;
        for (uint32_t group = segment_start; group < segment_end; ++group) {
            segment_rows += current_m[group];
            raw_rows += current_m_raw[group];
        }
        const uint32_t row_split_base = segment_rows / cfg.aiv_num;
        const uint32_t row_split_rem = segment_rows - row_split_base * cfg.aiv_num;
        const auto &entry = actual[segment];

        UpdateSwigluSegmentMetaReport(report, "segmentIdx", segment, segment, entry.segmentIdx);
        UpdateSwigluSegmentMetaReport(report, "segmentStartExpert", segment, segment_start, entry.segmentStartExpert);
        UpdateSwigluSegmentMetaReport(report, "segmentEndExpert", segment, segment_end, entry.segmentEndExpert);
        UpdateSwigluSegmentMetaReport(report, "segmentRowBase", segment, segment_row_base, entry.segmentRowBase);
        UpdateSwigluSegmentMetaReport(report, "segmentRows", segment, segment_rows, entry.segmentRows);
        UpdateSwigluSegmentMetaReport(report, "cumsumRows", segment, raw_rows, entry.cumsumRows);
        UpdateSwigluSegmentMetaReport(report, "expertTokenRows", segment, raw_rows, entry.expertTokenRows);
        UpdateSwigluSegmentMetaReport(report, "rowSplitBase", segment, row_split_base, entry.rowSplitBase);
        UpdateSwigluSegmentMetaReport(report, "rowSplitRem", segment, row_split_rem, entry.rowSplitRem);
        UpdateSwigluSegmentMetaReport(report, "valid", segment, 1U, entry.valid);
        UpdateSwigluSegmentMetaReport(report, "generation", segment, build.tiling.frontReorderTiling.stageNum,
                                      entry.generation);
        UpdateSwigluSegmentMetaReport(report, "producerCoreIdx", segment, 0U, entry.producerCoreIdx);
        UpdateSwigluSegmentMetaReport(report, "metadataMode", segment, kHostSwigluMetadataModeSharedSegmentMeta,
                                      entry.metadataMode);
        UpdateSwigluSegmentMetaReport(report, "segmentNum", segment, segment_count, entry.segmentNum);
        UpdateSwigluSegmentMetaReport(report, "epilogueGranularity", segment, epilogue_granularity,
                                      entry.epilogueGranularity);
        UpdateSwigluSegmentMetaReport(report, "marker", segment, 1U, entry.marker);

        segment_row_base += segment_rows;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

SwigluTaskReport CheckSwigluTask(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                 const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                 const std::vector<uint8_t> &expert_idx)
{
    constexpr uint64_t kHostSwigluTaskDebugMagic = 0x563553575441534bULL;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t segment_count = SwigluSegmentNum(expert_per_rank);
    const std::vector<DispatchFFNCombineSwigluTaskDebug> actual =
        ReadSwigluTaskDebugTable(build, workspace_dev, cfg.aiv_num);
    const std::vector<uint32_t> current_m = BuildExpectedGmm1CurrentM(cfg, build, runtime, expert_idx);

    SwigluTaskReport report;
    report.segment_count = segment_count;
    report.checked_count = actual.size();
    uint32_t segment_row_base = 0;
    for (uint32_t segment = 0; segment < segment_count; ++segment) {
        const uint32_t segment_start = SwigluSegmentStartExpert(expert_per_rank, segment);
        const uint32_t segment_end = SwigluSegmentEndExpert(expert_per_rank, segment);
        uint32_t segment_rows = 0;
        for (uint32_t group = segment_start; group < segment_end; ++group) {
            segment_rows += current_m[group];
        }
        const uint32_t row_split_base = segment_rows / cfg.aiv_num;
        const uint32_t row_split_rem = segment_rows - row_split_base * cfg.aiv_num;
        uint32_t covered_rows = 0;

        for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
            const size_t idx = static_cast<size_t>(segment) * cfg.aiv_num + core;
            const uint32_t local_rows = row_split_base + (core < row_split_rem ? 1U : 0U);
            const uint32_t local_row_start =
                segment_row_base + core * row_split_base + (core < row_split_rem ? core : row_split_rem);
            const auto &entry = actual[idx];
            UpdateSwigluTaskReport(report, "magic", idx, kHostSwigluTaskDebugMagic, entry.magic);
            UpdateSwigluTaskReport(report, "segmentIdx", idx, segment, entry.segmentIdx);
            UpdateSwigluTaskReport(report, "coreIdx", idx, core, entry.coreIdx);
            UpdateSwigluTaskReport(report, "rank", idx, rank, entry.rank);
            UpdateSwigluTaskReport(report, "rankSize", idx, rank_size, entry.rankSize);
            UpdateSwigluTaskReport(report, "segmentStartExpert", idx, segment_start, entry.segmentStartExpert);
            UpdateSwigluTaskReport(report, "segmentEndExpert", idx, segment_end, entry.segmentEndExpert);
            UpdateSwigluTaskReport(report, "segmentRowBase", idx, segment_row_base, entry.segmentRowBase);
            UpdateSwigluTaskReport(report, "segmentRows", idx, segment_rows, entry.segmentRows);
            UpdateSwigluTaskReport(report, "localRowStart", idx, local_row_start, entry.localRowStart);
            UpdateSwigluTaskReport(report, "localRows", idx, local_rows, entry.localRows);
            UpdateSwigluTaskReport(report, "rowSplitBase", idx, row_split_base, entry.rowSplitBase);
            UpdateSwigluTaskReport(report, "rowSplitRem", idx, row_split_rem, entry.rowSplitRem);
            UpdateSwigluTaskReport(report, "marker", idx, 1U, entry.marker);
            covered_rows += local_rows;
        }
        UpdateSwigluTaskReport(report, "coveredRows", segment, segment_rows, covered_rows);
        segment_row_base += segment_rows;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

SwigluDoneReport CheckSwigluDone(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                 const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                 const std::vector<uint8_t> &expert_idx)
{
    constexpr uint64_t kHostSwigluDoneDebugMagic = 0x56355357444f4e45ULL;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t segment_count = SwigluSegmentNum(expert_per_rank);
    const uint32_t expected_v2c_enabled = build.tiling.frontReorderTiling.stageNum >= 12U ? 1U : 0U;
    const std::vector<DispatchFFNCombineSwigluDoneDebug> actual =
        ReadSwigluDoneDebugTable(build, workspace_dev, cfg.aiv_num);
    const std::vector<uint32_t> current_m = BuildExpectedGmm1CurrentM(cfg, build, runtime, expert_idx);

    SwigluDoneReport report;
    report.segment_count = segment_count;
    report.checked_count = actual.size();
    uint32_t segment_row_base = 0;
    for (uint32_t segment = 0; segment < segment_count; ++segment) {
        const uint32_t segment_start = SwigluSegmentStartExpert(expert_per_rank, segment);
        const uint32_t segment_end = SwigluSegmentEndExpert(expert_per_rank, segment);
        uint32_t segment_rows = 0;
        for (uint32_t group = segment_start; group < segment_end; ++group) {
            segment_rows += current_m[group];
        }
        const uint32_t row_split_base = segment_rows / cfg.aiv_num;
        const uint32_t row_split_rem = segment_rows - row_split_base * cfg.aiv_num;
        for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
            const size_t idx = static_cast<size_t>(segment) * cfg.aiv_num + core;
            const uint32_t local_rows = row_split_base + (core < row_split_rem ? 1U : 0U);
            const uint32_t local_row_start =
                segment_row_base + core * row_split_base + (core < row_split_rem ? core : row_split_rem);
            const auto &entry = actual[idx];
            UpdateSwigluDoneReport(report, "magic", idx, kHostSwigluDoneDebugMagic, entry.magic);
            UpdateSwigluDoneReport(report, "segmentIdx", idx, segment, entry.segmentIdx);
            UpdateSwigluDoneReport(report, "coreIdx", idx, core, entry.coreIdx);
            UpdateSwigluDoneReport(report, "rank", idx, rank, entry.rank);
            UpdateSwigluDoneReport(report, "rankSize", idx, rank_size, entry.rankSize);
            UpdateSwigluDoneReport(report, "stageNum", idx, build.tiling.frontReorderTiling.stageNum, entry.stageNum);
            UpdateSwigluDoneReport(report, "segmentRowBase", idx, segment_row_base, entry.segmentRowBase);
            UpdateSwigluDoneReport(report, "segmentRows", idx, segment_rows, entry.segmentRows);
            UpdateSwigluDoneReport(report, "localRowStart", idx, local_row_start, entry.localRowStart);
            UpdateSwigluDoneReport(report, "localRows", idx, local_rows, entry.localRows);
            UpdateSwigluDoneReport(report, "aivSyncBeforeDone", idx, 1U, entry.aivSyncBeforeDone);
            UpdateSwigluDoneReport(report, "v2cFlagId", idx, V5_V2C_HARD_FLAG_BASE, entry.v2cFlagId);
            UpdateSwigluDoneReport(report, "v2cEnabled", idx, expected_v2c_enabled, entry.v2cEnabled);
            UpdateSwigluDoneReport(report, "marker", idx, 1U, entry.marker);
        }
        segment_row_base += segment_rows;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

SwigluFinalSyncReport CheckSwigluFinalSync(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                           const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev)
{
    constexpr uint64_t kHostSwigluFinalSyncDebugMagic = 0x56355357464e5359ULL;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t stage_num = build.tiling.frontReorderTiling.stageNum;
    const uint32_t segment_count = SwigluSegmentNum(cfg.expert_per_rank);
    const std::vector<DispatchFFNCombineSwigluFinalSyncDebug> actual =
        ReadSwigluFinalSyncDebugTable(build, workspace_dev, cfg.aiv_num);

    SwigluFinalSyncReport report;
    report.checked_count = actual.size();
    for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
        const auto &entry = actual[core];
        UpdateSwigluFinalSyncReport(report, "magic", core, kHostSwigluFinalSyncDebugMagic, entry.magic);
        UpdateSwigluFinalSyncReport(report, "coreIdx", core, core, entry.coreIdx);
        UpdateSwigluFinalSyncReport(report, "rank", core, rank, entry.rank);
        UpdateSwigluFinalSyncReport(report, "rankSize", core, rank_size, entry.rankSize);
        UpdateSwigluFinalSyncReport(report, "stageNum", core, stage_num, entry.stageNum);
        UpdateSwigluFinalSyncReport(report, "segmentNum", core, segment_count, entry.segmentNum);
        UpdateSwigluFinalSyncReport(report, "finalSyncEnabled", core, 1U, entry.finalSyncEnabled);
        UpdateSwigluFinalSyncReport(report, "finalSyncAfterAllSegments", core, 1U, entry.finalSyncAfterAllSegments);
        UpdateSwigluFinalSyncReport(report, "v2cEnabled", core, 0U, entry.v2cEnabled);
        UpdateSwigluFinalSyncReport(report, "marker", core, 1U, entry.marker);
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

SwigluFinalSyncReport CheckSwigluFinalSyncDisabled(const DispatchFFNCombineBuildResult &build,
                                                   const DeviceBuffer &workspace_dev, uint32_t aiv_num)
{
    const std::vector<DispatchFFNCombineSwigluFinalSyncDebug> actual =
        ReadSwigluFinalSyncDebugTable(build, workspace_dev, aiv_num);

    SwigluFinalSyncReport report;
    report.checked_count = actual.size();
    for (uint32_t core = 0; core < aiv_num; ++core) {
        const auto &entry = actual[core];
        UpdateSwigluFinalSyncReport(report, "magic", core, 0U, entry.magic);
        UpdateSwigluFinalSyncReport(report, "coreIdx", core, 0U, entry.coreIdx);
        UpdateSwigluFinalSyncReport(report, "rank", core, 0U, entry.rank);
        UpdateSwigluFinalSyncReport(report, "rankSize", core, 0U, entry.rankSize);
        UpdateSwigluFinalSyncReport(report, "stageNum", core, 0U, entry.stageNum);
        UpdateSwigluFinalSyncReport(report, "segmentNum", core, 0U, entry.segmentNum);
        UpdateSwigluFinalSyncReport(report, "finalSyncEnabled", core, 0U, entry.finalSyncEnabled);
        UpdateSwigluFinalSyncReport(report, "finalSyncAfterAllSegments", core, 0U, entry.finalSyncAfterAllSegments);
        UpdateSwigluFinalSyncReport(report, "v2cEnabled", core, 0U, entry.v2cEnabled);
        UpdateSwigluFinalSyncReport(report, "marker", core, 0U, entry.marker);
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

Gmm2V2CReport CheckGmm2V2COverlap(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                  const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev)
{
    constexpr uint64_t kHostGmm2V2CDebugMagic = 0x5635474d32563243ULL;
    constexpr uint32_t kHostGmm2WaitSourceV2COnly = 1U;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t segment_count = SwigluSegmentNum(cfg.expert_per_rank);
    const std::vector<DispatchFFNCombineGmm2V2CDebug> actual = ReadGmm2V2CDebugTable(build, workspace_dev);

    Gmm2V2CReport report;
    report.segment_count = segment_count;
    report.checked_count = actual.size();
    for (uint32_t segment = 0; segment < segment_count; ++segment) {
        for (uint32_t core = 0; core < build.block_dim; ++core) {
            const size_t idx = static_cast<size_t>(segment) * build.block_dim + core;
            const auto &entry = actual[idx];
            UpdateGmm2V2CReport(report, "magic", idx, kHostGmm2V2CDebugMagic, entry.magic);
            UpdateGmm2V2CReport(report, "segmentIdx", idx, segment, entry.segmentIdx);
            UpdateGmm2V2CReport(report, "coreIdx", idx, core, entry.coreIdx);
            UpdateGmm2V2CReport(report, "rank", idx, rank, entry.rank);
            UpdateGmm2V2CReport(report, "rankSize", idx, rank_size, entry.rankSize);
            UpdateGmm2V2CReport(report, "stageNum", idx, build.tiling.frontReorderTiling.stageNum, entry.stageNum);
            UpdateGmm2V2CReport(report, "waitSource", idx, kHostGmm2WaitSourceV2COnly, entry.waitSource);
            UpdateGmm2V2CReport(report, "v2cFlagId", idx, V5_V2C_HARD_FLAG_BASE, entry.v2cFlagId);
            UpdateGmm2V2CReport(report, "v2cEnabled", idx, 1U, entry.v2cEnabled);
            UpdateGmm2V2CReport(report, "swigluDoneConsumed", idx, 0U, entry.swigluDoneConsumed);
            UpdateGmm2V2CReport(report, "finalSyncConsumed", idx, 0U, entry.finalSyncConsumed);
            UpdateGmm2V2CReport(report, "marker", idx, 1U, entry.marker);
        }
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

Gmm2SegmentReport CheckGmm2Segment(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                   const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                   const std::vector<uint8_t> &expert_idx)
{
    constexpr uint64_t kHostGmm2SegmentDebugMagic = 0x5635474d3253474dULL;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t segment_count = SwigluSegmentNum(expert_per_rank);
    const std::vector<DispatchFFNCombineGmm2SegmentDebug> actual = ReadGmm2SegmentDebugTable(build, workspace_dev);
    const std::vector<uint32_t> current_m = BuildExpectedGmm1CurrentM(cfg, build, runtime, expert_idx);
    const std::vector<uint32_t> current_m_raw = BuildExpectedGmm1CurrentMRaw(cfg, build, runtime, expert_idx);

    Gmm2SegmentReport report;
    report.segment_count = segment_count;
    report.epilogue_granularity = SwigluEpilogueGranularity(expert_per_rank);
    uint32_t segment_row_base = 0;
    for (uint32_t segment = 0; segment < segment_count; ++segment) {
        const uint32_t segment_start = SwigluSegmentStartExpert(expert_per_rank, segment);
        const uint32_t segment_end = SwigluSegmentEndExpert(expert_per_rank, segment);
        uint32_t segment_rows = 0;
        uint32_t raw_rows = 0;
        for (uint32_t group = segment_start; group < segment_end; ++group) {
            segment_rows += current_m[group];
            raw_rows += current_m_raw[group];
        }

        const auto &entry = actual[segment];
        UpdateGmm2SegmentReport(report, "magic", segment, kHostGmm2SegmentDebugMagic, entry.magic);
        UpdateGmm2SegmentReport(report, "segmentIdx", segment, segment, entry.segmentIdx);
        UpdateGmm2SegmentReport(report, "rank", segment, rank, entry.rank);
        UpdateGmm2SegmentReport(report, "rankSize", segment, rank_size, entry.rankSize);
        UpdateGmm2SegmentReport(report, "segmentStartExpert", segment, segment_start, entry.segmentStartExpert);
        UpdateGmm2SegmentReport(report, "segmentEndExpert", segment, segment_end, entry.segmentEndExpert);
        UpdateGmm2SegmentReport(report, "segmentRowBase", segment, segment_row_base, entry.segmentRowBase);
        UpdateGmm2SegmentReport(report, "segmentRows", segment, segment_rows, entry.segmentRows);
        UpdateGmm2SegmentReport(report, "cumsumRows", segment, raw_rows, entry.cumsumRows);
        UpdateGmm2SegmentReport(report, "expertTokenRows", segment, raw_rows, entry.expertTokenRows);
        UpdateGmm2SegmentReport(report, "maxOutputSize", segment, cfg.max_output_size, entry.maxOutputSize);
        UpdateGmm2SegmentReport(report, "segmentNum", segment, segment_count, entry.segmentNum);
        UpdateGmm2SegmentReport(report, "epilogueGranularity", segment, SwigluEpilogueGranularity(expert_per_rank),
                                entry.epilogueGranularity);
        UpdateGmm2SegmentReport(report, "marker", segment, 1U, entry.marker);

        segment_row_base += segment_rows;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

Gmm2TaskReport CheckGmm2Task(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                             const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                             const std::vector<uint8_t> &expert_idx)
{
    constexpr uint64_t kHostGmm2TaskDebugMagic = 0x5635474d3254534bULL;
    const auto &gmm2 = build.tiling.gmm2Tiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t core_num = build.block_dim;
    const uint32_t tile_n = CeilDivU32(cfg.k, gmm2.l1TileN);
    const std::vector<uint32_t> current_m = BuildExpectedGmm1CurrentM(cfg, build, runtime, expert_idx);
    const std::vector<uint32_t> current_m_raw = BuildExpectedGmm1CurrentMRaw(cfg, build, runtime, expert_idx);
    const std::vector<DispatchFFNCombineGmm2TaskDebug> actual = ReadGmm2TaskDebugTable(build, workspace_dev);

    Gmm2TaskReport report;
    uint32_t group_base = 0;
    uint32_t start_core_idx = 0;
    for (uint32_t segment = 0; segment < SwigluSegmentNum(expert_per_rank); ++segment) {
        const uint32_t segment_start = SwigluSegmentStartExpert(expert_per_rank, segment);
        const uint32_t segment_end = SwigluSegmentEndExpert(expert_per_rank, segment);
        for (uint32_t group = segment_start; group < segment_end; ++group) {
            const uint32_t cur_m_raw = current_m_raw[group];
            const uint32_t cur_m = current_m[group];
            const uint32_t tile_m = CeilDivU32(cur_m, gmm2.l1TileM);
            const uint32_t core_loops = tile_m * tile_n;
            for (uint32_t core = 0; core < core_num; ++core) {
                const size_t idx = static_cast<size_t>(group) * core_num + core;
                DispatchFFNCombineGmm2TaskDebug expected;
                expected.magic = kHostGmm2TaskDebugMagic;
                expected.segmentIdx = segment;
                expected.groupIdx = group;
                expected.coreIdx = core;
                expected.rank = rank;
                expected.rankSize = rank_size;
                expected.groupBase = group_base;
                expected.currentMRaw = cur_m_raw;
                expected.currentM = cur_m;
                expected.expertTokenNums = cur_m_raw;
                expected.tileM = tile_m;
                expected.tileN = tile_n;
                expected.coreLoops = core_loops;
                expected.startCoreIdx = start_core_idx;
                expected.startLoopIdx = ((core < start_core_idx) ? (core + core_num) : core) - start_core_idx;
                if (expected.startLoopIdx < core_loops) {
                    expected.assignedTileCount = (core_loops - 1U - expected.startLoopIdx) / core_num + 1U;
                    expected.firstLoop = expected.startLoopIdx;
                    expected.lastLoop = expected.startLoopIdx + (expected.assignedTileCount - 1U) * core_num;
                    GetGmm1BlockCoordMN(expected.firstLoop, tile_m, tile_n, expected.firstBlockM, expected.firstBlockN);
                    GetGmm1BlockCoordMN(expected.lastLoop, tile_m, tile_n, expected.lastBlockM, expected.lastBlockN);
                    expected.firstActualM = (expected.firstBlockM + 1U == tile_m) ?
                                                (cur_m - expected.firstBlockM * gmm2.l1TileM) :
                                                gmm2.l1TileM;
                    expected.firstActualN = (expected.firstBlockN + 1U == tile_n) ?
                                                (cfg.k - expected.firstBlockN * gmm2.l1TileN) :
                                                gmm2.l1TileN;
                    expected.lastActualM = (expected.lastBlockM + 1U == tile_m) ?
                                               (cur_m - expected.lastBlockM * gmm2.l1TileM) :
                                               gmm2.l1TileM;
                    expected.lastActualN = (expected.lastBlockN + 1U == tile_n) ?
                                               (cfg.k - expected.lastBlockN * gmm2.l1TileN) :
                                               gmm2.l1TileN;
                } else {
                    expected.firstLoop = kHostDispatchInvalidTask;
                    expected.lastLoop = kHostDispatchInvalidTask;
                    expected.firstBlockM = kHostDispatchInvalidTask;
                    expected.firstBlockN = kHostDispatchInvalidTask;
                    expected.lastBlockM = kHostDispatchInvalidTask;
                    expected.lastBlockN = kHostDispatchInvalidTask;
                }
                expected.marker = 1U;

                const auto &entry = actual[idx];
                UpdateGmm2TaskReport(report, "magic", idx, expected.magic, entry.magic);
                UpdateGmm2TaskReport(report, "segmentIdx", idx, expected.segmentIdx, entry.segmentIdx);
                UpdateGmm2TaskReport(report, "groupIdx", idx, expected.groupIdx, entry.groupIdx);
                UpdateGmm2TaskReport(report, "coreIdx", idx, expected.coreIdx, entry.coreIdx);
                UpdateGmm2TaskReport(report, "rank", idx, expected.rank, entry.rank);
                UpdateGmm2TaskReport(report, "rankSize", idx, expected.rankSize, entry.rankSize);
                UpdateGmm2TaskReport(report, "groupBase", idx, expected.groupBase, entry.groupBase);
                UpdateGmm2TaskReport(report, "currentMRaw", idx, expected.currentMRaw, entry.currentMRaw);
                UpdateGmm2TaskReport(report, "currentM", idx, expected.currentM, entry.currentM);
                UpdateGmm2TaskReport(report, "expertTokenNums", idx, expected.expertTokenNums, entry.expertTokenNums);
                UpdateGmm2TaskReport(report, "tileM", idx, expected.tileM, entry.tileM);
                UpdateGmm2TaskReport(report, "tileN", idx, expected.tileN, entry.tileN);
                UpdateGmm2TaskReport(report, "coreLoops", idx, expected.coreLoops, entry.coreLoops);
                UpdateGmm2TaskReport(report, "startCoreIdx", idx, expected.startCoreIdx, entry.startCoreIdx);
                UpdateGmm2TaskReport(report, "startLoopIdx", idx, expected.startLoopIdx, entry.startLoopIdx);
                UpdateGmm2TaskReport(report, "assignedTileCount", idx, expected.assignedTileCount,
                                     entry.assignedTileCount);
                UpdateGmm2TaskReport(report, "firstLoop", idx, expected.firstLoop, entry.firstLoop);
                UpdateGmm2TaskReport(report, "lastLoop", idx, expected.lastLoop, entry.lastLoop);
                UpdateGmm2TaskReport(report, "firstBlockM", idx, expected.firstBlockM, entry.firstBlockM);
                UpdateGmm2TaskReport(report, "firstBlockN", idx, expected.firstBlockN, entry.firstBlockN);
                UpdateGmm2TaskReport(report, "lastBlockM", idx, expected.lastBlockM, entry.lastBlockM);
                UpdateGmm2TaskReport(report, "lastBlockN", idx, expected.lastBlockN, entry.lastBlockN);
                UpdateGmm2TaskReport(report, "firstActualM", idx, expected.firstActualM, entry.firstActualM);
                UpdateGmm2TaskReport(report, "firstActualN", idx, expected.firstActualN, entry.firstActualN);
                UpdateGmm2TaskReport(report, "lastActualM", idx, expected.lastActualM, entry.lastActualM);
                UpdateGmm2TaskReport(report, "lastActualN", idx, expected.lastActualN, entry.lastActualN);
                UpdateGmm2TaskReport(report, "marker", idx, expected.marker, entry.marker);
            }
            group_base += cur_m;
            start_core_idx = (start_core_idx + core_loops) % core_num;
        }
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

Gmm2DoneReport CheckGmm2Done(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                             const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev)
{
    constexpr uint64_t kHostGmm2DoneDebugMagic = 0x5635474d32444f4eULL;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t segment_count = SwigluSegmentNum(cfg.expert_per_rank);
    const uint32_t stage_num = build.tiling.frontReorderTiling.stageNum;
    const std::vector<DispatchFFNCombineGmm2DoneDebug> actual = ReadGmm2DoneDebugTable(build, workspace_dev);

    Gmm2DoneReport report;
    report.checked_count = actual.size();
    for (uint32_t core = 0; core < build.block_dim; ++core) {
        const auto &entry = actual[core];
        UpdateGmm2DoneReport(report, "magic", core, kHostGmm2DoneDebugMagic, entry.magic);
        UpdateGmm2DoneReport(report, "coreIdx", core, core, entry.coreIdx);
        UpdateGmm2DoneReport(report, "rank", core, rank, entry.rank);
        UpdateGmm2DoneReport(report, "rankSize", core, rank_size, entry.rankSize);
        UpdateGmm2DoneReport(report, "stageNum", core, stage_num, entry.stageNum);
        UpdateGmm2DoneReport(report, "segmentNum", core, segment_count, entry.segmentNum);
        UpdateGmm2DoneReport(report, "expertPerRank", core, cfg.expert_per_rank, entry.expertPerRank);
        UpdateGmm2DoneReport(report, "finalSyncEnabled", core, stage_num == 12U ? 1U : 0U, entry.finalSyncEnabled);
        UpdateGmm2DoneReport(report, "combineReadyEnabled", core, stage_num >= 13U ? 1U : 0U,
                             entry.combineReadyEnabled);
        UpdateGmm2DoneReport(report, "pipelineDrained", core, 1U, entry.pipelineDrained);
        UpdateGmm2DoneReport(report, "marker", core, 1U, entry.marker);
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

Gmm2FinalSyncReport CheckGmm2FinalSync(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                       const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev)
{
    constexpr uint64_t kHostGmm2FinalSyncDebugMagic = 0x5635474d32465359ULL;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t segment_count = SwigluSegmentNum(cfg.expert_per_rank);
    const uint32_t stage_num = build.tiling.frontReorderTiling.stageNum;
    const std::vector<DispatchFFNCombineGmm2FinalSyncDebug> actual = ReadGmm2FinalSyncDebugTable(build, workspace_dev);

    Gmm2FinalSyncReport report;
    report.checked_count = actual.size();
    for (uint32_t core = 0; core < build.block_dim; ++core) {
        const auto &entry = actual[core];
        UpdateGmm2FinalSyncReport(report, "magic", core, kHostGmm2FinalSyncDebugMagic, entry.magic);
        UpdateGmm2FinalSyncReport(report, "coreIdx", core, core, entry.coreIdx);
        UpdateGmm2FinalSyncReport(report, "rank", core, rank, entry.rank);
        UpdateGmm2FinalSyncReport(report, "rankSize", core, rank_size, entry.rankSize);
        UpdateGmm2FinalSyncReport(report, "stageNum", core, stage_num, entry.stageNum);
        UpdateGmm2FinalSyncReport(report, "segmentNum", core, segment_count, entry.segmentNum);
        UpdateGmm2FinalSyncReport(report, "expertPerRank", core, cfg.expert_per_rank, entry.expertPerRank);
        UpdateGmm2FinalSyncReport(report, "finalSyncEnabled", core, stage_num == 12U ? 1U : 0U, entry.finalSyncEnabled);
        UpdateGmm2FinalSyncReport(report, "finalSyncAfterAllGroups", core, stage_num == 12U ? 1U : 0U,
                                  entry.finalSyncAfterAllGroups);
        UpdateGmm2FinalSyncReport(report, "combineReadyEnabled", core, stage_num >= 13U ? 1U : 0U,
                                  entry.combineReadyEnabled);
        UpdateGmm2FinalSyncReport(report, "marker", core, 1U, entry.marker);
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

void UpdateSwigluOutputQuantReport(SwigluOutputReport &report, uint32_t segment, uint32_t row, uint32_t col,
                                   int32_t expected, int32_t actual)
{
    ++report.value_count;
    if (expected == actual) {
        return;
    }
    if (report.first_table.empty()) {
        report.first_table = "gmPermutedToken";
        report.first_segment = segment;
        report.first_row = row;
        report.first_col = col;
        report.expected_i32 = expected;
        report.actual_i32 = actual;
    }
    ++report.quant_mismatch_count;
}

void UpdateSwigluOutputScaleReport(SwigluOutputReport &report, uint32_t segment, uint32_t row, float expected,
                                   float actual)
{
    const bool invalid = std::isnan(expected) || std::isinf(expected) || std::isnan(actual) || std::isinf(actual);
    const double abs_err =
        invalid ? std::numeric_limits<double>::infinity() : std::fabs(static_cast<double>(actual) - expected);
    const double rel_denom = std::max(std::fabs(static_cast<double>(expected)), 1.0e-12);
    const double rel_err = invalid ? std::numeric_limits<double>::infinity() : abs_err / rel_denom;
    const double tolerance = 1.0e-5 + 1.0e-3 * std::fabs(static_cast<double>(expected));
    report.max_scale_abs_err = std::max(report.max_scale_abs_err, abs_err);
    report.max_scale_rel_err = std::max(report.max_scale_rel_err, rel_err);
    ++report.scale_count;
    if (!invalid && abs_err <= tolerance) {
        return;
    }
    if (report.first_table.empty()) {
        report.first_table = "ptrPerTokenScale2";
        report.first_segment = segment;
        report.first_row = row;
        report.expected_f32 = expected;
        report.actual_f32 = actual;
    }
    ++report.scale_mismatch_count;
}

SwigluOutputReport CheckSwigluOutput(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                     const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                     const std::vector<uint8_t> &expert_idx)
{
    const auto &dispatch = build.tiling.dispatchTiling;
    const auto &gmm1 = build.tiling.gmm1Tiling;
    const auto &swiglu = build.tiling.swigluTiling;
    const uint32_t output_n = cfg.n / 2U;
    const size_t gm_c_elems = static_cast<size_t>(cfg.max_output_size) * cfg.n;
    const size_t gm_c_bytes = gm_c_elems * sizeof(uint16_t);
    const size_t scale_elems = cfg.max_output_size;
    const size_t scale_bytes = scale_elems * sizeof(float);
    const size_t permuted_elems = static_cast<size_t>(cfg.max_output_size) * output_n;
    const size_t permuted_bytes = permuted_elems * sizeof(int8_t);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);

    std::vector<uint16_t> gm_c(gm_c_elems, 0);
    std::vector<float> per_token_scale(scale_elems, 0.0f);
    std::vector<int8_t> gm_permuted(permuted_elems, 0);
    std::vector<float> per_token_scale2(scale_elems, 0.0f);
    if (!gm_c.empty() && aclrtMemcpy(gm_c.data(), gm_c_bytes, workspace_base + gmm1.gmCOffset, gm_c_bytes,
                                     ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host swiglu gmC copy failed");
    }
    if (!per_token_scale.empty() &&
        aclrtMemcpy(per_token_scale.data(), scale_bytes, workspace_base + dispatch.perTokenScaleOffset, scale_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host swiglu ptrPerTokenScale copy failed");
    }
    if (!gm_permuted.empty() &&
        aclrtMemcpy(gm_permuted.data(), permuted_bytes, workspace_base + swiglu.gmPermutedTokenOffset, permuted_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host swiglu gmPermutedToken copy failed");
    }
    if (!per_token_scale2.empty() &&
        aclrtMemcpy(per_token_scale2.data(), scale_bytes, workspace_base + swiglu.perTokenScale2Offset, scale_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host swiglu ptrPerTokenScale2 copy failed");
    }

    SwigluOutputReport report;
    report.segment_count = SwigluSegmentNum(cfg.expert_per_rank);
    report.ub_bytes = SwigluFullRowUbBytes(cfg.n);
    const std::vector<uint32_t> current_m = BuildExpectedGmm1CurrentM(cfg, build, runtime, expert_idx);
    std::vector<float> swiglu_values(output_n, 0.0f);
    uint32_t segment_row_base = 0;
    for (uint32_t segment = 0; segment < report.segment_count; ++segment) {
        const uint32_t segment_start = SwigluSegmentStartExpert(cfg.expert_per_rank, segment);
        const uint32_t segment_end = SwigluSegmentEndExpert(cfg.expert_per_rank, segment);
        uint32_t segment_rows = 0;
        for (uint32_t group = segment_start; group < segment_end; ++group) {
            segment_rows += current_m[group];
        }
        for (uint32_t local_row = 0; local_row < segment_rows; ++local_row) {
            const uint32_t row = segment_row_base + local_row;
            const float row_scale = per_token_scale[row];
            float max_abs = 0.0f;
            for (uint32_t col = 0; col < output_n; ++col) {
                const float x = Fp16ToFloat(gm_c[static_cast<size_t>(row) * cfg.n + col]) * row_scale;
                const float gate = Fp16ToFloat(gm_c[static_cast<size_t>(row) * cfg.n + output_n + col]) * row_scale;
                const float denom = 1.0f + static_cast<float>(std::exp(-x));
                const float value = x / denom * gate;
                swiglu_values[col] = value;
                max_abs = std::max(max_abs, std::fabs(value));
            }
            const float expected_scale2 = max_abs > 0.0f ? max_abs / 127.0f : 1.0e-6f / 127.0f;
            UpdateSwigluOutputScaleReport(report, segment, row, expected_scale2, per_token_scale2[row]);
            for (uint32_t col = 0; col < output_n; ++col) {
                const float quant_input = max_abs > 0.0f ? swiglu_values[col] / expected_scale2 : 0.0f;
                const int32_t expected = static_cast<int32_t>(QuantizeFp32ViaHalfCastRint(quant_input));
                const int32_t actual = static_cast<int32_t>(gm_permuted[static_cast<size_t>(row) * output_n + col]);
                UpdateSwigluOutputQuantReport(report, segment, row, col, expected, actual);
            }
            ++report.row_count;
        }
        segment_row_base += segment_rows;
    }
    report.pass = report.quant_mismatch_count == 0 && report.scale_mismatch_count == 0 &&
                  report.value_count == report.row_count * output_n && report.scale_count == report.row_count;
    return report;
}

std::vector<DispatchFFNCombineGmm1DoneDebug> ReadGmm1DoneDebugTable(const DispatchFFNCombineBuildResult &build,
                                                                    const DeviceBuffer &workspace_dev,
                                                                    size_t done_count)
{
    const auto &gmm1 = build.tiling.gmm1Tiling;
    const size_t done_bytes = done_count * sizeof(DispatchFFNCombineGmm1DoneDebug);
    const uint64_t done_offset = Gmm1DoneDebugOffset(build);
    const uint64_t required_bytes = done_offset - gmm1.gmm1DebugOffset + done_bytes;
    if (gmm1.gmm1DebugBytes < required_bytes) {
        throw std::runtime_error("gmm1 debug area is too small for done table");
    }

    std::vector<DispatchFFNCombineGmm1DoneDebug> actual(done_count);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (!actual.empty() && aclrtMemcpy(actual.data(), done_bytes, workspace_base + done_offset, done_bytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host gmm1 done table copy failed");
    }
    return actual;
}

Gmm1SegmentPlanReport CheckGmm1SegmentPlan(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                           const DeviceBuffer &workspace_dev)
{
    constexpr uint64_t kHostGmm1DoneDebugMagic = 0x5635474d4d31444eULL;
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t core_num = build.block_dim;
    const uint32_t segment_count = SwigluSegmentNum(expert_per_rank);
    const size_t done_count = static_cast<size_t>(segment_count) * core_num;
    const std::vector<DispatchFFNCombineGmm1DoneDebug> actual =
        ReadGmm1DoneDebugTable(build, workspace_dev, done_count);

    Gmm1SegmentPlanReport report;
    report.segment_count = segment_count;
    report.epilogue_granularity = SwigluEpilogueGranularity(expert_per_rank);
    for (uint32_t segment = 0; segment < segment_count; ++segment) {
        const uint32_t segment_start = SwigluSegmentStartExpert(expert_per_rank, segment);
        const uint32_t segment_end = SwigluSegmentEndExpert(expert_per_rank, segment);
        for (uint32_t core = 0; core < core_num; ++core) {
            const size_t idx = static_cast<size_t>(segment) * core_num + core;
            const auto &entry = actual[idx];
            UpdateGmm1SegmentPlanReport(report, "magic", idx, kHostGmm1DoneDebugMagic, entry.magic);
            UpdateGmm1SegmentPlanReport(report, "segmentIdx", idx, segment, entry.segmentIdx);
            UpdateGmm1SegmentPlanReport(report, "segmentStartExpert", idx, segment_start, entry.segmentStartExpert);
            UpdateGmm1SegmentPlanReport(report, "segmentEndExpert", idx, segment_end, entry.segmentEndExpert);
            UpdateGmm1SegmentPlanReport(report, "c2vFlagId", idx, V5_C2V_HARD_FLAG_BASE, entry.c2vFlagId);
            UpdateGmm1SegmentPlanReport(report, "segmentPolicy", idx, kHostGmm1SegmentPolicyReferenceEpilogue,
                                        entry.segmentPolicy);
            UpdateGmm1SegmentPlanReport(report, "marker", idx, 1U, entry.marker);
        }
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

Gmm1DoneReport CheckGmm1Done(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                             const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                             const std::vector<uint8_t> &expert_idx)
{
    constexpr uint64_t kHostGmm1DoneDebugMagic = 0x5635474d4d31444eULL;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t core_num = build.block_dim;
    const uint32_t segment_count = SwigluSegmentNum(expert_per_rank);
    const size_t done_count = static_cast<size_t>(segment_count) * core_num;
    const std::vector<DispatchFFNCombineGmm1DoneDebug> actual =
        ReadGmm1DoneDebugTable(build, workspace_dev, done_count);
    const std::vector<uint32_t> current_m = BuildExpectedGmm1CurrentM(cfg, build, runtime, expert_idx);
    const uint32_t expected_c2v_enabled = build.tiling.frontReorderTiling.stageNum >= 11U ? 1U : 0U;

    Gmm1DoneReport report;
    report.segment_count = segment_count;
    uint32_t segment_row_base = 0;
    for (uint32_t segment = 0; segment < segment_count; ++segment) {
        const uint32_t segment_start = SwigluSegmentStartExpert(expert_per_rank, segment);
        const uint32_t segment_end = SwigluSegmentEndExpert(expert_per_rank, segment);
        uint32_t segment_rows = 0;
        for (uint32_t group = segment_start; group < segment_end; ++group) {
            segment_rows += current_m[group];
        }
        for (uint32_t core = 0; core < core_num; ++core) {
            const size_t idx = static_cast<size_t>(segment) * core_num + core;
            const auto &entry = actual[idx];
            UpdateGmm1DoneReport(report, "magic", idx, kHostGmm1DoneDebugMagic, entry.magic);
            UpdateGmm1DoneReport(report, "segmentIdx", idx, segment, entry.segmentIdx);
            UpdateGmm1DoneReport(report, "coreIdx", idx, core, entry.coreIdx);
            UpdateGmm1DoneReport(report, "rank", idx, rank, entry.rank);
            UpdateGmm1DoneReport(report, "rankSize", idx, rank_size, entry.rankSize);
            UpdateGmm1DoneReport(report, "segmentStartExpert", idx, segment_start, entry.segmentStartExpert);
            UpdateGmm1DoneReport(report, "segmentEndExpert", idx, segment_end, entry.segmentEndExpert);
            UpdateGmm1DoneReport(report, "segmentRowBase", idx, segment_row_base, entry.segmentRowBase);
            UpdateGmm1DoneReport(report, "segmentRows", idx, segment_rows, entry.segmentRows);
            UpdateGmm1DoneReport(report, "c2vFlagId", idx, V5_C2V_HARD_FLAG_BASE, entry.c2vFlagId);
            UpdateGmm1DoneReport(report, "c2vEnabled", idx, expected_c2v_enabled, entry.c2vEnabled);
            UpdateGmm1DoneReport(report, "marker", idx, 1U, entry.marker);
            UpdateGmm1DoneReport(report, "segmentPolicy", idx, kHostGmm1SegmentPolicyReferenceEpilogue,
                                 entry.segmentPolicy);
        }
        segment_row_base += segment_rows;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

size_t PackedWeightInt8OffsetShape(uint32_t k_rows, uint32_t n_cols, uint32_t expert, uint32_t k_idx, uint32_t n_idx)
{
    constexpr uint32_t kC0K = 16U;
    constexpr uint32_t kC0N = 32U;
    const uint32_t k_align = static_cast<uint32_t>(AlignUpU64(k_rows, kC0K));
    const uint32_t n_align = static_cast<uint32_t>(AlignUpU64(n_cols, kC0N));
    const size_t expert_stride = static_cast<size_t>(k_align) * n_align;
    return static_cast<size_t>(expert) * expert_stride + static_cast<size_t>(n_idx / kC0N) * k_align * kC0N +
           static_cast<size_t>(k_idx) * kC0N + n_idx % kC0N;
}

size_t PackedWeightInt8Offset(const CaseConfig &cfg, uint32_t expert, uint32_t k_idx, uint32_t n_idx)
{
    return PackedWeightInt8OffsetShape(cfg.k, cfg.n, expert, k_idx, n_idx);
}

float PackedScaleFp32At(const std::vector<uint8_t> &scale, uint32_t n, uint32_t expert, uint32_t col)
{
    const size_t offset = (static_cast<size_t>(expert) * n + col) * sizeof(uint64_t);
    float value = 0.0f;
    std::memcpy(&value, scale.data() + offset, sizeof(value));
    return value;
}

void UpdateGmm1OutputReport(Gmm1OutputReport &report, uint32_t group, uint32_t row, uint32_t col, float expected,
                            float actual)
{
    const bool invalid = std::isnan(expected) || std::isinf(expected) || std::isnan(actual) || std::isinf(actual);
    const double abs_err =
        invalid ? std::numeric_limits<double>::infinity() : std::fabs(static_cast<double>(actual) - expected);
    const double tolerance = 2.0e-2 + 3.0e-3 * std::fabs(static_cast<double>(expected));
    const double rel_denom = std::max(std::fabs(static_cast<double>(expected)), 1.0e-7);
    const double rel_err = invalid ? std::numeric_limits<double>::infinity() : abs_err / rel_denom;
    report.max_abs_err = std::max(report.max_abs_err, abs_err);
    report.max_rel_err = std::max(report.max_rel_err, rel_err);
    ++report.checked_count;
    if (!invalid && abs_err <= tolerance) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_group = group;
        report.first_row = row;
        report.first_col = col;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

void UpdateGmm2OutputReport(Gmm2OutputReport &report, uint32_t group, uint32_t row, uint32_t col, int32_t acc,
                            float expected_f32, uint16_t expected_half, uint16_t actual_half)
{
    const float expected_fp16 = Fp16ToFloat(expected_half);
    const float actual_f32 = Fp16ToFloat(actual_half);
    const bool invalid = std::isnan(expected_f32) || std::isinf(expected_f32) || std::isnan(expected_fp16) ||
                         std::isinf(expected_fp16) || std::isnan(actual_f32) || std::isinf(actual_f32);
    const double abs_err =
        invalid ? std::numeric_limits<double>::infinity() : std::fabs(static_cast<double>(actual_f32) - expected_f32);
    const double rel_denom = std::max(std::fabs(static_cast<double>(expected_f32)), 1.0e-7);
    const double rel_err = invalid ? std::numeric_limits<double>::infinity() : abs_err / rel_denom;
    const uint32_t ulp_err = HalfUlpDistance(expected_half, actual_half);
    report.max_abs_err = std::max(report.max_abs_err, abs_err);
    report.max_rel_err = std::max(report.max_rel_err, rel_err);
    report.max_ulp_err = std::max(report.max_ulp_err, ulp_err);
    ++report.checked_count;
    if (!invalid && expected_half == actual_half) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_group = group;
        report.first_row = row;
        report.first_col = col;
        report.expected_acc = acc;
        report.expected_half = expected_half;
        report.actual_half = actual_half;
        report.expected_f32 = expected_f32;
        report.expected_fp16 = expected_fp16;
        report.actual_f32 = actual_f32;
    }
    ++report.mismatch_count;
}

Gmm1OutputReport CheckGmm1Output(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                 const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                 const std::vector<uint8_t> &expert_idx, const std::vector<uint8_t> &weight1,
                                 const std::vector<uint8_t> &scale1, const std::string &case_dir)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &dispatch = build.tiling.dispatchTiling;
    const auto &gmm1 = build.tiling.gmm1Tiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const uint32_t local_begin = rank * expert_per_rank;
    constexpr uint32_t kC0K = 16U;
    constexpr uint32_t kC0N = 32U;
    const size_t packed_weight_bytes =
        static_cast<size_t>(expert_per_rank) * AlignUpU64(cfg.k, kC0K) * AlignUpU64(cfg.n, kC0N);
    const size_t packed_scale_bytes = static_cast<size_t>(expert_per_rank) * cfg.n * sizeof(uint64_t);
    if (weight1.size() < packed_weight_bytes) {
        throw std::runtime_error("weight1 file is too small for gmm1 output check");
    }
    if (scale1.size() < packed_scale_bytes) {
        throw std::runtime_error("scale1 file is too small for gmm1 output check");
    }

    const size_t gm_a_bytes = static_cast<size_t>(cfg.max_output_size) * cfg.k * sizeof(int8_t);
    const size_t gm_c_elems = static_cast<size_t>(cfg.max_output_size) * cfg.n;
    const size_t gm_c_bytes = gm_c_elems * sizeof(uint16_t);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    std::vector<int8_t> gm_a(gm_a_bytes, 0);
    std::vector<uint16_t> gm_c(gm_c_elems, 0);
    if (!gm_a.empty() && aclrtMemcpy(gm_a.data(), gm_a_bytes, workspace_base + dispatch.gmAOffset, gm_a_bytes,
                                     ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host gmm1 gmA copy failed");
    }
    if (!gm_c.empty() && aclrtMemcpy(gm_c.data(), gm_c_bytes, workspace_base + gmm1.gmCOffset, gm_c_bytes,
                                     ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host gmm1 gmC copy failed");
    }

    const bool dump_gmm1 = ParseEnvInt("DISPATCH_FFN_COMBINE_V8_DUMP_GMM1", 0) != 0;
    std::vector<uint16_t> expected_dump;
    if (dump_gmm1) {
        expected_dump.assign(gm_c_elems, 0);
    }

    const int8_t *weight_data = reinterpret_cast<const int8_t *>(weight1.data());
    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);

    Gmm1OutputReport report;
    uint32_t group_base = 0;
    for (uint32_t group = 0; group < expert_per_rank; ++group) {
        uint32_t current_m_raw = 0;
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            const size_t token_idx = static_cast<size_t>(src_rank) * expert_num_aligned + local_begin + group;
            current_m_raw += static_cast<uint32_t>(expected_tokens[token_idx]);
        }
        const uint32_t remaining = group_base >= cfg.max_output_size ? 0U : cfg.max_output_size - group_base;
        const uint32_t current_m = current_m_raw > remaining ? remaining : current_m_raw;
        for (uint32_t row = 0; row < current_m; ++row) {
            const uint32_t dst_row = group_base + row;
            for (uint32_t col = 0; col < cfg.n; ++col) {
                int32_t acc = 0;
                const size_t gm_a_row = static_cast<size_t>(dst_row) * cfg.k;
                for (uint32_t k_idx = 0; k_idx < cfg.k; ++k_idx) {
                    const int32_t a = static_cast<int32_t>(gm_a[gm_a_row + k_idx]);
                    const int32_t b = static_cast<int32_t>(weight_data[PackedWeightInt8Offset(cfg, group, k_idx, col)]);
                    acc += a * b;
                }
                const float scale = PackedScaleFp32At(scale1, cfg.n, group, col);
                const float expected = static_cast<float>(acc) * scale;
                const float actual = Fp16ToFloat(gm_c[static_cast<size_t>(dst_row) * cfg.n + col]);
                UpdateGmm1OutputReport(report, group, row, col, expected, actual);
                if (dump_gmm1) {
                    expected_dump[static_cast<size_t>(dst_row) * cfg.n + col] = FloatToHalfTrunc(expected);
                }
            }
        }
        group_base += current_m;
    }
    if (dump_gmm1) {
        WriteBinaryFile(case_dir + "/gmm1_expected_gmC_rank" + std::to_string(runtime.hccl.rank_id) + ".bin",
                        expected_dump.data(), expected_dump.size() * sizeof(uint16_t));
        WriteBinaryFile(case_dir + "/gmm1_actual_gmC_rank" + std::to_string(runtime.hccl.rank_id) + ".bin", gm_c.data(),
                        gm_c.size() * sizeof(uint16_t));
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

Gmm2OutputReport CheckGmm2Output(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                 const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                 const std::vector<uint8_t> &expert_idx, const std::vector<uint8_t> &weight2,
                                 const std::vector<uint8_t> &scale2)
{
    const auto &swiglu = build.tiling.swigluTiling;
    const auto &gmm2 = build.tiling.gmm2Tiling;
    const auto &combine = build.tiling.combineTiling;
    const uint32_t input_k = cfg.n / 2U;
    const uint32_t output_n = cfg.k;
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    if (combine.gmm2CombineCvMode == kHostGmm2CombineCvModeDirect) {
        Gmm2OutputReport report;
        report.skipped = true;
        report.skip_reason = "cv-direct-no-gm-gmm2-output";
        report.segment_count = SwigluSegmentNum(expert_per_rank);
        report.group_count = expert_per_rank;
        report.col_count = output_n;
        return report;
    }
    constexpr uint32_t kC0K = 16U;
    constexpr uint32_t kC0N = 32U;
    const size_t packed_weight_bytes =
        static_cast<size_t>(expert_per_rank) * AlignUpU64(input_k, kC0K) * AlignUpU64(output_n, kC0N);
    const size_t packed_scale_bytes = static_cast<size_t>(expert_per_rank) * output_n * sizeof(uint64_t);
    if (weight2.size() < packed_weight_bytes) {
        throw std::runtime_error("weight2 file is too small for gmm2 output check");
    }
    if (scale2.size() < packed_scale_bytes) {
        throw std::runtime_error("scale2 file is too small for gmm2 output check");
    }

    const size_t gm_permuted_elems = static_cast<size_t>(cfg.max_output_size) * input_k;
    const size_t gm_permuted_bytes = gm_permuted_elems * sizeof(int8_t);
    const size_t gmm2_output_elems = static_cast<size_t>(cfg.max_output_size) * output_n;
    const size_t gmm2_output_bytes = gmm2_output_elems * sizeof(uint16_t);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    std::vector<int8_t> gm_permuted(gm_permuted_elems, 0);
    std::vector<uint16_t> gmm2_output(gmm2_output_elems, 0);
    if (!gm_permuted.empty() &&
        aclrtMemcpy(gm_permuted.data(), gm_permuted_bytes, workspace_base + swiglu.gmPermutedTokenOffset,
                    gm_permuted_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host gmm2 gmPermutedToken copy failed");
    }
    if (!gmm2_output.empty() &&
        aclrtMemcpy(gmm2_output.data(), gmm2_output_bytes, workspace_base + gmm2.gmm2OutputOffset, gmm2_output_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host gmm2 gmm2Output copy failed");
    }

    const int8_t *weight_data = reinterpret_cast<const int8_t *>(weight2.data());
    const std::vector<uint32_t> current_m = BuildExpectedGmm1CurrentM(cfg, build, runtime, expert_idx);

    Gmm2OutputReport report;
    report.segment_count = SwigluSegmentNum(expert_per_rank);
    report.group_count = expert_per_rank;
    report.col_count = output_n;
    uint32_t group_base = 0;
    for (uint32_t group = 0; group < expert_per_rank; ++group) {
        const uint32_t cur_m = current_m[group];
        for (uint32_t row = 0; row < cur_m; ++row) {
            const uint32_t dst_row = group_base + row;
            const size_t gm_permuted_row = static_cast<size_t>(dst_row) * input_k;
            for (uint32_t col = 0; col < output_n; ++col) {
                int32_t acc = 0;
                for (uint32_t k_idx = 0; k_idx < input_k; ++k_idx) {
                    const int32_t a = static_cast<int32_t>(gm_permuted[gm_permuted_row + k_idx]);
                    const int32_t b = static_cast<int32_t>(
                        weight_data[PackedWeightInt8OffsetShape(input_k, output_n, group, k_idx, col)]);
                    acc += a * b;
                }
                const float scale = PackedScaleFp32At(scale2, output_n, group, col);
                const float expected_f32 = static_cast<float>(acc) * scale;
                const uint16_t expected_half = FloatToHalfRoundToNearestEven(expected_f32);
                const uint16_t actual_half = gmm2_output[static_cast<size_t>(dst_row) * output_n + col];
                UpdateGmm2OutputReport(report, group, row, col, acc, expected_f32, expected_half, actual_half);
            }
        }
        group_base += cur_m;
    }
    report.row_count = group_base;
    report.expected_checked_count = report.row_count * output_n;
    report.pass = report.mismatch_count == 0 && report.checked_count == report.expected_checked_count;
    return report;
}

void UpdateGmm1DetailCumsumReport(Gmm1DetailReport &report, size_t idx, int32_t expected, int32_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.first_table.empty()) {
        report.first_table = "cumsumMM";
        report.first_idx = idx;
        report.expected_i32 = expected;
        report.actual_i32 = actual;
    }
    ++report.cumsum_mismatch_count;
}

void UpdateGmm1DetailValueReport(Gmm1DetailReport &report, uint32_t group, uint32_t row, uint32_t col, int32_t acc,
                                 float expected_f32, uint16_t expected_half, uint16_t actual_half)
{
    const float expected_fp16 = Fp16ToFloat(expected_half);
    const float actual_f32 = Fp16ToFloat(actual_half);
    const bool invalid = std::isnan(expected_f32) || std::isinf(expected_f32) || std::isnan(expected_fp16) ||
                         std::isinf(expected_fp16) || std::isnan(actual_f32) || std::isinf(actual_f32);
    const double abs_err =
        invalid ? std::numeric_limits<double>::infinity() : std::fabs(static_cast<double>(actual_f32) - expected_f32);
    const double rel_denom = std::max(std::fabs(static_cast<double>(expected_f32)), 1.0e-7);
    const double rel_err = invalid ? std::numeric_limits<double>::infinity() : abs_err / rel_denom;
    const uint32_t ulp_err = HalfUlpDistance(expected_half, actual_half);
    report.max_abs_err = std::max(report.max_abs_err, abs_err);
    report.max_rel_err = std::max(report.max_rel_err, rel_err);
    report.max_ulp_err = std::max(report.max_ulp_err, ulp_err);
    ++report.checked_count;
    if (!invalid && expected_half == actual_half) {
        return;
    }
    if (report.first_table.empty()) {
        report.first_table = "gmC";
        report.first_group = group;
        report.first_row = row;
        report.first_col = col;
        report.expected_acc = acc;
        report.expected_half = expected_half;
        report.actual_half = actual_half;
        report.expected_f32 = expected_f32;
        report.expected_fp16 = expected_fp16;
        report.actual_f32 = actual_f32;
    }
    ++report.gm_c_mismatch_count;
}

Gmm1DetailReport CheckGmm1Detail(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                 const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                 const std::vector<uint8_t> &expert_idx, const std::vector<uint8_t> &weight1,
                                 const std::vector<uint8_t> &scale1)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &dispatch = build.tiling.dispatchTiling;
    const auto &gmm1 = build.tiling.gmm1Tiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const uint32_t local_begin = rank * expert_per_rank;
    constexpr uint32_t kC0K = 16U;
    constexpr uint32_t kC0N = 32U;
    const size_t packed_weight_bytes =
        static_cast<size_t>(expert_per_rank) * AlignUpU64(cfg.k, kC0K) * AlignUpU64(cfg.n, kC0N);
    const size_t packed_scale_bytes = static_cast<size_t>(expert_per_rank) * cfg.n * sizeof(uint64_t);
    if (weight1.size() < packed_weight_bytes) {
        throw std::runtime_error("weight1 file is too small for gmm1 detail check");
    }
    if (scale1.size() < packed_scale_bytes) {
        throw std::runtime_error("scale1 file is too small for gmm1 detail check");
    }

    const size_t gm_a_bytes = static_cast<size_t>(cfg.max_output_size) * cfg.k * sizeof(int8_t);
    const size_t gm_c_elems = static_cast<size_t>(cfg.max_output_size) * cfg.n;
    const size_t gm_c_bytes = gm_c_elems * sizeof(uint16_t);
    const size_t cumsum_elems = static_cast<size_t>(rank_size) * expert_per_rank;
    const size_t cumsum_bytes = cumsum_elems * sizeof(int32_t);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    std::vector<int8_t> gm_a(gm_a_bytes, 0);
    std::vector<uint16_t> gm_c(gm_c_elems, 0);
    std::vector<int32_t> actual_cumsum(cumsum_elems, 0);
    if (!gm_a.empty() && aclrtMemcpy(gm_a.data(), gm_a_bytes, workspace_base + dispatch.gmAOffset, gm_a_bytes,
                                     ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host gmm1 detail gmA copy failed");
    }
    if (!gm_c.empty() && aclrtMemcpy(gm_c.data(), gm_c_bytes, workspace_base + gmm1.gmCOffset, gm_c_bytes,
                                     ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host gmm1 detail gmC copy failed");
    }
    if (!actual_cumsum.empty() && aclrtMemcpy(actual_cumsum.data(), cumsum_bytes, workspace_base + front.cumsumMMOffset,
                                              cumsum_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host gmm1 detail cumsumMM copy failed");
    }

    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);
    std::vector<int32_t> expected_cumsum(cumsum_elems, 0);
    for (uint32_t local_expert = 0; local_expert < expert_per_rank; ++local_expert) {
        int32_t running_sum = 0;
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            const size_t token_idx = static_cast<size_t>(src_rank) * expert_num_aligned + local_begin + local_expert;
            running_sum += expected_tokens[token_idx];
            expected_cumsum[static_cast<size_t>(src_rank) * expert_per_rank + local_expert] = running_sum;
        }
    }

    Gmm1DetailReport report;
    report.group_count = expert_per_rank;
    report.col_count = cfg.n;
    for (size_t idx = 0; idx < cumsum_elems; ++idx) {
        UpdateGmm1DetailCumsumReport(report, idx, expected_cumsum[idx], actual_cumsum[idx]);
    }

    const int8_t *weight_data = reinterpret_cast<const int8_t *>(weight1.data());
    uint32_t group_base = 0;
    for (uint32_t group = 0; group < expert_per_rank; ++group) {
        const uint32_t current_m_raw = static_cast<uint32_t>(
            std::max<int32_t>(expected_cumsum[static_cast<size_t>(rank_size - 1U) * expert_per_rank + group], 0));
        const uint32_t remaining = group_base >= cfg.max_output_size ? 0U : cfg.max_output_size - group_base;
        const uint32_t current_m = current_m_raw > remaining ? remaining : current_m_raw;
        for (uint32_t row = 0; row < current_m; ++row) {
            const uint32_t dst_row = group_base + row;
            for (uint32_t col = 0; col < cfg.n; ++col) {
                int32_t acc = 0;
                const size_t gm_a_row = static_cast<size_t>(dst_row) * cfg.k;
                for (uint32_t k_idx = 0; k_idx < cfg.k; ++k_idx) {
                    const int32_t a = static_cast<int32_t>(gm_a[gm_a_row + k_idx]);
                    const int32_t b = static_cast<int32_t>(weight_data[PackedWeightInt8Offset(cfg, group, k_idx, col)]);
                    acc += a * b;
                }
                const float scale = PackedScaleFp32At(scale1, cfg.n, group, col);
                const float expected_f32 = static_cast<float>(acc) * scale;
                const uint16_t expected_half = FloatToHalfRoundToNearestEven(expected_f32);
                const uint16_t actual_half = gm_c[static_cast<size_t>(dst_row) * cfg.n + col];
                UpdateGmm1DetailValueReport(report, group, row, col, acc, expected_f32, expected_half, actual_half);
            }
        }
        group_base += current_m;
    }
    report.row_count = group_base;
    report.pass = report.cumsum_mismatch_count == 0 && report.gm_c_mismatch_count == 0 &&
                  report.checked_count == static_cast<size_t>(report.row_count) * cfg.n;
    return report;
}

void UpdateDispatchMetadataReport(DispatchMetadataReport &report, const std::string &field, size_t idx,
                                  uint64_t expected, uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

DispatchMetadataReport CheckDispatchMetadata(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                             const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                             const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &dispatch = build.tiling.dispatchTiling;
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const size_t metadata_elems = static_cast<size_t>(expert_per_rank) * rank_size;
    const size_t metadata_bytes = metadata_elems * sizeof(DispatchFFNCombineDispatchMetadataDebug);
    const uint64_t metadata_offset = dispatch.dispatchDebugOffset + sizeof(DispatchFFNCombineDispatchLayoutDebug);

    if (dispatch.dispatchDebugBytes < sizeof(DispatchFFNCombineDispatchLayoutDebug) + metadata_bytes) {
        throw std::runtime_error("dispatch debug area is too small for metadata table");
    }

    std::vector<DispatchFFNCombineDispatchMetadataDebug> actual(metadata_elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual.data(), metadata_bytes, workspace_base + metadata_offset, metadata_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host dispatch metadata copy failed");
    }

    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);

    std::vector<DispatchFFNCombineDispatchMetadataDebug> expected(metadata_elems);
    uint32_t group_base = 0;
    const uint32_t local_begin = rank * expert_per_rank;
    for (uint32_t group = 0; group < expert_per_rank; ++group) {
        uint32_t current_m = 0;
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            const size_t token_idx = static_cast<size_t>(src_rank) * expert_num_aligned + local_begin + group;
            current_m += static_cast<uint32_t>(expected_tokens[token_idx]);
        }

        uint32_t cumsum_before_src = 0;
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            const size_t meta_idx = static_cast<size_t>(group) * rank_size + src_rank;
            const size_t token_idx = static_cast<size_t>(src_rank) * expert_num_aligned + local_begin + group;
            const uint32_t raw_rows = static_cast<uint32_t>(expected_tokens[token_idx]);

            uint32_t src_row_base = 0;
            for (uint32_t expert = 0; expert < local_begin + group; ++expert) {
                src_row_base +=
                    static_cast<uint32_t>(expected_tokens[static_cast<size_t>(src_rank) * expert_num_aligned + expert]);
            }

            const uint32_t dst_row_base = group_base + cumsum_before_src;
            uint32_t rows = raw_rows;
            if (dst_row_base >= cfg.max_output_size) {
                rows = 0;
            } else if (dst_row_base + rows > cfg.max_output_size) {
                rows = cfg.max_output_size - dst_row_base;
            }

            expected[meta_idx].groupIdx = group;
            expected[meta_idx].srcRank = src_rank;
            expected[meta_idx].groupBase = group_base;
            expected[meta_idx].currentM = current_m;
            expected[meta_idx].rawRows = raw_rows;
            expected[meta_idx].rows = rows;
            expected[meta_idx].srcRowBase = src_row_base;
            expected[meta_idx].dstRowBase = dst_row_base;
            cumsum_before_src += raw_rows;
        }
        group_base += current_m;
    }

    DispatchMetadataReport report;
    for (size_t idx = 0; idx < metadata_elems; ++idx) {
        UpdateDispatchMetadataReport(report, "groupIdx", idx, expected[idx].groupIdx, actual[idx].groupIdx);
        UpdateDispatchMetadataReport(report, "srcRank", idx, expected[idx].srcRank, actual[idx].srcRank);
        UpdateDispatchMetadataReport(report, "groupBase", idx, expected[idx].groupBase, actual[idx].groupBase);
        UpdateDispatchMetadataReport(report, "currentM", idx, expected[idx].currentM, actual[idx].currentM);
        UpdateDispatchMetadataReport(report, "rawRows", idx, expected[idx].rawRows, actual[idx].rawRows);
        UpdateDispatchMetadataReport(report, "rows", idx, expected[idx].rows, actual[idx].rows);
        UpdateDispatchMetadataReport(report, "srcRowBase", idx, expected[idx].srcRowBase, actual[idx].srcRowBase);
        UpdateDispatchMetadataReport(report, "dstRowBase", idx, expected[idx].dstRowBase, actual[idx].dstRowBase);
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

void UpdateDispatchCopyChunkMetaReport(DispatchCopyChunkReport &report, const std::string &field, uint64_t expected,
                                       uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.first_table.empty()) {
        report.first_table = "metadata";
        report.first_field = field;
        report.expected_u64 = expected;
        report.actual_u64 = actual;
    }
    ++report.metadata_mismatch_count;
}

void UpdateDispatchCopyChunkIntReport(DispatchCopyChunkReport &report, const std::string &table, size_t idx,
                                      int32_t expected, int32_t actual, size_t &mismatch_count)
{
    if (expected == actual) {
        return;
    }
    if (report.first_table.empty()) {
        report.first_table = table;
        report.first_idx = idx;
        report.expected_i32 = expected;
        report.actual_i32 = actual;
    }
    ++mismatch_count;
}

void UpdateDispatchCopyChunkScaleReport(DispatchCopyChunkReport &report, size_t idx, float expected, float actual)
{
    const float tolerance = 1.0e-5f + 1.0e-3f * std::fabs(expected);
    if (std::isfinite(expected) && std::isfinite(actual) && std::fabs(actual - expected) <= tolerance) {
        return;
    }
    if (report.first_table.empty()) {
        report.first_table = "ptrPerTokenScale";
        report.first_idx = idx;
        report.expected_f32 = expected;
        report.actual_f32 = actual;
    }
    ++report.scale_mismatch_count;
}

DispatchFFNCombineDispatchCopyChunkDebug BuildExpectedDispatchCopyChunkRecord(
    const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build, const StandaloneRankRuntime &runtime,
    const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &dispatch = build.tiling.dispatchTiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const uint32_t local_begin = rank * expert_per_rank;
    const uint32_t packed_stride = cfg.k + UB_ALIGN;
    uint32_t tile_rows = static_cast<uint32_t>(dispatch.dispatchTileBytes / packed_stride);
    tile_rows = std::max(1U, tile_rows);

    DispatchFFNCombineDispatchCopyChunkDebug expected;
    expected.magic = kHostDispatchCopyChunkDebugMagic;
    expected.coreIdx = 0U;
    expected.coreNum = cfg.aiv_num;
    expected.tileRows = tile_rows;
    expected.packedStride = packed_stride;
    expected.marker = 1U;

    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);

    for (uint32_t pass = 0; pass < 2U; ++pass) {
        uint32_t group_base = 0;
        for (uint32_t group = 0; group < expert_per_rank; ++group) {
            uint32_t current_m = 0;
            for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
                const size_t token_idx = static_cast<size_t>(src_rank) * expert_num_aligned + local_begin + group;
                current_m += static_cast<uint32_t>(expected_tokens[token_idx]);
            }

            uint32_t cumsum_before_src = 0;
            for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
                if (pass == 0U && src_rank == rank) {
                    const size_t token_idx = static_cast<size_t>(src_rank) * expert_num_aligned + local_begin + group;
                    cumsum_before_src += static_cast<uint32_t>(expected_tokens[token_idx]);
                    continue;
                }
                const size_t token_idx = static_cast<size_t>(src_rank) * expert_num_aligned + local_begin + group;
                const uint32_t raw_rows = static_cast<uint32_t>(expected_tokens[token_idx]);
                const uint32_t dst_row_base = group_base + cumsum_before_src;
                uint32_t rows = raw_rows;
                if (dst_row_base >= cfg.max_output_size) {
                    rows = 0U;
                } else if (dst_row_base + rows > cfg.max_output_size) {
                    rows = cfg.max_output_size - dst_row_base;
                }
                if (rows != 0U) {
                    uint32_t src_row_base = 0;
                    for (uint32_t expert = 0; expert < local_begin + group; ++expert) {
                        src_row_base += static_cast<uint32_t>(
                            expected_tokens[static_cast<size_t>(src_rank) * expert_num_aligned + expert]);
                    }
                    expected.groupIdx = group;
                    expected.srcRank = src_rank;
                    expected.groupBase = group_base;
                    expected.currentM = current_m;
                    expected.rawRows = raw_rows;
                    expected.rows = rows;
                    expected.chunkRows = std::min(rows, tile_rows);
                    expected.srcRowBase = src_row_base;
                    expected.dstRowBase = dst_row_base;
                    expected.remoteOffsetBytes = static_cast<uint64_t>(src_row_base) * packed_stride;
                    expected.gmAOffsetBytes = static_cast<uint64_t>(dst_row_base) * cfg.k;
                    expected.scaleOffsetBytes = static_cast<uint64_t>(dst_row_base) * sizeof(float);
                    return expected;
                }
                cumsum_before_src += raw_rows;
            }
            group_base += current_m;
        }
    }
    return expected;
}

void CheckDispatchCopyChunkZeroProbe(DispatchCopyChunkReport &report, const CaseConfig &cfg,
                                     const DispatchFFNCombineBuildResult &build, const DeviceBuffer &workspace_dev,
                                     uint32_t row)
{
    const auto &dispatch = build.tiling.dispatchTiling;
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    const size_t row_bytes = static_cast<size_t>(cfg.k);
    std::vector<int8_t> gm_a(row_bytes, 0);
    float scale = 0.0f;
    if (aclrtMemcpy(gm_a.data(), row_bytes, workspace_base + dispatch.gmAOffset + static_cast<uint64_t>(row) * cfg.k,
                    row_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(&scale, sizeof(scale),
                    workspace_base + dispatch.perTokenScaleOffset + static_cast<uint64_t>(row) * sizeof(float),
                    sizeof(scale), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host dispatch copy zero probe failed");
    }
    ++report.zero_probe_count;
    for (size_t col = 0; col < gm_a.size(); ++col) {
        UpdateDispatchCopyChunkIntReport(report, "zeroGmA", static_cast<size_t>(row) * cfg.k + col, 0,
                                         static_cast<int32_t>(gm_a[col]), report.zero_mismatch_count);
        if (report.zero_mismatch_count != 0U) {
            break;
        }
    }
    if (scale != 0.0f) {
        if (report.first_table.empty()) {
            report.first_table = "zeroScale";
            report.first_idx = row;
            report.expected_f32 = 0.0f;
            report.actual_f32 = scale;
        }
        ++report.zero_mismatch_count;
    }
}

DispatchCopyChunkReport CheckDispatchCopyChunk(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                               const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                               const std::vector<uint8_t> &x_bytes,
                                               const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &dispatch = build.tiling.dispatchTiling;
    const uint32_t expert_num = front.expertNum;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const size_t metadata_bytes = static_cast<size_t>(cfg.expert_per_rank) * runtime.hccl.world_size *
                                  sizeof(DispatchFFNCombineDispatchMetadataDebug);
    const uint64_t copy_debug_offset =
        dispatch.dispatchDebugOffset + sizeof(DispatchFFNCombineDispatchLayoutDebug) + metadata_bytes;
    if (dispatch.dispatchDebugBytes < sizeof(DispatchFFNCombineDispatchLayoutDebug) + metadata_bytes +
                                          sizeof(DispatchFFNCombineDispatchCopyChunkDebug)) {
        throw std::runtime_error("dispatch debug area is too small for copy chunk debug");
    }

    DispatchFFNCombineDispatchCopyChunkDebug actual;
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(&actual, sizeof(actual), workspace_base + copy_debug_offset, sizeof(actual),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host dispatch copy chunk debug failed");
    }

    const DispatchFFNCombineDispatchCopyChunkDebug expected =
        BuildExpectedDispatchCopyChunkRecord(cfg, build, runtime, expert_idx);

    DispatchCopyChunkReport report;
    report.group = actual.groupIdx;
    report.src_rank = actual.srcRank;
    report.rows = actual.rows;
    report.chunk_rows = actual.chunkRows;
    report.src_row_base = actual.srcRowBase;
    report.dst_row_base = actual.dstRowBase;

    UpdateDispatchCopyChunkMetaReport(report, "magic", expected.magic, actual.magic);
    UpdateDispatchCopyChunkMetaReport(report, "marker", expected.marker, actual.marker);
    UpdateDispatchCopyChunkMetaReport(report, "groupIdx", expected.groupIdx, actual.groupIdx);
    UpdateDispatchCopyChunkMetaReport(report, "srcRank", expected.srcRank, actual.srcRank);
    UpdateDispatchCopyChunkMetaReport(report, "coreIdx", expected.coreIdx, actual.coreIdx);
    UpdateDispatchCopyChunkMetaReport(report, "coreNum", expected.coreNum, actual.coreNum);
    UpdateDispatchCopyChunkMetaReport(report, "groupBase", expected.groupBase, actual.groupBase);
    UpdateDispatchCopyChunkMetaReport(report, "currentM", expected.currentM, actual.currentM);
    UpdateDispatchCopyChunkMetaReport(report, "rawRows", expected.rawRows, actual.rawRows);
    UpdateDispatchCopyChunkMetaReport(report, "rows", expected.rows, actual.rows);
    UpdateDispatchCopyChunkMetaReport(report, "chunkRows", expected.chunkRows, actual.chunkRows);
    UpdateDispatchCopyChunkMetaReport(report, "tileRows", expected.tileRows, actual.tileRows);
    UpdateDispatchCopyChunkMetaReport(report, "srcRowBase", expected.srcRowBase, actual.srcRowBase);
    UpdateDispatchCopyChunkMetaReport(report, "dstRowBase", expected.dstRowBase, actual.dstRowBase);
    UpdateDispatchCopyChunkMetaReport(report, "packedStride", expected.packedStride, actual.packedStride);
    UpdateDispatchCopyChunkMetaReport(report, "remoteOffsetBytes", expected.remoteOffsetBytes,
                                      actual.remoteOffsetBytes);
    UpdateDispatchCopyChunkMetaReport(report, "gmAOffsetBytes", expected.gmAOffsetBytes, actual.gmAOffsetBytes);
    UpdateDispatchCopyChunkMetaReport(report, "scaleOffsetBytes", expected.scaleOffsetBytes, actual.scaleOffsetBytes);

    if (expected.chunkRows != 0U) {
        std::vector<int8_t> actual_gm_a(static_cast<size_t>(expected.chunkRows) * cfg.k, 0);
        std::vector<float> actual_scale(expected.chunkRows, 0.0f);
        if (aclrtMemcpy(actual_gm_a.data(), actual_gm_a.size(),
                        workspace_base + dispatch.gmAOffset + static_cast<uint64_t>(expected.dstRowBase) * cfg.k,
                        actual_gm_a.size(), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
            aclrtMemcpy(actual_scale.data(), actual_scale.size() * sizeof(float),
                        workspace_base + dispatch.perTokenScaleOffset +
                            static_cast<uint64_t>(expected.dstRowBase) * sizeof(float),
                        actual_scale.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            throw std::runtime_error("device->host dispatch copy chunk output failed");
        }

        const std::vector<uint8_t> all_x_bytes =
            GatherAllRankBytes(runtime.hccl.rank_id, runtime.hccl.world_size, x_bytes, "x");
        const std::vector<uint8_t> all_expert_idx_bytes =
            GatherAllRankBytes(runtime.hccl.rank_id, runtime.hccl.world_size, expert_idx, "expert_idx");
        const std::vector<uint16_t> all_x_bf16 = BytesToU16(all_x_bytes);
        const size_t route_elems = static_cast<size_t>(cfg.m) * cfg.topk;
        const size_t src_expert_offset = static_cast<size_t>(expected.srcRank) * route_elems * sizeof(int32_t);
        if (src_expert_offset + route_elems * sizeof(int32_t) > all_expert_idx_bytes.size()) {
            throw std::runtime_error("dispatch copy chunk source expert_idx slice is out of range");
        }
        std::vector<uint8_t> src_expert_idx(route_elems * sizeof(int32_t), 0);
        std::copy_n(all_expert_idx_bytes.begin() + static_cast<std::ptrdiff_t>(src_expert_offset),
                    src_expert_idx.size(), src_expert_idx.begin());
        const ExpectedFrontRoute src_route =
            BuildExpectedFrontRoute(src_expert_idx, route_elems, expert_num, expert_num_aligned);

        for (uint32_t row = 0; row < expected.chunkRows; ++row) {
            const size_t sorted_row = static_cast<size_t>(expected.srcRowBase) + row;
            if (sorted_row >= src_route.dst_to_src_route.size() || src_route.dst_to_src_route[sorted_row] < 0) {
                throw std::runtime_error("dispatch copy chunk sorted row is invalid");
            }
            const uint32_t src_route_idx = static_cast<uint32_t>(src_route.dst_to_src_route[sorted_row]);
            const uint32_t src_token = src_route_idx / cfg.topk;
            const uint32_t global_token = expected.srcRank * cfg.m + src_token;
            float expected_scale = 0.0f;
            const std::vector<int8_t> expected_quant =
                QuantizeBf16RowLikeV3(all_x_bf16, global_token, cfg.k, expected_scale);
            for (uint32_t col = 0; col < cfg.k; ++col) {
                const size_t idx = static_cast<size_t>(row) * cfg.k + col;
                UpdateDispatchCopyChunkIntReport(report, "gmA", idx, static_cast<int32_t>(expected_quant[col]),
                                                 static_cast<int32_t>(actual_gm_a[idx]), report.gm_a_mismatch_count);
            }
            UpdateDispatchCopyChunkScaleReport(report, row, expected_scale, actual_scale[row]);
            report.checked_bytes += cfg.k;
            ++report.checked_scales;
        }

        std::array<uint32_t, 4> probe_rows = {
            0U,
            expected.dstRowBase > 0U ? expected.dstRowBase - 1U : expected.dstRowBase + expected.chunkRows,
            expected.dstRowBase + expected.chunkRows,
            cfg.max_output_size == 0U ? 0U : cfg.max_output_size - 1U,
        };
        for (uint32_t row : probe_rows) {
            if (row >= cfg.max_output_size ||
                (row >= expected.dstRowBase && row < expected.dstRowBase + expected.chunkRows)) {
                continue;
            }
            CheckDispatchCopyChunkZeroProbe(report, cfg, build, workspace_dev, row);
        }
    }

    report.pass = report.metadata_mismatch_count == 0 && report.gm_a_mismatch_count == 0 &&
                  report.scale_mismatch_count == 0 && report.zero_mismatch_count == 0;
    return report;
}

void UpdateDispatchTaskReport(DispatchTaskReport &report, const std::string &field, size_t idx, uint64_t expected,
                              uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

uint32_t CountHostDispatchRowBlocks(const std::vector<uint32_t> &rows_per_src, uint32_t row_block_rows)
{
    if (row_block_rows == 0U) {
        return 0U;
    }

    uint32_t block_count = 0U;
    for (const uint32_t rows : rows_per_src) {
        if (rows != 0U) {
            block_count += CeilDivU32(rows, row_block_rows);
        }
    }
    return block_count;
}

uint32_t DecideHostDispatchRowBlockRows(const std::vector<uint32_t> &rows_per_src, uint32_t current_m,
                                        uint32_t packed_stride, uint32_t aiv_num)
{
    if (current_m == 0U) {
        return 1U;
    }

    const uint32_t min_block_rows = std::max(1U, CeilDivU32(kHostDispatchMinRowBlockBytes, packed_stride));
    uint32_t max_blocks_per_group = aiv_num * kHostDispatchRowBlockBlocksPerCore;
    max_blocks_per_group = std::max(1U, max_blocks_per_group);

    uint32_t non_empty_source_ranks = 0U;
    for (const uint32_t rows : rows_per_src) {
        if (rows != 0U) {
            ++non_empty_source_ranks;
        }
    }
    const uint32_t effective_max_blocks = std::max(max_blocks_per_group, non_empty_source_ranks);
    const uint32_t natural_blocks_by_min_bytes = CeilDivU32(current_m, min_block_rows);
    uint32_t target_blocks = std::min(natural_blocks_by_min_bytes, effective_max_blocks);
    target_blocks = std::max(1U, target_blocks);

    uint32_t row_block_rows = std::max(min_block_rows, CeilDivU32(current_m, target_blocks));
    while (CountHostDispatchRowBlocks(rows_per_src, row_block_rows) > effective_max_blocks &&
           row_block_rows < current_m) {
        row_block_rows = row_block_rows > current_m / 2U ? current_m : row_block_rows * 2U;
    }
    return std::max(1U, row_block_rows);
}

DispatchTaskReport CheckDispatchTaskSplit(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                          const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                          const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &dispatch = build.tiling.dispatchTiling;
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t aiv_num = cfg.aiv_num;
    const size_t metadata_bytes =
        static_cast<size_t>(expert_per_rank) * rank_size * sizeof(DispatchFFNCombineDispatchMetadataDebug);
    const size_t task_stats_elems = static_cast<size_t>(expert_per_rank) * aiv_num;
    const size_t task_stats_bytes = task_stats_elems * sizeof(DispatchFFNCombineDispatchTaskStatsDebug);
    const uint64_t task_stats_offset =
        dispatch.dispatchDebugOffset + sizeof(DispatchFFNCombineDispatchLayoutDebug) + metadata_bytes;

    if (dispatch.dispatchDebugBytes <
        sizeof(DispatchFFNCombineDispatchLayoutDebug) + metadata_bytes + task_stats_bytes) {
        throw std::runtime_error("dispatch debug area is too small for task stats table");
    }

    std::vector<DispatchFFNCombineDispatchTaskStatsDebug> actual(task_stats_elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual.data(), task_stats_bytes, workspace_base + task_stats_offset, task_stats_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host dispatch task stats copy failed");
    }
    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);

    std::vector<DispatchFFNCombineDispatchTaskStatsDebug> expected(task_stats_elems);
    uint32_t group_base = 0;
    const uint32_t local_begin = rank * expert_per_rank;
    const uint32_t packed_stride = cfg.k + UB_ALIGN;
    const uint32_t max_blocks_per_group = std::max(1U, aiv_num * kHostDispatchRowBlockBlocksPerCore);
    for (uint32_t group = 0; group < expert_per_rank; ++group) {
        std::vector<uint32_t> rows_per_src(rank_size, 0);
        std::vector<uint32_t> src_row_base_per_src(rank_size, 0);
        std::vector<uint32_t> dst_row_base_per_src(rank_size, 0);
        uint32_t current_m = 0;
        uint32_t cumsum_before_src = 0;
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            const size_t token_idx = static_cast<size_t>(src_rank) * expert_num_aligned + local_begin + group;
            const uint32_t raw_rows = static_cast<uint32_t>(expected_tokens[token_idx]);
            const uint32_t dst_row_base = group_base + cumsum_before_src;
            uint32_t rows = raw_rows;
            if (dst_row_base >= cfg.max_output_size) {
                rows = 0;
            } else if (dst_row_base + rows > cfg.max_output_size) {
                rows = cfg.max_output_size - dst_row_base;
            }
            rows_per_src[src_rank] = rows;
            uint32_t src_row_base = 0;
            for (uint32_t expert = 0; expert < local_begin + group; ++expert) {
                src_row_base +=
                    static_cast<uint32_t>(expected_tokens[static_cast<size_t>(src_rank) * expert_num_aligned + expert]);
            }
            src_row_base_per_src[src_rank] = src_row_base;
            dst_row_base_per_src[src_rank] = dst_row_base;
            current_m += raw_rows;
            cumsum_before_src += raw_rows;
        }

        const uint32_t row_block_rows = DecideHostDispatchRowBlockRows(rows_per_src, current_m, packed_stride, aiv_num);
        for (uint32_t core = 0; core < aiv_num; ++core) {
            const size_t idx = static_cast<size_t>(group) * aiv_num + core;
            expected[idx].groupIdx = group;
            expected[idx].coreIdx = core;
            expected[idx].rankSize = rank_size;
            expected[idx].groupBase = group_base;
            expected[idx].currentM = current_m;
            expected[idx].firstSrcRank = kHostDispatchInvalidTask;
            expected[idx].lastSrcRank = kHostDispatchInvalidTask;
            expected[idx].rowBlockRows = row_block_rows;
            expected[idx].maxBlocksPerGroup = max_blocks_per_group;

            uint32_t linear_block = 0U;
            for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
                const uint32_t rows = rows_per_src[src_rank];
                for (uint32_t block_row = 0; block_row < rows; block_row += row_block_rows) {
                    const uint32_t block_rows =
                        (rows - block_row > row_block_rows) ? row_block_rows : (rows - block_row);
                    if (linear_block % aiv_num == core) {
                        if (expected[idx].firstSrcRank == kHostDispatchInvalidTask) {
                            expected[idx].firstSrcRank = src_rank;
                            expected[idx].firstRows = block_rows;
                            expected[idx].firstSrcRowBase = src_row_base_per_src[src_rank] + block_row;
                            expected[idx].firstDstRowBase = dst_row_base_per_src[src_rank] + block_row;
                        }
                        expected[idx].lastSrcRank = src_rank;
                        expected[idx].lastRows = block_rows;
                        expected[idx].lastSrcRowBase = src_row_base_per_src[src_rank] + block_row;
                        expected[idx].lastDstRowBase = dst_row_base_per_src[src_rank] + block_row;
                        ++expected[idx].assignedBlockCount;
                        expected[idx].coreRowCount += block_rows;
                    }
                    ++linear_block;
                }
            }
        }
        group_base += current_m;
    }

    DispatchTaskReport report;
    for (size_t idx = 0; idx < task_stats_elems; ++idx) {
        report.total_blocks += actual[idx].assignedBlockCount;
        report.total_rows += actual[idx].coreRowCount;
        if (actual[idx].assignedBlockCount != 0U) {
            ++report.active_slots;
        }
        report.max_rows_per_slot = std::max<uint64_t>(report.max_rows_per_slot, actual[idx].coreRowCount);
        report.min_row_block_rows = std::min<uint64_t>(report.min_row_block_rows, actual[idx].rowBlockRows);
        report.max_row_block_rows = std::max<uint64_t>(report.max_row_block_rows, actual[idx].rowBlockRows);
        report.max_blocks_per_group = std::max<uint64_t>(report.max_blocks_per_group, actual[idx].maxBlocksPerGroup);

        UpdateDispatchTaskReport(report, "groupIdx", idx, expected[idx].groupIdx, actual[idx].groupIdx);
        UpdateDispatchTaskReport(report, "coreIdx", idx, expected[idx].coreIdx, actual[idx].coreIdx);
        UpdateDispatchTaskReport(report, "rankSize", idx, expected[idx].rankSize, actual[idx].rankSize);
        UpdateDispatchTaskReport(report, "assignedBlockCount", idx, expected[idx].assignedBlockCount,
                                 actual[idx].assignedBlockCount);
        UpdateDispatchTaskReport(report, "coreRowCount", idx, expected[idx].coreRowCount, actual[idx].coreRowCount);
        UpdateDispatchTaskReport(report, "groupBase", idx, expected[idx].groupBase, actual[idx].groupBase);
        UpdateDispatchTaskReport(report, "currentM", idx, expected[idx].currentM, actual[idx].currentM);
        UpdateDispatchTaskReport(report, "firstSrcRank", idx, expected[idx].firstSrcRank, actual[idx].firstSrcRank);
        UpdateDispatchTaskReport(report, "firstRows", idx, expected[idx].firstRows, actual[idx].firstRows);
        UpdateDispatchTaskReport(report, "firstSrcRowBase", idx, expected[idx].firstSrcRowBase,
                                 actual[idx].firstSrcRowBase);
        UpdateDispatchTaskReport(report, "firstDstRowBase", idx, expected[idx].firstDstRowBase,
                                 actual[idx].firstDstRowBase);
        UpdateDispatchTaskReport(report, "lastSrcRank", idx, expected[idx].lastSrcRank, actual[idx].lastSrcRank);
        UpdateDispatchTaskReport(report, "lastRows", idx, expected[idx].lastRows, actual[idx].lastRows);
        UpdateDispatchTaskReport(report, "lastSrcRowBase", idx, expected[idx].lastSrcRowBase,
                                 actual[idx].lastSrcRowBase);
        UpdateDispatchTaskReport(report, "lastDstRowBase", idx, expected[idx].lastDstRowBase,
                                 actual[idx].lastDstRowBase);
        UpdateDispatchTaskReport(report, "rowBlockRows", idx, expected[idx].rowBlockRows, actual[idx].rowBlockRows);
        UpdateDispatchTaskReport(report, "maxBlocksPerGroup", idx, expected[idx].maxBlocksPerGroup,
                                 actual[idx].maxBlocksPerGroup);
    }
    report.group_count = expert_per_rank;
    if (report.min_row_block_rows == std::numeric_limits<uint64_t>::max()) {
        report.min_row_block_rows = 0;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

void UpdateDispatchGatherIntReport(DispatchGatherReport &report, const std::string &table, size_t idx, int32_t expected,
                                   int32_t actual, size_t &mismatch_count)
{
    if (actual == expected) {
        return;
    }
    if (report.first_table.empty()) {
        report.first_table = table;
        report.first_idx = idx;
        report.expected_i32 = expected;
        report.actual_i32 = actual;
    }
    ++mismatch_count;
}

void UpdateDispatchGatherScaleReport(DispatchGatherReport &report, size_t idx, float expected, float actual)
{
    const float tolerance = 1.0e-5f + 1.0e-3f * std::fabs(expected);
    if (std::isfinite(expected) && std::isfinite(actual) && std::fabs(actual - expected) <= tolerance) {
        return;
    }
    if (report.first_table.empty()) {
        report.first_table = "ptrPerTokenScale";
        report.first_idx = idx;
        report.expected_f32 = expected;
        report.actual_f32 = actual;
    }
    ++report.scale_mismatch_count;
}

std::vector<uint8_t> GatherAllPackedOffsetARows(const CaseConfig &cfg, const StandaloneRankRuntime &runtime,
                                                size_t packed_rows)
{
    const size_t packed_stride = static_cast<size_t>(cfg.k) + 32U;
    const size_t local_packed_bytes = packed_rows * packed_stride;
    if (local_packed_bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("packed offsetA section is too large for MPI gather");
    }

    std::vector<uint8_t> local_packed(local_packed_bytes, 0);
    const auto *window_base = reinterpret_cast<const uint8_t *>(runtime.hccl.WindowIn(runtime.hccl.rank_id));
    if (!local_packed.empty() && aclrtMemcpy(local_packed.data(), local_packed.size(), window_base, local_packed.size(),
                                             ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host local packed offsetA copy failed");
    }

    std::vector<uint8_t> all_packed(local_packed_bytes * static_cast<size_t>(runtime.hccl.world_size), 0);
    if (all_packed.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("all-rank packed offsetA section is too large for MPI bcast");
    }
    const int packed_bytes_i32 = static_cast<int>(local_packed_bytes);
    CommMpiGather(local_packed.data(), packed_bytes_i32, COMM_MPI_CHAR,
                  runtime.hccl.rank_id == 0 ? all_packed.data() : nullptr, packed_bytes_i32, COMM_MPI_CHAR, 0);
    CommMpiBcast(all_packed.data(), static_cast<int>(all_packed.size()), COMM_MPI_CHAR, 0);
    return all_packed;
}

std::vector<uint8_t> GatherAllPackedOffsetA(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                            const StandaloneRankRuntime &runtime)
{
    (void)build;
    return GatherAllPackedOffsetARows(cfg, runtime, cfg.max_output_size);
}

DispatchGatherReport CheckDispatchGather(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                         const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                         const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &dispatch = build.tiling.dispatchTiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const uint32_t local_begin = rank * expert_per_rank;
    const size_t packed_stride = static_cast<size_t>(cfg.k) + 32U;
    const size_t local_packed_bytes = static_cast<size_t>(cfg.max_output_size) * packed_stride;
    const std::vector<uint8_t> all_packed = GatherAllPackedOffsetA(cfg, build, runtime);

    std::vector<int8_t> expected_gm_a(static_cast<size_t>(cfg.max_output_size) * cfg.k, 0);
    std::vector<float> expected_scale(cfg.max_output_size, 0.0f);

    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);

    uint32_t group_base = 0;
    for (uint32_t group = 0; group < expert_per_rank; ++group) {
        uint32_t current_m = 0;
        uint32_t cumsum_before_src = 0;
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            const size_t token_idx = static_cast<size_t>(src_rank) * expert_num_aligned + local_begin + group;
            const uint32_t raw_rows = static_cast<uint32_t>(expected_tokens[token_idx]);
            const uint32_t dst_row_base = group_base + cumsum_before_src;
            uint32_t rows = raw_rows;
            if (dst_row_base >= cfg.max_output_size) {
                rows = 0;
            } else if (dst_row_base + rows > cfg.max_output_size) {
                rows = cfg.max_output_size - dst_row_base;
            }

            uint32_t src_row_base = 0;
            for (uint32_t expert = 0; expert < local_begin + group; ++expert) {
                src_row_base +=
                    static_cast<uint32_t>(expected_tokens[static_cast<size_t>(src_rank) * expert_num_aligned + expert]);
            }

            for (uint32_t row = 0; row < rows; ++row) {
                const size_t src_packed_offset = static_cast<size_t>(src_rank) * local_packed_bytes +
                                                 (static_cast<size_t>(src_row_base) + row) * packed_stride;
                const size_t dst_row = static_cast<size_t>(dst_row_base) + row;
                const size_t dst_offset = dst_row * cfg.k;
                if (src_packed_offset + cfg.k + sizeof(float) > all_packed.size() ||
                    dst_offset + cfg.k > expected_gm_a.size() || dst_row >= expected_scale.size()) {
                    throw std::runtime_error("dispatch gather golden index is out of range");
                }
                std::copy_n(reinterpret_cast<const int8_t *>(all_packed.data() + src_packed_offset), cfg.k,
                            expected_gm_a.begin() + dst_offset);
                std::memcpy(&expected_scale[dst_row], all_packed.data() + src_packed_offset + cfg.k, sizeof(float));
            }

            current_m += raw_rows;
            cumsum_before_src += raw_rows;
        }
        group_base += current_m;
    }

    std::vector<int8_t> actual_gm_a(expected_gm_a.size(), 0);
    std::vector<float> actual_scale(expected_scale.size(), 0.0f);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    const size_t gm_a_bytes = actual_gm_a.size() * sizeof(int8_t);
    const size_t scale_bytes = actual_scale.size() * sizeof(float);
    if ((!actual_gm_a.empty() && aclrtMemcpy(actual_gm_a.data(), gm_a_bytes, workspace_base + dispatch.gmAOffset,
                                             gm_a_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) ||
        (!actual_scale.empty() &&
         aclrtMemcpy(actual_scale.data(), scale_bytes, workspace_base + dispatch.perTokenScaleOffset, scale_bytes,
                     ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)) {
        throw std::runtime_error("device->host dispatch gather output copy failed");
    }
    DispatchGatherReport report;
    for (size_t idx = 0; idx < expected_gm_a.size(); ++idx) {
        UpdateDispatchGatherIntReport(report, "gmA", idx, static_cast<int32_t>(expected_gm_a[idx]),
                                      static_cast<int32_t>(actual_gm_a[idx]), report.gm_a_mismatch_count);
    }
    for (size_t idx = 0; idx < expected_scale.size(); ++idx) {
        UpdateDispatchGatherScaleReport(report, idx, expected_scale[idx], actual_scale[idx]);
    }
    report.pass = report.gm_a_mismatch_count == 0 && report.scale_mismatch_count == 0;
    return report;
}

void UpdateDispatchGatherDetailIntReport(DispatchGatherDetailReport &report, uint32_t group, uint32_t src_rank,
                                         uint32_t local_row, uint32_t dst_row, uint32_t col, int32_t expected,
                                         int32_t actual);
void UpdateDispatchGatherDetailScaleReport(DispatchGatherDetailReport &report, uint32_t group, uint32_t src_rank,
                                           uint32_t local_row, uint32_t dst_row, float expected, float actual);
void UpdateDispatchGatherDetailCoverageReport(DispatchGatherDetailReport &report, uint32_t group, uint32_t src_rank,
                                              uint32_t local_row, uint32_t dst_row, uint64_t expected, uint64_t actual);
void SetDispatchGatherDetailFirst(DispatchGatherDetailReport &report, const std::string &table, uint32_t group,
                                  uint32_t src_rank, uint32_t local_row, uint32_t dst_row, uint32_t col);

uint64_t HashDispatchBytes(const uint8_t *data, size_t bytes, uint64_t seed = 1469598103934665603ULL)
{
    uint64_t hash = seed;
    size_t idx = 0U;
    while (idx + sizeof(uint64_t) <= bytes) {
        uint64_t word = 0U;
        std::memcpy(&word, data + idx, sizeof(uint64_t));
        hash ^= word + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        idx += sizeof(uint64_t);
    }
    while (idx < bytes) {
        hash ^= static_cast<uint64_t>(data[idx]) + 0x9e3779b97f4a7c15ULL + (hash << 6U) + (hash >> 2U);
        ++idx;
    }
    return hash;
}

DispatchGatherDetailReport CheckDispatchGatherRankSplit(
    const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build, const StandaloneRankRuntime &runtime,
    const DeviceBuffer &workspace_dev, const std::vector<uint8_t> &x_bytes, const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &dispatch = build.tiling.dispatchTiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const uint32_t local_begin = rank * expert_per_rank;
    constexpr uint32_t kSampleSlots = 2U;
    constexpr bool trace = false;
    constexpr bool trace_detail = false;
    constexpr bool trace_group0 = false;
    auto trace_phase = [trace, rank](const char *phase) {
        if (trace) {
            std::cerr << "dispatchGatherRankSplitTrace rank=" << rank << " phase=" << phase << std::endl;
        }
    };

    trace_phase("begin");
    const std::vector<uint16_t> x_bf16 = BytesToU16(x_bytes);
    const ExpectedFrontRoute expected_route =
        BuildExpectedFrontRoute(expert_idx, front.routeElems, front.expertNum, front.expertNumAligned);
    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);
    trace_phase("expected_tokens");

    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    const size_t sample_count_per_rank = static_cast<size_t>(rank_size) * expert_per_rank * kSampleSlots;
    auto sample_index = [expert_per_rank](uint32_t dst_rank, uint32_t group, uint32_t slot) {
        return (static_cast<size_t>(dst_rank) * expert_per_rank + group) * kSampleSlots + slot;
    };

    std::vector<uint64_t> local_sample_hash(sample_count_per_rank, 0U);
    for (uint32_t dst_rank = 0U; dst_rank < rank_size; ++dst_rank) {
        const uint32_t dst_local_begin = dst_rank * expert_per_rank;
        uint32_t src_prev_sum = 0U;
        for (uint32_t expert = 0U; expert < dst_local_begin; ++expert) {
            src_prev_sum +=
                static_cast<uint32_t>(expected_tokens[static_cast<size_t>(rank) * expert_num_aligned + expert]);
        }

        uint32_t group_base = 0U;
        for (uint32_t group = 0U; group < expert_per_rank; ++group) {
            uint32_t current_m = 0U;
            uint32_t cumsum_before_src = 0U;
            for (uint32_t src_rank = 0U; src_rank < rank_size; ++src_rank) {
                const uint32_t raw = static_cast<uint32_t>(
                    expected_tokens[static_cast<size_t>(src_rank) * expert_num_aligned + dst_local_begin + group]);
                current_m += raw;
                if (src_rank < rank) {
                    cumsum_before_src += raw;
                }
            }
            const uint32_t raw_rows = static_cast<uint32_t>(
                expected_tokens[static_cast<size_t>(rank) * expert_num_aligned + dst_local_begin + group]);
            const uint32_t dst_row_base = group_base + cumsum_before_src;
            const uint32_t src_row_base = src_prev_sum;
            uint32_t rows = 0U;
            if (dst_row_base < cfg.max_output_size) {
                rows = raw_rows;
                if (dst_row_base + rows > cfg.max_output_size) {
                    rows = cfg.max_output_size - dst_row_base;
                }
                src_prev_sum += rows;
            }
            if (rows != 0U) {
                const uint32_t sample_rows[kSampleSlots] = {0U, rows - 1U};
                for (uint32_t slot = 0U; slot < kSampleSlots; ++slot) {
                    const uint32_t src_row = src_row_base + sample_rows[slot];
                    if (src_row >= expected_route.dst_to_src_route.size()) {
                        throw std::runtime_error("dispatch sample source row exceeds offsetA rows");
                    }
                    const int32_t src_route = expected_route.dst_to_src_route[src_row];
                    if (src_route < 0 || static_cast<size_t>(src_route) >= expected_route.expanded_row_idx.size()) {
                        throw std::runtime_error("dispatch sample source route is invalid");
                    }
                    const uint32_t token = static_cast<uint32_t>(src_route) / cfg.topk;
                    float expected_scale = 0.0f;
                    const std::vector<int8_t> expected_quant =
                        QuantizeBf16RowLikeV3(x_bf16, token, cfg.k, expected_scale);
                    uint64_t hash = HashDispatchBytes(reinterpret_cast<const uint8_t *>(expected_quant.data()), cfg.k);
                    hash = HashDispatchBytes(reinterpret_cast<const uint8_t *>(&expected_scale), sizeof(float), hash);
                    local_sample_hash[sample_index(dst_rank, group, slot)] = hash;
                }
            }
            group_base += current_m;
        }
    }
    trace_phase("local_hash");

    const size_t hash_bytes = local_sample_hash.size() * sizeof(uint64_t);
    if (hash_bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("dispatch offsetA sample hash table is too large for MPI gather");
    }
    std::vector<uint64_t> local_source_hash(sample_count_per_rank, 0U);
    const size_t source_packed_stride = static_cast<size_t>(cfg.k) + kHostFrontPackedScaleBytes;
    std::vector<uint8_t> local_packed_offset_a(static_cast<size_t>(front.routeElems) * source_packed_stride, 0U);
    if (!local_packed_offset_a.empty() &&
        aclrtMemcpy(local_packed_offset_a.data(), local_packed_offset_a.size(),
                    reinterpret_cast<const uint8_t *>(runtime.hccl.WindowIn(runtime.hccl.rank_id)),
                    local_packed_offset_a.size(), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host dispatch local source offsetA copy failed");
    }
    for (uint32_t dst_rank = 0U; dst_rank < rank_size; ++dst_rank) {
        const uint32_t dst_local_begin = dst_rank * expert_per_rank;
        uint32_t group_base = 0U;
        uint32_t src_prev_sum = 0U;
        for (uint32_t expert = 0U; expert < dst_local_begin; ++expert) {
            src_prev_sum +=
                static_cast<uint32_t>(expected_tokens[static_cast<size_t>(rank) * expert_num_aligned + expert]);
        }
        for (uint32_t group = 0U; group < expert_per_rank; ++group) {
            uint32_t current_m = 0U;
            uint32_t cumsum_before_src = 0U;
            for (uint32_t src_rank = 0U; src_rank < rank_size; ++src_rank) {
                const uint32_t raw = static_cast<uint32_t>(
                    expected_tokens[static_cast<size_t>(src_rank) * expert_num_aligned + dst_local_begin + group]);
                current_m += raw;
                if (src_rank < rank) {
                    cumsum_before_src += raw;
                }
            }
            const uint32_t raw_rows = static_cast<uint32_t>(
                expected_tokens[static_cast<size_t>(rank) * expert_num_aligned + dst_local_begin + group]);
            const uint32_t dst_row_base = group_base + cumsum_before_src;
            uint32_t rows = 0U;
            if (dst_row_base < cfg.max_output_size) {
                rows = raw_rows;
                if (dst_row_base + rows > cfg.max_output_size) {
                    rows = cfg.max_output_size - dst_row_base;
                }
                if (rows != 0U) {
                    const uint32_t sample_rows[kSampleSlots] = {0U, rows - 1U};
                    for (uint32_t slot = 0U; slot < kSampleSlots; ++slot) {
                        const uint32_t src_row = src_prev_sum + sample_rows[slot];
                        const uint8_t *packed_row =
                            local_packed_offset_a.data() + static_cast<size_t>(src_row) * source_packed_stride;
                        uint64_t hash = HashDispatchBytes(packed_row, cfg.k);
                        hash = HashDispatchBytes(packed_row + cfg.k, sizeof(float), hash);
                        local_source_hash[sample_index(dst_rank, group, slot)] = hash;
                    }
                }
                src_prev_sum += rows;
            }
            group_base += current_m;
        }
    }

    std::vector<uint64_t> all_sample_hash(static_cast<size_t>(rank_size) * sample_count_per_rank, 0U);
    std::vector<uint64_t> all_source_hash(static_cast<size_t>(rank_size) * sample_count_per_rank, 0U);
    trace_phase("mpi_gather_begin");
    CommMpiGather(local_sample_hash.data(), static_cast<int>(hash_bytes), COMM_MPI_CHAR,
                  rank == 0U ? all_sample_hash.data() : nullptr, static_cast<int>(hash_bytes), COMM_MPI_CHAR, 0);
    CommMpiGather(local_source_hash.data(), static_cast<int>(hash_bytes), COMM_MPI_CHAR,
                  rank == 0U ? all_source_hash.data() : nullptr, static_cast<int>(hash_bytes), COMM_MPI_CHAR, 0);
    trace_phase("mpi_gather_done");
    CommMpiBcast(all_sample_hash.data(), static_cast<int>(all_sample_hash.size() * sizeof(uint64_t)), COMM_MPI_CHAR, 0);
    CommMpiBcast(all_source_hash.data(), static_cast<int>(all_source_hash.size() * sizeof(uint64_t)), COMM_MPI_CHAR, 0);
    trace_phase("mpi_bcast_done");

    uint32_t total_output_rows = 0U;
    for (uint32_t group = 0U; group < expert_per_rank; ++group) {
        for (uint32_t src_rank = 0U; src_rank < rank_size; ++src_rank) {
            total_output_rows += static_cast<uint32_t>(
                expected_tokens[static_cast<size_t>(src_rank) * expert_num_aligned + local_begin + group]);
        }
    }
    total_output_rows = std::min(total_output_rows, cfg.max_output_size);

    std::vector<int32_t> row_owner(total_output_rows, -1);
    std::vector<uint32_t> prev_sum_by_src(rank_size, 0U);
    for (uint32_t src_rank = 0U; src_rank < rank_size; ++src_rank) {
        for (uint32_t expert = 0U; expert < local_begin; ++expert) {
            prev_sum_by_src[src_rank] +=
                static_cast<uint32_t>(expected_tokens[static_cast<size_t>(src_rank) * expert_num_aligned + expert]);
        }
    }

    DispatchGatherDetailReport report;
    std::vector<int8_t> actual_payload_row(cfg.k, 0);
    float actual_scale_value = 0.0f;
    uint32_t group_base = 0U;
    trace_phase("actual_compare_begin");
    for (uint32_t group = 0U; group < expert_per_rank; ++group) {
        uint32_t current_m = 0U;
        uint32_t cumsum_before_src = 0U;
        for (uint32_t src_rank = 0U; src_rank < rank_size; ++src_rank) {
            ++report.segment_count;
            const uint32_t raw_rows = static_cast<uint32_t>(
                expected_tokens[static_cast<size_t>(src_rank) * expert_num_aligned + local_begin + group]);
            const uint32_t dst_row_base = group_base + cumsum_before_src;
            const uint32_t src_row_base = prev_sum_by_src[src_rank];
            uint32_t rows = 0U;
            if (dst_row_base < cfg.max_output_size) {
                rows = raw_rows;
                if (dst_row_base + rows > cfg.max_output_size) {
                    rows = cfg.max_output_size - dst_row_base;
                }
                prev_sum_by_src[src_rank] += rows;
            }
            if (rows != 0U) {
                ++report.non_empty_segment_count;
            }

            if (rows != 0U) {
                if (static_cast<uint64_t>(src_row_base) + rows > front.routeElems) {
                    throw std::runtime_error("dispatch rank-split source row range exceeds offsetA rows");
                }

                for (uint32_t row = 0U; row < rows; ++row) {
                    const uint32_t dst_row = dst_row_base + row;
                    const uint32_t owner_id = group * rank_size + src_rank;
                    UpdateDispatchGatherDetailCoverageReport(
                        report, group, src_rank, row, dst_row, owner_id,
                        static_cast<uint64_t>(row_owner[dst_row] < 0 ? owner_id : row_owner[dst_row]));
                    row_owner[dst_row] = static_cast<int32_t>(owner_id);
                    ++report.row_count;
                    report.byte_count += cfg.k;
                    ++report.scale_count;
                }
                const uint32_t sample_rows[kSampleSlots] = {0U, rows - 1U};
                for (uint32_t slot = 0U; slot < kSampleSlots; ++slot) {
                    const uint32_t local_row = sample_rows[slot];
                    const uint32_t dst_row = dst_row_base + local_row;
                    if (aclrtMemcpy(actual_payload_row.data(), actual_payload_row.size(),
                                    workspace_base + dispatch.gmAOffset + static_cast<uint64_t>(dst_row) * cfg.k,
                                    actual_payload_row.size(), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                        aclrtMemcpy(&actual_scale_value, sizeof(actual_scale_value),
                                    workspace_base + dispatch.perTokenScaleOffset +
                                        static_cast<uint64_t>(dst_row) * sizeof(float),
                                    sizeof(actual_scale_value), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                        throw std::runtime_error("device->host dispatch sample destination row copy failed");
                    }
                    uint64_t got_hash =
                        HashDispatchBytes(reinterpret_cast<const uint8_t *>(actual_payload_row.data()), cfg.k);
                    got_hash = HashDispatchBytes(reinterpret_cast<const uint8_t *>(&actual_scale_value), sizeof(float),
                                                 got_hash);
                    const uint64_t expected_hash =
                        all_sample_hash[static_cast<size_t>(src_rank) * sample_count_per_rank +
                                        sample_index(rank, group, slot)];
                    const uint64_t source_hash = all_source_hash[static_cast<size_t>(src_rank) * sample_count_per_rank +
                                                                 sample_index(rank, group, slot)];
                    if (trace_group0 && group == 0U) {
                        std::cerr << "dispatchGatherRankSplitDetail rank=" << rank << " group=" << group
                                  << " srcRank=" << src_rank << " slot=" << slot << " localRow=" << local_row
                                  << " dstRow=" << dst_row << " expectedHash=" << expected_hash
                                  << " actualHash=" << got_hash << " actualScale=" << std::fixed << std::setprecision(8)
                                  << actual_scale_value << std::endl;
                    }
                    if (expected_hash != got_hash) {
                        if (trace_detail) {
                            bool found_match = false;
                            uint32_t match_rank = 0U;
                            uint32_t match_group = 0U;
                            uint32_t match_slot = 0U;
                            for (uint32_t probe_rank = 0U; probe_rank < rank_size && !found_match; ++probe_rank) {
                                for (uint32_t probe_group = 0U; probe_group < expert_per_rank && !found_match;
                                     ++probe_group) {
                                    for (uint32_t probe_slot = 0U; probe_slot < kSampleSlots; ++probe_slot) {
                                        const uint64_t probe_hash =
                                            all_sample_hash[static_cast<size_t>(src_rank) * sample_count_per_rank +
                                                            sample_index(probe_rank, probe_group, probe_slot)];
                                        if (probe_hash == got_hash) {
                                            found_match = true;
                                            match_rank = probe_rank;
                                            match_group = probe_group;
                                            match_slot = probe_slot;
                                            break;
                                        }
                                    }
                                }
                            }
                            std::cerr << "dispatchGatherRankSplitMismatchDetail rank=" << rank << " group=" << group
                                      << " srcRank=" << src_rank << " slot=" << slot << " localRow=" << local_row
                                      << " dstRow=" << dst_row << " expectedHash=" << expected_hash
                                      << " sourceHash=" << source_hash << " actualHash=" << got_hash
                                      << " sourceMatchesExpected=" << (source_hash == expected_hash ? 1 : 0)
                                      << " actualMatchesSource=" << (got_hash == source_hash ? 1 : 0)
                                      << " match=" << (found_match ? 1 : 0) << " matchDstRank=" << match_rank
                                      << " matchGroup=" << match_group << " matchSlot=" << match_slot
                                      << " actualScale=" << std::fixed << std::setprecision(8) << actual_scale_value;
                            const uint32_t src_row = src_row_base + local_row;
                            const int32_t src_route = expected_route.dst_to_src_route[src_row];
                            const uint32_t token = static_cast<uint32_t>(src_route) / cfg.topk;
                            float expected_scale = 0.0f;
                            const std::vector<int8_t> expected_quant =
                                QuantizeBf16RowLikeV3(x_bf16, token, cfg.k, expected_scale);
                            uint32_t actual_prefix_offset = cfg.k;
                            if (cfg.k >= 16U) {
                                for (uint32_t probe = 0U; probe + 16U <= cfg.k; ++probe) {
                                    bool same_prefix = true;
                                    for (uint32_t byte = 0U; byte < 16U; ++byte) {
                                        if (expected_quant[probe + byte] != actual_payload_row[byte]) {
                                            same_prefix = false;
                                            break;
                                        }
                                    }
                                    if (same_prefix) {
                                        actual_prefix_offset = probe;
                                        break;
                                    }
                                }
                            }
                            std::cerr << " expectedScale=" << expected_scale << " token=" << token
                                      << " srcRow=" << src_row << " actualPrefixOffset=" << actual_prefix_offset
                                      << " expBytes=";
                            for (uint32_t byte = 0U; byte < 16U && byte < cfg.k; ++byte) {
                                std::cerr << (byte == 0U ? "" : ",") << static_cast<int32_t>(expected_quant[byte]);
                            }
                            std::cerr << " actBytes=";
                            for (uint32_t byte = 0U; byte < 16U && byte < cfg.k; ++byte) {
                                std::cerr << (byte == 0U ? "" : ",") << static_cast<int32_t>(actual_payload_row[byte]);
                            }
                            std::cerr << std::endl;
                        }
                        if (report.first_table.empty()) {
                            SetDispatchGatherDetailFirst(report, "hash", group, src_rank, local_row, dst_row, 0U);
                            report.expected_u64 = expected_hash;
                            report.actual_u64 = got_hash;
                        }
                        ++report.gm_a_mismatch_count;
                    }
                }
            }
            current_m += raw_rows;
            cumsum_before_src += raw_rows;
        }
        group_base += current_m;
    }

    for (uint32_t row = 0U; row < total_output_rows; ++row) {
        UpdateDispatchGatherDetailCoverageReport(report, 0U, 0U, row, row, 1U, row_owner[row] >= 0 ? 1U : 0U);
    }
    report.pass =
        report.gm_a_mismatch_count == 0 && report.scale_mismatch_count == 0 && report.coverage_mismatch_count == 0;
    trace_phase("done");
    return report;
}

void SetDispatchGatherDetailFirst(DispatchGatherDetailReport &report, const std::string &table, uint32_t group,
                                  uint32_t src_rank, uint32_t local_row, uint32_t dst_row, uint32_t col)
{
    if (!report.first_table.empty()) {
        return;
    }
    report.first_table = table;
    report.first_group = group;
    report.first_src_rank = src_rank;
    report.first_local_row = local_row;
    report.first_dst_row = dst_row;
    report.first_col = col;
}

void UpdateDispatchGatherDetailIntReport(DispatchGatherDetailReport &report, uint32_t group, uint32_t src_rank,
                                         uint32_t local_row, uint32_t dst_row, uint32_t col, int32_t expected,
                                         int32_t actual)
{
    if (actual == expected) {
        return;
    }
    if (report.first_table.empty()) {
        SetDispatchGatherDetailFirst(report, "gmA", group, src_rank, local_row, dst_row, col);
        report.expected_i32 = expected;
        report.actual_i32 = actual;
    }
    ++report.gm_a_mismatch_count;
}

void UpdateDispatchGatherDetailScaleReport(DispatchGatherDetailReport &report, uint32_t group, uint32_t src_rank,
                                           uint32_t local_row, uint32_t dst_row, float expected, float actual)
{
    const float tolerance = 1.0e-5f + 1.0e-3f * std::fabs(expected);
    if (std::isfinite(expected) && std::isfinite(actual) && std::fabs(actual - expected) <= tolerance) {
        return;
    }
    if (report.first_table.empty()) {
        SetDispatchGatherDetailFirst(report, "ptrPerTokenScale", group, src_rank, local_row, dst_row, 0);
        report.expected_f32 = expected;
        report.actual_f32 = actual;
    }
    ++report.scale_mismatch_count;
}

void UpdateDispatchGatherDetailCoverageReport(DispatchGatherDetailReport &report, uint32_t group, uint32_t src_rank,
                                              uint32_t local_row, uint32_t dst_row, uint64_t expected, uint64_t actual)
{
    if (actual == expected) {
        return;
    }
    if (report.first_table.empty()) {
        SetDispatchGatherDetailFirst(report, "coverage", group, src_rank, local_row, dst_row, 0);
        report.expected_u64 = expected;
        report.actual_u64 = actual;
    }
    ++report.coverage_mismatch_count;
}

DispatchGatherDetailReport CheckDispatchGatherDetail(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                                     const StandaloneRankRuntime &runtime,
                                                     const DeviceBuffer &workspace_dev,
                                                     const std::vector<uint8_t> &x_bytes,
                                                     const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &dispatch = build.tiling.dispatchTiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const uint32_t local_begin = rank * expert_per_rank;
    const std::vector<uint8_t> all_x_bytes =
        GatherAllRankBytes(runtime.hccl.rank_id, runtime.hccl.world_size, x_bytes, "x");
    const std::vector<uint8_t> all_expert_idx_bytes =
        GatherAllRankBytes(runtime.hccl.rank_id, runtime.hccl.world_size, expert_idx, "expert_idx");
    const std::vector<uint16_t> all_x_bf16 = BytesToU16(all_x_bytes);
    const std::vector<int32_t> all_expert_idx_i32 = BytesToI32(all_expert_idx_bytes);
    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);

    std::vector<int8_t> actual_gm_a(static_cast<size_t>(cfg.max_output_size) * cfg.k, 0);
    std::vector<float> actual_scale(cfg.max_output_size, 0.0f);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    const size_t gm_a_bytes = actual_gm_a.size() * sizeof(int8_t);
    const size_t scale_bytes = actual_scale.size() * sizeof(float);
    if ((!actual_gm_a.empty() && aclrtMemcpy(actual_gm_a.data(), gm_a_bytes, workspace_base + dispatch.gmAOffset,
                                             gm_a_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) ||
        (!actual_scale.empty() &&
         aclrtMemcpy(actual_scale.data(), scale_bytes, workspace_base + dispatch.perTokenScaleOffset, scale_bytes,
                     ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)) {
        throw std::runtime_error("device->host dispatch gather detail output copy failed");
    }

    DispatchGatherDetailReport report;
    std::vector<int32_t> row_owner(cfg.max_output_size, -1);
    uint32_t group_base = 0;
    for (uint32_t group = 0; group < expert_per_rank; ++group) {
        uint32_t current_m = 0;
        uint32_t cumsum_before_src = 0;
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            ++report.segment_count;
            const size_t token_idx = static_cast<size_t>(src_rank) * expert_num_aligned + local_begin + group;
            const uint32_t raw_rows = static_cast<uint32_t>(expected_tokens[token_idx]);
            const uint32_t dst_row_base = group_base + cumsum_before_src;
            uint32_t rows = raw_rows;
            if (dst_row_base >= cfg.max_output_size) {
                rows = 0;
            } else if (dst_row_base + rows > cfg.max_output_size) {
                rows = cfg.max_output_size - dst_row_base;
            }

            if (rows != 0U) {
                ++report.non_empty_segment_count;
            }
            for (uint32_t row = 0; row < rows; ++row) {
                const uint32_t dst_row = dst_row_base + row;
                const uint32_t owner_id = group * rank_size + src_rank;
                if (dst_row >= row_owner.size()) {
                    throw std::runtime_error("dispatch gather detail dst row is out of range");
                }
                UpdateDispatchGatherDetailCoverageReport(
                    report, group, src_rank, row, dst_row, owner_id,
                    static_cast<uint64_t>(row_owner[dst_row] < 0 ? owner_id : row_owner[dst_row]));
                row_owner[dst_row] = static_cast<int32_t>(owner_id);
                ++report.row_count;
                report.byte_count += cfg.k;
                ++report.scale_count;
            }

            uint32_t emitted_rows = 0;
            const uint32_t global_expert = local_begin + group;
            for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
                uint32_t token_start = 0;
                uint32_t token_end = 0;
                GetExpectedTokenRange(cfg, core, token_start, token_end);
                for (uint32_t token = token_start; token < token_end; ++token) {
                    const uint32_t global_token = src_rank * cfg.m + token;
                    float expected_scale = 0.0f;
                    const std::vector<int8_t> expected_quant =
                        QuantizeBf16RowLikeV3(all_x_bf16, global_token, cfg.k, expected_scale);
                    for (uint32_t topk_idx = 0; topk_idx < cfg.topk; ++topk_idx) {
                        const size_t expert_idx_offset = static_cast<size_t>(global_token) * cfg.topk + topk_idx;
                        if (expert_idx_offset >= all_expert_idx_i32.size() ||
                            all_expert_idx_i32[expert_idx_offset] != static_cast<int32_t>(global_expert)) {
                            continue;
                        }

                        if (emitted_rows >= rows) {
                            ++emitted_rows;
                            continue;
                        }
                        const uint32_t dst_row = dst_row_base + emitted_rows;
                        const uint32_t local_row = emitted_rows;
                        const size_t dst_offset = static_cast<size_t>(dst_row) * cfg.k;
                        if (dst_offset + cfg.k > actual_gm_a.size() || dst_row >= actual_scale.size()) {
                            throw std::runtime_error("dispatch gather detail activation index is out of range");
                        }
                        for (uint32_t col = 0; col < cfg.k; ++col) {
                            const int32_t expected = static_cast<int32_t>(expected_quant[col]);
                            const int32_t actual = static_cast<int32_t>(actual_gm_a[dst_offset + col]);
                            UpdateDispatchGatherDetailIntReport(report, group, src_rank, local_row, dst_row, col,
                                                                expected, actual);
                        }
                        UpdateDispatchGatherDetailScaleReport(report, group, src_rank, local_row, dst_row,
                                                              expected_scale, actual_scale[dst_row]);
                        ++emitted_rows;
                    }
                }
            }
            UpdateDispatchGatherDetailCoverageReport(report, group, src_rank, rows, dst_row_base + rows, raw_rows,
                                                     emitted_rows);

            current_m += raw_rows;
            cumsum_before_src += raw_rows;
        }
        group_base += current_m;
    }

    const uint32_t expected_output_rows = static_cast<uint32_t>(
        std::min<uint64_t>(static_cast<uint64_t>(group_base), static_cast<uint64_t>(cfg.max_output_size)));
    for (uint32_t row = 0; row < expected_output_rows; ++row) {
        UpdateDispatchGatherDetailCoverageReport(report, 0, 0, row, row, 1U, row_owner[row] >= 0 ? 1U : 0U);
    }
    report.pass =
        report.gm_a_mismatch_count == 0 && report.scale_mismatch_count == 0 && report.coverage_mismatch_count == 0;
    return report;
}

void UpdateDispatchGroupDoneReport(DispatchGroupDoneReport &report, const std::string &field, size_t idx,
                                   uint64_t expected, uint64_t actual)
{
    if (actual == expected) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.first_idx = idx;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

DispatchGroupDoneReport CheckDispatchGroupDone(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                               const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                               const std::vector<uint8_t> &expert_idx)
{
    constexpr uint64_t kHostDispatchGroupDoneDebugMagic = 0x5635444752444f4eULL;
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &dispatch = build.tiling.dispatchTiling;
    const uint32_t expert_per_rank = cfg.expert_per_rank;
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const size_t metadata_bytes =
        static_cast<size_t>(expert_per_rank) * rank_size * sizeof(DispatchFFNCombineDispatchMetadataDebug);
    const size_t task_stats_bytes =
        static_cast<size_t>(expert_per_rank) * cfg.aiv_num * sizeof(DispatchFFNCombineDispatchTaskStatsDebug);
    const size_t group_done_elems = expert_per_rank;
    const size_t group_done_bytes = group_done_elems * sizeof(DispatchFFNCombineDispatchGroupDoneDebug);
    const uint64_t group_done_offset = dispatch.dispatchDebugOffset + sizeof(DispatchFFNCombineDispatchLayoutDebug) +
                                       metadata_bytes + task_stats_bytes;

    if (dispatch.dispatchDebugBytes <
        sizeof(DispatchFFNCombineDispatchLayoutDebug) + metadata_bytes + task_stats_bytes + group_done_bytes) {
        throw std::runtime_error("dispatch debug area is too small for group done table");
    }

    std::vector<DispatchFFNCombineDispatchGroupDoneDebug> actual(group_done_elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual.data(), group_done_bytes, workspace_base + group_done_offset, group_done_bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host dispatch group done copy failed");
    }
    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);

    DispatchGroupDoneReport report;
    uint32_t group_base = 0;
    const uint32_t local_begin = rank * expert_per_rank;
    for (uint32_t group = 0; group < expert_per_rank; ++group) {
        uint32_t current_m = 0;
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            const size_t token_idx = static_cast<size_t>(src_rank) * expert_num_aligned + local_begin + group;
            current_m += static_cast<uint32_t>(expected_tokens[token_idx]);
        }

        UpdateDispatchGroupDoneReport(report, "magic", group, kHostDispatchGroupDoneDebugMagic, actual[group].magic);
        UpdateDispatchGroupDoneReport(report, "groupIdx", group, group, actual[group].groupIdx);
        UpdateDispatchGroupDoneReport(report, "rank", group, rank, actual[group].rank);
        UpdateDispatchGroupDoneReport(report, "rankSize", group, rank_size, actual[group].rankSize);
        UpdateDispatchGroupDoneReport(report, "groupBase", group, group_base, actual[group].groupBase);
        UpdateDispatchGroupDoneReport(report, "currentM", group, current_m, actual[group].currentM);
        UpdateDispatchGroupDoneReport(report, "marker", group, 1U, actual[group].marker);
        UpdateDispatchGroupDoneReport(report, "reserved0", group, 0U, actual[group].reserved0);
        group_base += current_m;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

void UpdateCombineLayoutReport(CombineLayoutReport &report, const std::string &field, uint64_t expected,
                               uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

void UpdateUnpermuteLayoutReport(UnpermuteLayoutReport &report, const std::string &field, uint64_t expected,
                                 uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_field = field;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

template <typename Report>
void UpdateCombineIndexedReport(Report &report, const std::string &field, size_t idx, uint64_t expected,
                                uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_idx = idx;
        report.first_field = field;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

template <typename Report>
void UpdateUnpermuteIndexedReport(Report &report, const std::string &field, size_t idx, uint64_t expected,
                                  uint64_t actual)
{
    if (expected == actual) {
        return;
    }
    if (report.mismatch_count == 0) {
        report.first_idx = idx;
        report.first_field = field;
        report.expected = expected;
        report.actual = actual;
    }
    ++report.mismatch_count;
}

uint32_t FloatBitsHost(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

uint64_t HostPeerOffsetD(uint64_t window_bytes)
{
    return AlignUpU64(window_bytes / 3U, 512U) + kHostMiB;
}

uint64_t HostPeerOffsetScale2(uint64_t window_bytes)
{
    return AlignUpU64(window_bytes / 3U, 512U);
}

uint64_t HostPeerOffsetTokenPerExpert(uint64_t window_bytes)
{
    return window_bytes - 2U * kHostMiB;
}

uint64_t HostPeerSignalBase(uint64_t window_bytes)
{
    return window_bytes - kHostMiB;
}

uint32_t ExpectedRowsClipped(uint32_t src_row_offset, uint32_t rows_raw, uint32_t max_output_size)
{
    if (src_row_offset >= max_output_size) {
        return 0U;
    }
    const uint32_t remaining = max_output_size - src_row_offset;
    return rows_raw > remaining ? remaining : rows_raw;
}

bool HostCombineSmallTokenPath(const CaseConfig &cfg, uint32_t rank_size)
{
    (void)rank_size;
    const uint64_t token_volume = static_cast<uint64_t>(cfg.m) * cfg.topk;
    return token_volume <= kHostCombineSmallTokenThreshold;
}

uint32_t HostCombineSubtileFirstIndex(uint32_t subtile_count, uint32_t aiv_sub_core_idx)
{
    const uint32_t first_sub_core_count = (subtile_count + 1U) / 2U;
    return aiv_sub_core_idx == 0U ? 0U : first_sub_core_count;
}

uint32_t HostCombineSubtileAssignedCount(uint32_t subtile_count, uint32_t aiv_sub_core_idx)
{
    const uint32_t first_sub_core_count = (subtile_count + 1U) / 2U;
    return aiv_sub_core_idx == 0U ? first_sub_core_count : subtile_count - first_sub_core_count;
}

uint32_t HostCombineDebugIoSlots(uint32_t rank_size, uint32_t aiv_num)
{
    return std::max(rank_size, aiv_num);
}

uint32_t ExpectedCombineTileRows(const DispatchFFNCombineCombineTiling &combine)
{
    (void)combine;
    return 0U;
}

uint32_t ExpectedCombineEffectiveTileRows(uint32_t rows, uint32_t tile_rows, uint32_t tile_cols, uint32_t core_num)
{
    (void)tile_rows;
    (void)tile_cols;
    if (rows <= 1U) {
        return 1U;
    }
    uint32_t target_blocks = rows;
    if (target_blocks > core_num) {
        target_blocks = core_num;
    }
    if (target_blocks == 0U) {
        target_blocks = 1U;
    }
    const uint32_t effective_rows = CeilDivU32(rows, target_blocks);
    return effective_rows == 0U ? 1U : effective_rows;
}

uint32_t ExpectedCombineBlockCount(uint32_t rows, uint32_t tile_rows, uint32_t tile_cols, uint32_t core_num)
{
    if (rows == 0U) {
        return 0U;
    }
    return CeilDivU32(rows, ExpectedCombineEffectiveTileRows(rows, tile_rows, tile_cols, core_num));
}

uint32_t HostGmmCommonStartLoopIdx(uint32_t core_idx, uint32_t core_num, uint32_t start_core_idx)
{
    return ((core_idx < start_core_idx) ? (core_idx + core_num) : core_idx) - start_core_idx;
}

void GetGmmCommonActualBlockShapeMN(uint32_t current_m, uint32_t problem_n, uint32_t l1_tile_m, uint32_t l1_tile_n,
                                    uint32_t tile_m, uint32_t tile_n, uint32_t block_m, uint32_t block_n,
                                    uint32_t &actual_m, uint32_t &actual_n)
{
    actual_m = (block_m + 1U == tile_m) ? (current_m - block_m * l1_tile_m) : l1_tile_m;
    actual_n = (block_n + 1U == tile_n) ? (problem_n - block_n * l1_tile_n) : l1_tile_n;
}

uint32_t ExpectedSrcRankDstRowOffset(const std::vector<int32_t> &expected_tokens, uint32_t expert_num_aligned,
                                     uint32_t src_rank, uint32_t global_expert)
{
    uint32_t dst_row_offset = 0;
    for (uint32_t expert = 0; expert < global_expert; ++expert) {
        dst_row_offset +=
            static_cast<uint32_t>(expected_tokens[static_cast<size_t>(src_rank) * expert_num_aligned + expert]);
    }
    return dst_row_offset;
}

bool HostCombineSmallCoreHasAssignedSubtile(const DispatchFFNCombineGmm2Tiling &gmm2, uint32_t problem_n,
                                            uint32_t current_m, uint32_t tile_m, uint32_t tile_n, uint32_t core_loops,
                                            uint32_t start_core_idx, uint32_t aic_core_num, uint32_t aic_core_idx,
                                            uint32_t aiv_sub_core_idx)
{
    if (aic_core_num == 0U) {
        return false;
    }
    const uint32_t start_loop_idx = HostGmmCommonStartLoopIdx(aic_core_idx, aic_core_num, start_core_idx);
    for (uint32_t loop = start_loop_idx; loop < core_loops; loop += aic_core_num) {
        uint32_t block_m = 0;
        uint32_t block_n = 0;
        GetGmm1BlockCoordMN(loop, tile_m, tile_n, block_m, block_n);
        uint32_t actual_m = 0;
        uint32_t actual_n = 0;
        GetGmmCommonActualBlockShapeMN(current_m, problem_n, gmm2.l1TileM, gmm2.l1TileN, tile_m, tile_n, block_m,
                                       block_n, actual_m, actual_n);
        const uint32_t subtile_count = CeilDivU32(actual_m, kHostCombineSmallTokenSubtileRows);
        const uint32_t assigned_subtiles = HostCombineSubtileAssignedCount(subtile_count, aiv_sub_core_idx);
        if (assigned_subtiles != 0U) {
            return true;
        }
    }
    return false;
}

uint64_t HostCombineDirectLargeRequiredUbBytes(uint32_t k)
{
    return kHostCombineDirectLargeUbStages * (AlignUpU64(static_cast<uint64_t>(k) * sizeof(uint16_t), UB_ALIGN) +
                                              AlignUpU64(static_cast<uint64_t>(k) * sizeof(uint16_t), UB_ALIGN) +
                                              AlignUpU64(static_cast<uint64_t>(k) * sizeof(float), UB_ALIGN));
}

bool HostCombineDirectLargeUbPlanOk(uint32_t k)
{
    return k != 0U && k <= kHostCombineVecTileElems && (k * sizeof(uint16_t)) % UB_ALIGN == 0U &&
           HostCombineDirectLargeRequiredUbBytes(k) <= AtlasA5::UB_SIZE;
}

uint64_t HostCombineDirectSmallRequiredUbBytes()
{
    return kHostCombineDirectSmallUbStages *
           (AlignUpU64(static_cast<uint64_t>(kHostCombineSmallMaxElems) * sizeof(uint16_t), UB_ALIGN) +
            AlignUpU64(static_cast<uint64_t>(kHostCombineSmallMaxElems) * sizeof(uint16_t), UB_ALIGN) +
            AlignUpU64(static_cast<uint64_t>(kHostCombineSmallMaxElems) * sizeof(float), UB_ALIGN) +
            AlignUpU64(static_cast<uint64_t>(kHostCombineSmallScaleElems) * sizeof(float), UB_ALIGN));
}

bool HostCombineDirectSmallUbPlanOk()
{
    return kHostCombineSmallTokenSubtileCols != 0U && kHostCombineSmallScaleElems != 0U &&
           HostCombineDirectSmallRequiredUbBytes() <= AtlasA5::UB_SIZE;
}

uint64_t HostGmm2CombineCvSlotOffset()
{
    return 2U * (static_cast<uint64_t>(kHostCombineSmallMaxElems) * sizeof(uint16_t) +
                 static_cast<uint64_t>(kHostCombineSmallMaxElems) * sizeof(uint16_t) +
                 static_cast<uint64_t>(kHostCombineSmallMaxElems) * sizeof(float) +
                 static_cast<uint64_t>(kHostCombineSmallScaleElems) * sizeof(float));
}

uint64_t HostGmm2CombineCvTileScaleOffset()
{
    return HostGmm2CombineCvSlotOffset() + kHostGmm2CombineCvSlotBytes;
}

uint64_t HostGmm2CombineCvRequiredUbBytes()
{
    return HostGmm2CombineCvTileScaleOffset() +
           static_cast<uint64_t>(kHostGmm2CombineCvScaleCacheElems) * sizeof(float);
}

CombineLayoutReport CheckCombineLayout(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                       const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev)
{
    const auto &combine = build.tiling.combineTiling;
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    DispatchFFNCombineCombineLayoutDebug actual;
    std::memset(&actual, 0, sizeof(actual));
    if (combine.combineDebugBytes < sizeof(actual)) {
        throw std::runtime_error("combine debug area is too small for layout header");
    }
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(&actual, sizeof(actual), workspace_base + combine.combineDebugOffset, sizeof(actual),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host combine layout header copy failed");
    }

    const uint64_t gmm2_output_bytes = static_cast<uint64_t>(cfg.max_output_size) * cfg.k * sizeof(uint16_t);
    const uint64_t scale_bytes = static_cast<uint64_t>(cfg.max_output_size) * sizeof(float);
    const uint64_t offset_d_bytes = static_cast<uint64_t>(cfg.max_output_size) * cfg.k * sizeof(uint16_t);
    const uint64_t window_bytes = runtime.hccl.WindowBytes();
    const uint64_t token_volume = static_cast<uint64_t>(cfg.m) * cfg.topk;
    const uint32_t is_large_path = token_volume > kHostCombineSmallTokenThreshold ? 1U : 0U;
    const bool direct_large_selected = is_large_path != 0U;
    const bool direct_small_selected = is_large_path == 0U;
    const bool direct_subtile_selected = direct_large_selected || direct_small_selected;
    CombineLayoutReport report;
    UpdateCombineLayoutReport(report, "magic", kHostCombineLayoutDebugMagic, actual.magic);
    UpdateCombineLayoutReport(report, "workspaceBase", reinterpret_cast<uint64_t>(workspace_dev.ptr),
                              actual.workspaceBase);
    UpdateCombineLayoutReport(report, "gmm2OutputOffset", combine.gmm2OutputOffset, actual.gmm2OutputOffset);
    UpdateCombineLayoutReport(report, "perTokenScale2Offset", combine.perTokenScale2Offset,
                              actual.perTokenScale2Offset);
    UpdateCombineLayoutReport(report, "cumsumMMOffset", front.cumsumMMOffset, actual.cumsumMMOffset);
    UpdateCombineLayoutReport(report, "preSumBeforeRankOffset", front.preSumBeforeRankOffset,
                              actual.preSumBeforeRankOffset);
    UpdateCombineLayoutReport(report, "combineScratchOffset", combine.combineScratchOffset,
                              actual.combineScratchOffset);
    UpdateCombineLayoutReport(report, "combineScratchBytes", combine.combineScratchBytes, actual.combineScratchBytes);
    UpdateCombineLayoutReport(report, "combineScratchBytesPerAiv", combine.combineScratchBytesPerAiv,
                              actual.combineScratchBytesPerAiv);
    UpdateCombineLayoutReport(report, "combineScratchCore0Offset", combine.combineScratchOffset,
                              actual.combineScratchCoreOffset);
    UpdateCombineLayoutReport(report, "combineDebugOffset", combine.combineDebugOffset, actual.combineDebugOffset);
    UpdateCombineLayoutReport(report, "combineDebugBytes", combine.combineDebugBytes, actual.combineDebugBytes);
    UpdateCombineLayoutReport(report, "peerOffsetD", HostPeerOffsetD(window_bytes), actual.peerOffsetD);
    UpdateCombineLayoutReport(report, "peerOffsetPeerTokenPerExpert", HostPeerOffsetTokenPerExpert(window_bytes),
                              actual.peerOffsetPeerTokenPerExpert);
    UpdateCombineLayoutReport(report, "peerOffsetScale2", HostPeerOffsetScale2(window_bytes), actual.peerOffsetScale2);
    UpdateCombineLayoutReport(report, "peerSignalBaseOffset", HostPeerSignalBase(window_bytes),
                              actual.peerSignalBaseOffset);
    UpdateCombineLayoutReport(report, "gmm2OutputBytes", gmm2_output_bytes, actual.gmm2OutputBytes);
    UpdateCombineLayoutReport(report, "perTokenScale2Bytes", scale_bytes, actual.perTokenScale2Bytes);
    UpdateCombineLayoutReport(report, "offsetDBytes", offset_d_bytes, actual.offsetDBytes);
    UpdateCombineLayoutReport(report, "offsetScale2Bytes", scale_bytes, actual.offsetScale2Bytes);
    UpdateCombineLayoutReport(report, "offsetScale2CapacityBytes", kHostMiB, actual.offsetScale2CapacityBytes);
    UpdateCombineLayoutReport(report, "rank", static_cast<uint64_t>(runtime.hccl.rank_id), actual.rank);
    UpdateCombineLayoutReport(report, "rankSize", static_cast<uint64_t>(runtime.hccl.world_size), actual.rankSize);
    UpdateCombineLayoutReport(report, "coreIdx", 0, actual.coreIdx);
    UpdateCombineLayoutReport(report, "stageNum", front.stageNum, actual.stageNum);
    UpdateCombineLayoutReport(report, "problemK", cfg.k, actual.problemK);
    UpdateCombineLayoutReport(report, "maxOutputSize", cfg.max_output_size, actual.maxOutputSize);
    UpdateCombineLayoutReport(report, "expertPerRank", cfg.expert_per_rank, actual.expertPerRank);
    UpdateCombineLayoutReport(report, "combineTileCols", combine.combineTileCols, actual.combineTileCols);
    UpdateCombineLayoutReport(report, "combineTileRows", ExpectedCombineTileRows(combine), actual.combineTileRows);
    UpdateCombineLayoutReport(report, "debugMode", combine.combineDebugMode, actual.debugMode);
    UpdateCombineLayoutReport(report, "outputElementBytes", sizeof(uint16_t), actual.outputElementBytes);
    UpdateCombineLayoutReport(report, "gmm2OutputRowBytes", cfg.k * sizeof(uint16_t), actual.gmm2OutputRowBytes);
    UpdateCombineLayoutReport(report, "offsetDRowBytes", cfg.k * sizeof(uint16_t), actual.offsetDRowBytes);
    UpdateCombineLayoutReport(report, "tileBytes", combine.combineTileCols * sizeof(uint16_t), actual.tileBytes);
    UpdateCombineLayoutReport(report, "scratchBytesPerBuffer", combine.combineScratchBytesPerAiv / 2U,
                              actual.scratchBytesPerBuffer);
    UpdateCombineLayoutReport(report, "layoutVersion", kHostCombineLayoutVersion, actual.layoutVersion);
    UpdateCombineLayoutReport(report, "marker", 1U, actual.marker);
    UpdateCombineLayoutReport(report, "scratchOffsetAlignment", 0, combine.combineScratchOffset % 512U);
    UpdateCombineLayoutReport(report, "scratchCapacity", 0, combine.combineScratchBytes > kHostMiB ? 1U : 0U);
    UpdateCombineLayoutReport(report, "gmm2OutputRowAlignment", 0, (cfg.k * sizeof(uint16_t)) % 32U);
    UpdateCombineLayoutReport(report, "offsetDRowAlignment", 0, (cfg.k * sizeof(uint16_t)) % 32U);
    UpdateCombineLayoutReport(report, "offsetScale2CapacityOk", scale_bytes <= kHostMiB ? 1U : 0U,
                              actual.offsetScale2CapacityOk);
    UpdateCombineLayoutReport(report, "offsetScale2DNoOverlap",
                              HostPeerOffsetScale2(window_bytes) + kHostMiB <= HostPeerOffsetD(window_bytes) ? 1U : 0U,
                              actual.offsetScale2DNoOverlap);
    UpdateCombineLayoutReport(report, "tokenVolume", token_volume, actual.tokenVolume);
    UpdateCombineLayoutReport(report, "isLargePath", is_large_path, actual.isLargePath);
    UpdateCombineLayoutReport(report, "combineImplMode", combine.combineImplMode, actual.combineImplMode);
    UpdateCombineLayoutReport(report, "combineStopStep", combine.combineStopStep, actual.combineStopStep);
    UpdateCombineLayoutReport(report, "gmm2CombineCvMode", kHostGmm2CombineCvModeDirect,
                              actual.gmm2CombineCvMode);
    UpdateCombineLayoutReport(report, "gmm2CombineCvDebugMode", combine.gmm2CombineCvDebugMode,
                              actual.gmm2CombineCvDebugMode);
    UpdateCombineLayoutReport(report, "gmm2CombineCvReadyFlag", V8_GMM2_COMBINE_CV_READY_HARD_FLAG,
                              actual.gmm2CombineCvReadyFlag);
    UpdateCombineLayoutReport(report, "gmm2CombineCvFreeFlag", V8_GMM2_COMBINE_CV_FREE_HARD_FLAG,
                              actual.gmm2CombineCvFreeFlag);
    UpdateCombineLayoutReport(report, "gmm2CombineCvSlotOffset", HostGmm2CombineCvSlotOffset(),
                              actual.gmm2CombineCvSlotOffset);
    UpdateCombineLayoutReport(report, "gmm2CombineCvTileScaleOffset", HostGmm2CombineCvTileScaleOffset(),
                              actual.gmm2CombineCvTileScaleOffset);
    UpdateCombineLayoutReport(report, "gmm2CombineCvRequiredUbBytes", HostGmm2CombineCvRequiredUbBytes(),
                              actual.gmm2CombineCvRequiredUbBytes);
    UpdateCombineLayoutReport(report, "directLargeEnabled", direct_large_selected ? 1U : 0U, actual.directLargeEnabled);
    UpdateCombineLayoutReport(report, "directSmallEnabled", direct_small_selected ? 1U : 0U, actual.directSmallEnabled);
    if (combine.combineStopStep == 1U && direct_large_selected) {
        UpdateCombineLayoutReport(report, "directLargeRequiredUbBytes", 0U, actual.directLargeRequiredUbBytes);
        UpdateCombineLayoutReport(report, "directLargeUbCOffset0", 0U, actual.directLargeUbCOffset0);
        UpdateCombineLayoutReport(report, "directLargeUbDOffset0", 0U, actual.directLargeUbDOffset0);
        UpdateCombineLayoutReport(report, "directLargeUbFp32Offset0", 0U, actual.directLargeUbFp32Offset0);
        UpdateCombineLayoutReport(report, "directLargeUbCOffset1", 0U, actual.directLargeUbCOffset1);
        UpdateCombineLayoutReport(report, "directLargeUbDOffset1", 0U, actual.directLargeUbDOffset1);
        UpdateCombineLayoutReport(report, "directLargeUbFp32Offset1", 0U, actual.directLargeUbFp32Offset1);
        UpdateCombineLayoutReport(report, "directLargeTileCols", 0U, actual.directLargeTileCols);
        UpdateCombineLayoutReport(report, "directLargeUbStages", 0U, actual.directLargeUbStages);
        UpdateCombineLayoutReport(report, "directLargeRequiredUbBytesOk", 0U, actual.directLargeRequiredUbBytesOk);
        UpdateCombineLayoutReport(report, "usesLaneSplit", 0U, actual.usesLaneSplit);
        UpdateCombineLayoutReport(report, "usesColumnFallback", 0U, actual.usesColumnFallback);
        UpdateCombineLayoutReport(report, "usesOldBusinessHelper", 0U, actual.usesOldBusinessHelper);
    }
    if (combine.combineStopStep == 1U && direct_subtile_selected) {
        const uint64_t c0 = 0U;
        const uint64_t d0 =
            c0 + AlignUpU64(static_cast<uint64_t>(kHostCombineSmallMaxElems) * sizeof(uint16_t), UB_ALIGN);
        const uint64_t fp0 =
            d0 + AlignUpU64(static_cast<uint64_t>(kHostCombineSmallMaxElems) * sizeof(uint16_t), UB_ALIGN);
        const uint64_t scale0 =
            fp0 + AlignUpU64(static_cast<uint64_t>(kHostCombineSmallMaxElems) * sizeof(float), UB_ALIGN);
        const uint64_t c1 =
            scale0 + AlignUpU64(static_cast<uint64_t>(kHostCombineSmallScaleElems) * sizeof(float), UB_ALIGN);
        const uint64_t d1 =
            c1 + AlignUpU64(static_cast<uint64_t>(kHostCombineSmallMaxElems) * sizeof(uint16_t), UB_ALIGN);
        const uint64_t fp1 =
            d1 + AlignUpU64(static_cast<uint64_t>(kHostCombineSmallMaxElems) * sizeof(uint16_t), UB_ALIGN);
        const uint64_t scale1 =
            fp1 + AlignUpU64(static_cast<uint64_t>(kHostCombineSmallMaxElems) * sizeof(float), UB_ALIGN);
        UpdateCombineLayoutReport(report, "directSmallRequiredUbBytes", HostCombineDirectSmallRequiredUbBytes(),
                                  actual.directSmallRequiredUbBytes);
        UpdateCombineLayoutReport(report, "directSmallUbCOffset0", c0, actual.directSmallUbCOffset0);
        UpdateCombineLayoutReport(report, "directSmallUbDOffset0", d0, actual.directSmallUbDOffset0);
        UpdateCombineLayoutReport(report, "directSmallUbFp32Offset0", fp0, actual.directSmallUbFp32Offset0);
        UpdateCombineLayoutReport(report, "directSmallUbScaleOffset0", scale0, actual.directSmallUbScaleOffset0);
        UpdateCombineLayoutReport(report, "directSmallUbCOffset1", c1, actual.directSmallUbCOffset1);
        UpdateCombineLayoutReport(report, "directSmallUbDOffset1", d1, actual.directSmallUbDOffset1);
        UpdateCombineLayoutReport(report, "directSmallUbFp32Offset1", fp1, actual.directSmallUbFp32Offset1);
        UpdateCombineLayoutReport(report, "directSmallUbScaleOffset1", scale1, actual.directSmallUbScaleOffset1);
        UpdateCombineLayoutReport(report, "directSmallTileRows", kHostCombineSmallTokenSubtileRows,
                                  actual.directSmallTileRows);
        UpdateCombineLayoutReport(report, "directSmallTileCols", kHostCombineSmallTokenSubtileCols,
                                  actual.directSmallTileCols);
        UpdateCombineLayoutReport(report, "directSmallN0", kHostCombineSmallTokenSubtileCols, actual.directSmallN0);
        UpdateCombineLayoutReport(report, "directSmallUbStages", kHostCombineDirectSmallUbStages,
                                  actual.directSmallUbStages);
        UpdateCombineLayoutReport(report, "directSmallMaxElems", kHostCombineSmallMaxElems, actual.directSmallMaxElems);
        UpdateCombineLayoutReport(report, "directSmallScaleElems", kHostCombineSmallScaleElems,
                                  actual.directSmallScaleElems);
        UpdateCombineLayoutReport(report, "directSmallRequiredUbBytesOk", HostCombineDirectSmallUbPlanOk() ? 1U : 0U,
                                  actual.directSmallRequiredUbBytesOk);
        UpdateCombineLayoutReport(report, "usesLaneSplit", 0U, actual.usesLaneSplit);
        UpdateCombineLayoutReport(report, "usesColumnFallback", 0U, actual.usesColumnFallback);
        UpdateCombineLayoutReport(report, "usesOldBusinessHelper", 0U, actual.usesOldBusinessHelper);
        UpdateCombineLayoutReport(report, "usesFrontCaseBranch", 0U, actual.usesFrontCaseBranch);
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

CombineReadyReport CheckCombineSmallReady(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                          const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                          const std::vector<uint8_t> &expert_idx)
{
    struct SmallGroupPlan {
        uint32_t currentM = 0;
        uint32_t tileM = 0;
        uint32_t tileN = 0;
        uint32_t coreLoops = 0;
        uint32_t startCoreIdx = 0;
    };

    const auto &combine = build.tiling.combineTiling;
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &gmm2 = build.tiling.gmm2Tiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t local_begin = rank * cfg.expert_per_rank;
    const size_t elems = static_cast<size_t>(cfg.expert_per_rank) * cfg.aiv_num;
    const size_t bytes = elems * sizeof(DispatchFFNCombineCombineGmm2ReadyDebug);
    const uint64_t offset = combine.combineDebugOffset + sizeof(DispatchFFNCombineCombineLayoutDebug);
    std::vector<DispatchFFNCombineCombineGmm2ReadyDebug> actual(elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (!actual.empty() &&
        aclrtMemcpy(actual.data(), bytes, workspace_base + offset, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host combine small ready table copy failed");
    }

    (void)cfg;
    (void)build;
    (void)runtime;
    (void)expert_idx;

    CombineReadyReport report;
    for (uint32_t group = 0; group < cfg.expert_per_rank; ++group) {
        for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
            const size_t idx = static_cast<size_t>(group) * cfg.aiv_num + core;
            if (actual[idx].magic == kHostCombineReadyDebugMagic) {
                ++report.marked_count;
                if (group == 0U && core < 64U) {
                    report.group0_magic_mask |= (1ULL << core);
                }
                if (actual[idx].coreIdx != core) {
                    if (report.marked_coreidx_mismatch_count == 0U) {
                        report.first_marked_core_slot = core;
                        report.first_marked_core_value = actual[idx].coreIdx;
                    }
                    ++report.marked_coreidx_mismatch_count;
                }
            }
            UpdateCombineIndexedReport(report, "magic", idx, kHostCombineReadyDebugMagic, actual[idx].magic);
            UpdateCombineIndexedReport(report, "groupIdx", idx, group, actual[idx].groupIdx);
            UpdateCombineIndexedReport(report, "coreIdx", idx, core, actual[idx].coreIdx);
            UpdateCombineIndexedReport(report, "flagId", idx,
                                       V5_GMM2_TO_COMBINE_HARD_FLAG_BASE + group / CROSS_CORE_FLAG_MAX_SET_COUNT,
                                       actual[idx].flagId);
            UpdateCombineIndexedReport(report, "aivSyncAfterWait", idx, 0U, actual[idx].aivSyncAfterWait);
            UpdateCombineIndexedReport(report, "initialEventMask", idx, kHostCombineInitialEventMask,
                                       actual[idx].initialEventMask);
            UpdateCombineIndexedReport(report, "waitStartSyscntNonzero", idx, 1U,
                                       actual[idx].waitStartSyscnt != 0U ? 1U : 0U);
            UpdateCombineIndexedReport(report, "waitOrder", idx, 1U,
                                       actual[idx].waitStartSyscnt <= actual[idx].waitEndSyscnt ? 1U : 0U);
            UpdateCombineIndexedReport(report, "syncStartIsWaitEnd", idx, 1U,
                                       actual[idx].syncAllStartSyscnt == actual[idx].waitEndSyscnt ? 1U : 0U);
            UpdateCombineIndexedReport(report, "syncEndIsWaitEnd", idx, 1U,
                                       actual[idx].syncAllEndSyscnt == actual[idx].waitEndSyscnt ? 1U : 0U);
            UpdateCombineIndexedReport(report, "marker", idx, 1U, actual[idx].marker);
            ++report.checked_count;
        }
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

CombineReadyReport CheckCombineGmm2Ready(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                         const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                         const std::vector<uint8_t> &expert_idx)
{
    const auto &combine = build.tiling.combineTiling;
    if (combine.gmm2CombineCvMode == kHostGmm2CombineCvModeDirect) {
        CombineReadyReport report;
        report.pass = true;
        return report;
    }
    if (combine.combineStopStep >= 3U) {
        return CheckCombineSmallReady(cfg, build, runtime, workspace_dev, expert_idx);
    }
    const size_t elems = static_cast<size_t>(cfg.expert_per_rank) * cfg.aiv_num;
    const size_t bytes = elems * sizeof(DispatchFFNCombineCombineGmm2ReadyDebug);
    const uint64_t offset = combine.combineDebugOffset + sizeof(DispatchFFNCombineCombineLayoutDebug);
    std::vector<DispatchFFNCombineCombineGmm2ReadyDebug> actual(elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual.data(), bytes, workspace_base + offset, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host combine ready table copy failed");
    }
    CombineReadyReport report;
    const uint32_t expected_aiv_sync =
        HostCombineSmallTokenPath(cfg, static_cast<uint32_t>(runtime.hccl.world_size)) ? 0U : 1U;
    for (uint32_t group = 0; group < cfg.expert_per_rank; ++group) {
        for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
            const size_t idx = static_cast<size_t>(group) * cfg.aiv_num + core;
            if (actual[idx].magic == kHostCombineReadyDebugMagic) {
                ++report.marked_count;
                if (group == 0U && core < 64U) {
                    report.group0_magic_mask |= (1ULL << core);
                }
                if (actual[idx].coreIdx != core) {
                    if (report.marked_coreidx_mismatch_count == 0U) {
                        report.first_marked_core_slot = core;
                        report.first_marked_core_value = actual[idx].coreIdx;
                    }
                    ++report.marked_coreidx_mismatch_count;
                }
            }
            UpdateCombineIndexedReport(report, "magic", idx, kHostCombineReadyDebugMagic, actual[idx].magic);
            UpdateCombineIndexedReport(report, "groupIdx", idx, group, actual[idx].groupIdx);
            UpdateCombineIndexedReport(report, "coreIdx", idx, core, actual[idx].coreIdx);
            UpdateCombineIndexedReport(report, "flagId", idx,
                                       V5_GMM2_TO_COMBINE_HARD_FLAG_BASE + group / CROSS_CORE_FLAG_MAX_SET_COUNT,
                                       actual[idx].flagId);
            UpdateCombineIndexedReport(report, "aivSyncAfterWait", idx, expected_aiv_sync,
                                       actual[idx].aivSyncAfterWait);
            UpdateCombineIndexedReport(report, "initialEventMask", idx, kHostCombineInitialEventMask,
                                       actual[idx].initialEventMask);
            UpdateCombineIndexedReport(report, "waitStartSyscntNonzero", idx, 1U,
                                       actual[idx].waitStartSyscnt != 0U ? 1U : 0U);
            UpdateCombineIndexedReport(report, "waitOrder", idx, 1U,
                                       actual[idx].waitStartSyscnt <= actual[idx].waitEndSyscnt ? 1U : 0U);
            UpdateCombineIndexedReport(report, "syncOrder", idx, 1U,
                                       actual[idx].syncAllStartSyscnt <= actual[idx].syncAllEndSyscnt ? 1U : 0U);
            const uint32_t expected_wait_before_sync =
                expected_aiv_sync != 0U ? (actual[idx].waitEndSyscnt <= actual[idx].syncAllStartSyscnt ? 1U : 0U) :
                                          (actual[idx].syncAllStartSyscnt == actual[idx].waitEndSyscnt &&
                                                   actual[idx].syncAllEndSyscnt == actual[idx].waitEndSyscnt ?
                                               1U :
                                               0U);
            UpdateCombineIndexedReport(report, "waitBeforeSync", idx, 1U, expected_wait_before_sync);
            UpdateCombineIndexedReport(report, "marker", idx, 1U, actual[idx].marker);
            ++report.checked_count;
        }
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

CombineMetaReport CheckCombineMeta(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                   const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                   const std::vector<uint8_t> &expert_idx)
{
    const auto &combine = build.tiling.combineTiling;
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t local_begin = rank * cfg.expert_per_rank;
    const bool direct_large_metadata =
        build.tiling.combineTiling.combineStopStep >= 3U && !HostCombineSmallTokenPath(cfg, rank_size);
    const size_t ready_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * cfg.aiv_num * sizeof(DispatchFFNCombineCombineGmm2ReadyDebug);
    const uint64_t meta_offset =
        combine.combineDebugOffset + sizeof(DispatchFFNCombineCombineLayoutDebug) + ready_bytes;
    const size_t elems = static_cast<size_t>(cfg.expert_per_rank) * rank_size;
    const size_t bytes = elems * sizeof(DispatchFFNCombineCombineMetadataDebug);
    std::vector<DispatchFFNCombineCombineMetadataDebug> actual(elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual.data(), bytes, workspace_base + meta_offset, bytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS) {
        throw std::runtime_error("device->host combine metadata table copy failed");
    }

    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);
    CombineMetaReport report;
    uint32_t group_base = 0;
    for (uint32_t group = 0; group < cfg.expert_per_rank; ++group) {
        uint32_t current_m = 0;
        uint32_t cumsum_before_src = 0;
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            const uint32_t global_expert = local_begin + group;
            const size_t token_idx = static_cast<size_t>(src_rank) * front.expertNumAligned + global_expert;
            const uint32_t rows_raw = static_cast<uint32_t>(expected_tokens[token_idx]);
            const uint32_t src_row_offset = group_base + cumsum_before_src;
            const uint32_t rows = ExpectedRowsClipped(src_row_offset, rows_raw, cfg.max_output_size);
            const uint32_t skip_reason = src_row_offset >= cfg.max_output_size ? 1U : 0U;
            const uint32_t dst_row_offset =
                ExpectedSrcRankDstRowOffset(expected_tokens, front.expertNumAligned, src_rank, global_expert);
            const size_t idx = static_cast<size_t>(group) * rank_size + src_rank;
            UpdateCombineIndexedReport(report, "magic", idx, kHostCombineMetaDebugMagic, actual[idx].magic);
            UpdateCombineIndexedReport(report, "groupIdx", idx, group, actual[idx].groupIdx);
            UpdateCombineIndexedReport(report, "srcRank", idx, src_rank, actual[idx].srcRank);
            if (direct_large_metadata) {
                UpdateCombineIndexedReport(report, "coreIdx", idx, src_rank % cfg.aiv_num, actual[idx].coreIdx);
                UpdateCombineIndexedReport(report, "coreNum", idx, cfg.aiv_num, actual[idx].coreNum);
                ++report.owner_checked_count;
            }
            UpdateCombineIndexedReport(report, "groupBase", idx, group_base, actual[idx].groupBase);
            UpdateCombineIndexedReport(report, "rowsRaw", idx, rows_raw, actual[idx].rowsRaw);
            UpdateCombineIndexedReport(report, "rows", idx, rows, actual[idx].rows);
            UpdateCombineIndexedReport(report, "srcRowOffset", idx, src_row_offset, actual[idx].srcRowOffset);
            UpdateCombineIndexedReport(report, "dstRowOffset", idx, dst_row_offset, actual[idx].dstRowOffset);
            UpdateCombineIndexedReport(report, "cumsumBeforeSrc", idx, cumsum_before_src, actual[idx].cumsumBeforeSrc);
            UpdateCombineIndexedReport(report, "clipped", idx, rows != rows_raw ? 1U : 0U, actual[idx].clipped);
            UpdateCombineIndexedReport(report, "skipReason", idx, skip_reason, actual[idx].skipReason);
            UpdateCombineIndexedReport(report, "marker", idx, 1U, actual[idx].marker);
            current_m += rows_raw;
            cumsum_before_src += rows_raw;
            ++report.segment_count;
        }
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            const size_t idx = static_cast<size_t>(group) * rank_size + src_rank;
            UpdateCombineIndexedReport(report, "groupBaseAfter", idx, group_base + current_m,
                                       actual[idx].groupBaseAfter);
        }
        group_base += current_m;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

CombineLoadReport CheckCombineSmallLoad(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                        const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                        const std::vector<uint8_t> &expert_idx)
{
    struct SmallGroupPlan {
        uint32_t groupBase = 0;
        uint32_t currentM = 0;
        uint32_t tileM = 0;
        uint32_t tileN = 0;
        uint32_t coreLoops = 0;
        uint32_t startCoreIdx = 0;
    };
    struct SmallLoadExpected {
        uint32_t marker = 0;
        uint32_t firstLoopIdx = kHostDispatchInvalidTask;
        uint32_t firstSrcRow = 0;
        uint32_t firstCol = 0;
        uint32_t firstRows = 0;
        uint32_t firstScaleOffset = 0;
        uint32_t totalCLoadBytes = 0;
        uint32_t firstBufferId = 0;
        uint32_t scaleLoadCount = 0;
        uint32_t scaleCacheHitCount = 0;
        uint32_t loadCount = 0;
        uint32_t firstScaleBits = 0;
        uint32_t firstCBits = 0;
        uint32_t lastCBits = 0;
    };

    const auto &combine = build.tiling.combineTiling;
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &gmm2 = build.tiling.gmm2Tiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t local_begin = rank * cfg.expert_per_rank;
    const size_t ready_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * cfg.aiv_num * sizeof(DispatchFFNCombineCombineGmm2ReadyDebug);
    const size_t meta_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * rank_size * sizeof(DispatchFFNCombineCombineMetadataDebug);
    const size_t io_slots = HostCombineDebugIoSlots(rank_size, cfg.aiv_num);
    const uint64_t load_offset =
        combine.combineDebugOffset + sizeof(DispatchFFNCombineCombineLayoutDebug) + ready_bytes + meta_bytes;
    const size_t table_elems = static_cast<size_t>(cfg.expert_per_rank) * io_slots;
    const size_t bytes = table_elems * sizeof(DispatchFFNCombineCombineLoadDebug);
    if (combine.combineDebugBytes < load_offset - combine.combineDebugOffset + bytes) {
        throw std::runtime_error("combine debug area is too small for small load table");
    }
    std::vector<DispatchFFNCombineCombineLoadDebug> actual(table_elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (!actual.empty() && aclrtMemcpy(actual.data(), bytes, workspace_base + load_offset, bytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host combine small load table copy failed");
    }

    const size_t gmm2_output_elems = static_cast<size_t>(cfg.max_output_size) * cfg.k;
    const size_t gmm2_output_bytes = gmm2_output_elems * sizeof(uint16_t);
    const size_t scale_bytes = static_cast<size_t>(cfg.max_output_size) * sizeof(float);
    std::vector<uint16_t> gmm2_output(gmm2_output_elems, 0);
    std::vector<uint8_t> scale_bytes_host(scale_bytes, 0);
    if ((!gmm2_output.empty() &&
         aclrtMemcpy(gmm2_output.data(), gmm2_output_bytes, workspace_base + combine.gmm2OutputOffset,
                     gmm2_output_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) ||
        (!scale_bytes_host.empty() &&
         aclrtMemcpy(scale_bytes_host.data(), scale_bytes, workspace_base + combine.perTokenScale2Offset, scale_bytes,
                     ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)) {
        throw std::runtime_error("device->host combine small load source buffer copy failed");
    }

    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);
    std::vector<SmallGroupPlan> group_plan(cfg.expert_per_rank);
    uint32_t group_base = 0;
    uint32_t start_core_idx = 0;
    for (uint32_t group = 0; group < cfg.expert_per_rank; ++group) {
        const uint32_t global_expert = local_begin + group;
        uint32_t current_m_raw = 0;
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            current_m_raw += static_cast<uint32_t>(
                expected_tokens[static_cast<size_t>(src_rank) * front.expertNumAligned + global_expert]);
        }
        const uint32_t remaining = group_base >= cfg.max_output_size ? 0U : cfg.max_output_size - group_base;
        SmallGroupPlan &plan = group_plan[group];
        plan.groupBase = group_base;
        plan.currentM = current_m_raw > remaining ? remaining : current_m_raw;
        plan.tileM = CeilDivU32(plan.currentM, gmm2.l1TileM);
        plan.tileN = CeilDivU32(cfg.k, gmm2.l1TileN);
        plan.coreLoops = plan.tileM * plan.tileN;
        plan.startCoreIdx = start_core_idx;
        start_core_idx = build.block_dim == 0U ? 0U : (start_core_idx + plan.coreLoops) % build.block_dim;
        group_base += plan.currentM;
    }

    const size_t expected_elems = static_cast<size_t>(cfg.expert_per_rank) * cfg.aiv_num;
    std::vector<SmallLoadExpected> expected(expected_elems);
    std::vector<uint32_t> pingpong(cfg.aiv_num, 0U);
    std::vector<std::array<uint32_t, 2>> scale_source(cfg.aiv_num);
    for (auto &source : scale_source) {
        source = {kHostDispatchInvalidTask, kHostDispatchInvalidTask};
    }
    const uint32_t aic_core_num = build.block_dim;
    for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
        const uint32_t aic_core_idx = aic_core_num == 0U ? 0U : core % aic_core_num;
        const uint32_t aiv_sub_core_idx = (aic_core_num == 0U || core < aic_core_num) ? 0U : 1U;
        for (uint32_t group = 0; group < cfg.expert_per_rank; ++group) {
            const SmallGroupPlan &plan = group_plan[group];
            const uint32_t start_loop_idx =
                aic_core_num == 0U ? 0U : HostGmmCommonStartLoopIdx(aic_core_idx, aic_core_num, plan.startCoreIdx);
            SmallLoadExpected &entry = expected[static_cast<size_t>(group) * cfg.aiv_num + core];
            for (uint32_t loop = start_loop_idx; aic_core_num != 0U && loop < plan.coreLoops; loop += aic_core_num) {
                uint32_t block_m = 0;
                uint32_t block_n = 0;
                GetGmm1BlockCoordMN(loop, plan.tileM, plan.tileN, block_m, block_n);
                uint32_t actual_m = 0;
                uint32_t actual_n = 0;
                GetGmmCommonActualBlockShapeMN(plan.currentM, cfg.k, gmm2.l1TileM, gmm2.l1TileN, plan.tileM, plan.tileN,
                                               block_m, block_n, actual_m, actual_n);
                const uint32_t subtile_count = CeilDivU32(actual_m, kHostCombineSmallTokenSubtileRows);
                const uint32_t first_subtile = HostCombineSubtileFirstIndex(subtile_count, aiv_sub_core_idx);
                const uint32_t assigned_subtiles =
                    HostCombineSubtileAssignedCount(subtile_count, aiv_sub_core_idx);
                for (uint32_t subtile = 0; subtile < assigned_subtiles; ++subtile) {
                    const uint32_t subtile_idx = first_subtile + subtile;
                    const uint32_t row_in_tile = subtile_idx * kHostCombineSmallTokenSubtileRows;
                    if (row_in_tile >= actual_m) {
                        continue;
                    }
                    const uint32_t rows = std::min<uint32_t>(kHostCombineSmallTokenSubtileRows, actual_m - row_in_tile);
                    const uint32_t src_row = plan.groupBase + block_m * gmm2.l1TileM + row_in_tile;
                    const uint32_t col = block_n * gmm2.l1TileN;
                    const uint32_t buffer_id = pingpong[core];
                    pingpong[core] = (pingpong[core] + 1U) % 2U;
                    const bool cache_hit = scale_source[core][buffer_id] == src_row;
                    if (!cache_hit) {
                        scale_source[core][buffer_id] = src_row;
                    }
                    if (entry.loadCount == 0U) {
                        float scale_value = 0.0f;
                        std::memcpy(&scale_value,
                                    scale_bytes_host.data() + static_cast<size_t>(src_row) * sizeof(float),
                                    sizeof(float));
                        entry.marker = 1U;
                        entry.firstLoopIdx = loop;
                        entry.firstSrcRow = src_row;
                        entry.firstCol = col;
                        entry.firstRows = rows;
                        entry.firstScaleOffset = src_row;
                        entry.firstBufferId = buffer_id;
                        entry.firstScaleBits = FloatBitsHost(scale_value);
                        entry.firstCBits =
                            FloatBitsHost(Fp16ToFloat(gmm2_output[static_cast<size_t>(src_row) * cfg.k + col]));
                    }
                    entry.totalCLoadBytes += rows * actual_n * sizeof(uint16_t);
                    entry.scaleLoadCount += cache_hit ? 0U : 1U;
                    entry.scaleCacheHitCount += cache_hit ? 1U : 0U;
                    ++entry.loadCount;
                    const size_t last_idx = static_cast<size_t>(src_row + rows - 1U) * cfg.k + col + actual_n - 1U;
                    entry.lastCBits = FloatBitsHost(Fp16ToFloat(gmm2_output[last_idx]));
                }
            }
        }
    }

    CombineLoadReport report;
    for (uint32_t group = 0; group < cfg.expert_per_rank; ++group) {
        for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
            const size_t idx = static_cast<size_t>(group) * cfg.aiv_num + core;
            const size_t actual_idx = static_cast<size_t>(group) * io_slots + core;
            const SmallLoadExpected &exp = expected[idx];
            if (exp.marker == 0U) {
                UpdateCombineIndexedReport(report, "magic", idx, 0U, actual[actual_idx].magic);
                UpdateCombineIndexedReport(report, "marker", idx, 0U, actual[actual_idx].marker);
                ++report.segment_count;
                continue;
            }
            const uint32_t expected_mode = exp.firstBufferId |
                                           (exp.scaleLoadCount << kHostCombineSmallLoadModeScaleLoadShift) |
                                           (exp.scaleCacheHitCount << kHostCombineSmallLoadModeScaleCacheHitShift) |
                                           (exp.loadCount << kHostCombineSmallLoadModeLoadCountShift);
            UpdateCombineIndexedReport(report, "magic", idx, kHostCombineLoadDebugMagic, actual[actual_idx].magic);
            UpdateCombineIndexedReport(report, "groupIdx", idx, group, actual[actual_idx].groupIdx);
            UpdateCombineIndexedReport(report, "srcRank", idx, exp.firstLoopIdx, actual[actual_idx].srcRank);
            UpdateCombineIndexedReport(report, "coreIdx", idx, core, actual[actual_idx].coreIdx);
            UpdateCombineIndexedReport(report, "coreNum", idx, cfg.aiv_num, actual[actual_idx].coreNum);
            UpdateCombineIndexedReport(report, "srcRow", idx, exp.firstSrcRow, actual[actual_idx].srcRow);
            UpdateCombineIndexedReport(report, "dstRow", idx, exp.firstCol, actual[actual_idx].dstRow);
            UpdateCombineIndexedReport(report, "rows", idx, exp.firstRows, actual[actual_idx].rows);
            UpdateCombineIndexedReport(report, "cLoadBytes", idx, exp.totalCLoadBytes, actual[actual_idx].cLoadBytes);
            UpdateCombineIndexedReport(report, "scaleOffsetBytes", idx, exp.firstScaleOffset * sizeof(float),
                                       actual[actual_idx].scaleOffsetBytes);
            UpdateCombineIndexedReport(report, "scaleBits", idx, exp.firstScaleBits, actual[actual_idx].scaleBits);
            UpdateCombineIndexedReport(report, "cFirstBits", idx, exp.firstCBits, actual[actual_idx].cFirstBits);
            UpdateCombineIndexedReport(report, "cLastBits", idx, exp.lastCBits, actual[actual_idx].cLastBits);
            UpdateCombineIndexedReport(report, "scaleReadMode", idx, expected_mode, actual[actual_idx].scaleReadMode);
            UpdateCombineIndexedReport(report, "marker", idx, 1U, actual[actual_idx].marker);
            report.loaded_count += exp.loadCount;
            ++report.segment_count;
        }
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

CombineLoadReport CheckCombineLoad(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                   const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                   const std::vector<uint8_t> &expert_idx)
{
    const auto &combine = build.tiling.combineTiling;
    if (combine.gmm2CombineCvMode == kHostGmm2CombineCvModeDirect) {
        CombineLoadReport report;
        report.pass = true;
        return report;
    }
    if (combine.combineStopStep >= 4U) {
        return CheckCombineSmallLoad(cfg, build, runtime, workspace_dev, expert_idx);
    }
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t local_begin = rank * cfg.expert_per_rank;
    const size_t ready_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * cfg.aiv_num * sizeof(DispatchFFNCombineCombineGmm2ReadyDebug);
    const size_t meta_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * rank_size * sizeof(DispatchFFNCombineCombineMetadataDebug);
    const size_t io_slots = HostCombineDebugIoSlots(rank_size, cfg.aiv_num);
    const uint64_t load_offset =
        combine.combineDebugOffset + sizeof(DispatchFFNCombineCombineLayoutDebug) + ready_bytes + meta_bytes;
    const size_t table_elems = static_cast<size_t>(cfg.expert_per_rank) * io_slots;
    const size_t bytes = table_elems * sizeof(DispatchFFNCombineCombineLoadDebug);
    std::vector<DispatchFFNCombineCombineLoadDebug> actual(table_elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual.data(), bytes, workspace_base + load_offset, bytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS) {
        throw std::runtime_error("device->host combine load table copy failed");
    }

    const size_t gmm2_output_elems = static_cast<size_t>(cfg.max_output_size) * cfg.k;
    const size_t gmm2_output_bytes = gmm2_output_elems * sizeof(uint16_t);
    const size_t scale_bytes = static_cast<size_t>(cfg.max_output_size) * sizeof(float);
    std::vector<uint16_t> gmm2_output(gmm2_output_elems, 0);
    std::vector<uint8_t> scale_bytes_host(scale_bytes, 0);
    if ((!gmm2_output.empty() &&
         aclrtMemcpy(gmm2_output.data(), gmm2_output_bytes, workspace_base + combine.gmm2OutputOffset,
                     gmm2_output_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) ||
        (!scale_bytes_host.empty() &&
         aclrtMemcpy(scale_bytes_host.data(), scale_bytes, workspace_base + combine.perTokenScale2Offset, scale_bytes,
                     ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)) {
        throw std::runtime_error("device->host combine load source buffer copy failed");
    }

    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);

    CombineLoadReport report;
    uint32_t group_base = 0;
    for (uint32_t group = 0; group < cfg.expert_per_rank; ++group) {
        uint32_t current_m = 0;
        uint32_t cumsum_before_src = 0;
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            const uint32_t global_expert = local_begin + group;
            const size_t token_idx = static_cast<size_t>(src_rank) * front.expertNumAligned + global_expert;
            const uint32_t rows_raw = static_cast<uint32_t>(expected_tokens[token_idx]);
            const uint32_t src_row = group_base + cumsum_before_src;
            const uint32_t rows = ExpectedRowsClipped(src_row, rows_raw, cfg.max_output_size);
            const uint32_t dst_row =
                ExpectedSrcRankDstRowOffset(expected_tokens, front.expertNumAligned, src_rank, global_expert);
            uint32_t scale_bits = 0;
            uint32_t c_first_bits = 0;
            uint32_t c_last_bits = 0;
            if (rows != 0U) {
                float scale_value = 0.0f;
                std::memcpy(&scale_value, scale_bytes_host.data() + static_cast<size_t>(src_row) * sizeof(float),
                            sizeof(float));
                scale_bits = FloatBitsHost(scale_value);
                c_first_bits = FloatBitsHost(Fp16ToFloat(gmm2_output[static_cast<size_t>(src_row) * cfg.k]));
                c_last_bits =
                    FloatBitsHost(Fp16ToFloat(gmm2_output[static_cast<size_t>(src_row) * cfg.k + cfg.k - 1U]));
                ++report.loaded_count;
            }
            const size_t idx = static_cast<size_t>(group) * rank_size + src_rank;
            const size_t actual_idx = static_cast<size_t>(group) * io_slots + src_rank;
            UpdateCombineIndexedReport(report, "magic", idx, kHostCombineLoadDebugMagic, actual[actual_idx].magic);
            UpdateCombineIndexedReport(report, "groupIdx", idx, group, actual[actual_idx].groupIdx);
            UpdateCombineIndexedReport(report, "srcRank", idx, src_rank, actual[actual_idx].srcRank);
            UpdateCombineIndexedReport(report, "coreIdx", idx, src_rank % cfg.aiv_num, actual[actual_idx].coreIdx);
            UpdateCombineIndexedReport(report, "coreNum", idx, cfg.aiv_num, actual[actual_idx].coreNum);
            UpdateCombineIndexedReport(report, "srcRow", idx, src_row, actual[actual_idx].srcRow);
            UpdateCombineIndexedReport(report, "dstRow", idx, dst_row, actual[actual_idx].dstRow);
            UpdateCombineIndexedReport(report, "rows", idx, rows, actual[actual_idx].rows);
            UpdateCombineIndexedReport(report, "cLoadBytes", idx, rows == 0U ? 0U : cfg.k * sizeof(uint16_t),
                                       actual[actual_idx].cLoadBytes);
            UpdateCombineIndexedReport(report, "scaleOffsetBytes", idx, rows == 0U ? 0U : src_row * sizeof(float),
                                       actual[actual_idx].scaleOffsetBytes);
            UpdateCombineIndexedReport(report, "scaleBits", idx, scale_bits, actual[actual_idx].scaleBits);
            UpdateCombineIndexedReport(report, "cFirstBits", idx, c_first_bits, actual[actual_idx].cFirstBits);
            UpdateCombineIndexedReport(report, "cLastBits", idx, c_last_bits, actual[actual_idx].cLastBits);
            UpdateCombineIndexedReport(report, "scaleReadMode", idx, kHostCombineScaleReadDirectScalar,
                                       actual[actual_idx].scaleReadMode);
            UpdateCombineIndexedReport(report, "marker", idx, 1U, actual[actual_idx].marker);
            current_m += rows_raw;
            cumsum_before_src += rows_raw;
            ++report.segment_count;
        }
        group_base += current_m;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

CombineDequantReport CheckCombineSmallDequant(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                              const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                              const std::vector<uint8_t> &expert_idx)
{
    struct SmallGroupPlan {
        uint32_t groupBase = 0;
        uint32_t currentM = 0;
        uint32_t tileM = 0;
        uint32_t tileN = 0;
        uint32_t coreLoops = 0;
        uint32_t startCoreIdx = 0;
    };
    struct SmallDequantExpected {
        uint32_t marker = 0;
        uint32_t firstLoopIdx = kHostDispatchInvalidTask;
        uint32_t firstSrcRow = 0;
        uint32_t firstRows = 0;
        uint32_t scaleBits = 0;
        uint32_t beforeBits = 0;
        uint32_t afterFirstBits = 0;
        uint32_t afterLastBits = 0;
        uint32_t dFirstBits = 0;
        uint32_t dLastBits = 0;
        uint32_t loadCount = 0;
        uint32_t scaleLoadCount = 0;
        uint32_t scaleCacheHitCount = 0;
        uint32_t mulsRowCount = 0;
    };

    const auto &combine = build.tiling.combineTiling;
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &gmm2 = build.tiling.gmm2Tiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t local_begin = rank * cfg.expert_per_rank;
    const size_t ready_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * cfg.aiv_num * sizeof(DispatchFFNCombineCombineGmm2ReadyDebug);
    const size_t meta_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * rank_size * sizeof(DispatchFFNCombineCombineMetadataDebug);
    const size_t io_slots = HostCombineDebugIoSlots(rank_size, cfg.aiv_num);
    const size_t load_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * io_slots * sizeof(DispatchFFNCombineCombineLoadDebug);
    const uint64_t dequant_offset = combine.combineDebugOffset + sizeof(DispatchFFNCombineCombineLayoutDebug) +
                                    ready_bytes + meta_bytes + load_bytes;
    const size_t elems = static_cast<size_t>(cfg.expert_per_rank) * cfg.aiv_num;
    const size_t table_elems = static_cast<size_t>(cfg.expert_per_rank) * io_slots;
    const size_t bytes = table_elems * sizeof(DispatchFFNCombineCombineDequantDebug);
    if (combine.combineDebugBytes < dequant_offset - combine.combineDebugOffset + bytes) {
        throw std::runtime_error("combine debug area is too small for small dequant table");
    }
    std::vector<DispatchFFNCombineCombineDequantDebug> actual(table_elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (!actual.empty() && aclrtMemcpy(actual.data(), bytes, workspace_base + dequant_offset, bytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host combine small dequant table copy failed");
    }

    const size_t gmm2_output_elems = static_cast<size_t>(cfg.max_output_size) * cfg.k;
    const size_t gmm2_output_bytes = gmm2_output_elems * sizeof(uint16_t);
    const size_t scale_bytes = static_cast<size_t>(cfg.max_output_size) * sizeof(float);
    std::vector<uint16_t> gmm2_output(gmm2_output_elems, 0);
    std::vector<uint8_t> scale_bytes_host(scale_bytes, 0);
    if ((!gmm2_output.empty() &&
         aclrtMemcpy(gmm2_output.data(), gmm2_output_bytes, workspace_base + combine.gmm2OutputOffset,
                     gmm2_output_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) ||
        (!scale_bytes_host.empty() &&
         aclrtMemcpy(scale_bytes_host.data(), scale_bytes, workspace_base + combine.perTokenScale2Offset, scale_bytes,
                     ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)) {
        throw std::runtime_error("device->host combine small dequant source buffer copy failed");
    }

    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);
    std::vector<SmallGroupPlan> group_plan(cfg.expert_per_rank);
    uint32_t group_base = 0;
    uint32_t start_core_idx = 0;
    for (uint32_t group = 0; group < cfg.expert_per_rank; ++group) {
        const uint32_t global_expert = local_begin + group;
        uint32_t current_m_raw = 0;
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            current_m_raw += static_cast<uint32_t>(
                expected_tokens[static_cast<size_t>(src_rank) * front.expertNumAligned + global_expert]);
        }
        const uint32_t remaining = group_base >= cfg.max_output_size ? 0U : cfg.max_output_size - group_base;
        SmallGroupPlan &plan = group_plan[group];
        plan.groupBase = group_base;
        plan.currentM = current_m_raw > remaining ? remaining : current_m_raw;
        plan.tileM = CeilDivU32(plan.currentM, gmm2.l1TileM);
        plan.tileN = CeilDivU32(cfg.k, gmm2.l1TileN);
        plan.coreLoops = plan.tileM * plan.tileN;
        plan.startCoreIdx = start_core_idx;
        start_core_idx = build.block_dim == 0U ? 0U : (start_core_idx + plan.coreLoops) % build.block_dim;
        group_base += plan.currentM;
    }

    std::vector<SmallDequantExpected> expected(elems);
    std::vector<uint32_t> pingpong(cfg.aiv_num, 0U);
    std::vector<std::array<uint32_t, 2>> scale_source(cfg.aiv_num);
    for (auto &source : scale_source) {
        source = {kHostDispatchInvalidTask, kHostDispatchInvalidTask};
    }
    const uint32_t aic_core_num = build.block_dim;
    for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
        const uint32_t aic_core_idx = aic_core_num == 0U ? 0U : core % aic_core_num;
        const uint32_t aiv_sub_core_idx = (aic_core_num == 0U || core < aic_core_num) ? 0U : 1U;
        for (uint32_t group = 0; group < cfg.expert_per_rank; ++group) {
            const SmallGroupPlan &plan = group_plan[group];
            const uint32_t start_loop_idx =
                aic_core_num == 0U ? 0U : HostGmmCommonStartLoopIdx(aic_core_idx, aic_core_num, plan.startCoreIdx);
            SmallDequantExpected &entry = expected[static_cast<size_t>(group) * cfg.aiv_num + core];
            for (uint32_t loop = start_loop_idx; aic_core_num != 0U && loop < plan.coreLoops; loop += aic_core_num) {
                uint32_t block_m = 0;
                uint32_t block_n = 0;
                GetGmm1BlockCoordMN(loop, plan.tileM, plan.tileN, block_m, block_n);
                uint32_t actual_m = 0;
                uint32_t actual_n = 0;
                GetGmmCommonActualBlockShapeMN(plan.currentM, cfg.k, gmm2.l1TileM, gmm2.l1TileN, plan.tileM, plan.tileN,
                                               block_m, block_n, actual_m, actual_n);
                const uint32_t subtile_count = CeilDivU32(actual_m, kHostCombineSmallTokenSubtileRows);
                const uint32_t first_subtile = HostCombineSubtileFirstIndex(subtile_count, aiv_sub_core_idx);
                const uint32_t assigned_subtiles =
                    HostCombineSubtileAssignedCount(subtile_count, aiv_sub_core_idx);
                for (uint32_t subtile = 0; subtile < assigned_subtiles; ++subtile) {
                    const uint32_t subtile_idx = first_subtile + subtile;
                    const uint32_t row_in_tile = subtile_idx * kHostCombineSmallTokenSubtileRows;
                    if (row_in_tile >= actual_m) {
                        continue;
                    }
                    const uint32_t rows = std::min<uint32_t>(kHostCombineSmallTokenSubtileRows, actual_m - row_in_tile);
                    const uint32_t src_row = plan.groupBase + block_m * gmm2.l1TileM + row_in_tile;
                    const uint32_t col = block_n * gmm2.l1TileN;
                    const uint32_t buffer_id = pingpong[core];
                    pingpong[core] = (pingpong[core] + 1U) % 2U;
                    const bool cache_hit = scale_source[core][buffer_id] == src_row;
                    if (!cache_hit) {
                        scale_source[core][buffer_id] = src_row;
                    }
                    float first_scale = 0.0f;
                    std::memcpy(&first_scale, scale_bytes_host.data() + static_cast<size_t>(src_row) * sizeof(float),
                                sizeof(float));
                    const float before_first = Fp16ToFloat(gmm2_output[static_cast<size_t>(src_row) * cfg.k + col]);
                    const float after_first = before_first * first_scale;
                    const size_t last_idx = static_cast<size_t>(src_row + rows - 1U) * cfg.k + col + actual_n - 1U;
                    float last_scale = 0.0f;
                    std::memcpy(&last_scale,
                                scale_bytes_host.data() + static_cast<size_t>(src_row + rows - 1U) * sizeof(float),
                                sizeof(float));
                    const float after_last = Fp16ToFloat(gmm2_output[last_idx]) * last_scale;
                    const uint16_t d_first = FloatToHalfRoundToNearestEven(after_first);
                    const uint16_t d_last = FloatToHalfRoundToNearestEven(after_last);
                    if (entry.loadCount == 0U) {
                        entry.marker = 1U;
                        entry.firstLoopIdx = loop;
                        entry.firstSrcRow = src_row;
                        entry.firstRows = rows;
                        entry.scaleBits = FloatBitsHost(first_scale);
                        entry.beforeBits = FloatBitsHost(before_first);
                        entry.afterFirstBits = FloatBitsHost(after_first);
                        entry.dFirstBits = FloatBitsHost(Fp16ToFloat(d_first));
                    }
                    entry.afterLastBits = FloatBitsHost(after_last);
                    entry.dLastBits = FloatBitsHost(Fp16ToFloat(d_last));
                    entry.scaleLoadCount += cache_hit ? 0U : 1U;
                    entry.scaleCacheHitCount += cache_hit ? 1U : 0U;
                    ++entry.loadCount;
                    entry.mulsRowCount += rows;
                }
            }
        }
    }

    CombineDequantReport report;
    for (uint32_t group = 0; group < cfg.expert_per_rank; ++group) {
        for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
            const size_t idx = static_cast<size_t>(group) * cfg.aiv_num + core;
            const size_t actual_idx = static_cast<size_t>(group) * io_slots + core;
            const SmallDequantExpected &exp = expected[idx];
            if (exp.marker == 0U) {
                UpdateCombineIndexedReport(report, "magic", idx, 0U, actual[actual_idx].magic);
                UpdateCombineIndexedReport(report, "marker", idx, 0U, actual[actual_idx].marker);
                ++report.segment_count;
                continue;
            }
            const uint32_t expected_op_counts =
                exp.loadCount | (exp.scaleLoadCount << kHostCombineSmallLoadModeScaleLoadShift) |
                (exp.scaleCacheHitCount << kHostCombineSmallLoadModeScaleCacheHitShift) |
                (exp.mulsRowCount << kHostCombineSmallDequantOpMulsRowsShift);
            UpdateCombineIndexedReport(report, "magic", idx, kHostCombineDequantDebugMagic, actual[actual_idx].magic);
            UpdateCombineIndexedReport(report, "groupIdx", idx, group, actual[actual_idx].groupIdx);
            UpdateCombineIndexedReport(report, "srcRank", idx, exp.firstLoopIdx, actual[actual_idx].srcRank);
            UpdateCombineIndexedReport(report, "coreIdx", idx, core, actual[actual_idx].coreIdx);
            UpdateCombineIndexedReport(report, "coreNum", idx, cfg.aiv_num, actual[actual_idx].coreNum);
            UpdateCombineIndexedReport(report, "srcRow", idx, exp.firstSrcRow, actual[actual_idx].srcRow);
            UpdateCombineIndexedReport(report, "rows", idx, exp.firstRows, actual[actual_idx].rows);
            UpdateCombineIndexedReport(report, "scaleBits", idx, exp.scaleBits, actual[actual_idx].scaleBits);
            UpdateCombineIndexedReport(report, "fp32BeforeFirstBits", idx, exp.beforeBits,
                                       actual[actual_idx].fp32BeforeFirstBits);
            UpdateCombineIndexedReport(report, "fp32AfterFirstBits", idx, exp.afterFirstBits,
                                       actual[actual_idx].fp32AfterFirstBits);
            UpdateCombineIndexedReport(report, "fp32AfterLastBits", idx, exp.afterLastBits,
                                       actual[actual_idx].fp32AfterLastBits);
            UpdateCombineIndexedReport(report, "dFirstBits", idx, exp.dFirstBits, actual[actual_idx].dFirstBits);
            UpdateCombineIndexedReport(report, "dLastBits", idx, exp.dLastBits, actual[actual_idx].dLastBits);
            UpdateCombineIndexedReport(report, "opCounts", idx, expected_op_counts, actual[actual_idx].opCounts);
            UpdateCombineIndexedReport(report, "marker", idx, 1U, actual[actual_idx].marker);
            report.dequant_count += exp.loadCount;
            ++report.segment_count;
        }
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

CombineDequantReport CheckCombineDequant(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                         const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                         const std::vector<uint8_t> &expert_idx)
{
    const auto &combine = build.tiling.combineTiling;
    if (combine.gmm2CombineCvMode == kHostGmm2CombineCvModeDirect) {
        CombineDequantReport report;
        report.pass = true;
        return report;
    }
    if (combine.combineStopStep >= 5U) {
        return CheckCombineSmallDequant(cfg, build, runtime, workspace_dev, expert_idx);
    }
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t local_begin = rank * cfg.expert_per_rank;
    const size_t ready_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * cfg.aiv_num * sizeof(DispatchFFNCombineCombineGmm2ReadyDebug);
    const size_t meta_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * rank_size * sizeof(DispatchFFNCombineCombineMetadataDebug);
    const size_t io_slots = HostCombineDebugIoSlots(rank_size, cfg.aiv_num);
    const size_t load_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * io_slots * sizeof(DispatchFFNCombineCombineLoadDebug);
    const uint64_t dequant_offset = combine.combineDebugOffset + sizeof(DispatchFFNCombineCombineLayoutDebug) +
                                    ready_bytes + meta_bytes + load_bytes;
    const size_t table_elems = static_cast<size_t>(cfg.expert_per_rank) * io_slots;
    const size_t bytes = table_elems * sizeof(DispatchFFNCombineCombineDequantDebug);
    std::vector<DispatchFFNCombineCombineDequantDebug> actual(table_elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual.data(), bytes, workspace_base + dequant_offset, bytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS) {
        throw std::runtime_error("device->host combine dequant table copy failed");
    }

    const size_t gmm2_output_elems = static_cast<size_t>(cfg.max_output_size) * cfg.k;
    const size_t gmm2_output_bytes = gmm2_output_elems * sizeof(uint16_t);
    const size_t scale_bytes = static_cast<size_t>(cfg.max_output_size) * sizeof(float);
    std::vector<uint16_t> gmm2_output(gmm2_output_elems, 0);
    std::vector<uint8_t> scale_bytes_host(scale_bytes, 0);
    if ((!gmm2_output.empty() &&
         aclrtMemcpy(gmm2_output.data(), gmm2_output_bytes, workspace_base + combine.gmm2OutputOffset,
                     gmm2_output_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) ||
        (!scale_bytes_host.empty() &&
         aclrtMemcpy(scale_bytes_host.data(), scale_bytes, workspace_base + combine.perTokenScale2Offset, scale_bytes,
                     ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)) {
        throw std::runtime_error("device->host combine dequant source buffer copy failed");
    }

    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);

    CombineDequantReport report;
    uint32_t group_base = 0;
    for (uint32_t group = 0; group < cfg.expert_per_rank; ++group) {
        uint32_t current_m = 0;
        uint32_t cumsum_before_src = 0;
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            const uint32_t global_expert = local_begin + group;
            const size_t token_idx = static_cast<size_t>(src_rank) * front.expertNumAligned + global_expert;
            const uint32_t rows_raw = static_cast<uint32_t>(expected_tokens[token_idx]);
            const uint32_t src_row = group_base + cumsum_before_src;
            const uint32_t rows = ExpectedRowsClipped(src_row, rows_raw, cfg.max_output_size);
            uint32_t scale_bits = 0;
            uint32_t before_first_bits = 0;
            uint32_t after_first_bits = 0;
            uint32_t after_last_bits = 0;
            uint32_t d_first_bits = 0;
            uint32_t d_last_bits = 0;
            uint32_t op_counts = 0;
            if (rows != 0U) {
                float scale_value = 0.0f;
                std::memcpy(&scale_value, scale_bytes_host.data() + static_cast<size_t>(src_row) * sizeof(float),
                            sizeof(float));
                const float before_first = Fp16ToFloat(gmm2_output[static_cast<size_t>(src_row) * cfg.k]);
                const float before_last = Fp16ToFloat(gmm2_output[static_cast<size_t>(src_row) * cfg.k + cfg.k - 1U]);
                const float after_first = before_first * scale_value;
                const float after_last = before_last * scale_value;
                const uint16_t d_first = FloatToHalfRoundToNearestEven(after_first);
                const uint16_t d_last = FloatToHalfRoundToNearestEven(after_last);
                scale_bits = FloatBitsHost(scale_value);
                before_first_bits = FloatBitsHost(before_first);
                after_first_bits = FloatBitsHost(after_first);
                after_last_bits = FloatBitsHost(after_last);
                d_first_bits = FloatBitsHost(Fp16ToFloat(d_first));
                d_last_bits = FloatBitsHost(Fp16ToFloat(d_last));
                op_counts = kHostCombineDequantOpCounts;
                ++report.dequant_count;
            }
            const size_t idx = static_cast<size_t>(group) * rank_size + src_rank;
            const size_t actual_idx = static_cast<size_t>(group) * io_slots + src_rank;
            UpdateCombineIndexedReport(report, "magic", idx, kHostCombineDequantDebugMagic, actual[actual_idx].magic);
            UpdateCombineIndexedReport(report, "groupIdx", idx, group, actual[actual_idx].groupIdx);
            UpdateCombineIndexedReport(report, "srcRank", idx, src_rank, actual[actual_idx].srcRank);
            UpdateCombineIndexedReport(report, "coreIdx", idx, src_rank % cfg.aiv_num, actual[actual_idx].coreIdx);
            UpdateCombineIndexedReport(report, "coreNum", idx, cfg.aiv_num, actual[actual_idx].coreNum);
            UpdateCombineIndexedReport(report, "srcRow", idx, src_row, actual[actual_idx].srcRow);
            UpdateCombineIndexedReport(report, "rows", idx, rows, actual[actual_idx].rows);
            UpdateCombineIndexedReport(report, "scaleBits", idx, scale_bits, actual[actual_idx].scaleBits);
            UpdateCombineIndexedReport(report, "fp32BeforeFirstBits", idx, before_first_bits,
                                       actual[actual_idx].fp32BeforeFirstBits);
            UpdateCombineIndexedReport(report, "fp32AfterFirstBits", idx, after_first_bits,
                                       actual[actual_idx].fp32AfterFirstBits);
            UpdateCombineIndexedReport(report, "fp32AfterLastBits", idx, after_last_bits,
                                       actual[actual_idx].fp32AfterLastBits);
            UpdateCombineIndexedReport(report, "dFirstBits", idx, d_first_bits, actual[actual_idx].dFirstBits);
            UpdateCombineIndexedReport(report, "dLastBits", idx, d_last_bits, actual[actual_idx].dLastBits);
            UpdateCombineIndexedReport(report, "opCounts", idx, op_counts, actual[actual_idx].opCounts);
            UpdateCombineIndexedReport(report, "marker", idx, 1U, actual[actual_idx].marker);
            current_m += rows_raw;
            cumsum_before_src += rows_raw;
            ++report.segment_count;
        }
        group_base += current_m;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

CombineStoreReport CheckCombineSmallStore(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                          const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                          const std::vector<uint8_t> &expert_idx)
{
    struct SmallGroupPlan {
        uint32_t groupBase = 0;
        uint32_t currentM = 0;
        uint32_t tileM = 0;
        uint32_t tileN = 0;
        uint32_t coreLoops = 0;
        uint32_t startCoreIdx = 0;
    };
    struct SmallStoreExpected {
        uint32_t marker = 0;
        uint32_t firstRemoteRank = 0;
        uint32_t firstSrcRow = 0;
        uint32_t firstDstRow = 0;
        uint32_t firstRows = 0;
        uint32_t firstCol = 0;
        uint32_t lastRemoteRank = 0;
        uint32_t lastDstRow = 0;
        uint32_t lastCol = 0;
        uint32_t storeCount = 0;
        uint32_t totalStoreBytes = 0;
        uint64_t firstDstOffsetBytes = 0;
        uint32_t firstBits = 0;
        uint32_t lastBits = 0;
        uint16_t firstHalf = 0;
        uint16_t lastHalf = 0;
    };

    const auto &combine = build.tiling.combineTiling;
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &gmm2 = build.tiling.gmm2Tiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const size_t ready_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * cfg.aiv_num * sizeof(DispatchFFNCombineCombineGmm2ReadyDebug);
    const size_t meta_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * rank_size * sizeof(DispatchFFNCombineCombineMetadataDebug);
    const size_t io_slots = HostCombineDebugIoSlots(rank_size, cfg.aiv_num);
    const size_t load_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * io_slots * sizeof(DispatchFFNCombineCombineLoadDebug);
    const size_t dequant_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * io_slots * sizeof(DispatchFFNCombineCombineDequantDebug);
    const uint64_t store_offset = combine.combineDebugOffset + sizeof(DispatchFFNCombineCombineLayoutDebug) +
                                  ready_bytes + meta_bytes + load_bytes + dequant_bytes;
    const size_t elems = static_cast<size_t>(cfg.expert_per_rank) * cfg.aiv_num;
    const size_t table_elems = static_cast<size_t>(cfg.expert_per_rank) * io_slots;
    const size_t bytes = table_elems * sizeof(DispatchFFNCombineCombineStoreDebug);
    if (combine.combineDebugBytes < store_offset - combine.combineDebugOffset + bytes) {
        throw std::runtime_error("combine debug area is too small for small store table");
    }
    std::vector<DispatchFFNCombineCombineStoreDebug> actual(table_elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    CommMpiBarrier();
    if (!actual.empty() && aclrtMemcpy(actual.data(), bytes, workspace_base + store_offset, bytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host combine small store table copy failed");
    }

    const size_t gmm2_output_elems = static_cast<size_t>(cfg.max_output_size) * cfg.k;
    const size_t gmm2_output_bytes = gmm2_output_elems * sizeof(uint16_t);
    const size_t scale_bytes = static_cast<size_t>(cfg.max_output_size) * sizeof(float);
    std::vector<uint16_t> gmm2_output(gmm2_output_elems, 0);
    std::vector<uint8_t> scale_bytes_host(scale_bytes, 0);
    if ((!gmm2_output.empty() &&
         aclrtMemcpy(gmm2_output.data(), gmm2_output_bytes, workspace_base + combine.gmm2OutputOffset,
                     gmm2_output_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) ||
        (!scale_bytes_host.empty() &&
         aclrtMemcpy(scale_bytes_host.data(), scale_bytes, workspace_base + combine.perTokenScale2Offset, scale_bytes,
                     ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)) {
        throw std::runtime_error("device->host combine small store source buffer copy failed");
    }

    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);

    auto build_group_plan = [&](uint32_t executor_rank) {
        std::vector<SmallGroupPlan> group_plan(cfg.expert_per_rank);
        uint32_t group_base = 0;
        uint32_t start_core_idx = 0;
        const uint32_t executor_begin = executor_rank * cfg.expert_per_rank;
        for (uint32_t group = 0; group < cfg.expert_per_rank; ++group) {
            const uint32_t global_expert = executor_begin + group;
            uint32_t current_m_raw = 0;
            for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
                current_m_raw += static_cast<uint32_t>(
                    expected_tokens[static_cast<size_t>(src_rank) * front.expertNumAligned + global_expert]);
            }
            const uint32_t remaining = group_base >= cfg.max_output_size ? 0U : cfg.max_output_size - group_base;
            SmallGroupPlan &plan = group_plan[group];
            plan.groupBase = group_base;
            plan.currentM = current_m_raw > remaining ? remaining : current_m_raw;
            plan.tileM = CeilDivU32(plan.currentM, gmm2.l1TileM);
            plan.tileN = CeilDivU32(cfg.k, gmm2.l1TileN);
            plan.coreLoops = plan.tileM * plan.tileN;
            plan.startCoreIdx = start_core_idx;
            start_core_idx = build.block_dim == 0U ? 0U : (start_core_idx + plan.coreLoops) % build.block_dim;
            group_base += plan.currentM;
        }
        return group_plan;
    };

    auto dequant_half = [&](uint32_t src_row, uint32_t col) {
        float scale_value = 0.0f;
        std::memcpy(&scale_value, scale_bytes_host.data() + static_cast<size_t>(src_row) * sizeof(float),
                    sizeof(float));
        const float value = Fp16ToFloat(gmm2_output[static_cast<size_t>(src_row) * cfg.k + col]) * scale_value;
        return FloatToHalfRoundToNearestEven(value);
    };

    auto build_expected = [&](uint32_t executor_rank, bool with_values) {
        std::vector<SmallStoreExpected> expected(elems);
        const std::vector<SmallGroupPlan> group_plan = build_group_plan(executor_rank);
        const uint32_t executor_begin = executor_rank * cfg.expert_per_rank;
        const uint32_t aic_core_num = build.block_dim;
        for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
            const uint32_t aic_core_idx = aic_core_num == 0U ? 0U : core % aic_core_num;
            const uint32_t aiv_sub_core_idx = (aic_core_num == 0U || core < aic_core_num) ? 0U : 1U;
            for (uint32_t group = 0; group < cfg.expert_per_rank; ++group) {
                const SmallGroupPlan &plan = group_plan[group];
                const uint32_t global_expert = executor_begin + group;
                const uint32_t start_loop_idx =
                    aic_core_num == 0U ? 0U : HostGmmCommonStartLoopIdx(aic_core_idx, aic_core_num, plan.startCoreIdx);
                SmallStoreExpected &entry = expected[static_cast<size_t>(group) * cfg.aiv_num + core];
                for (uint32_t loop = start_loop_idx; aic_core_num != 0U && loop < plan.coreLoops;
                     loop += aic_core_num) {
                    uint32_t block_m = 0;
                    uint32_t block_n = 0;
                    GetGmm1BlockCoordMN(loop, plan.tileM, plan.tileN, block_m, block_n);
                    uint32_t actual_m = 0;
                    uint32_t actual_n = 0;
                    GetGmmCommonActualBlockShapeMN(plan.currentM, cfg.k, gmm2.l1TileM, gmm2.l1TileN, plan.tileM,
                                                   plan.tileN, block_m, block_n, actual_m, actual_n);
                    const uint32_t subtile_count = CeilDivU32(actual_m, kHostCombineSmallTokenSubtileRows);
                    const uint32_t first_subtile = HostCombineSubtileFirstIndex(subtile_count, aiv_sub_core_idx);
                    const uint32_t assigned_subtiles =
                        HostCombineSubtileAssignedCount(subtile_count, aiv_sub_core_idx);
                    for (uint32_t subtile = 0; subtile < assigned_subtiles; ++subtile) {
                        const uint32_t subtile_idx = first_subtile + subtile;
                        const uint32_t row_in_tile = subtile_idx * kHostCombineSmallTokenSubtileRows;
                        if (row_in_tile >= actual_m) {
                            continue;
                        }
                        const uint32_t rows =
                            std::min<uint32_t>(kHostCombineSmallTokenSubtileRows, actual_m - row_in_tile);
                        const uint32_t st_tile = block_m * gmm2.l1TileM + row_in_tile;
                        const uint32_t ed_tile = st_tile + rows;
                        const uint32_t col = block_n * gmm2.l1TileN;
                        uint32_t pre_sum_rank_in_expert = 0;
                        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
                            const uint32_t len_rank_in_expert = static_cast<uint32_t>(
                                expected_tokens[static_cast<size_t>(src_rank) * front.expertNumAligned +
                                                global_expert]);
                            const uint32_t dst_expert_offset = ExpectedSrcRankDstRowOffset(
                                expected_tokens, front.expertNumAligned, src_rank, global_expert);
                            const uint32_t st_rank_in_expert = pre_sum_rank_in_expert;
                            const uint32_t ed_rank_in_expert = st_rank_in_expert + len_rank_in_expert;
                            pre_sum_rank_in_expert += len_rank_in_expert;
                            if (st_rank_in_expert >= ed_tile) {
                                break;
                            }
                            if (ed_rank_in_expert <= st_tile) {
                                continue;
                            }
                            const uint32_t st_data = std::max(st_rank_in_expert, st_tile);
                            const uint32_t ed_data = std::min(ed_rank_in_expert, ed_tile);
                            if (ed_data <= st_data) {
                                continue;
                            }
                            const uint32_t len_data = ed_data - st_data;
                            const uint32_t dst_offset_in_expert =
                                st_tile > st_rank_in_expert ? st_tile - st_rank_in_expert : 0U;
                            const uint32_t dst_row = dst_expert_offset + dst_offset_in_expert;
                            const uint32_t src_row = plan.groupBase + st_data;
                            const uint32_t last_src_row = plan.groupBase + ed_data - 1U;
                            const uint32_t last_dst_row = dst_row + len_data - 1U;
                            const uint32_t last_col = col + actual_n - 1U;
                            uint16_t first_half = 0;
                            uint16_t last_half = 0;
                            if (with_values) {
                                first_half = dequant_half(src_row, col);
                                last_half = dequant_half(last_src_row, last_col);
                            }
                            if (entry.storeCount == 0U) {
                                entry.marker = 1U;
                                entry.firstRemoteRank = src_rank;
                                entry.firstSrcRow = src_row;
                                entry.firstDstRow = dst_row;
                                entry.firstRows = len_data;
                                entry.firstCol = col;
                                entry.firstDstOffsetBytes =
                                    (static_cast<uint64_t>(dst_row) * cfg.k + col) * sizeof(uint16_t);
                                entry.firstHalf = first_half;
                                entry.firstBits = FloatBitsHost(Fp16ToFloat(first_half));
                            }
                            entry.lastRemoteRank = src_rank;
                            entry.lastDstRow = last_dst_row;
                            entry.lastCol = last_col;
                            entry.lastHalf = last_half;
                            entry.lastBits = FloatBitsHost(Fp16ToFloat(last_half));
                            entry.totalStoreBytes += len_data * actual_n * sizeof(uint16_t);
                            ++entry.storeCount;
                        }
                    }
                }
            }
        }
        return expected;
    };

    const size_t sample_elems_per_rank = static_cast<size_t>(rank_size) * cfg.expert_per_rank * cfg.aiv_num * 2U;
    std::vector<uint16_t> local_offset_d_samples(sample_elems_per_rank, 0);
    const auto *window_base = reinterpret_cast<const uint8_t *>(runtime.hccl.WindowIn(runtime.hccl.rank_id));
    const auto *offset_d_base = window_base + HostPeerOffsetD(runtime.hccl.WindowBytes());
    for (uint32_t executor_rank = 0; executor_rank < rank_size; ++executor_rank) {
        const std::vector<SmallStoreExpected> sample_expected = build_expected(executor_rank, false);
        for (uint32_t group = 0; group < cfg.expert_per_rank; ++group) {
            for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
                const size_t idx = static_cast<size_t>(group) * cfg.aiv_num + core;
                const SmallStoreExpected &exp = sample_expected[idx];
                if (exp.marker == 0U) {
                    continue;
                }
                const size_t sample_idx =
                    ((static_cast<size_t>(executor_rank) * cfg.expert_per_rank + group) * cfg.aiv_num + core) * 2U;
                if (exp.firstRemoteRank == rank) {
                    const size_t offset =
                        (static_cast<size_t>(exp.firstDstRow) * cfg.k + exp.firstCol) * sizeof(uint16_t);
                    if (aclrtMemcpy(&local_offset_d_samples[sample_idx], sizeof(uint16_t), offset_d_base + offset,
                                    sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                        throw std::runtime_error("device->host combine small offsetD first sample copy failed");
                    }
                }
                if (exp.lastRemoteRank == rank) {
                    const size_t offset =
                        (static_cast<size_t>(exp.lastDstRow) * cfg.k + exp.lastCol) * sizeof(uint16_t);
                    if (aclrtMemcpy(&local_offset_d_samples[sample_idx + 1U], sizeof(uint16_t), offset_d_base + offset,
                                    sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                        throw std::runtime_error("device->host combine small offsetD last sample copy failed");
                    }
                }
            }
        }
    }
    std::vector<uint8_t> local_sample_bytes(local_offset_d_samples.size() * sizeof(uint16_t), 0);
    std::memcpy(local_sample_bytes.data(), local_offset_d_samples.data(), local_sample_bytes.size());
    const std::vector<uint8_t> all_sample_bytes = GatherAllRankBytes(
        runtime.hccl.rank_id, runtime.hccl.world_size, local_sample_bytes, "combine small offsetD sample");
    const auto *all_offset_d_samples = reinterpret_cast<const uint16_t *>(all_sample_bytes.data());

    const std::vector<SmallStoreExpected> expected = build_expected(rank, true);
    CombineStoreReport report;
    for (uint32_t group = 0; group < cfg.expert_per_rank; ++group) {
        for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
            const size_t idx = static_cast<size_t>(group) * cfg.aiv_num + core;
            const size_t actual_idx = static_cast<size_t>(group) * io_slots + core;
            const SmallStoreExpected &exp = expected[idx];
            if (exp.marker == 0U) {
                UpdateCombineIndexedReport(report, "magic", idx, 0U, actual[actual_idx].magic);
                UpdateCombineIndexedReport(report, "remoteBaseValid", idx, 0U, actual[actual_idx].remoteBaseValid);
                ++report.segment_count;
                continue;
            }
            const uint32_t local_or_remote = exp.firstRemoteRank == rank ? 0U : 1U;
            UpdateCombineIndexedReport(report, "magic", idx, kHostCombineStoreDebugMagic, actual[actual_idx].magic);
            UpdateCombineIndexedReport(report, "groupIdx", idx, group, actual[actual_idx].groupIdx);
            UpdateCombineIndexedReport(report, "srcRank", idx, exp.firstRemoteRank, actual[actual_idx].srcRank);
            UpdateCombineIndexedReport(report, "coreIdx", idx, core, actual[actual_idx].coreIdx);
            UpdateCombineIndexedReport(report, "srcRow", idx, exp.firstSrcRow, actual[actual_idx].srcRow);
            UpdateCombineIndexedReport(report, "dstRow", idx, exp.firstDstRow, actual[actual_idx].dstRow);
            UpdateCombineIndexedReport(report, "rows", idx, exp.firstRows, actual[actual_idx].rows);
            UpdateCombineIndexedReport(report, "remoteRank", idx, exp.firstRemoteRank, actual[actual_idx].remoteRank);
            UpdateCombineIndexedReport(report, "remoteBaseValid", idx, 1U, actual[actual_idx].remoteBaseValid);
            UpdateCombineIndexedReport(report, "dstGmOffsetBytes", idx, exp.firstDstOffsetBytes,
                                       actual[actual_idx].dstGmOffsetBytes);
            UpdateCombineIndexedReport(report, "storeBytes", idx, exp.totalStoreBytes, actual[actual_idx].storeBytes);
            UpdateCombineIndexedReport(report, "storedFirstBits", idx, exp.firstBits,
                                       actual[actual_idx].storedFirstBits);
            UpdateCombineIndexedReport(report, "storedLastBits", idx, exp.lastBits, actual[actual_idx].storedLastBits);
            UpdateCombineIndexedReport(report, "localOrRemote", idx, local_or_remote, actual[actual_idx].localOrRemote);
            const size_t first_sample_idx =
                static_cast<size_t>(exp.firstRemoteRank) * sample_elems_per_rank +
                ((static_cast<size_t>(rank) * cfg.expert_per_rank + group) * cfg.aiv_num + core) * 2U;
            const size_t last_sample_idx =
                static_cast<size_t>(exp.lastRemoteRank) * sample_elems_per_rank +
                ((static_cast<size_t>(rank) * cfg.expert_per_rank + group) * cfg.aiv_num + core) * 2U + 1U;
            UpdateCombineIndexedReport(report, "remoteActualFirstHalf", idx, exp.firstHalf,
                                       all_offset_d_samples[first_sample_idx]);
            UpdateCombineIndexedReport(report, "remoteActualLastHalf", idx, exp.lastHalf,
                                       all_offset_d_samples[last_sample_idx]);
            report.stored_count += exp.storeCount;
            ++report.segment_count;
        }
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

CombineStoreReport CheckCombineStore(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                     const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                     const std::vector<uint8_t> &expert_idx)
{
    const auto &combine = build.tiling.combineTiling;
    if (combine.gmm2CombineCvMode == kHostGmm2CombineCvModeDirect) {
        CombineStoreReport report;
        report.pass = true;
        return report;
    }
    if (combine.combineStopStep >= 6U) {
        return CheckCombineSmallStore(cfg, build, runtime, workspace_dev, expert_idx);
    }
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t local_begin = rank * cfg.expert_per_rank;
    const size_t ready_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * cfg.aiv_num * sizeof(DispatchFFNCombineCombineGmm2ReadyDebug);
    const size_t meta_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * rank_size * sizeof(DispatchFFNCombineCombineMetadataDebug);
    const size_t io_slots = HostCombineDebugIoSlots(rank_size, cfg.aiv_num);
    const size_t load_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * io_slots * sizeof(DispatchFFNCombineCombineLoadDebug);
    const size_t dequant_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * io_slots * sizeof(DispatchFFNCombineCombineDequantDebug);
    const uint64_t store_offset = combine.combineDebugOffset + sizeof(DispatchFFNCombineCombineLayoutDebug) +
                                  ready_bytes + meta_bytes + load_bytes + dequant_bytes;
    const size_t table_elems = static_cast<size_t>(cfg.expert_per_rank) * io_slots;
    const size_t bytes = table_elems * sizeof(DispatchFFNCombineCombineStoreDebug);
    std::vector<DispatchFFNCombineCombineStoreDebug> actual(table_elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    CommMpiBarrier();
    if (aclrtMemcpy(actual.data(), bytes, workspace_base + store_offset, bytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS) {
        throw std::runtime_error("device->host combine store table copy failed");
    }

    const size_t gmm2_output_elems = static_cast<size_t>(cfg.max_output_size) * cfg.k;
    const size_t gmm2_output_bytes = gmm2_output_elems * sizeof(uint16_t);
    const size_t scale_bytes = static_cast<size_t>(cfg.max_output_size) * sizeof(float);
    std::vector<uint16_t> gmm2_output(gmm2_output_elems, 0);
    std::vector<uint8_t> scale_bytes_host(scale_bytes, 0);
    if ((!gmm2_output.empty() &&
         aclrtMemcpy(gmm2_output.data(), gmm2_output_bytes, workspace_base + combine.gmm2OutputOffset,
                     gmm2_output_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) ||
        (!scale_bytes_host.empty() &&
         aclrtMemcpy(scale_bytes_host.data(), scale_bytes, workspace_base + combine.perTokenScale2Offset, scale_bytes,
                     ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)) {
        throw std::runtime_error("device->host combine store source buffer copy failed");
    }

    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);

    const size_t local_sample_elems = static_cast<size_t>(rank_size) * cfg.expert_per_rank * 2U;
    std::vector<uint16_t> local_offset_d_samples(local_sample_elems, 0);
    const auto *window_base = reinterpret_cast<const uint8_t *>(runtime.hccl.WindowIn(runtime.hccl.rank_id));
    const auto *offset_d_base = window_base + HostPeerOffsetD(runtime.hccl.WindowBytes());
    for (uint32_t executor_rank = 0; executor_rank < rank_size; ++executor_rank) {
        const uint32_t executor_begin = executor_rank * cfg.expert_per_rank;
        uint32_t executor_group_base = 0;
        for (uint32_t group = 0; group < cfg.expert_per_rank; ++group) {
            const uint32_t global_expert = executor_begin + group;
            uint32_t current_m = 0;
            uint32_t cumsum_before_this_src = 0;
            for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
                const uint32_t rows_raw = static_cast<uint32_t>(
                    expected_tokens[static_cast<size_t>(src_rank) * front.expertNumAligned + global_expert]);
                if (src_rank < rank) {
                    cumsum_before_this_src += rows_raw;
                }
                current_m += rows_raw;
            }
            const uint32_t rows_raw_this_src = static_cast<uint32_t>(
                expected_tokens[static_cast<size_t>(rank) * front.expertNumAligned + global_expert]);
            const uint32_t src_row = executor_group_base + cumsum_before_this_src;
            const uint32_t rows = ExpectedRowsClipped(src_row, rows_raw_this_src, cfg.max_output_size);
            if (rows != 0U) {
                const uint32_t dst_row =
                    ExpectedSrcRankDstRowOffset(expected_tokens, front.expertNumAligned, rank, global_expert);
                const size_t sample_idx = (static_cast<size_t>(executor_rank) * cfg.expert_per_rank + group) * 2U;
                const size_t row_offset_bytes = static_cast<size_t>(dst_row) * cfg.k * sizeof(uint16_t);
                if (aclrtMemcpy(&local_offset_d_samples[sample_idx], sizeof(uint16_t), offset_d_base + row_offset_bytes,
                                sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
                    aclrtMemcpy(&local_offset_d_samples[sample_idx + 1U], sizeof(uint16_t),
                                offset_d_base + row_offset_bytes + static_cast<size_t>(cfg.k - 1U) * sizeof(uint16_t),
                                sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
                    throw std::runtime_error("device->host combine offsetD sample copy failed");
                }
            }
            executor_group_base += current_m;
        }
    }
    std::vector<uint8_t> local_sample_bytes(local_offset_d_samples.size() * sizeof(uint16_t), 0);
    std::memcpy(local_sample_bytes.data(), local_offset_d_samples.data(), local_sample_bytes.size());
    const std::vector<uint8_t> all_sample_bytes =
        GatherAllRankBytes(runtime.hccl.rank_id, runtime.hccl.world_size, local_sample_bytes, "combine offsetD sample");
    const auto *all_offset_d_samples = reinterpret_cast<const uint16_t *>(all_sample_bytes.data());

    CombineStoreReport report;
    uint32_t group_base = 0;
    for (uint32_t group = 0; group < cfg.expert_per_rank; ++group) {
        uint32_t current_m = 0;
        uint32_t cumsum_before_src = 0;
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            const uint32_t global_expert = local_begin + group;
            const size_t token_idx = static_cast<size_t>(src_rank) * front.expertNumAligned + global_expert;
            const uint32_t rows_raw = static_cast<uint32_t>(expected_tokens[token_idx]);
            const uint32_t src_row = group_base + cumsum_before_src;
            const uint32_t rows = ExpectedRowsClipped(src_row, rows_raw, cfg.max_output_size);
            const uint32_t dst_row =
                ExpectedSrcRankDstRowOffset(expected_tokens, front.expertNumAligned, src_rank, global_expert);
            const uint32_t local_or_remote = src_rank == rank ? 0U : 1U;
            uint32_t expected_store_bits = 0;
            uint32_t expected_last_bits = 0;
            uint16_t expected_first_half = 0;
            uint16_t expected_last_half = 0;
            if (rows != 0U) {
                float scale_value = 0.0f;
                std::memcpy(&scale_value, scale_bytes_host.data() + static_cast<size_t>(src_row) * sizeof(float),
                            sizeof(float));
                const float c_first = Fp16ToFloat(gmm2_output[static_cast<size_t>(src_row) * cfg.k]);
                const float c_last = Fp16ToFloat(gmm2_output[static_cast<size_t>(src_row) * cfg.k + cfg.k - 1U]);
                expected_first_half = FloatToHalfRoundToNearestEven(c_first * scale_value);
                expected_last_half = FloatToHalfRoundToNearestEven(c_last * scale_value);
                expected_store_bits = FloatBitsHost(Fp16ToFloat(expected_first_half));
                expected_last_bits = FloatBitsHost(Fp16ToFloat(expected_last_half));
                ++report.stored_count;
            }
            const size_t idx = static_cast<size_t>(group) * rank_size + src_rank;
            const size_t actual_idx = static_cast<size_t>(group) * io_slots + src_rank;
            UpdateCombineIndexedReport(report, "magic", idx, kHostCombineStoreDebugMagic, actual[actual_idx].magic);
            UpdateCombineIndexedReport(report, "groupIdx", idx, group, actual[actual_idx].groupIdx);
            UpdateCombineIndexedReport(report, "srcRank", idx, src_rank, actual[actual_idx].srcRank);
            UpdateCombineIndexedReport(report, "coreIdx", idx, src_rank % cfg.aiv_num, actual[actual_idx].coreIdx);
            UpdateCombineIndexedReport(report, "srcRow", idx, src_row, actual[actual_idx].srcRow);
            UpdateCombineIndexedReport(report, "dstRow", idx, dst_row, actual[actual_idx].dstRow);
            UpdateCombineIndexedReport(report, "rows", idx, rows, actual[actual_idx].rows);
            UpdateCombineIndexedReport(report, "remoteRank", idx, src_rank, actual[actual_idx].remoteRank);
            UpdateCombineIndexedReport(report, "remoteBaseValid", idx, rows != 0U ? 1U : 0U,
                                       actual[actual_idx].remoteBaseValid);
            UpdateCombineIndexedReport(report, "dstGmOffsetBytes", idx,
                                       rows != 0U ? static_cast<uint64_t>(dst_row) * cfg.k * sizeof(uint16_t) : 0U,
                                       actual[actual_idx].dstGmOffsetBytes);
            UpdateCombineIndexedReport(report, "storeBytes", idx,
                                       rows != 0U ? static_cast<uint64_t>(cfg.k * sizeof(uint16_t)) : 0U,
                                       actual[actual_idx].storeBytes);
            UpdateCombineIndexedReport(report, "storedFirstBits", idx, expected_store_bits,
                                       actual[actual_idx].storedFirstBits);
            UpdateCombineIndexedReport(report, "storedLastBits", idx, expected_last_bits,
                                       actual[actual_idx].storedLastBits);
            UpdateCombineIndexedReport(report, "localOrRemote", idx, local_or_remote, actual[actual_idx].localOrRemote);
            if (rows != 0U) {
                const size_t sample_idx =
                    ((static_cast<size_t>(src_rank) * rank_size + rank) * cfg.expert_per_rank + group) * 2U;
                UpdateCombineIndexedReport(report, "remoteActualFirstHalf", idx, expected_first_half,
                                           all_offset_d_samples[sample_idx]);
                UpdateCombineIndexedReport(report, "remoteActualLastHalf", idx, expected_last_half,
                                           all_offset_d_samples[sample_idx + 1U]);
            }
            current_m += rows_raw;
            cumsum_before_src += rows_raw;
            ++report.segment_count;
        }
        group_base += current_m;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

CombineFinalizeReport CheckCombineFinalize(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                           const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev)
{
    const auto &combine = build.tiling.combineTiling;
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const size_t io_slots = HostCombineDebugIoSlots(rank_size, cfg.aiv_num);
    const size_t ready_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * cfg.aiv_num * sizeof(DispatchFFNCombineCombineGmm2ReadyDebug);
    const size_t meta_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * rank_size * sizeof(DispatchFFNCombineCombineMetadataDebug);
    const size_t load_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * io_slots * sizeof(DispatchFFNCombineCombineLoadDebug);
    const size_t dequant_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * io_slots * sizeof(DispatchFFNCombineCombineDequantDebug);
    const size_t store_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * io_slots * sizeof(DispatchFFNCombineCombineStoreDebug);
    const uint64_t finalize_offset = combine.combineDebugOffset + sizeof(DispatchFFNCombineCombineLayoutDebug) +
                                     ready_bytes + meta_bytes + load_bytes + dequant_bytes + store_bytes;
    const size_t elems = cfg.aiv_num;
    const size_t bytes = elems * sizeof(DispatchFFNCombineCombineFinalizeDebug);
    std::vector<DispatchFFNCombineCombineFinalizeDebug> actual(elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual.data(), bytes, workspace_base + finalize_offset, bytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS) {
        throw std::runtime_error("device->host combine finalize table copy failed");
    }

    const uint32_t reset_elems = rank_size * front.expertNumAligned;
    CombineFinalizeReport report;
    for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
        const auto &entry = actual[core];
        UpdateCombineIndexedReport(report, "magic", core, kHostCombineFinalizeDebugMagic, entry.magic);
        UpdateCombineIndexedReport(report, "coreIdx", core, core, entry.coreIdx);
        UpdateCombineIndexedReport(report, "coreNum", core, cfg.aiv_num, entry.coreNum);
        UpdateCombineIndexedReport(report, "resetElems", core, reset_elems, entry.resetElems);
        UpdateCombineIndexedReport(report, "resetWriter", core, 0U, entry.resetWriter);
        UpdateCombineIndexedReport(report, "tokenPerExpertBaseOffsetBytes", core,
                                   HostPeerOffsetTokenPerExpert(runtime.hccl.WindowBytes()),
                                   entry.tokenPerExpertBaseOffsetBytes);
        UpdateCombineIndexedReport(report, "tokenPerExpertResetFirstSample", core, 0U,
                                   entry.tokenPerExpertResetFirstSample);
        UpdateCombineIndexedReport(report, "tokenPerExpertResetLastSample", core, 0U,
                                   entry.tokenPerExpertResetLastSample);
        UpdateCombineIndexedReport(report, "boundaryDoneMarker", core, 1U, entry.boundaryDoneMarker);
        UpdateCombineIndexedReport(report, "crossRankSyncAfterReset", core, 1U, entry.crossRankSyncAfterReset);
        UpdateCombineIndexedReport(report, "marker", core, 1U, entry.marker);
        UpdateCombineIndexedReport(report, "finalizeWaitStartNonzero", core, 1U,
                                   entry.finalizeWaitStartSyscnt != 0U ? 1U : 0U);
        UpdateCombineIndexedReport(report, "finalizeWaitOrder", core, 1U,
                                   entry.finalizeWaitEndSyscnt >= entry.finalizeWaitStartSyscnt ? 1U : 0U);
        UpdateCombineIndexedReport(report, "syncAllOrder", core, 1U,
                                   entry.syncAllEndSyscnt >= entry.syncAllStartSyscnt ? 1U : 0U);
        UpdateCombineIndexedReport(report, "resetOrder", core, 1U,
                                   entry.resetEndSyscnt >= entry.resetStartSyscnt ? 1U : 0U);
        UpdateCombineIndexedReport(report, "crossRankOrder", core, 1U,
                                   entry.crossRankSyncEndSyscnt >= entry.crossRankSyncStartSyscnt ? 1U : 0U);
        ++report.checked_count;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

CombineTaskReport CheckCombineSmallTask(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                        const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                        const std::vector<uint8_t> &expert_idx)
{
    const auto &combine = build.tiling.combineTiling;
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &gmm2 = build.tiling.gmm2Tiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t local_begin = rank * cfg.expert_per_rank;
    const size_t ready_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * cfg.aiv_num * sizeof(DispatchFFNCombineCombineGmm2ReadyDebug);
    const size_t meta_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * rank_size * sizeof(DispatchFFNCombineCombineMetadataDebug);
    const size_t io_slots = HostCombineDebugIoSlots(rank_size, cfg.aiv_num);
    const size_t load_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * io_slots * sizeof(DispatchFFNCombineCombineLoadDebug);
    const size_t dequant_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * io_slots * sizeof(DispatchFFNCombineCombineDequantDebug);
    const size_t store_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * io_slots * sizeof(DispatchFFNCombineCombineStoreDebug);
    const size_t finalize_bytes = static_cast<size_t>(cfg.aiv_num) * sizeof(DispatchFFNCombineCombineFinalizeDebug);
    const uint64_t task_offset = combine.combineDebugOffset + sizeof(DispatchFFNCombineCombineLayoutDebug) +
                                 ready_bytes + meta_bytes + load_bytes + dequant_bytes + store_bytes + finalize_bytes;
    const size_t elems = static_cast<size_t>(cfg.expert_per_rank) * cfg.aiv_num;
    const size_t bytes = elems * sizeof(DispatchFFNCombineCombineTaskStatsDebug);
    if (combine.combineDebugBytes < task_offset - combine.combineDebugOffset + bytes) {
        throw std::runtime_error("combine debug area is too small for small task table");
    }

    std::vector<DispatchFFNCombineCombineTaskStatsDebug> actual(elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (!actual.empty() && aclrtMemcpy(actual.data(), bytes, workspace_base + task_offset, bytes,
                                       ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host combine small task table copy failed");
    }

    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);
    const uint32_t aic_core_num = build.block_dim;
    const uint32_t tile_n = CeilDivU32(cfg.k, gmm2.l1TileN);

    CombineTaskReport report;
    uint32_t group_base = 0;
    uint32_t start_core_idx = 0;
    for (uint32_t group = 0; group < cfg.expert_per_rank; ++group) {
        const uint32_t global_expert = local_begin + group;
        uint32_t current_m_raw = 0;
        for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
            current_m_raw += static_cast<uint32_t>(
                expected_tokens[static_cast<size_t>(src_rank) * front.expertNumAligned + global_expert]);
        }
        const uint32_t remaining = group_base >= cfg.max_output_size ? 0U : cfg.max_output_size - group_base;
        const uint32_t current_m = current_m_raw > remaining ? remaining : current_m_raw;
        const uint32_t tile_m = CeilDivU32(current_m, gmm2.l1TileM);
        const uint32_t core_loops = tile_m * tile_n;

        for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
            const uint32_t aic_core_idx = aic_core_num == 0U ? 0U : core % aic_core_num;
            const uint32_t aiv_sub_core_idx = (aic_core_num == 0U || core < aic_core_num) ? 0U : 1U;
            const uint32_t start_loop_idx =
                aic_core_num == 0U ? 0U : HostGmmCommonStartLoopIdx(aic_core_idx, aic_core_num, start_core_idx);
            uint32_t assigned_subtile_count = 0;
            uint32_t assigned_rows = 0;
            uint32_t first_loop_idx = kHostDispatchInvalidTask;
            uint32_t first_rows = 0;
            uint32_t first_row_start = 0;
            uint32_t first_col_start = 0;
            uint32_t first_actual_m = 0;
            uint32_t first_actual_n = 0;
            uint32_t last_loop_idx = kHostDispatchInvalidTask;
            uint32_t last_rows = 0;
            uint32_t last_row_start = 0;
            uint32_t last_col_start = 0;
            uint32_t last_actual_m = 0;
            uint32_t last_actual_n = 0;

            for (uint32_t loop = start_loop_idx; aic_core_num != 0U && loop < core_loops; loop += aic_core_num) {
                uint32_t block_m = 0;
                uint32_t block_n = 0;
                GetGmm1BlockCoordMN(loop, tile_m, tile_n, block_m, block_n);
                uint32_t actual_m = 0;
                uint32_t actual_n = 0;
                GetGmmCommonActualBlockShapeMN(current_m, cfg.k, gmm2.l1TileM, gmm2.l1TileN, tile_m, tile_n, block_m,
                                               block_n, actual_m, actual_n);
                const uint32_t subtile_count = CeilDivU32(actual_m, kHostCombineSmallTokenSubtileRows);
                const uint32_t first_subtile = HostCombineSubtileFirstIndex(subtile_count, aiv_sub_core_idx);
                const uint32_t assigned_subtiles =
                    HostCombineSubtileAssignedCount(subtile_count, aiv_sub_core_idx);
                for (uint32_t subtile = 0; subtile < assigned_subtiles; ++subtile) {
                    const uint32_t subtile_idx = first_subtile + subtile;
                    const uint32_t row_in_tile = subtile_idx * kHostCombineSmallTokenSubtileRows;
                    if (row_in_tile >= actual_m) {
                        continue;
                    }
                    const uint32_t rows = std::min<uint32_t>(kHostCombineSmallTokenSubtileRows, actual_m - row_in_tile);
                    const uint32_t row_start = block_m * gmm2.l1TileM + row_in_tile;
                    const uint32_t col_start = block_n * gmm2.l1TileN;
                    if (first_loop_idx == kHostDispatchInvalidTask) {
                        first_loop_idx = loop;
                        first_rows = rows;
                        first_row_start = row_start;
                        first_col_start = col_start;
                        first_actual_m = actual_m;
                        first_actual_n = actual_n;
                    }
                    last_loop_idx = loop;
                    last_rows = rows;
                    last_row_start = row_start;
                    last_col_start = col_start;
                    last_actual_m = actual_m;
                    last_actual_n = actual_n;
                    ++assigned_subtile_count;
                    assigned_rows += rows;
                }
            }

            const size_t idx = static_cast<size_t>(group) * cfg.aiv_num + core;
            const auto &entry = actual[idx];
            UpdateCombineIndexedReport(report, "magic", idx, kHostCombineTaskDebugMagic, entry.magic);
            UpdateCombineIndexedReport(report, "groupIdx", idx, group, entry.groupIdx);
            UpdateCombineIndexedReport(report, "coreIdx", idx, core, entry.coreIdx);
            UpdateCombineIndexedReport(report, "rankSize", idx, rank_size, entry.rankSize);
            UpdateCombineIndexedReport(report, "assignedRankCount", idx, assigned_subtile_count,
                                       entry.assignedRankCount);
            UpdateCombineIndexedReport(report, "coreRowCount", idx, assigned_rows, entry.coreRowCount);
            UpdateCombineIndexedReport(report, "groupBase", idx, group_base, entry.groupBase);
            UpdateCombineIndexedReport(report, "currentM", idx, current_m, entry.currentM);
            UpdateCombineIndexedReport(report, "firstSrcRank", idx, first_loop_idx, entry.firstSrcRank);
            UpdateCombineIndexedReport(report, "firstRows", idx, first_rows, entry.firstRows);
            UpdateCombineIndexedReport(report, "firstSrcRowOffset", idx, first_row_start, entry.firstSrcRowOffset);
            UpdateCombineIndexedReport(report, "firstDstRowOffset", idx, first_col_start, entry.firstDstRowOffset);
            UpdateCombineIndexedReport(report, "lastSrcRank", idx, last_loop_idx, entry.lastSrcRank);
            UpdateCombineIndexedReport(report, "lastRows", idx, last_rows, entry.lastRows);
            UpdateCombineIndexedReport(report, "lastSrcRowOffset", idx, last_row_start, entry.lastSrcRowOffset);
            UpdateCombineIndexedReport(report, "lastDstRowOffset", idx, last_col_start, entry.lastDstRowOffset);
            UpdateCombineIndexedReport(report, "marker", idx, 1U, entry.marker);
            UpdateCombineIndexedReport(report, "currentMRaw", idx, current_m_raw, entry.currentMRaw);
            UpdateCombineIndexedReport(report, "coreLoops", idx, core_loops, entry.coreLoops);
            UpdateCombineIndexedReport(report, "startCoreIdx", idx, start_core_idx, entry.startCoreIdx);
            UpdateCombineIndexedReport(report, "startLoopIdx", idx, start_loop_idx, entry.startLoopIdx);
            UpdateCombineIndexedReport(report, "aicCoreIdx", idx, aic_core_idx, entry.aicCoreIdx);
            UpdateCombineIndexedReport(report, "aicCoreNum", idx, aic_core_num, entry.aicCoreNum);
            UpdateCombineIndexedReport(report, "aivSubCoreIdx", idx, aiv_sub_core_idx, entry.aivSubCoreIdx);
            UpdateCombineIndexedReport(report, "tileM", idx, tile_m, entry.tileM);
            UpdateCombineIndexedReport(report, "tileN", idx, tile_n, entry.tileN);
            UpdateCombineIndexedReport(report, "firstActualM", idx, first_actual_m, entry.firstActualM);
            UpdateCombineIndexedReport(report, "firstActualN", idx, first_actual_n, entry.firstActualN);
            UpdateCombineIndexedReport(report, "lastActualM", idx, last_actual_m, entry.lastActualM);
            UpdateCombineIndexedReport(report, "lastActualN", idx, last_actual_n, entry.lastActualN);
            UpdateCombineIndexedReport(report, "reserved0", idx, 0U, entry.reserved0);
            ++report.checked_count;
        }
        start_core_idx = aic_core_num == 0U ? 0U : (start_core_idx + core_loops) % aic_core_num;
        group_base += current_m;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

CombineTaskReport CheckCombineTask(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                   const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                   const std::vector<uint8_t> &expert_idx)
{
    return CheckCombineSmallTask(cfg, build, runtime, workspace_dev, expert_idx);
}

CombineDoneReport CheckCombineDone(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                   const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                   const std::vector<uint8_t> &expert_idx)
{
    const auto &combine = build.tiling.combineTiling;
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const uint32_t rank = static_cast<uint32_t>(runtime.hccl.rank_id);
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    (void)expert_idx;
    const size_t ready_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * cfg.aiv_num * sizeof(DispatchFFNCombineCombineGmm2ReadyDebug);
    const size_t meta_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * rank_size * sizeof(DispatchFFNCombineCombineMetadataDebug);
    const size_t io_slots = HostCombineDebugIoSlots(rank_size, cfg.aiv_num);
    const size_t load_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * io_slots * sizeof(DispatchFFNCombineCombineLoadDebug);
    const size_t dequant_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * io_slots * sizeof(DispatchFFNCombineCombineDequantDebug);
    const size_t store_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * io_slots * sizeof(DispatchFFNCombineCombineStoreDebug);
    const size_t finalize_bytes = static_cast<size_t>(cfg.aiv_num) * sizeof(DispatchFFNCombineCombineFinalizeDebug);
    const size_t task_bytes =
        static_cast<size_t>(cfg.expert_per_rank) * cfg.aiv_num * sizeof(DispatchFFNCombineCombineTaskStatsDebug);
    const uint64_t done_offset = combine.combineDebugOffset + sizeof(DispatchFFNCombineCombineLayoutDebug) +
                                 ready_bytes + meta_bytes + load_bytes + dequant_bytes + store_bytes + finalize_bytes +
                                 task_bytes;
    const size_t elems = cfg.aiv_num;
    const size_t bytes = elems * sizeof(DispatchFFNCombineCombineDoneDebug);
    std::vector<DispatchFFNCombineCombineDoneDebug> actual(elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual.data(), bytes, workspace_base + done_offset, bytes, ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS) {
        throw std::runtime_error("device->host combine done table copy failed");
    }

    CombineDoneReport report;
    for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
        const uint32_t local_block_count = actual[core].localSegmentCount;
        const uint32_t remote_block_count = actual[core].remoteSegmentCount;
        const uint32_t local_rows = actual[core].localRows;
        const uint32_t remote_rows = actual[core].remoteRows;
        const uint32_t local_bytes = actual[core].localBytes;
        const uint32_t remote_bytes = actual[core].remoteBytes;
        UpdateCombineIndexedReport(report, "magic", core, kHostCombineDoneDebugMagic, actual[core].magic);
        UpdateCombineIndexedReport(report, "coreIdx", core, core, actual[core].coreIdx);
        UpdateCombineIndexedReport(report, "rank", core, rank, actual[core].rank);
        UpdateCombineIndexedReport(report, "rankSize", core, rank_size, actual[core].rankSize);
        UpdateCombineIndexedReport(report, "stageNum", core, front.stageNum, actual[core].stageNum);
        UpdateCombineIndexedReport(report, "expertPerRank", core, cfg.expert_per_rank, actual[core].expertPerRank);
        UpdateCombineIndexedReport(report, "syncBeforeDone", core, 1U, actual[core].syncBeforeDone);
        UpdateCombineIndexedReport(report, "crossRankSync", core, 1U, actual[core].crossRankSync);
        UpdateCombineIndexedReport(report, "marker", core, 1U, actual[core].marker);
        UpdateCombineIndexedReport(report, "localSegmentCount", core, local_block_count,
                                   actual[core].localSegmentCount);
        UpdateCombineIndexedReport(report, "remoteSegmentCount", core, remote_block_count,
                                   actual[core].remoteSegmentCount);
        UpdateCombineIndexedReport(report, "localRows", core, local_rows, actual[core].localRows);
        UpdateCombineIndexedReport(report, "remoteRows", core, remote_rows, actual[core].remoteRows);
        UpdateCombineIndexedReport(report, "localBytes", core, local_bytes, actual[core].localBytes);
        UpdateCombineIndexedReport(report, "remoteBytes", core, remote_bytes, actual[core].remoteBytes);
        ++report.checked_count;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

CombineDetailReport CheckCombineDetail(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                       const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                       const std::vector<uint8_t> &expert_idx)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &combine = build.tiling.combineTiling;
    const uint32_t rank_size = static_cast<uint32_t>(runtime.hccl.world_size);
    const uint32_t expert_num_aligned = front.expertNumAligned;
    const size_t gmm2_output_bytes = static_cast<size_t>(cfg.max_output_size) * cfg.k * sizeof(uint16_t);
    const size_t scale_bytes = static_cast<size_t>(cfg.max_output_size) * sizeof(float);
    const size_t offset_d_bytes = static_cast<size_t>(cfg.max_output_size) * cfg.k * sizeof(uint16_t);
    CombineDetailReport report;
    if (combine.gmm2CombineCvMode == kHostGmm2CombineCvModeDirect) {
        report.skipped = true;
        report.skip_reason = "cv-direct-no-gm-gmm2-output";
        return report;
    }
    const size_t mpi_limit = static_cast<size_t>(std::numeric_limits<int>::max());
    const auto too_large_for_all_rank_gather = [mpi_limit, rank_size](size_t local_bytes) {
        return local_bytes > mpi_limit || (rank_size != 0U && local_bytes > mpi_limit / rank_size);
    };
    if (too_large_for_all_rank_gather(gmm2_output_bytes) || too_large_for_all_rank_gather(scale_bytes) ||
        too_large_for_all_rank_gather(offset_d_bytes)) {
        report.skipped = true;
        report.skip_reason = "all-rank-buffer-too-large";
        return report;
    }
    std::vector<uint8_t> local_gmm2_output(gmm2_output_bytes, 0);
    std::vector<uint8_t> local_scale(scale_bytes, 0);
    std::vector<uint8_t> local_offset_d(offset_d_bytes, 0);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    const auto *window_base = reinterpret_cast<const uint8_t *>(runtime.hccl.WindowIn(runtime.hccl.rank_id));
    if ((!local_gmm2_output.empty() &&
         aclrtMemcpy(local_gmm2_output.data(), gmm2_output_bytes, workspace_base + combine.gmm2OutputOffset,
                     gmm2_output_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) ||
        (!local_scale.empty() &&
         aclrtMemcpy(local_scale.data(), scale_bytes, workspace_base + combine.perTokenScale2Offset, scale_bytes,
                     ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) ||
        (!local_offset_d.empty() &&
         aclrtMemcpy(local_offset_d.data(), offset_d_bytes, window_base + HostPeerOffsetD(runtime.hccl.WindowBytes()),
                     offset_d_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)) {
        throw std::runtime_error("device->host combine detail buffers copy failed");
    }

    const std::vector<uint8_t> all_gmm2_output =
        GatherAllRankBytes(runtime.hccl.rank_id, runtime.hccl.world_size, local_gmm2_output, "combine gmm2Output");
    const std::vector<uint8_t> all_scale =
        GatherAllRankBytes(runtime.hccl.rank_id, runtime.hccl.world_size, local_scale, "combine scale2");
    const std::vector<uint8_t> all_offset_d =
        GatherAllRankBytes(runtime.hccl.rank_id, runtime.hccl.world_size, local_offset_d, "combine offsetD");
    const auto *all_gmm2_output_u16 = reinterpret_cast<const uint16_t *>(all_gmm2_output.data());
    const auto *all_offset_d_u16 = reinterpret_cast<const uint16_t *>(all_offset_d.data());

    const std::vector<int32_t> local_row = BuildExpectedLocalTokenPerExpert(cfg, build, expert_idx);
    const std::vector<int32_t> expected_tokens =
        GatherExpectedTokenRows(runtime.hccl.rank_id, runtime.hccl.world_size, local_row);

    for (uint32_t executor_rank = 0; executor_rank < rank_size; ++executor_rank) {
        const uint32_t local_begin = executor_rank * cfg.expert_per_rank;
        uint32_t group_base = 0;
        for (uint32_t group = 0; group < cfg.expert_per_rank; ++group) {
            const uint32_t global_expert = local_begin + group;
            uint32_t current_m = 0;
            uint32_t cumsum_before_src = 0;
            for (uint32_t src_rank = 0; src_rank < rank_size; ++src_rank) {
                ++report.segment_count;
                const uint32_t rows_raw = static_cast<uint32_t>(
                    expected_tokens[static_cast<size_t>(src_rank) * expert_num_aligned + global_expert]);
                const uint32_t src_row_offset = group_base + cumsum_before_src;
                const uint32_t rows = ExpectedRowsClipped(src_row_offset, rows_raw, cfg.max_output_size);
                const uint32_t dst_row_offset =
                    ExpectedSrcRankDstRowOffset(expected_tokens, expert_num_aligned, src_rank, global_expert);
                if (rows != 0U) {
                    ++report.non_empty_segment_count;
                }
                for (uint32_t row = 0; row < rows; ++row) {
                    const size_t exec_c2_row =
                        static_cast<size_t>(executor_rank) * cfg.max_output_size + src_row_offset + row;
                    const size_t src_d_row = static_cast<size_t>(src_rank) * cfg.max_output_size + dst_row_offset + row;
                    if (exec_c2_row >= static_cast<size_t>(rank_size) * cfg.max_output_size ||
                        src_d_row >= static_cast<size_t>(rank_size) * cfg.max_output_size) {
                        ++report.coverage_mismatch_count;
                        continue;
                    }
                    float scale_value = 0.0f;
                    std::memcpy(&scale_value, all_scale.data() + exec_c2_row * sizeof(float), sizeof(float));
                    for (uint32_t col = 0; col < cfg.k; ++col) {
                        const uint16_t c2_half = all_gmm2_output_u16[exec_c2_row * cfg.k + col];
                        const float expected_f32 = Fp16ToFloat(c2_half) * scale_value;
                        const uint16_t expected_half = FloatToHalfRoundToNearestEven(expected_f32);
                        const uint16_t actual_half = all_offset_d_u16[src_d_row * cfg.k + col];
                        const float expected_half_f32 = Fp16ToFloat(expected_half);
                        const float actual_f32 = Fp16ToFloat(actual_half);
                        const double abs_err = std::fabs(static_cast<double>(expected_half_f32) - actual_f32);
                        const double rel_err =
                            abs_err / std::max(std::fabs(static_cast<double>(expected_half_f32)), 1.0e-7);
                        report.max_abs_err = std::max(report.max_abs_err, abs_err);
                        report.max_rel_err = std::max(report.max_rel_err, rel_err);
                        if (expected_half != actual_half) {
                            if (report.mismatch_count == 0) {
                                report.first_executor_rank = executor_rank;
                                report.first_src_rank = src_rank;
                                report.first_group = group;
                                report.first_src_row = src_row_offset + row;
                                report.first_dst_row = dst_row_offset + row;
                                report.first_col = col;
                                report.first_c2_half = c2_half;
                                report.first_scale = scale_value;
                                report.expected_half = expected_half;
                                report.actual_half = actual_half;
                                report.expected_f32 = expected_half_f32;
                                report.actual_f32 = actual_f32;
                            }
                            ++report.mismatch_count;
                        }
                        ++report.value_count;
                    }
                    ++report.row_count;
                    report.byte_count += cfg.k * sizeof(uint16_t);
                }
                current_m += rows_raw;
                cumsum_before_src += rows_raw;
            }
            group_base += current_m;
        }
    }
    report.pass = report.mismatch_count == 0 && report.coverage_mismatch_count == 0;
    return report;
}

uint64_t UnpermuteTaskDebugOffset(const DispatchFFNCombineBuildResult &build)
{
    return build.tiling.unpermuteTiling.unpermuteDebugOffset + sizeof(DispatchFFNCombineUnpermuteLayoutDebug);
}

uint64_t UnpermuteMetaDebugOffset(const DispatchFFNCombineBuildResult &build)
{
    return UnpermuteTaskDebugOffset(build) + static_cast<uint64_t>(build.tiling.dispatchFFNCombineInfo.aivNum) *
                                                 sizeof(DispatchFFNCombineUnpermuteTaskDebug);
}

uint64_t UnpermuteAccumDebugOffset(const DispatchFFNCombineBuildResult &build)
{
    return UnpermuteMetaDebugOffset(build) + static_cast<uint64_t>(build.tiling.dispatchFFNCombineInfo.aivNum) *
                                                 sizeof(DispatchFFNCombineUnpermuteMetaDebug);
}

uint64_t UnpermuteOutputDebugOffset(const DispatchFFNCombineBuildResult &build)
{
    return UnpermuteAccumDebugOffset(build) + static_cast<uint64_t>(build.tiling.dispatchFFNCombineInfo.aivNum) *
                                                  sizeof(DispatchFFNCombineUnpermuteAccumDebug);
}

uint64_t UnpermuteDoneDebugOffset(const DispatchFFNCombineBuildResult &build)
{
    return UnpermuteOutputDebugOffset(build) + static_cast<uint64_t>(build.tiling.dispatchFFNCombineInfo.aivNum) *
                                                   sizeof(DispatchFFNCombineUnpermuteOutputDebug);
}

uint64_t HostUnpermuteMainUbBytes(uint32_t token_batch, uint32_t topk, uint32_t tile_cols)
{
    uint64_t ub_offset = 0;
    for (uint32_t i = 0; i < 2U; ++i) {
        ub_offset += AlignUpU64(static_cast<uint64_t>(token_batch) * topk * sizeof(int32_t), UB_ALIGN);
        ub_offset += AlignUpU64(static_cast<uint64_t>(token_batch) * topk * sizeof(float), UB_ALIGN);
    }
    ub_offset += AlignUpU64(static_cast<uint64_t>(tile_cols) * sizeof(float), UB_ALIGN);
    for (uint32_t i = 0; i < 2U; ++i) {
        ub_offset += AlignUpU64(static_cast<uint64_t>(tile_cols) * sizeof(uint16_t), UB_ALIGN);
        ub_offset += AlignUpU64(static_cast<uint64_t>(tile_cols) * sizeof(float), UB_ALIGN);
    }
    ub_offset += AlignUpU64(static_cast<uint64_t>(tile_cols) * sizeof(uint16_t), UB_ALIGN);
    return ub_offset;
}

UnpermuteLayoutReport CheckUnpermuteLayout(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                           const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                           const DeviceBuffer &probs_dev, const DeviceBuffer &out_dev)
{
    const auto &front = build.tiling.frontReorderTiling;
    [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
    const auto &unpermute = build.tiling.unpermuteTiling;
    DispatchFFNCombineUnpermuteLayoutDebug actual;
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(&actual, sizeof(actual), workspace_base + unpermute.unpermuteDebugOffset, sizeof(actual),
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host unpermute layout copy failed");
    }

    const uint32_t expanded_rows_valid = std::min<uint32_t>(cfg.m * cfg.topk, cfg.max_output_size);
    const uint64_t window_base = reinterpret_cast<uint64_t>(runtime.hccl.WindowIn(runtime.hccl.rank_id));
    UnpermuteLayoutReport report;
    UpdateUnpermuteLayoutReport(report, "magic", kHostUnpermuteLayoutDebugMagic, actual.magic);
    UpdateUnpermuteLayoutReport(report, "workspaceBase", reinterpret_cast<uint64_t>(workspace_dev.ptr),
                                actual.workspaceBase);
    UpdateUnpermuteLayoutReport(report, "offsetDBase", window_base + HostPeerOffsetD(runtime.hccl.WindowBytes()),
                                actual.offsetDBase);
    UpdateUnpermuteLayoutReport(report, "expandedRowIdxBase",
                                reinterpret_cast<uint64_t>(workspace_dev.ptr) + front.expandedRowIdxOffset,
                                actual.expandedRowIdxBase);
    UpdateUnpermuteLayoutReport(report, "probsBase", reinterpret_cast<uint64_t>(probs_dev.ptr), actual.probsBase);
    UpdateUnpermuteLayoutReport(report, "outBase", reinterpret_cast<uint64_t>(out_dev.ptr), actual.outBase);
    UpdateUnpermuteLayoutReport(report, "unpermuteDebugOffset", unpermute.unpermuteDebugOffset,
                                actual.unpermuteDebugOffset);
    UpdateUnpermuteLayoutReport(report, "unpermuteDebugBytes", unpermute.unpermuteDebugBytes,
                                actual.unpermuteDebugBytes);
    UpdateUnpermuteLayoutReport(report, "peerOffsetD", HostPeerOffsetD(runtime.hccl.WindowBytes()), actual.peerOffsetD);
    UpdateUnpermuteLayoutReport(report, "offsetScale2Base",
                                window_base + HostPeerOffsetScale2(runtime.hccl.WindowBytes()),
                                actual.offsetScale2Base);
    UpdateUnpermuteLayoutReport(report, "peerOffsetScale2", HostPeerOffsetScale2(runtime.hccl.WindowBytes()),
                                actual.peerOffsetScale2);
    UpdateUnpermuteLayoutReport(report, "offsetDBytes",
                                static_cast<uint64_t>(cfg.max_output_size) * cfg.k * sizeof(uint16_t),
                                actual.offsetDBytes);
    UpdateUnpermuteLayoutReport(report, "offsetScale2Bytes", static_cast<uint64_t>(cfg.max_output_size) * sizeof(float),
                                actual.offsetScale2Bytes);
    UpdateUnpermuteLayoutReport(report, "offsetScale2CapacityBytes", kHostMiB, actual.offsetScale2CapacityBytes);
    UpdateUnpermuteLayoutReport(report, "outBytes", static_cast<uint64_t>(cfg.m) * cfg.k * sizeof(uint16_t),
                                actual.outBytes);
    UpdateUnpermuteLayoutReport(report, "expandedRowIdxBytes",
                                static_cast<uint64_t>(AlignUpU64(cfg.m, 256U)) * cfg.topk * sizeof(int32_t),
                                actual.expandedRowIdxBytes);
    UpdateUnpermuteLayoutReport(report, "probsBytes", static_cast<uint64_t>(cfg.m) * cfg.topk * sizeof(float),
                                actual.probsBytes);
    UpdateUnpermuteLayoutReport(report, "frontWorkspaceBytes", front.frontWorkspaceBytes, actual.frontWorkspaceBytes);
    UpdateUnpermuteLayoutReport(report, "rank", runtime.hccl.rank_id, actual.rank);
    UpdateUnpermuteLayoutReport(report, "rankSize", runtime.hccl.world_size, actual.rankSize);
    UpdateUnpermuteLayoutReport(report, "coreIdx", 0U, actual.coreIdx);
    UpdateUnpermuteLayoutReport(report, "coreNum", cfg.aiv_num, actual.coreNum);
    UpdateUnpermuteLayoutReport(report, "stageNum", front.stageNum, actual.stageNum);
    UpdateUnpermuteLayoutReport(report, "problemM", cfg.m, actual.problemM);
    UpdateUnpermuteLayoutReport(report, "problemK", cfg.k, actual.problemK);
    UpdateUnpermuteLayoutReport(report, "topK", cfg.topk, actual.topK);
    UpdateUnpermuteLayoutReport(report, "maxOutputSize", cfg.max_output_size, actual.maxOutputSize);
    UpdateUnpermuteLayoutReport(report, "expandedRowsValid", expanded_rows_valid, actual.expandedRowsValid);
    UpdateUnpermuteLayoutReport(report, "tileCols", unpermute.unpermuteTileCols, actual.tileCols);
    UpdateUnpermuteLayoutReport(report, "tokenBatch", unpermute.unpermuteTokenBatch, actual.tokenBatch);
    UpdateUnpermuteLayoutReport(report, "debugMode", unpermute.unpermuteDebugMode, actual.debugMode);
    UpdateUnpermuteLayoutReport(report, "layoutVersion", kHostUnpermuteLayoutVersion, actual.layoutVersion);
    UpdateUnpermuteLayoutReport(report, "outputElementBytes", sizeof(uint16_t), actual.outputElementBytes);
    UpdateUnpermuteLayoutReport(report, "offsetDRowBytes", cfg.k * sizeof(uint16_t), actual.offsetDRowBytes);
    UpdateUnpermuteLayoutReport(report, "outRowBytes", cfg.k * sizeof(uint16_t), actual.outRowBytes);
    UpdateUnpermuteLayoutReport(report, "tileBytes", unpermute.unpermuteTileCols * sizeof(uint16_t), actual.tileBytes);
    UpdateUnpermuteLayoutReport(
        report, "metadataBytesPerBatch",
        unpermute.unpermuteTokenBatch * cfg.topk * static_cast<uint32_t>(sizeof(int32_t) + sizeof(float)),
        actual.metadataBytesPerBatch);
    UpdateUnpermuteLayoutReport(
        report, "ubMainBytes",
        HostUnpermuteMainUbBytes(unpermute.unpermuteTokenBatch, cfg.topk, unpermute.unpermuteTileCols),
        actual.ubMainBytes);
    UpdateUnpermuteLayoutReport(report, "metadataBufferNum", 2U, actual.metadataBufferNum);
    UpdateUnpermuteLayoutReport(report, "tokenBufferNum", 2U, actual.tokenBufferNum);
    UpdateUnpermuteLayoutReport(report, "syncBoundaryInCombine", 1U, actual.syncBoundaryInCombine);
    UpdateUnpermuteLayoutReport(report, "crossRankSyncInUnpermute", 0U, actual.crossRankSyncInUnpermute);
    UpdateUnpermuteLayoutReport(report, "taskSplitMode", 1U, actual.taskSplitMode);
    UpdateUnpermuteLayoutReport(report, "kTileMode", 1U, actual.kTileMode);
    UpdateUnpermuteLayoutReport(report, "probsDtypeBytes", sizeof(float), actual.probsDtypeBytes);
    UpdateUnpermuteLayoutReport(report, "indexDtypeBytes", sizeof(int32_t), actual.indexDtypeBytes);
    UpdateUnpermuteLayoutReport(report, "offsetScale2CapacityOk",
                                static_cast<uint64_t>(cfg.max_output_size) * sizeof(float) <= kHostMiB ? 1U : 0U,
                                actual.offsetScale2CapacityOk);
    UpdateUnpermuteLayoutReport(
        report, "offsetScale2DNoOverlap",
        HostPeerOffsetScale2(runtime.hccl.WindowBytes()) + kHostMiB <= HostPeerOffsetD(runtime.hccl.WindowBytes()) ?
            1U :
            0U,
        actual.offsetScale2DNoOverlap);
    UpdateUnpermuteLayoutReport(report, "marker", 1U, actual.marker);
    report.pass = report.mismatch_count == 0;
    return report;
}

UnpermuteIndexedReport CheckUnpermuteTask(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                          const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev)
{
    const size_t elems = cfg.aiv_num;
    const size_t bytes = elems * sizeof(DispatchFFNCombineUnpermuteTaskDebug);
    std::vector<DispatchFFNCombineUnpermuteTaskDebug> actual(elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual.data(), bytes, workspace_base + UnpermuteTaskDebugOffset(build), bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host unpermute task copy failed");
    }
    const uint32_t split_base = cfg.m / cfg.aiv_num;
    const uint32_t split_rem = cfg.m % cfg.aiv_num;
    UnpermuteIndexedReport report;
    for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
        const uint32_t token_start = core * split_base + (core < split_rem ? core : split_rem);
        const uint32_t token_count = split_base + (core < split_rem ? 1U : 0U);
        UpdateUnpermuteIndexedReport(report, "magic", core, kHostUnpermuteTaskDebugMagic, actual[core].magic);
        UpdateUnpermuteIndexedReport(report, "coreIdx", core, core, actual[core].coreIdx);
        UpdateUnpermuteIndexedReport(report, "rank", core, runtime.hccl.rank_id, actual[core].rank);
        UpdateUnpermuteIndexedReport(report, "rankSize", core, runtime.hccl.world_size, actual[core].rankSize);
        UpdateUnpermuteIndexedReport(report, "stageNum", core, build.tiling.frontReorderTiling.stageNum,
                                     actual[core].stageNum);
        UpdateUnpermuteIndexedReport(report, "tokenStart", core, token_start, actual[core].tokenStart);
        UpdateUnpermuteIndexedReport(report, "tokenCount", core, token_count, actual[core].tokenCount);
        UpdateUnpermuteIndexedReport(report, "expandedStart", core, token_start * cfg.topk, actual[core].expandedStart);
        UpdateUnpermuteIndexedReport(report, "expandedCount", core, token_count * cfg.topk, actual[core].expandedCount);
        UpdateUnpermuteIndexedReport(report, "splitBase", core, split_base, actual[core].splitBase);
        UpdateUnpermuteIndexedReport(report, "splitRem", core, split_rem, actual[core].splitRem);
        UpdateUnpermuteIndexedReport(report, "problemM", core, cfg.m, actual[core].problemM);
        UpdateUnpermuteIndexedReport(report, "topK", core, cfg.topk, actual[core].topK);
        UpdateUnpermuteIndexedReport(report, "marker", core, 1U, actual[core].marker);
        ++report.checked_count;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

UnpermuteIndexedReport CheckUnpermuteMeta(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                          const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                          const std::vector<uint8_t> &probs_bytes)
{
    (void)runtime;
    const size_t elems = cfg.aiv_num;
    const size_t bytes = elems * sizeof(DispatchFFNCombineUnpermuteMetaDebug);
    std::vector<DispatchFFNCombineUnpermuteMetaDebug> actual(elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual.data(), bytes, workspace_base + UnpermuteMetaDebugOffset(build), bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host unpermute meta copy failed");
    }
    const size_t expanded_elems = static_cast<size_t>(cfg.m) * cfg.topk;
    std::vector<int32_t> expanded(expanded_elems, 0);
    if (!expanded.empty() && aclrtMemcpy(expanded.data(), expanded.size() * sizeof(int32_t),
                                         workspace_base + build.tiling.frontReorderTiling.expandedRowIdxOffset,
                                         expanded.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host unpermute expandedRowIdx copy failed");
    }
    const std::vector<float> probs = BytesToF32(probs_bytes);
    const uint32_t split_base = cfg.m / cfg.aiv_num;
    const uint32_t split_rem = cfg.m % cfg.aiv_num;
    const uint32_t expanded_rows_valid = std::min<uint32_t>(cfg.m * cfg.topk, cfg.max_output_size);
    UnpermuteIndexedReport report;
    for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
        const uint32_t token_start = core * split_base + (core < split_rem ? core : split_rem);
        const uint32_t token_count = split_base + (core < split_rem ? 1U : 0U);
        if (token_count == 0U) {
            continue;
        }
        const size_t slot = static_cast<size_t>(token_start) * cfg.topk;
        const int32_t expanded_row = expanded[slot];
        const float prob = probs[slot];
        const uint32_t valid = expanded_row >= 0 && static_cast<uint32_t>(expanded_row) < expanded_rows_valid ? 1U : 0U;
        UpdateUnpermuteIndexedReport(report, "magic", core, kHostUnpermuteMetaDebugMagic, actual[core].magic);
        UpdateUnpermuteIndexedReport(report, "coreIdx", core, core, actual[core].coreIdx);
        UpdateUnpermuteIndexedReport(report, "token", core, token_start, actual[core].token);
        UpdateUnpermuteIndexedReport(report, "topkIdx", core, 0U, actual[core].topkIdx);
        UpdateUnpermuteIndexedReport(report, "expandedRow", core, static_cast<uint32_t>(expanded_row),
                                     actual[core].expandedRow);
        UpdateUnpermuteIndexedReport(report, "valid", core, valid, actual[core].valid);
        UpdateUnpermuteIndexedReport(report, "probBits", core, FloatBitsHost(prob), actual[core].probBits);
        UpdateUnpermuteIndexedReport(report, "batchStart", core, token_start, actual[core].batchStart);
        UpdateUnpermuteIndexedReport(report, "localToken", core, 0U, actual[core].localToken);
        UpdateUnpermuteIndexedReport(report, "expandedRowsValid", core, expanded_rows_valid,
                                     actual[core].expandedRowsValid);
        UpdateUnpermuteIndexedReport(report, "marker", core, 1U, actual[core].marker);
        ++report.checked_count;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

UnpermuteIndexedReport CheckUnpermuteAccum(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                           const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev)
{
    (void)runtime;
    const size_t elems = cfg.aiv_num;
    const size_t bytes = elems * sizeof(DispatchFFNCombineUnpermuteAccumDebug);
    std::vector<DispatchFFNCombineUnpermuteAccumDebug> actual(elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual.data(), bytes, workspace_base + UnpermuteAccumDebugOffset(build), bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host unpermute accum copy failed");
    }
    const uint32_t split_base = cfg.m / cfg.aiv_num;
    const uint32_t split_rem = cfg.m % cfg.aiv_num;
    UnpermuteIndexedReport report;
    for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
        const uint32_t token_start = core * split_base + (core < split_rem ? core : split_rem);
        const uint32_t token_count = split_base + (core < split_rem ? 1U : 0U);
        if (token_count == 0U) {
            continue;
        }
        UpdateUnpermuteIndexedReport(report, "magic", core, kHostUnpermuteAccumDebugMagic, actual[core].magic);
        UpdateUnpermuteIndexedReport(report, "coreIdx", core, core, actual[core].coreIdx);
        UpdateUnpermuteIndexedReport(report, "token", core, token_start, actual[core].token);
        UpdateUnpermuteIndexedReport(report, "col", core, 0U, actual[core].col);
        UpdateUnpermuteIndexedReport(
            report, "cols", core, std::min(cfg.k, build.tiling.unpermuteTiling.unpermuteTileCols), actual[core].cols);
        UpdateUnpermuteIndexedReport(report, "topkProcessed", core, actual[core].validTopk, actual[core].topkProcessed);
        UpdateUnpermuteIndexedReport(report, "marker", core, 1U, actual[core].marker);
        ++report.checked_count;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

UnpermuteIndexedReport CheckUnpermuteOutput(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                            const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev)
{
    (void)runtime;
    const size_t elems = cfg.aiv_num;
    const size_t bytes = elems * sizeof(DispatchFFNCombineUnpermuteOutputDebug);
    std::vector<DispatchFFNCombineUnpermuteOutputDebug> actual(elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual.data(), bytes, workspace_base + UnpermuteOutputDebugOffset(build), bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host unpermute output copy failed");
    }
    const uint32_t split_base = cfg.m / cfg.aiv_num;
    const uint32_t split_rem = cfg.m % cfg.aiv_num;
    UnpermuteIndexedReport report;
    for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
        const uint32_t token_start = core * split_base + (core < split_rem ? core : split_rem);
        const uint32_t token_count = split_base + (core < split_rem ? 1U : 0U);
        if (token_count == 0U) {
            continue;
        }
        UpdateUnpermuteIndexedReport(report, "magic", core, kHostUnpermuteOutputDebugMagic, actual[core].magic);
        UpdateUnpermuteIndexedReport(report, "coreIdx", core, core, actual[core].coreIdx);
        UpdateUnpermuteIndexedReport(report, "token", core, token_start, actual[core].token);
        UpdateUnpermuteIndexedReport(report, "col", core, 0U, actual[core].col);
        UpdateUnpermuteIndexedReport(
            report, "cols", core, std::min(cfg.k, build.tiling.unpermuteTiling.unpermuteTileCols), actual[core].cols);
        UpdateUnpermuteIndexedReport(report, "topkProcessed", core, actual[core].validTopk, actual[core].topkProcessed);
        UpdateUnpermuteIndexedReport(report, "marker", core, 1U, actual[core].marker);
        ++report.checked_count;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

UnpermuteIndexedReport CheckUnpermuteDone(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                          const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev)
{
    const size_t elems = cfg.aiv_num;
    const size_t bytes = elems * sizeof(DispatchFFNCombineUnpermuteDoneDebug);
    std::vector<DispatchFFNCombineUnpermuteDoneDebug> actual(elems);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    if (aclrtMemcpy(actual.data(), bytes, workspace_base + UnpermuteDoneDebugOffset(build), bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        throw std::runtime_error("device->host unpermute done copy failed");
    }
    const uint32_t split_base = cfg.m / cfg.aiv_num;
    const uint32_t split_rem = cfg.m % cfg.aiv_num;
    UnpermuteIndexedReport report;
    for (uint32_t core = 0; core < cfg.aiv_num; ++core) {
        const uint32_t token_start = core * split_base + (core < split_rem ? core : split_rem);
        const uint32_t token_count = split_base + (core < split_rem ? 1U : 0U);
        const uint32_t values = token_count * cfg.k;
        UpdateUnpermuteIndexedReport(report, "magic", core, kHostUnpermuteDoneDebugMagic, actual[core].magic);
        UpdateUnpermuteIndexedReport(report, "coreIdx", core, core, actual[core].coreIdx);
        UpdateUnpermuteIndexedReport(report, "rank", core, runtime.hccl.rank_id, actual[core].rank);
        UpdateUnpermuteIndexedReport(report, "rankSize", core, runtime.hccl.world_size, actual[core].rankSize);
        UpdateUnpermuteIndexedReport(report, "stageNum", core, build.tiling.frontReorderTiling.stageNum,
                                     actual[core].stageNum);
        UpdateUnpermuteIndexedReport(report, "tokenStart", core, token_start, actual[core].tokenStart);
        UpdateUnpermuteIndexedReport(report, "tokenCount", core, token_count, actual[core].tokenCount);
        UpdateUnpermuteIndexedReport(report, "syncBeforeDone", core, 1U, actual[core].syncBeforeDone);
        UpdateUnpermuteIndexedReport(report, "marker", core, 1U, actual[core].marker);
        UpdateUnpermuteIndexedReport(report, "rows", core, token_count, actual[core].rows);
        UpdateUnpermuteIndexedReport(report, "values", core, values, actual[core].values);
        UpdateUnpermuteIndexedReport(report, "bytes", core, values * static_cast<uint32_t>(sizeof(uint16_t)),
                                     actual[core].bytes);
        ++report.checked_count;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

UnpermuteDetailReport CheckUnpermuteDetail(const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                                           const StandaloneRankRuntime &runtime, const DeviceBuffer &workspace_dev,
                                           const std::vector<uint8_t> &probs_bytes,
                                           const std::vector<uint16_t> &actual_out)
{
    const size_t expanded_elems = static_cast<size_t>(cfg.m) * cfg.topk;
    const size_t offset_d_elems = static_cast<size_t>(cfg.max_output_size) * cfg.k;
    const size_t offset_d_bytes = offset_d_elems * sizeof(uint16_t);
    std::vector<int32_t> expanded(expanded_elems, 0);
    std::vector<uint16_t> offset_d(offset_d_elems, 0);
    const auto *workspace_base = reinterpret_cast<const uint8_t *>(workspace_dev.ptr);
    const auto *window_base = reinterpret_cast<const uint8_t *>(runtime.hccl.WindowIn(runtime.hccl.rank_id));
    if ((!expanded.empty() &&
         aclrtMemcpy(expanded.data(), expanded.size() * sizeof(int32_t),
                     workspace_base + build.tiling.frontReorderTiling.expandedRowIdxOffset,
                     expanded.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) ||
        (!offset_d.empty() &&
         aclrtMemcpy(offset_d.data(), offset_d_bytes, window_base + HostPeerOffsetD(runtime.hccl.WindowBytes()),
                     offset_d_bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)) {
        throw std::runtime_error("device->host unpermute detail buffers copy failed");
    }
    const std::vector<float> probs = BytesToF32(probs_bytes);
    if (probs.size() < expanded_elems || actual_out.size() < static_cast<size_t>(cfg.m) * cfg.k) {
        throw std::runtime_error("unpermute detail input sizes are too small");
    }
    const uint32_t expanded_rows_valid = std::min<uint32_t>(cfg.m * cfg.topk, cfg.max_output_size);
    UnpermuteDetailReport report;
    for (uint32_t token = 0; token < cfg.m; ++token) {
        for (uint32_t col = 0; col < cfg.k; ++col) {
            float acc = 0.0f;
            uint32_t first_valid_topk = 0;
            int32_t first_expanded_row = -1;
            float first_prob = 0.0f;
            for (uint32_t topk = 0; topk < cfg.topk; ++topk) {
                const size_t slot = static_cast<size_t>(token) * cfg.topk + topk;
                const int32_t row = expanded[slot];
                if (row < 0 || static_cast<uint32_t>(row) >= expanded_rows_valid) {
                    ++report.invalid_row_count;
                    continue;
                }
                const float prob = probs[slot];
                acc += Fp16ToFloat(offset_d[static_cast<size_t>(row) * cfg.k + col]) * prob;
                if (first_expanded_row < 0) {
                    first_valid_topk = topk;
                    first_expanded_row = row;
                    first_prob = prob;
                }
            }
            const uint16_t expected_half = FloatToHalfRoundToNearestEven(acc);
            const uint16_t actual_half = actual_out[static_cast<size_t>(token) * cfg.k + col];
            const float expected_f32 = Fp16ToFloat(expected_half);
            const float actual_f32 = Fp16ToFloat(actual_half);
            const bool invalid = std::isnan(expected_f32) || std::isinf(expected_f32) || std::isnan(actual_f32) ||
                                 std::isinf(actual_f32);
            const double abs_err = invalid ? std::numeric_limits<double>::infinity() :
                                             std::fabs(static_cast<double>(expected_f32) - actual_f32);
            const double rel_err = invalid ? std::numeric_limits<double>::infinity() :
                                             abs_err / std::max(std::fabs(static_cast<double>(expected_f32)), 1.0e-7);
            const double tolerance = 1.0e-5 + 1.0e-3 * std::fabs(static_cast<double>(expected_f32));
            const uint32_t ulp_err = HalfUlpDistance(expected_half, actual_half);
            report.max_abs_err = std::max(report.max_abs_err, abs_err);
            report.max_rel_err = std::max(report.max_rel_err, rel_err);
            report.max_ulp_err = std::max(report.max_ulp_err, ulp_err);
            if (invalid || abs_err > tolerance) {
                if (report.mismatch_count == 0) {
                    report.first_token = token;
                    report.first_col = col;
                    report.first_topk = first_valid_topk;
                    report.first_expanded_row = first_expanded_row;
                    report.first_prob = first_prob;
                    report.expected_half = expected_half;
                    report.actual_half = actual_half;
                    report.expected_f32 = expected_f32;
                    report.actual_f32 = actual_f32;
                }
                ++report.mismatch_count;
            }
            ++report.value_count;
        }
        ++report.row_count;
    }
    report.pass = report.mismatch_count == 0;
    return report;
}

std::string BuildCoreCountReportText(int rank_id, const CoreCountReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " coreCount "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count));
    if (!report.pass) {
        os << " first_idx=" << report.first_mismatch << " expected=" << report.expected << " actual=" << report.actual;
        os << " expected_sum=" << report.expected_sum << " actual_sum=" << report.actual_sum
           << " actual_nonzero=" << report.actual_nonzero_count;
        if (report.actual_nonzero_count != 0) {
            os << " first_actual_nonzero=" << report.first_actual_nonzero;
        }
    }
    return os.str();
}

std::string BuildBaseCursorReportText(int rank_id, const BaseCursorReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " baseCursor "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count));
    if (!report.pass) {
        os << " table=" << report.first_table << " first_idx=" << report.first_idx << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildFrontSortReportText(int rank_id, const FrontSortReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " frontSort "
       << (report.pass ? "PASS" :
                         "FAIL sortedKey=" + std::to_string(report.sorted_key_mismatch_count) +
                             " dstToSrc=" + std::to_string(report.dst_to_src_mismatch_count) +
                             " expanded=" + std::to_string(report.expanded_mismatch_count) +
                             " count=" + std::to_string(report.count_mismatch_count))
       << " activeWorkers=" << report.active_workers << " subblock0=" << report.subblock0_workers
       << " subblock1=" << report.subblock1_workers << " routeElems=" << report.route_elems
       << " aligned=" << report.aligned_route_elems << " keyMode=expert";
    if (!report.pass) {
        os << " table=" << report.first_table << " worker=" << report.first_worker << " first_idx=" << report.first_idx
           << " expected=" << report.expected << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildFrontRouteReportText(int rank_id, const FrontSortReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " frontRoute "
       << (report.pass ? "PASS" :
                         "FAIL expanded=" + std::to_string(report.expanded_mismatch_count) +
                             " dstToSrc=" + std::to_string(report.dst_to_src_mismatch_count) +
                             " count=" + std::to_string(report.count_mismatch_count))
       << " allActiveWorkers=" << (report.pass ? 1 : 0);
    if (!report.pass) {
        os << " table=" << report.first_table << " worker=" << report.first_worker << " first_idx=" << report.first_idx
           << " expected=" << report.expected << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildFrontSortCheckReportText(int rank_id, const FrontSortCheckReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " frontSortCheck "
       << (report.pass ? "PASS" :
                         "FAIL sortedExpert=" + std::to_string(report.sorted_expert_mismatch_count) +
                             " sortedPayload=" + std::to_string(report.sorted_payload_mismatch_count))
       << " routeElems=" << report.route_elems << " aligned=" << report.aligned_route_elems;
    if (!report.pass) {
        os << " table=" << report.first_table << " idx=" << report.first_idx << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildFrontMergeOutReportText(int rank_id, const FrontSortCheckReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " frontMergeOut "
       << (report.pass ? "PASS" :
                         "FAIL sortedExpert=" + std::to_string(report.sorted_expert_mismatch_count) +
                             " sortedPayload=" + std::to_string(report.sorted_payload_mismatch_count))
       << " routeElems=" << report.route_elems << " aligned=" << report.aligned_route_elems;
    if (!report.pass) {
        os << " table=" << report.first_table << " idx=" << report.first_idx << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildFrontMetadataReportText(int rank_id, const FrontMetadataReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " frontMetadata "
       << (report.pass ? "PASS" :
                         "FAIL expanded=" + std::to_string(report.expanded_mismatch_count) +
                             " count=" + std::to_string(report.count_mismatch_count) +
                             " base=" + std::to_string(report.base_mismatch_count) +
                             " coreCount=" + std::to_string(report.core_count_mismatch_count) +
                             " permutation=" + std::to_string(report.permutation_mismatch_count))
       << " routeElems=" << report.route_elems << " expertNumAligned=" << report.expert_num_aligned
       << " countSum=" << report.count_sum;
    if (report.metadata_core_count != 0U) {
        os << " metadataCores=" << report.metadata_core_count
           << " coreCountNonzero=" << report.metadata_core_nonzero_count << " coreCountSum=" << report.core_count_sum;
    }
    if (!report.pass) {
        os << " table=" << report.first_table << " idx=" << report.first_idx << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildFrontRunSortReportText(int rank_id, const FrontRunSortReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " frontRunSort "
       << (report.pass ? "PASS" :
                         "FAIL sortedExpert=" + std::to_string(report.sorted_expert_mismatch_count) +
                             " sortedPayload=" + std::to_string(report.sorted_payload_mismatch_count))
       << " routeElems=" << report.route_elems << " runs=" << report.sort_need_core_num;
    if (!report.pass) {
        os << " table=" << report.first_table << " worker=" << report.first_worker << " idx=" << report.first_idx
           << " expected=" << report.expected << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildFrontLocalMergeReportText(int rank_id, const FrontRunSortReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " frontLocalMerge "
       << (report.pass ? "PASS" :
                         "FAIL sortedExpert=" + std::to_string(report.sorted_expert_mismatch_count) +
                             " sortedPayload=" + std::to_string(report.sorted_payload_mismatch_count))
       << " routeElems=" << report.route_elems << " runs=" << report.sort_need_core_num;
    if (!report.pass) {
        os << " table=" << report.first_table << " worker=" << report.first_worker << " idx=" << report.first_idx
           << " expected=" << report.expected << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildFrontMiddleMergeReportText(int rank_id, const FrontRunSortReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " frontMiddleMerge "
       << (report.pass ? "PASS" :
                         "FAIL sortedExpert=" + std::to_string(report.sorted_expert_mismatch_count) +
                             " sortedPayload=" + std::to_string(report.sorted_payload_mismatch_count))
       << " routeElems=" << report.route_elems << " runs=" << report.sort_need_core_num;
    if (!report.pass) {
        os << " table=" << report.first_table << " worker=" << report.first_worker << " idx=" << report.first_idx
           << " expected=" << report.expected << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildFrontQuantReportText(int rank_id, const ScatterQuantReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " frontQuant "
       << (report.pass ? "PASS" :
                         "FAIL expanded=" + std::to_string(report.expanded_mismatch_count) +
                             " payload=" + std::to_string(report.offset_mismatch_count) +
                             " scale=" + std::to_string(report.scale_mismatch_count));
    if (!report.pass) {
        os << " table=" << report.first_table << " first_idx=" << report.first_idx;
        if (report.first_table == "scale") {
            os << " expected=" << report.expected_f32 << " actual=" << report.actual_f32;
        } else {
            os << " expected=" << report.expected_i32 << " actual=" << report.actual_i32;
        }
    }
    return os.str();
}

std::string BuildScatterQuantReportText(int rank_id, const ScatterQuantReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " scatterQuant "
       << (report.pass ? "PASS" :
                         "FAIL expanded=" + std::to_string(report.expanded_mismatch_count) +
                             " offsetA=" + std::to_string(report.offset_mismatch_count) +
                             " scale=" + std::to_string(report.scale_mismatch_count));
    if (!report.pass) {
        os << " table=" << report.first_table << " first_idx=" << report.first_idx;
        if (report.first_table == "scale") {
            os << " expected=" << report.expected_f32 << " actual=" << report.actual_f32;
        } else {
            os << " expected=" << report.expected_i32 << " actual=" << report.actual_i32;
        }
    }
    return os.str();
}

std::string BuildCountExchangeReportText(int rank_id, const CountExchangeReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " countExchange "
       << (report.pass ? "PASS" :
                         "FAIL tokenPerExpert=" + std::to_string(report.token_mismatch_count) +
                             " preSumBeforeRank=" + std::to_string(report.presum_mismatch_count) +
                             " process=" + std::to_string(report.process_mismatch_count));
    if (!report.pass) {
        os << " table=" << report.first_table << " first_idx=" << report.first_idx
           << " expected=" << report.expected_u64 << " actual=" << report.actual_u64;
    }
    return os.str();
}

std::string BuildCumsumReportText(int rank_id, const CumsumReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " cumsum "
       << (report.pass ? "PASS" :
                         "FAIL cumsumMM=" + std::to_string(report.cumsum_mismatch_count) +
                             " expertTokenNums=" + std::to_string(report.expert_token_nums_mismatch_count) +
                             " process=" + std::to_string(report.process_mismatch_count));
    if (!report.pass) {
        os << " table=" << report.first_table << " first_idx=" << report.first_idx
           << " expected=" << report.expected_u64 << " actual=" << report.actual_u64;
    }
    return os.str();
}

std::string BuildFrontDispatchContractReportText(int rank_id, const FrontDispatchContractReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " frontDispatchContract "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " checked=" << report.checked_count;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildFrontDoneReportText(int rank_id, const FrontDoneReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " frontDone "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count));
    if (!report.pass) {
        os << " src_rank=" << report.first_rank << " expected=" << report.expected << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildFrontDoneInactiveReportText(int rank_id, const FrontDoneReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " frontDoneInactive "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count));
    if (!report.pass) {
        os << " src_rank=" << report.first_rank << " expected=" << report.expected << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildDispatchLayoutReportText(int rank_id, const DispatchLayoutReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " dispatchLayout "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count));
    if (!report.pass) {
        os << " field=" << report.first_field << " expected=" << report.expected << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildFrontLayoutReportText(int rank_id, const DispatchLayoutReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " frontLayout "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count));
    if (!report.pass) {
        os << " field=" << report.first_field << " expected=" << report.expected << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildDispatchGatherLayoutReportText(int rank_id, const DispatchLayoutReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " dispatchGatherLayout "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count));
    if (!report.pass) {
        os << " field=" << report.first_field << " expected=" << report.expected << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildDispatchGatherRankSplitReportText(int rank_id, const DispatchGatherDetailReport &report,
                                                   uint32_t active_copy_cores, uint32_t inactive_copy_cores)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " dispatchGatherRankSplit "
       << (report.pass ? "PASS" :
                         "FAIL gmA=" + std::to_string(report.gm_a_mismatch_count) +
                             " scale=" + std::to_string(report.scale_mismatch_count) +
                             " coverage=" + std::to_string(report.coverage_mismatch_count))
       << " activeCopyCores=" << active_copy_cores << " inactiveCopyCores=" << inactive_copy_cores
       << " segments=" << report.segment_count << " nonEmpty=" << report.non_empty_segment_count
       << " rows=" << report.row_count << " bytes=" << report.byte_count << " scales=" << report.scale_count;
    if (!report.pass) {
        os << " table=" << report.first_table << " group=" << report.first_group << " srcRank=" << report.first_src_rank
           << " localRow=" << report.first_local_row << " dstRow=" << report.first_dst_row
           << " col=" << report.first_col;
        if (report.first_table == "gmA") {
            os << " expected=" << report.expected_i32 << " actual=" << report.actual_i32;
        } else if (report.first_table == "ptrPerTokenScale") {
            os << std::fixed << std::setprecision(8) << " expected=" << report.expected_f32
               << " actual=" << report.actual_f32;
        } else {
            os << " expected=" << report.expected_u64 << " actual=" << report.actual_u64;
        }
    }
    return os.str();
}

std::string BuildFrontCheckLayoutReportText(int rank_id, const FrontCheckLayoutReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " frontCheckLayout "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " helperMask=" << report.helper_debug_mask << " helperFailures=" << report.helper_debug_failures;
    if (report.quant_marker != 0) {
        os << " quantRows=" << report.quant_actual_rows << "/" << report.quant_expected_rows
           << " quantStores=" << report.quant_actual_stores << "/" << report.quant_expected_stores
           << " quantCores=" << report.quant_active_cores
           << " quantMode=" << (report.quant_full_row_mode != 0 ? "fullRow" : "columnTiled");
    }
    if (!report.pass) {
        os << " field=" << report.first_field << " expected=" << report.expected << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildGmm1LayoutReportText(int rank_id, const Gmm1LayoutReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " gmm1Layout "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count));
    if (!report.pass) {
        os << " field=" << report.first_field << " expected=" << report.expected << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildGmm2LayoutReportText(int rank_id, const Gmm2LayoutReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " gmm2Layout "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count));
    if (!report.pass) {
        os << " field=" << report.first_field << " expected=" << report.expected << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildGmm2V2CReportText(int rank_id, const Gmm2V2CReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " gmm2V2COverlap "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " segments=" << report.segment_count << " checked=" << report.checked_count;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildGmm2SegmentReportText(int rank_id, const Gmm2SegmentReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " gmm2Segment "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " segments=" << report.segment_count << " split=" << report.epilogue_granularity;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildGmm2TaskReportText(int rank_id, const Gmm2TaskReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " gmm2Task "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count));
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildGmm2DoneReportText(int rank_id, const Gmm2DoneReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " gmm2Done "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " checked=" << report.checked_count;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildGmm2FinalSyncReportText(int rank_id, const Gmm2FinalSyncReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " gmm2FinalSync "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " checked=" << report.checked_count;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildGmm2FinalSyncDisabledReportText(int rank_id, const Gmm2FinalSyncReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " gmm2FinalSyncDisabled "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " checked=" << report.checked_count;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildSwigluLayoutReportText(int rank_id, const SwigluLayoutReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " swigluLayout "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count));
    if (!report.pass) {
        os << " field=" << report.first_field << " expected=" << report.expected << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildSwigluC2VReportText(int rank_id, const SwigluC2VReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " swigluC2VOverlap "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " segments=" << report.segment_count << " checked=" << report.checked_count;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildSwigluSegmentReportText(int rank_id, const SwigluSegmentReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " swigluSegment "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " segments=" << report.segment_count << " split=" << report.epilogue_granularity;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildSwigluMetadataModeReportText(int rank_id, const SwigluMetadataModeReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " swigluMetadataMode "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " mode=" << report.actual_mode << " expected=" << report.expected_mode;
    if (!report.pass) {
        os << " field=" << report.first_field << " expected=" << report.expected << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildSwigluSegmentMetaReportText(int rank_id, const SwigluSegmentMetaReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " swigluSegmentMeta "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " mode=" << report.metadata_mode << " segments=" << report.segment_count
       << " checked=" << report.checked_count;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildSwigluTaskReportText(int rank_id, const SwigluTaskReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " swigluTask "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " segments=" << report.segment_count << " checked=" << report.checked_count;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildSwigluOutputReportText(int rank_id, const SwigluOutputReport &report)
{
    std::ostringstream os;
    os << std::setprecision(6) << "rank=" << rank_id << " swigluOutput "
       << (report.pass ? "PASS" :
                         "FAIL quant=" + std::to_string(report.quant_mismatch_count) +
                             " scale=" + std::to_string(report.scale_mismatch_count))
       << " segments=" << report.segment_count << " rows=" << report.row_count << " values=" << report.value_count
       << " scales=" << report.scale_count << " ub_bytes=" << report.ub_bytes
       << " max_scale_abs_err=" << report.max_scale_abs_err << " max_scale_rel_err=" << report.max_scale_rel_err;
    if (!report.pass) {
        os << " table=" << report.first_table << " segment=" << report.first_segment << " row=" << report.first_row
           << " col=" << report.first_col;
        if (report.first_table == "gmPermutedToken") {
            os << " expected=" << report.expected_i32 << " actual=" << report.actual_i32;
        } else {
            os << std::fixed << std::setprecision(10) << " expected=" << report.expected_f32
               << " actual=" << report.actual_f32;
        }
    }
    return os.str();
}

std::string BuildSwigluDetailReportText(int rank_id, const SwigluOutputReport &report)
{
    std::ostringstream os;
    os << std::setprecision(6) << "rank=" << rank_id << " swigluDetail "
       << (report.pass ? "PASS" :
                         "FAIL quant=" + std::to_string(report.quant_mismatch_count) +
                             " scale=" + std::to_string(report.scale_mismatch_count))
       << " segments=" << report.segment_count << " rows=" << report.row_count << " values=" << report.value_count
       << " scales=" << report.scale_count << " max_scale_abs_err=" << report.max_scale_abs_err
       << " max_scale_rel_err=" << report.max_scale_rel_err;
    if (!report.pass) {
        os << " table=" << report.first_table << " segment=" << report.first_segment << " row=" << report.first_row
           << " col=" << report.first_col;
        if (report.first_table == "gmPermutedToken") {
            os << " expected=" << report.expected_i32 << " actual=" << report.actual_i32;
        } else {
            os << std::fixed << std::setprecision(10) << " expected=" << report.expected_f32
               << " actual=" << report.actual_f32;
        }
    }
    return os.str();
}

std::string BuildSwigluDoneReportText(int rank_id, const SwigluDoneReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " swigluDone "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " segments=" << report.segment_count << " checked=" << report.checked_count;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildSwigluFinalSyncReportText(int rank_id, const SwigluFinalSyncReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " swigluFinalSync "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " checked=" << report.checked_count;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildSwigluFinalSyncDisabledReportText(int rank_id, const SwigluFinalSyncReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " swigluFinalSyncDisabled "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " checked=" << report.checked_count;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildGmm1SyncReportText(int rank_id, const Gmm1SyncReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " gmm1Sync "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " checked=" << report.checked_count;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildGmm1TaskReportText(int rank_id, const Gmm1TaskReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " gmm1Task "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count));
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildGmm1OutputReportText(int rank_id, const Gmm1OutputReport &report)
{
    std::ostringstream os;
    os << std::setprecision(6) << "rank=" << rank_id << " gmm1Output "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " checked=" << report.checked_count << " max_abs_err=" << report.max_abs_err
       << " max_rel_err=" << report.max_rel_err;
    if (!report.pass) {
        os << " group=" << report.first_group << " row=" << report.first_row << " col=" << report.first_col
           << " expected=" << report.expected << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildGmm2OutputReportText(int rank_id, const Gmm2OutputReport &report)
{
    std::ostringstream os;
    if (report.skipped) {
        os << "rank=" << rank_id << " gmm2Output SKIP reason=" << report.skip_reason
           << " checked=" << report.checked_count;
        return os.str();
    }
    os << std::setprecision(6) << "rank=" << rank_id << " gmm2Output "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " checked=" << report.checked_count << " max_abs_err=" << report.max_abs_err
       << " max_rel_err=" << report.max_rel_err << " max_ulp_err=" << report.max_ulp_err;
    if (!report.pass) {
        os << " group=" << report.first_group << " row=" << report.first_row << " col=" << report.first_col
           << " acc=" << report.expected_acc << " expected_half=" << report.expected_half
           << " actual_half=" << report.actual_half << " expected=" << report.expected_f32
           << " actual=" << report.actual_f32;
    }
    return os.str();
}

std::string BuildGmm2DetailReportText(int rank_id, const Gmm2OutputReport &report)
{
    std::ostringstream os;
    if (report.skipped) {
        os << "rank=" << rank_id << " gmm2Detail SKIP reason=" << report.skip_reason
           << " segments=" << report.segment_count << " groups=" << report.group_count
           << " rows=" << report.row_count << " cols=" << report.col_count;
        return os.str();
    }
    os << std::setprecision(6) << "rank=" << rank_id << " gmm2Detail "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " segments=" << report.segment_count << " groups=" << report.group_count << " rows=" << report.row_count
       << " cols=" << report.col_count << " checked=" << report.checked_count
       << " expected_checked=" << report.expected_checked_count << " max_abs_err=" << report.max_abs_err
       << " max_rel_err=" << report.max_rel_err << " max_ulp_err=" << report.max_ulp_err;
    if (!report.pass) {
        os << " group=" << report.first_group << " row=" << report.first_row << " col=" << report.first_col
           << " acc=" << report.expected_acc << " expected_half=" << report.expected_half
           << " actual_half=" << report.actual_half << " expected=" << report.expected_f32
           << " actual=" << report.actual_f32;
    }
    return os.str();
}

std::string BuildGmm1DoneReportText(int rank_id, const Gmm1DoneReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " gmm1Done "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " segments=" << report.segment_count;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildGmm1SegmentPlanReportText(int rank_id, const Gmm1SegmentPlanReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " gmm1SegmentPlan "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " segments=" << report.segment_count << " split=" << report.epilogue_granularity
       << " policy=REFERENCE_EPILOGUE";
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildGmm1DetailReportText(int rank_id, const Gmm1DetailReport &report)
{
    std::ostringstream os;
    os << std::setprecision(6) << "rank=" << rank_id << " gmm1Detail "
       << (report.pass ? "PASS" :
                         "FAIL cumsum=" + std::to_string(report.cumsum_mismatch_count) +
                             " gmC=" + std::to_string(report.gm_c_mismatch_count))
       << " groups=" << report.group_count << " rows=" << report.row_count << " cols=" << report.col_count
       << " checked=" << report.checked_count << " max_abs_err=" << report.max_abs_err
       << " max_rel_err=" << report.max_rel_err << " max_ulp_err=" << report.max_ulp_err;
    if (!report.pass) {
        os << " table=" << report.first_table;
        if (report.first_table == "cumsumMM") {
            os << " idx=" << report.first_idx << " expected=" << report.expected_i32 << " actual=" << report.actual_i32;
        } else {
            os << " group=" << report.first_group << " row=" << report.first_row << " col=" << report.first_col
               << " acc=" << report.expected_acc << " expected_f32=" << report.expected_f32
               << " expected_fp16=" << report.expected_fp16 << " actual=" << report.actual_f32
               << " expected_half=" << report.expected_half << " actual_half=" << report.actual_half;
        }
    }
    return os.str();
}

std::string BuildDispatchMetadataReportText(int rank_id, const DispatchMetadataReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " dispatchMeta "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count));
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildDispatchTaskReportText(int rank_id, const DispatchTaskReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " dispatchTask "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " groups=" << report.group_count << " blocks=" << report.total_blocks
       << " activeSlots=" << report.active_slots << " rows=" << report.total_rows
       << " rowBlockRows=" << report.min_row_block_rows << "-" << report.max_row_block_rows
       << " maxRowsPerSlot=" << report.max_rows_per_slot << " maxBlocksPerGroup=" << report.max_blocks_per_group;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildDispatchGatherReportText(int rank_id, const DispatchGatherReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " dispatchGather "
       << (report.pass ? "PASS" :
                         "FAIL gmA=" + std::to_string(report.gm_a_mismatch_count) +
                             " scale=" + std::to_string(report.scale_mismatch_count));
    if (!report.pass) {
        os << " table=" << report.first_table << " idx=" << report.first_idx;
        if (report.first_table == "gmA") {
            os << " expected=" << report.expected_i32 << " actual=" << report.actual_i32;
        } else {
            os << std::fixed << std::setprecision(8) << " expected=" << report.expected_f32
               << " actual=" << report.actual_f32;
        }
    }
    return os.str();
}

std::string BuildDispatchGatherDetailReportText(int rank_id, const DispatchGatherDetailReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " dispatchGatherDetail "
       << (report.pass ? "PASS" :
                         "FAIL gmA=" + std::to_string(report.gm_a_mismatch_count) +
                             " scale=" + std::to_string(report.scale_mismatch_count) +
                             " coverage=" + std::to_string(report.coverage_mismatch_count))
       << " segments=" << report.segment_count << " nonEmpty=" << report.non_empty_segment_count
       << " rows=" << report.row_count << " bytes=" << report.byte_count << " scales=" << report.scale_count;
    if (!report.pass) {
        os << " table=" << report.first_table << " group=" << report.first_group << " srcRank=" << report.first_src_rank
           << " localRow=" << report.first_local_row << " dstRow=" << report.first_dst_row
           << " col=" << report.first_col;
        if (report.first_table == "gmA") {
            os << " expected=" << report.expected_i32 << " actual=" << report.actual_i32;
        } else if (report.first_table == "ptrPerTokenScale") {
            os << std::fixed << std::setprecision(8) << " expected=" << report.expected_f32
               << " actual=" << report.actual_f32;
        } else {
            os << " expected=" << report.expected_u64 << " actual=" << report.actual_u64;
        }
    }
    return os.str();
}

std::string BuildDispatchCopyChunkReportText(int rank_id, const DispatchCopyChunkReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " dispatchCopyChunk "
       << (report.pass ? "PASS" :
                         "FAIL meta=" + std::to_string(report.metadata_mismatch_count) +
                             " gmA=" + std::to_string(report.gm_a_mismatch_count) +
                             " scale=" + std::to_string(report.scale_mismatch_count) +
                             " zero=" + std::to_string(report.zero_mismatch_count))
       << " group=" << report.group << " srcRank=" << report.src_rank << " rows=" << report.rows
       << " chunkRows=" << report.chunk_rows << " srcRowBase=" << report.src_row_base
       << " dstRowBase=" << report.dst_row_base << " bytes=" << report.checked_bytes
       << " scales=" << report.checked_scales << " zeroProbes=" << report.zero_probe_count;
    if (!report.pass) {
        os << " table=" << report.first_table;
        if (report.first_table == "metadata") {
            os << " field=" << report.first_field << " expected=" << report.expected_u64
               << " actual=" << report.actual_u64;
        } else if (report.first_table == "gmA" || report.first_table == "zeroGmA") {
            os << " idx=" << report.first_idx << " expected=" << report.expected_i32 << " actual=" << report.actual_i32;
        } else {
            os << " idx=" << report.first_idx << std::fixed << std::setprecision(8)
               << " expected=" << report.expected_f32 << " actual=" << report.actual_f32;
        }
    }
    return os.str();
}

std::string BuildDispatchCopyAllReportText(int rank_id, const DispatchGatherDetailReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " dispatchCopyAll "
       << (report.pass ? "PASS" :
                         "FAIL gmA=" + std::to_string(report.gm_a_mismatch_count) +
                             " scale=" + std::to_string(report.scale_mismatch_count) +
                             " coverage=" + std::to_string(report.coverage_mismatch_count))
       << " segments=" << report.segment_count << " nonEmpty=" << report.non_empty_segment_count
       << " rows=" << report.row_count << " bytes=" << report.byte_count << " scales=" << report.scale_count;
    if (!report.pass) {
        os << " table=" << report.first_table << " group=" << report.first_group << " srcRank=" << report.first_src_rank
           << " localRow=" << report.first_local_row << " dstRow=" << report.first_dst_row
           << " col=" << report.first_col;
        if (report.first_table == "gmA") {
            os << " expected=" << report.expected_i32 << " actual=" << report.actual_i32;
        } else if (report.first_table == "ptrPerTokenScale") {
            os << std::fixed << std::setprecision(8) << " expected=" << report.expected_f32
               << " actual=" << report.actual_f32;
        } else {
            os << " expected=" << report.expected_u64 << " actual=" << report.actual_u64;
        }
    }
    return os.str();
}

std::string BuildDispatchGroupDoneReportText(int rank_id, const DispatchGroupDoneReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " dispatchGroupDone "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count));
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildCombineLayoutReportText(int rank_id, const CombineLayoutReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " combineLayout "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count));
    if (!report.pass) {
        os << " field=" << report.first_field << " expected=" << report.expected << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildCombineReadyReportText(int rank_id, const CombineReadyReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " combineGmm2Ready "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " checked=" << report.checked_count << " marked=" << report.marked_count << " group0Mask=0x" << std::hex
       << report.group0_magic_mask << std::dec;
    if (report.marked_coreidx_mismatch_count != 0U) {
        os << " markedCoreIdxMismatch=" << report.marked_coreidx_mismatch_count
           << " firstMarkedSlot=" << report.first_marked_core_slot
           << " firstMarkedCore=" << report.first_marked_core_value;
    }
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildCombineMetaReportText(int rank_id, const CombineMetaReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " combineMeta "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " segments=" << report.segment_count;
    if (report.owner_checked_count != 0U) {
        os << " ownerChecked=" << report.owner_checked_count;
    }
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildCombineLoadReportText(int rank_id, const CombineLoadReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " combineLoad "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " segments=" << report.segment_count << " loaded=" << report.loaded_count;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildCombineDequantReportText(int rank_id, const CombineDequantReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " combineDequant "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " segments=" << report.segment_count << " dequant=" << report.dequant_count;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildCombineStoreReportText(int rank_id, const CombineStoreReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " combineStore "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " segments=" << report.segment_count << " stored=" << report.stored_count;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildCombineFinalizeReportText(int rank_id, const CombineFinalizeReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " combineFinalize "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " checked=" << report.checked_count << " resetZero=" << report.reset_zero_count;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildCombineTaskReportText(int rank_id, const CombineTaskReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " combineTask "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " checked=" << report.checked_count;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildCombineDoneReportText(int rank_id, const CombineDoneReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " combineDone "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " checked=" << report.checked_count;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildCombineDetailReportText(int rank_id, const CombineDetailReport &report)
{
    std::ostringstream os;
    if (report.skipped) {
        os << "rank=" << rank_id << " combineDetail SKIP reason=" << report.skip_reason << " rows=" << report.row_count
           << " values=" << report.value_count << " bytes=" << report.byte_count;
        return os.str();
    }
    os << std::setprecision(6) << "rank=" << rank_id << " combineDetail "
       << (report.pass ? "PASS" :
                         "FAIL mismatch=" + std::to_string(report.mismatch_count) +
                             " coverage=" + std::to_string(report.coverage_mismatch_count))
       << " segments=" << report.segment_count << " nonEmpty=" << report.non_empty_segment_count
       << " rows=" << report.row_count << " values=" << report.value_count << " bytes=" << report.byte_count
       << " max_abs_err=" << report.max_abs_err << " max_rel_err=" << report.max_rel_err;
    if (!report.pass) {
        os << " executorRank=" << report.first_executor_rank << " srcRank=" << report.first_src_rank
           << " group=" << report.first_group << " srcRow=" << report.first_src_row
           << " dstRow=" << report.first_dst_row << " col=" << report.first_col << " c2_half=" << report.first_c2_half
           << " scale=" << report.first_scale << " expected_half=" << report.expected_half
           << " actual_half=" << report.actual_half << " expected=" << report.expected_f32
           << " actual=" << report.actual_f32;
    }
    return os.str();
}

std::string BuildCombineStageReportText(int rank_id, const char *stage_name, const CombineDetailReport &report)
{
    std::ostringstream os;
    if (report.skipped) {
        os << "rank=" << rank_id << " " << stage_name << " SKIP reason=" << report.skip_reason
           << " rows=" << report.row_count << " values=" << report.value_count << " bytes=" << report.byte_count;
        return os.str();
    }
    os << std::setprecision(6) << "rank=" << rank_id << " " << stage_name << " "
       << (report.pass ? "PASS" :
                         "FAIL mismatch=" + std::to_string(report.mismatch_count) +
                             " coverage=" + std::to_string(report.coverage_mismatch_count))
       << " rows=" << report.row_count << " values=" << report.value_count << " bytes=" << report.byte_count
       << " max_abs_err=" << report.max_abs_err << " max_rel_err=" << report.max_rel_err;
    if (!report.pass) {
        os << " executorRank=" << report.first_executor_rank << " srcRank=" << report.first_src_rank
           << " group=" << report.first_group << " srcRow=" << report.first_src_row
           << " dstRow=" << report.first_dst_row << " col=" << report.first_col << " expected=" << report.expected_f32
           << " actual=" << report.actual_f32;
    }
    return os.str();
}

std::string BuildUnpermuteLayoutReportText(int rank_id, const UnpermuteLayoutReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " unpermuteLayout "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count));
    if (!report.pass) {
        os << " field=" << report.first_field << " expected=" << report.expected << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildUnpermuteIndexedReportText(int rank_id, const char *name, const UnpermuteIndexedReport &report)
{
    std::ostringstream os;
    os << "rank=" << rank_id << " " << name << " "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " checked=" << report.checked_count;
    if (!report.pass) {
        os << " idx=" << report.first_idx << " field=" << report.first_field << " expected=" << report.expected
           << " actual=" << report.actual;
    }
    return os.str();
}

std::string BuildUnpermuteDetailReportText(int rank_id, const UnpermuteDetailReport &report)
{
    std::ostringstream os;
    os << std::setprecision(6) << "rank=" << rank_id << " unpermuteDetail "
       << (report.pass ? "PASS" : "FAIL mismatch=" + std::to_string(report.mismatch_count))
       << " rows=" << report.row_count << " values=" << report.value_count
       << " invalidRows=" << report.invalid_row_count << " max_abs_err=" << report.max_abs_err
       << " max_rel_err=" << report.max_rel_err << " max_ulp_err=" << report.max_ulp_err;
    if (!report.pass) {
        os << " token=" << report.first_token << " col=" << report.first_col << " topk=" << report.first_topk
           << " expandedRow=" << report.first_expanded_row << " prob=" << report.first_prob
           << " expected_half=" << report.expected_half << " actual_half=" << report.actual_half
           << " expected=" << report.expected_f32 << " actual=" << report.actual_f32;
    }
    return os.str();
}

} // namespace

std::string BuildStageProfileReportText(int rank_id, const DebugHostBuffer &profile_host, uint32_t block_dim,
                                        double sys_cnt_multiple_ns, bool start_sync_debug, const CaseConfig *cfg,
                                        const DispatchFFNCombineBuildResult *build,
                                        const std::vector<int32_t> *expert_token_nums)
{
    std::ostringstream os;
    os << std::fixed << std::setprecision(2);
    if (profile_host.ptr == nullptr || profile_host.bytes == 0 || block_dim == 0) {
        os << "rank=" << rank_id << " stageProfile EMPTY";
        return os.str();
    }

    const auto *profile = static_cast<const uint8_t *>(profile_host.ptr);
    uint64_t kernel_start_min = std::numeric_limits<uint64_t>::max();
    uint64_t kernel_end_max = 0;
    std::array<StageProfileEnvelope, kDispatchFFNCombineProfileStageCount> envelopes{};
    std::array<StageProfileEnvelope, kDispatchFFNCombineProfileReadyStageCount> ready_envelopes{};
    std::array<StageProfileEnvelope, kDispatchFFNCombineProfileFrontDoneStepCount> front_done_envelopes{};
    std::array<CombineDetailProfileEnvelope, kDispatchFFNCombineProfileCombineDetailCount> combine_detail_envelopes{};
    std::array<Gmm1DetailProfileEnvelope, kDispatchFFNCombineProfileGmm1DetailMaxExperts> gmm1_detail_envelopes{};
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
    std::array<DispatchV2CTraceProfileEnvelope, kDispatchFFNCombineProfileDispatchV2CTraceCount>
        dispatch_v2c_set_trace_envelopes{};
    std::array<DispatchV2CTraceProfileEnvelope, kDispatchFFNCombineProfileDispatchV2CTraceCount>
        dispatch_v2c_wait_trace_envelopes{};
    std::array<DispatchV2CTraceProfileEnvelope, kDispatchFFNCombineProfileGmm2ToCombineTraceCount>
        gmm2_to_combine_set_trace_envelopes{};
    std::array<DispatchV2CTraceProfileEnvelope, kDispatchFFNCombineProfileGmm2ToCombineTraceCount>
        gmm2_to_combine_wait_trace_envelopes{};
    std::array<CombineDetailProfileEnvelope, kDispatchFFNCombineProfileGmm2CvDetailCount> gmm2_cv_detail_envelopes{};
    std::array<ExpertWallProfile, kDispatchFFNCombineProfileGmm1DetailMaxExperts> gmm2_aic0_wall_profiles{};
    std::array<ExpertWallProfile, kDispatchFFNCombineProfileGmm1DetailMaxExperts> combine_aiv0_wall_profiles{};
#endif
    std::array<CombineDetailProfileEnvelope, kDispatchFFNCombineProfileFrontDetailCount> front_detail_envelopes{};
    std::array<CombineDetailProfileEnvelope, kDispatchFFNCombineProfileDispatchDetailCount> dispatch_detail_envelopes{};
    std::array<CombineDetailProfileEnvelope, kDispatchFFNCombineProfileUnpermuteDetailCount>
        unpermute_detail_envelopes{};
    StageProfileEnvelope front_done_total_envelope{};
    bool combine_detail_valid = false;
    bool gmm1_detail_valid = false;
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
    bool dispatch_v2c_trace_valid = false;
    bool gmm2_to_combine_trace_valid = false;
    bool gmm2_cv_detail_valid = false;
    bool gmm2_aic0_wall_valid = false;
    bool combine_aiv0_wall_valid = false;
#endif
    bool front_detail_valid = false;
    bool dispatch_detail_valid = false;
    bool unpermute_detail_valid = false;
    const size_t gmm1_profile_experts =
        cfg == nullptr ? kDispatchFFNCombineProfileGmm1DetailMaxExperts :
                         std::min<size_t>(cfg->expert_per_rank, kDispatchFFNCombineProfileGmm1DetailMaxExperts);

    for (uint32_t block = 0; block < block_dim; ++block) {
        for (size_t profile_idx = 0; profile_idx < kDispatchFFNCombineProfileEntriesPerBlock; ++profile_idx) {
            const uint64_t *entry = reinterpret_cast<const uint64_t *>(
                profile + static_cast<size_t>(block) * kDispatchFFNCombineProfileBytesPerBlock +
                profile_idx * kDispatchFFNCombineProfileEntryBytes);
            const uint64_t kernel_start = entry[kDispatchFFNCombineProfileKernelStart];
            const uint64_t kernel_end = entry[kDispatchFFNCombineProfileKernelEnd];
            if (kernel_start == 0U && kernel_end == 0U) {
                continue;
            }
            kernel_start_min = std::min(kernel_start_min, kernel_start);
            kernel_end_max = std::max(kernel_end_max, kernel_end);

            for (size_t stage = 0; stage < kDispatchFFNCombineProfileStageCount; ++stage) {
                if (!StageProfileEntryParticipates(stage, profile_idx)) {
                    continue;
                }
                if (!ProfileIntervalValid(entry, stage)) {
                    continue;
                }
                const uint64_t start = entry[DispatchFFNCombineProfileStageStartIndex(stage)];
                const uint64_t end = entry[DispatchFFNCombineProfileStageEndIndex(stage)];
                UpdateProfileEnvelope(envelopes[stage], start, end);
            }
            for (size_t ready_stage = 0; ready_stage < kDispatchFFNCombineProfileReadyStageCount; ++ready_stage) {
                const size_t stage = ReadyStageProfileStage(ready_stage);
                if (!StageProfileEntryParticipates(stage, profile_idx)) {
                    continue;
                }
                if (!ReadyStageProfileIntervalValid(entry, ready_stage)) {
                    continue;
                }
                const uint64_t start = entry[DispatchFFNCombineProfileReadyStageStartIndex(ready_stage)];
                const uint64_t end = entry[DispatchFFNCombineProfileStageEndIndex(stage)];
                UpdateProfileEnvelope(ready_envelopes[ready_stage], start, end);
            }

            if (profile_idx == 0U) {
                for (size_t expert = 0; expert < gmm1_profile_experts; ++expert) {
                    if (Gmm1DetailProfileValid(entry, expert)) {
                        const uint64_t start = entry[DispatchFFNCombineProfileGmm1DetailStartIndex(expert)];
                        const uint64_t end = entry[DispatchFFNCombineProfileGmm1DetailEndIndex(expert)];
                        const uint64_t total = entry[DispatchFFNCombineProfileGmm1DetailTotalIndex(expert)];
                        const uint64_t count = entry[DispatchFFNCombineProfileGmm1DetailCountIndex(expert)];
                        UpdateGmm1DetailEnvelope(gmm1_detail_envelopes[expert], start, end, total, count);
                        gmm1_detail_valid = true;
                    }
                }
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
                for (size_t event = 0; event < kDispatchFFNCombineProfileDispatchV2CTraceCount; ++event) {
                    if (!DispatchV2CTraceProfileValid(entry, event)) {
                        continue;
                    }
                    const uint64_t start = entry[DispatchFFNCombineProfileDispatchV2CTraceStartIndex(event)];
                    const uint64_t end = entry[DispatchFFNCombineProfileDispatchV2CTraceEndIndex(event)];
                    const uint64_t flagId = entry[DispatchFFNCombineProfileDispatchV2CTraceFlagIndex(event)];
                    const uint64_t count = entry[DispatchFFNCombineProfileDispatchV2CTraceCountIndex(event)];
                    UpdateDispatchV2CTraceEnvelope(dispatch_v2c_wait_trace_envelopes[event], start, end, flagId, count);
                    dispatch_v2c_trace_valid = true;
                }
                for (size_t event = 0; event < kDispatchFFNCombineProfileGmm2ToCombineTraceCount; ++event) {
                    if (!Gmm2ToCombineTraceProfileValid(entry, event)) {
                        continue;
                    }
                    const uint64_t start = entry[DispatchFFNCombineProfileGmm2ToCombineTraceStartIndex(event)];
                    const uint64_t end = entry[DispatchFFNCombineProfileGmm2ToCombineTraceEndIndex(event)];
                    const uint64_t flagId = entry[DispatchFFNCombineProfileGmm2ToCombineTraceFlagIndex(event)];
                    const uint64_t count = entry[DispatchFFNCombineProfileGmm2ToCombineTraceCountIndex(event)];
                    UpdateDispatchV2CTraceEnvelope(gmm2_to_combine_set_trace_envelopes[event], start, end, flagId,
                                                   count);
                    gmm2_to_combine_trace_valid = true;
                }
                for (size_t detail = 0; detail < kDispatchFFNCombineProfileGmm2CvDetailCount; ++detail) {
                    if (!Gmm2CvDetailProfileValid(entry, detail)) {
                        continue;
                    }
                    const uint64_t start = entry[DispatchFFNCombineProfileGmm2CvDetailStartIndex(detail)];
                    const uint64_t end = entry[DispatchFFNCombineProfileGmm2CvDetailEndIndex(detail)];
                    const uint64_t total = entry[DispatchFFNCombineProfileGmm2CvDetailTotalIndex(detail)];
                    const uint64_t count = entry[DispatchFFNCombineProfileGmm2CvDetailCountIndex(detail)];
                    UpdateCombineDetailEnvelope(gmm2_cv_detail_envelopes[detail], start, end, total, count);
                    gmm2_cv_detail_valid = true;
                }
                if (block == 0U) {
                    for (size_t expert = 0; expert < gmm1_profile_experts; ++expert) {
                        if (!Gmm2ExpertWallProfileValid(entry, expert)) {
                            continue;
                        }
                        gmm2_aic0_wall_profiles[expert].valid = true;
                        gmm2_aic0_wall_profiles[expert].start =
                            entry[DispatchFFNCombineProfileGmm2ExpertWallStartIndex(expert)];
                        gmm2_aic0_wall_profiles[expert].end =
                            entry[DispatchFFNCombineProfileGmm2ExpertWallEndIndex(expert)];
                        gmm2_aic0_wall_valid = true;
                    }
                }
#endif
                continue;
            }
            for (size_t detail = 0; detail < kDispatchFFNCombineProfileFrontDetailCount; ++detail) {
                if (!FrontDetailProfileValid(entry, detail)) {
                    continue;
                }
                const uint64_t start = entry[DispatchFFNCombineProfileFrontDetailStartIndex(detail)];
                const uint64_t end = entry[DispatchFFNCombineProfileFrontDetailEndIndex(detail)];
                const uint64_t total = entry[DispatchFFNCombineProfileFrontDetailTotalIndex(detail)];
                const uint64_t count = entry[DispatchFFNCombineProfileFrontDetailCountIndex(detail)];
                UpdateCombineDetailEnvelope(front_detail_envelopes[detail], start, end, total, count);
                front_detail_valid = true;
            }
            for (size_t detail = 0; detail < kDispatchFFNCombineProfileDispatchDetailCount; ++detail) {
                if (!DispatchDetailProfileValid(entry, detail)) {
                    continue;
                }
                const uint64_t start = entry[DispatchFFNCombineProfileDispatchDetailStartIndex(detail)];
                const uint64_t end = entry[DispatchFFNCombineProfileDispatchDetailEndIndex(detail)];
                const uint64_t total = entry[DispatchFFNCombineProfileDispatchDetailTotalIndex(detail)];
                const uint64_t count = entry[DispatchFFNCombineProfileDispatchDetailCountIndex(detail)];
                UpdateCombineDetailEnvelope(dispatch_detail_envelopes[detail], start, end, total, count);
                dispatch_detail_valid = true;
            }
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
            for (size_t event = 0; event < kDispatchFFNCombineProfileDispatchV2CTraceCount; ++event) {
                if (!DispatchV2CTraceProfileValid(entry, event)) {
                    continue;
                }
                const uint64_t start = entry[DispatchFFNCombineProfileDispatchV2CTraceStartIndex(event)];
                const uint64_t end = entry[DispatchFFNCombineProfileDispatchV2CTraceEndIndex(event)];
                const uint64_t flagId = entry[DispatchFFNCombineProfileDispatchV2CTraceFlagIndex(event)];
                const uint64_t count = entry[DispatchFFNCombineProfileDispatchV2CTraceCountIndex(event)];
                UpdateDispatchV2CTraceEnvelope(dispatch_v2c_set_trace_envelopes[event], start, end, flagId, count);
                dispatch_v2c_trace_valid = true;
            }
            for (size_t event = 0; event < kDispatchFFNCombineProfileGmm2ToCombineTraceCount; ++event) {
                if (!Gmm2ToCombineTraceProfileValid(entry, event)) {
                    continue;
                }
                const uint64_t start = entry[DispatchFFNCombineProfileGmm2ToCombineTraceStartIndex(event)];
                const uint64_t end = entry[DispatchFFNCombineProfileGmm2ToCombineTraceEndIndex(event)];
                const uint64_t flagId = entry[DispatchFFNCombineProfileGmm2ToCombineTraceFlagIndex(event)];
                const uint64_t count = entry[DispatchFFNCombineProfileGmm2ToCombineTraceCountIndex(event)];
                UpdateDispatchV2CTraceEnvelope(gmm2_to_combine_wait_trace_envelopes[event], start, end, flagId, count);
                gmm2_to_combine_trace_valid = true;
            }
#endif
            for (size_t detail = 0; detail < kDispatchFFNCombineProfileCombineDetailCount; ++detail) {
                if (!CombineDetailProfileValid(entry, detail)) {
                    continue;
                }
                const uint64_t start = entry[DispatchFFNCombineProfileCombineDetailStartIndex(detail)];
                const uint64_t end = entry[DispatchFFNCombineProfileCombineDetailEndIndex(detail)];
                const uint64_t total = entry[DispatchFFNCombineProfileCombineDetailTotalIndex(detail)];
                const uint64_t count = entry[DispatchFFNCombineProfileCombineDetailCountIndex(detail)];
                UpdateCombineDetailEnvelope(combine_detail_envelopes[detail], start, end, total, count);
                combine_detail_valid = true;
            }
            for (size_t detail = 0; detail < kDispatchFFNCombineProfileUnpermuteDetailCount; ++detail) {
                if (!UnpermuteDetailProfileValid(entry, detail)) {
                    continue;
                }
                const uint64_t start = entry[DispatchFFNCombineProfileUnpermuteDetailStartIndex(detail)];
                const uint64_t end = entry[DispatchFFNCombineProfileUnpermuteDetailEndIndex(detail)];
                const uint64_t total = entry[DispatchFFNCombineProfileUnpermuteDetailTotalIndex(detail)];
                const uint64_t count = entry[DispatchFFNCombineProfileUnpermuteDetailCountIndex(detail)];
                UpdateCombineDetailEnvelope(unpermute_detail_envelopes[detail], start, end, total, count);
                unpermute_detail_valid = true;
            }
            for (size_t step = 0; step < kDispatchFFNCombineProfileFrontDoneStepCount; ++step) {
                if (!FrontDoneProfileIntervalValid(entry, step)) {
                    continue;
                }
                const uint64_t start = entry[DispatchFFNCombineProfileFrontDoneStartIndex(step)];
                const uint64_t end = entry[DispatchFFNCombineProfileFrontDoneEndIndex(step)];
                UpdateProfileEnvelope(front_done_envelopes[step], start, end);
            }
            if (FrontDoneProfileTotalValid(entry)) {
                const uint64_t start = entry[DispatchFFNCombineProfileFrontDoneStartIndex(
                    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DONE_STEP7_ENTRY_SYNC)];
                const uint64_t end = entry[DispatchFFNCombineProfileFrontDoneEndIndex(
                    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DONE_STEP10_EXIT_SYNC)];
                UpdateProfileEnvelope(front_done_total_envelope, start, end);
            }
#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
            if (block == 0U && profile_idx == 1U) {
                for (size_t expert = 0; expert < gmm1_profile_experts; ++expert) {
                    if (CombineExpertWallProfileValid(entry, expert)) {
                        combine_aiv0_wall_profiles[expert].valid = true;
                        combine_aiv0_wall_profiles[expert].start =
                            entry[DispatchFFNCombineProfileCombineExpertWallStartIndex(expert)];
                        combine_aiv0_wall_profiles[expert].end =
                            entry[DispatchFFNCombineProfileCombineExpertWallEndIndex(expert)];
                        combine_aiv0_wall_valid = true;
                    }
                }
            }
#endif
        }
    }

    if (kernel_start_min == std::numeric_limits<uint64_t>::max()) {
        os << "rank=" << rank_id << " stageProfile EMPTY";
        return os.str();
    }

    os << "rank=" << rank_id << " stageProfileEnvelope base=syscnt_min_kernel_start participant_cores_only=1"
       << (start_sync_debug ? " start_sync_all_cores=1" : "")
       << " kernel_us=" << SysCntTicksToUs(kernel_end_max - kernel_start_min, sys_cnt_multiple_ns) << '\n';
    for (size_t stage = 0; stage < kDispatchFFNCombineProfileStageCount; ++stage) {
        const StageProfileEnvelope &env = envelopes[stage];
        os << "  " << StageProfileName(stage);
        if (!env.valid) {
            os << " inactive\n";
            continue;
        }
        os << " active=" << env.active_entries
           << " start_us=" << SysCntTicksToUs(env.start_min - kernel_start_min, sys_cnt_multiple_ns)
           << " end_us=" << SysCntTicksToUs(env.end_max - kernel_start_min, sys_cnt_multiple_ns)
           << " envelope_us=" << SysCntTicksToUs(env.end_max - env.start_min, sys_cnt_multiple_ns)
           << " max_core_us=" << SysCntTicksToUs(env.max_core_ticks, sys_cnt_multiple_ns);
        const size_t ready_stage = ReadyStageForProfileStage(stage);
        if (ready_stage < kDispatchFFNCombineProfileReadyStageCount) {
            const StageProfileEnvelope &ready_env = ready_envelopes[ready_stage];
            if (ready_env.valid) {
                os << " ready_start_us=" << SysCntTicksToUs(ready_env.start_min - kernel_start_min, sys_cnt_multiple_ns)
                   << " ready_envelope_us="
                   << SysCntTicksToUs(ready_env.end_max - ready_env.start_min, sys_cnt_multiple_ns)
                   << " ready_max_core_us=" << SysCntTicksToUs(ready_env.max_core_ticks, sys_cnt_multiple_ns);
            }
        }
        os << '\n';
    }

    if (front_done_total_envelope.valid) {
        os << "rank=" << rank_id << " frontDoneDetailEnvelope base=syscnt_min_kernel_start active_aiv_only=1\n";
        os << "  frontDoneTotal active=" << front_done_total_envelope.active_entries << " start_us="
           << SysCntTicksToUs(front_done_total_envelope.start_min - kernel_start_min, sys_cnt_multiple_ns)
           << " end_us=" << SysCntTicksToUs(front_done_total_envelope.end_max - kernel_start_min, sys_cnt_multiple_ns)
           << " envelope_us="
           << SysCntTicksToUs(front_done_total_envelope.end_max - front_done_total_envelope.start_min,
                              sys_cnt_multiple_ns)
           << " max_core_us=" << SysCntTicksToUs(front_done_total_envelope.max_core_ticks, sys_cnt_multiple_ns) << '\n';
        for (size_t step = 0; step < kDispatchFFNCombineProfileFrontDoneStepCount; ++step) {
            const StageProfileEnvelope &env = front_done_envelopes[step];
            os << "  " << FrontDoneProfileName(step);
            if (!env.valid) {
                os << " inactive\n";
                continue;
            }
            os << " active=" << env.active_entries
               << " start_us=" << SysCntTicksToUs(env.start_min - kernel_start_min, sys_cnt_multiple_ns)
               << " end_us=" << SysCntTicksToUs(env.end_max - kernel_start_min, sys_cnt_multiple_ns)
               << " envelope_us=" << SysCntTicksToUs(env.end_max - env.start_min, sys_cnt_multiple_ns)
               << " max_core_us=" << SysCntTicksToUs(env.max_core_ticks, sys_cnt_multiple_ns) << '\n';
        }
    }

    if (front_detail_valid) {
        os << "rank=" << rank_id
           << " frontStepDetailEnvelope base=syscnt_min_kernel_start active_aiv_only=1 fields=firstLastTotalCount"
           << " nested_intervals=1\n";
        for (size_t detail = 0; detail < kDispatchFFNCombineProfileFrontDetailCount; ++detail) {
            const CombineDetailProfileEnvelope &env = front_detail_envelopes[detail];
            if (!env.valid) {
                continue;
            }
            os << "  " << FrontDetailProfileName(detail);
            os << " active=" << env.active_entries
               << " start_us=" << SysCntTicksToUs(env.start_min - kernel_start_min, sys_cnt_multiple_ns)
               << " end_us=" << SysCntTicksToUs(env.end_max - kernel_start_min, sys_cnt_multiple_ns)
               << " envelope_us=" << SysCntTicksToUs(env.end_max - env.start_min, sys_cnt_multiple_ns)
               << " max_core_us=" << SysCntTicksToUs(env.max_core_ticks, sys_cnt_multiple_ns)
               << " sum_core_us=" << SysCntTicksToUs(env.sum_core_ticks, sys_cnt_multiple_ns)
               << " count=" << env.call_count << " max_count=" << env.max_call_count;
            if (env.call_count != 0U) {
                os << " avg_call_us="
                   << SysCntTicksToUs(env.sum_core_ticks, sys_cnt_multiple_ns) / static_cast<double>(env.call_count);
            }
            os << '\n';
        }
    }

    if (dispatch_detail_valid) {
        os << "rank=" << rank_id
           << " dispatchDetailEnvelope base=syscnt_min_kernel_start active_aiv_only=1 fields=firstLastTotalCount"
           << " nested_intervals=1\n";
        for (size_t detail = 0; detail < kDispatchFFNCombineProfileDispatchDetailCount; ++detail) {
            const CombineDetailProfileEnvelope &env = dispatch_detail_envelopes[detail];
            if (!env.valid) {
                continue;
            }
            os << "  " << DispatchDetailProfileName(detail);
            os << " active=" << env.active_entries
               << " start_us=" << SysCntTicksToUs(env.start_min - kernel_start_min, sys_cnt_multiple_ns)
               << " end_us=" << SysCntTicksToUs(env.end_max - kernel_start_min, sys_cnt_multiple_ns)
               << " envelope_us=" << SysCntTicksToUs(env.end_max - env.start_min, sys_cnt_multiple_ns)
               << " max_core_us=" << SysCntTicksToUs(env.max_core_ticks, sys_cnt_multiple_ns)
               << " sum_core_us=" << SysCntTicksToUs(env.sum_core_ticks, sys_cnt_multiple_ns)
               << " count=" << env.call_count << " max_count=" << env.max_call_count;
            if (env.call_count != 0U) {
                os << " avg_call_us="
                   << SysCntTicksToUs(env.sum_core_ticks, sys_cnt_multiple_ns) / static_cast<double>(env.call_count);
            }
            os << '\n';
        }
    }

#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
    auto print_first_last_total_count = [&](const char *name, const CombineDetailProfileEnvelope &env,
                                            const char *extra) {
        if (!env.valid) {
            return;
        }
        os << "  " << name;
        if (extra != nullptr && extra[0] != '\0') {
            os << ' ' << extra;
        }
        os << " active=" << env.active_entries
           << " start_us=" << SysCntTicksToUs(env.start_min - kernel_start_min, sys_cnt_multiple_ns)
           << " end_us=" << SysCntTicksToUs(env.end_max - kernel_start_min, sys_cnt_multiple_ns)
           << " envelope_us=" << SysCntTicksToUs(env.end_max - env.start_min, sys_cnt_multiple_ns)
           << " max_core_us=" << SysCntTicksToUs(env.max_core_ticks, sys_cnt_multiple_ns)
           << " sum_core_us=" << SysCntTicksToUs(env.sum_core_ticks, sys_cnt_multiple_ns) << " count=" << env.call_count
           << " max_count=" << env.max_call_count;
        if (env.call_count != 0U) {
            os << " avg_call_us="
               << SysCntTicksToUs(env.sum_core_ticks, sys_cnt_multiple_ns) / static_cast<double>(env.call_count);
        }
        os << '\n';
    };
    auto print_dispatch_v2c_after_side = [&](const char *prefix, const DispatchV2CTraceProfileEnvelope &env) {
        os << ' ' << prefix << "_active=" << env.active_entries;
        if (!env.valid) {
            return;
        }
        os << ' ' << prefix << "_after_min_us=" << SysCntTicksToUs(env.end_min - kernel_start_min, sys_cnt_multiple_ns)
           << ' ' << prefix << "_after_max_us=" << SysCntTicksToUs(env.end_max - kernel_start_min, sys_cnt_multiple_ns)
           << ' ' << prefix << "_count=" << env.call_count;
    };
    if (dispatch_v2c_trace_valid) {
        os << "rank=" << rank_id
           << " dispatchV2cAfterTrace base=syscnt_min_kernel_start fields=groupFlagSetAfterWaitAfter"
           << " producer=set_from_aiv consumer=wait_on_aic traced_group_count="
           << kDispatchFFNCombineProfileDispatchV2CTraceGroupCount << '\n';
        for (size_t event = 0; event < kDispatchFFNCombineProfileDispatchV2CTraceCount; ++event) {
            const DispatchV2CTraceProfileEnvelope &set_env = dispatch_v2c_set_trace_envelopes[event];
            const DispatchV2CTraceProfileEnvelope &wait_env = dispatch_v2c_wait_trace_envelopes[event];
            os << "  group_idx=" << event - DISPATCH_FFN_COMBINE_PROFILE_DISPATCH_V2C_TRACE_GROUP_BASE;
            if (set_env.valid) {
                os << " set_flag=" << set_env.flag_id;
                if (set_env.mixed_flag) {
                    os << " set_mixed_flag=1";
                }
            }
            if (wait_env.valid) {
                os << " wait_flag=" << wait_env.flag_id;
                if (wait_env.mixed_flag) {
                    os << " wait_mixed_flag=1";
                }
            }
            if (set_env.valid && wait_env.valid) {
                os << " flag_mismatch=" << (set_env.flag_id == wait_env.flag_id ? 0 : 1)
                   << " wait_after_max_before_set_after_min=" << (wait_env.end_max < set_env.end_min ? 1 : 0)
                   << " wait_after_min_after_set_after_max=" << (wait_env.end_min > set_env.end_max ? 1 : 0);
            }
            print_dispatch_v2c_after_side("set", set_env);
            print_dispatch_v2c_after_side("wait", wait_env);
            os << '\n';
        }
    }
    if (gmm2_to_combine_trace_valid) {
        os << "rank=" << rank_id
           << " gmm2ToCombineAfterTrace base=syscnt_min_kernel_start fields=groupFlagSetAfterWaitAfter"
           << " producer=set_from_aic consumer=wait_on_aiv traced_group_count="
           << kDispatchFFNCombineProfileGmm2ToCombineTraceGroupCount << '\n';
        for (size_t event = 0; event < kDispatchFFNCombineProfileGmm2ToCombineTraceCount; ++event) {
            const DispatchV2CTraceProfileEnvelope &set_env = gmm2_to_combine_set_trace_envelopes[event];
            const DispatchV2CTraceProfileEnvelope &wait_env = gmm2_to_combine_wait_trace_envelopes[event];
            os << "  group_idx=" << event - DISPATCH_FFN_COMBINE_PROFILE_GMM2_TO_COMBINE_TRACE_GROUP_BASE;
            if (set_env.valid) {
                os << " set_flag=" << set_env.flag_id;
                if (set_env.mixed_flag) {
                    os << " set_mixed_flag=1";
                }
            }
            if (wait_env.valid) {
                os << " wait_flag=" << wait_env.flag_id;
                if (wait_env.mixed_flag) {
                    os << " wait_mixed_flag=1";
                }
            }
            if (set_env.valid && wait_env.valid) {
                os << " flag_mismatch=" << (set_env.flag_id == wait_env.flag_id ? 0 : 1)
                   << " wait_after_max_before_set_after_min=" << (wait_env.end_max < set_env.end_min ? 1 : 0)
                   << " wait_after_min_after_set_after_max=" << (wait_env.end_min > set_env.end_max ? 1 : 0);
            }
            print_dispatch_v2c_after_side("set", set_env);
            print_dispatch_v2c_after_side("wait", wait_env);
            os << '\n';
        }
    }
    if (gmm2_cv_detail_valid) {
        os << "rank=" << rank_id
           << " gmm2CvDetailEnvelope base=syscnt_min_kernel_start active_aic_only=1"
           << " fields=firstLastTotalCount producer=aic_store_to_dual_aiv_ub\n";
        for (size_t detail = 0; detail < kDispatchFFNCombineProfileGmm2CvDetailCount; ++detail) {
            print_first_last_total_count(Gmm2CvDetailProfileName(detail), gmm2_cv_detail_envelopes[detail], "");
        }
    }
#endif

    if (gmm1_detail_valid) {
        os << "rank=" << rank_id << " gmm1DetailEnvelope base=syscnt_min_kernel_start active_aic_only=1"
           << " fields=firstLastTotalL1TileCount"
           << " timing=async_l1_mn_tile_issue_plus_segment_drain"
           << " v2_compare_boundary=blockMmad\n";
        if (cfg == nullptr || build == nullptr || expert_token_nums == nullptr) {
            for (size_t expert = 0; expert < gmm1_profile_experts; ++expert) {
                const Gmm1DetailProfileEnvelope &env = gmm1_detail_envelopes[expert];
                os << "  expert=" << expert;
                if (!env.valid) {
                    os << " inactive\n";
                    continue;
                }
                os << " active=" << env.active_entries
                   << " start_us=" << SysCntTicksToUs(env.start_min - kernel_start_min, sys_cnt_multiple_ns)
                   << " end_us=" << SysCntTicksToUs(env.end_max - kernel_start_min, sys_cnt_multiple_ns)
                   << " envelope_us=" << SysCntTicksToUs(env.end_max - env.start_min, sys_cnt_multiple_ns)
                   << " max_core_us=" << SysCntTicksToUs(env.max_core_ticks, sys_cnt_multiple_ns)
                   << " sum_core_us=" << SysCntTicksToUs(env.sum_core_ticks, sys_cnt_multiple_ns)
                   << " l1Tiles=" << env.l1_tile_count << " max_l1Tiles=" << env.max_l1_tile_count;
                if (env.l1_tile_count != 0U) {
                    os << " avg_l1_tile_us="
                       << SysCntTicksToUs(env.sum_core_ticks, sys_cnt_multiple_ns) /
                              static_cast<double>(env.l1_tile_count);
                }
                os << '\n';
            }
        } else {
            const auto &gmm1 = build->tiling.gmm1Tiling;
            const uint32_t tile_n = CeilDivU32(cfg->n, gmm1.l1TileN);
            const uint32_t l1_k_blocks = CeilDivU32(cfg->k, gmm1.l1TileK);
            const uint32_t l0_k_steps_per_l1k = CeilDivU32(gmm1.l1TileK, gmm1.l0TileK);
            const uint32_t l0_k_steps_per_mn_tile = CeilDivU32(cfg->k, gmm1.l0TileK);
            const uint32_t packed_weight_k = static_cast<uint32_t>(AlignUpU64(cfg->k, kHostPackedWeightKAlign));
            const uint32_t packed_weight_n = static_cast<uint32_t>(AlignUpU64(cfg->n, kHostPackedWeightInt8NAlign));
            uint32_t group_base = 0;
            for (size_t expert = 0; expert < gmm1_profile_experts; ++expert) {
                const uint32_t current_m_raw =
                    expert < expert_token_nums->size() ? static_cast<uint32_t>((*expert_token_nums)[expert]) : 0U;
                const uint32_t remaining = group_base >= cfg->max_output_size ? 0U : cfg->max_output_size - group_base;
                const uint32_t current_m = current_m_raw > remaining ? remaining : current_m_raw;
                const uint32_t tile_m = CeilDivU32(current_m, gmm1.l1TileM);
                const uint32_t mn_tiles = tile_m * tile_n;
                const uint64_t l0_k_steps_total =
                    static_cast<uint64_t>(mn_tiles) * static_cast<uint64_t>(l0_k_steps_per_mn_tile);
                const Gmm1DetailProfileEnvelope &env = gmm1_detail_envelopes[expert];
                os << "  expert=" << expert << " X=(" << current_m << "," << cfg->k << ")"
                   << " W=(" << cfg->k << "," << cfg->n << ")"
                   << " Wpacked=(" << packed_weight_k << "," << packed_weight_n << ")"
                   << " C=(" << current_m << "," << cfg->n << ")"
                   << " currentMRaw=" << current_m_raw << " groupBase=" << group_base << " l1Tile=(" << gmm1.l1TileM
                   << "," << gmm1.l1TileN << "," << gmm1.l1TileK << ")"
                   << " l0Tile=(" << gmm1.l0TileM << "," << gmm1.l0TileN << "," << gmm1.l0TileK << ")"
                   << " tileGrid=(" << tile_m << "," << tile_n << ")"
                   << " mnTiles=" << mn_tiles << " l1KBlocks=" << l1_k_blocks
                   << " l0KStepsPerL1K=" << l0_k_steps_per_l1k << " l0KStepsPerMnTile=" << l0_k_steps_per_mn_tile
                   << " l0KStepsTotal=" << l0_k_steps_total;
                if (!env.valid) {
                    os << " inactive\n";
                    group_base += current_m;
                    continue;
                }
                os << " active=" << env.active_entries
                   << " start_us=" << SysCntTicksToUs(env.start_min - kernel_start_min, sys_cnt_multiple_ns)
                   << " end_us=" << SysCntTicksToUs(env.end_max - kernel_start_min, sys_cnt_multiple_ns)
                   << " envelope_us=" << SysCntTicksToUs(env.end_max - env.start_min, sys_cnt_multiple_ns)
                   << " max_core_us=" << SysCntTicksToUs(env.max_core_ticks, sys_cnt_multiple_ns)
                   << " sum_core_us=" << SysCntTicksToUs(env.sum_core_ticks, sys_cnt_multiple_ns)
                   << " timedL1Tiles=" << env.l1_tile_count << " max_l1Tiles=" << env.max_l1_tile_count;
                if (env.l1_tile_count != 0U) {
                    const double avg_l1_tile_us = SysCntTicksToUs(env.sum_core_ticks, sys_cnt_multiple_ns) /
                                                  static_cast<double>(env.l1_tile_count);
                    os << " avg_l1_tile_us=" << avg_l1_tile_us;
                    if (l0_k_steps_per_mn_tile != 0U) {
                        os << " derived_avg_l0k_step_us="
                           << avg_l1_tile_us / static_cast<double>(l0_k_steps_per_mn_tile);
                    }
                }
                if (env.sum_core_ticks != 0U) {
                    const double sum_core_us = SysCntTicksToUs(env.sum_core_ticks, sys_cnt_multiple_ns);
                    const double ops = 2.0 * static_cast<double>(current_m) * static_cast<double>(cfg->k) *
                                       static_cast<double>(cfg->n);
                    os << " sum_core_tflops=" << (ops / sum_core_us / 1.0e6);
                }
                os << '\n';
                group_base += current_m;
            }
        }
    }

#if DISPATCH_FFN_COMBINE_V8_ENABLE_DETAIL_PROFILE
    auto print_expert_wall_section =
        [&](const char *name, const char *core_kind,
            const std::array<ExpertWallProfile, kDispatchFFNCombineProfileGmm1DetailMaxExperts> &profiles) {
            os << "rank=" << rank_id << ' ' << name << ' ' << core_kind
               << " base=syscnt_min_kernel_start fields=expertStartEndWallUs\n";
            for (size_t expert = 0; expert < gmm1_profile_experts; ++expert) {
                const ExpertWallProfile &profile = profiles[expert];
                if (!profile.valid) {
                    continue;
                }
                os << "  expert=" << expert
                   << " start_us=" << SysCntTicksToUs(profile.start - kernel_start_min, sys_cnt_multiple_ns)
                   << " end_us=" << SysCntTicksToUs(profile.end - kernel_start_min, sys_cnt_multiple_ns)
                   << " wall_us=" << SysCntTicksToUs(profile.end - profile.start, sys_cnt_multiple_ns) << '\n';
            }
        };
    if (gmm2_aic0_wall_valid) {
        print_expert_wall_section("gmm2ExpertWall", "aic block=0", gmm2_aic0_wall_profiles);
    }
    if (combine_aiv0_wall_valid) {
        print_expert_wall_section("combineExpertWall", "aiv block=0 sub=0", combine_aiv0_wall_profiles);
    }
#endif

    if (combine_detail_valid) {
        os << "rank=" << rank_id
           << " combineDetailEnvelope base=syscnt_min_kernel_start active_aiv_only=1 fields=firstLastTotalCount"
           << " nested_intervals=1\n";
        for (size_t detail = 0; detail < kDispatchFFNCombineProfileCombineDetailCount; ++detail) {
            const CombineDetailProfileEnvelope &env = combine_detail_envelopes[detail];
            if (!env.valid) {
                continue;
            }
            os << "  " << CombineDetailProfileName(detail);
            os << " active=" << env.active_entries
               << " start_us=" << SysCntTicksToUs(env.start_min - kernel_start_min, sys_cnt_multiple_ns)
               << " end_us=" << SysCntTicksToUs(env.end_max - kernel_start_min, sys_cnt_multiple_ns)
               << " envelope_us=" << SysCntTicksToUs(env.end_max - env.start_min, sys_cnt_multiple_ns)
               << " max_core_us=" << SysCntTicksToUs(env.max_core_ticks, sys_cnt_multiple_ns)
               << " sum_core_us=" << SysCntTicksToUs(env.sum_core_ticks, sys_cnt_multiple_ns)
               << " count=" << env.call_count << " max_count=" << env.max_call_count;
            if (env.call_count != 0U) {
                os << " avg_call_us="
                   << SysCntTicksToUs(env.sum_core_ticks, sys_cnt_multiple_ns) / static_cast<double>(env.call_count);
            }
            os << '\n';
        }
    }

    if (unpermute_detail_valid) {
        os << "rank=" << rank_id
           << " unpermuteDetailEnvelope base=syscnt_min_kernel_start active_aiv_only=1 fields=firstLastTotalCount"
           << " nested_intervals=1\n";
        for (size_t detail = 0; detail < kDispatchFFNCombineProfileUnpermuteDetailCount; ++detail) {
            const CombineDetailProfileEnvelope &env = unpermute_detail_envelopes[detail];
            if (!env.valid) {
                continue;
            }
            os << "  " << UnpermuteDetailProfileName(detail);
            os << " active=" << env.active_entries
               << " start_us=" << SysCntTicksToUs(env.start_min - kernel_start_min, sys_cnt_multiple_ns)
               << " end_us=" << SysCntTicksToUs(env.end_max - kernel_start_min, sys_cnt_multiple_ns)
               << " envelope_us=" << SysCntTicksToUs(env.end_max - env.start_min, sys_cnt_multiple_ns)
               << " max_core_us=" << SysCntTicksToUs(env.max_core_ticks, sys_cnt_multiple_ns)
               << " sum_core_us=" << SysCntTicksToUs(env.sum_core_ticks, sys_cnt_multiple_ns)
               << " count=" << env.call_count << " max_count=" << env.max_call_count;
            if (env.call_count != 0U) {
                os << " avg_call_us="
                   << SysCntTicksToUs(env.sum_core_ticks, sys_cnt_multiple_ns) / static_cast<double>(env.call_count);
            }
            os << '\n';
        }
    }

    os << "rank=" << rank_id << " stageProfileCore fields=stageStartReadyStartEndReadyDuration\n";
    for (uint32_t block = 0; block < block_dim; ++block) {
        for (size_t profile_idx = 0; profile_idx < kDispatchFFNCombineProfileEntriesPerBlock; ++profile_idx) {
            const uint64_t *entry = reinterpret_cast<const uint64_t *>(
                profile + static_cast<size_t>(block) * kDispatchFFNCombineProfileBytesPerBlock +
                profile_idx * kDispatchFFNCombineProfileEntryBytes);
            const uint64_t kernel_start = entry[kDispatchFFNCombineProfileKernelStart];
            const uint64_t kernel_end = entry[kDispatchFFNCombineProfileKernelEnd];
            if (kernel_start == 0U && kernel_end == 0U) {
                continue;
            }
            os << "  " << ProfileEntryKind(profile_idx) << " block=" << block;
            if (profile_idx != 0U) {
                os << " sub=" << ProfileEntrySubblock(profile_idx);
            }
            if (kernel_end >= kernel_start) {
                os << " kernel=(" << SysCntTicksToUs(kernel_start - kernel_start_min, sys_cnt_multiple_ns) << ','
                   << SysCntTicksToUs(kernel_end - kernel_start_min, sys_cnt_multiple_ns) << ','
                   << SysCntTicksToUs(kernel_end - kernel_start, sys_cnt_multiple_ns) << ')';
            }
            for (size_t stage = 0; stage < kDispatchFFNCombineProfileStageCount; ++stage) {
                os << ' ' << StageProfileName(stage) << '=';
                if (!ProfileIntervalValid(entry, stage)) {
                    os << '-';
                    continue;
                }
                const uint64_t start = entry[DispatchFFNCombineProfileStageStartIndex(stage)];
                const uint64_t end = entry[DispatchFFNCombineProfileStageEndIndex(stage)];
                const size_t ready_stage = ReadyStageForProfileStage(stage);
                uint64_t duration_start = start;
                os << '(' << SysCntTicksToUs(start - kernel_start_min, sys_cnt_multiple_ns) << ',';
                if (ready_stage < kDispatchFFNCombineProfileReadyStageCount &&
                    ReadyStageProfileIntervalValid(entry, ready_stage)) {
                    const uint64_t ready_start = entry[DispatchFFNCombineProfileReadyStageStartIndex(ready_stage)];
                    duration_start = ready_start;
                    os << SysCntTicksToUs(ready_start - kernel_start_min, sys_cnt_multiple_ns);
                } else {
                    os << '-';
                }
                os << ',' << SysCntTicksToUs(end - kernel_start_min, sys_cnt_multiple_ns) << ','
                   << SysCntTicksToUs(end - duration_start, sys_cnt_multiple_ns) << ')';
            }
            if (profile_idx != 0U && FrontDoneProfileTotalValid(entry)) {
                const uint64_t total_start = entry[DispatchFFNCombineProfileFrontDoneStartIndex(
                    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DONE_STEP7_ENTRY_SYNC)];
                const uint64_t total_end = entry[DispatchFFNCombineProfileFrontDoneEndIndex(
                    DISPATCH_FFN_COMBINE_PROFILE_FRONT_DONE_STEP10_EXIT_SYNC)];
                os << " frontDoneTotal=(" << SysCntTicksToUs(total_start - kernel_start_min, sys_cnt_multiple_ns) << ','
                   << SysCntTicksToUs(total_end - kernel_start_min, sys_cnt_multiple_ns) << ','
                   << SysCntTicksToUs(total_end - total_start, sys_cnt_multiple_ns) << ')';
                for (size_t step = 0; step < kDispatchFFNCombineProfileFrontDoneStepCount; ++step) {
                    os << ' ' << FrontDoneProfileName(step) << '=';
                    if (!FrontDoneProfileIntervalValid(entry, step)) {
                        os << '-';
                        continue;
                    }
                    const uint64_t start = entry[DispatchFFNCombineProfileFrontDoneStartIndex(step)];
                    const uint64_t end = entry[DispatchFFNCombineProfileFrontDoneEndIndex(step)];
                    os << '(' << SysCntTicksToUs(start - kernel_start_min, sys_cnt_multiple_ns) << ','
                       << SysCntTicksToUs(end - kernel_start_min, sys_cnt_multiple_ns) << ','
                       << SysCntTicksToUs(end - start, sys_cnt_multiple_ns) << ')';
                }
            }
            os << '\n';
        }
    }
    return os.str();
}

bool RunStageDebugChecks(int rank_id, int world_size, const CaseConfig &cfg, const DispatchFFNCombineBuildResult &build,
                         const StandaloneRankRuntime &runtime, const DebugDeviceBuffer &workspace_dev,
                         const DebugDeviceBuffer &expert_token_nums_dev, const DebugDeviceBuffer &weight1_dev,
                         const DebugDeviceBuffer &scale1_dev, const DebugDeviceBuffer &weight2_dev,
                         const DebugDeviceBuffer &scale2_dev, const DebugDeviceBuffer &probs_dev,
                         const DebugDeviceBuffer &out_dev, const std::vector<uint8_t> &expert_idx,
                         const std::vector<uint8_t> &x, const std::vector<uint8_t> &weight1,
                         const std::vector<uint8_t> &scale1, const std::vector<uint8_t> &weight2,
                         const std::vector<uint8_t> &scale2, const std::vector<uint8_t> &probs,
                         const std::vector<uint16_t> &actual_out, const std::string &case_dir, bool skip_accuracy)
{
    bool stage_check_pass = true;
    if (skip_accuracy) {
        const auto &front = build.tiling.frontReorderTiling;
        [[maybe_unused]] const auto &frontDebug = build.tiling.frontDebugTiling;
        const auto &dispatch = build.tiling.dispatchTiling;
        {
            std::ostringstream os;
            os << "rank=" << rank_id << " stageDebugGate"
               << " stage=" << front.stageNum << " frontDebug=" << frontDebug.frontDebugMode
               << " frontStop=" << frontDebug.frontStopStep
               << " dispatchGatherDebug=" << dispatch.dispatchGatherDebugMode
               << " dispatchGatherStop=" << dispatch.dispatchGatherStopStep
               << " gmm1Debug=" << build.tiling.gmm1Tiling.gmm1DebugMode
               << " swigluDebug=" << build.tiling.swigluTiling.swigluDebugMode
               << " gmm2Debug=" << build.tiling.gmm2Tiling.gmm2DebugMode
               << " combineDebug=" << build.tiling.combineTiling.combineDebugMode
               << " combineStop=" << build.tiling.combineTiling.combineStopStep
               << " unpermuteDebug=" << build.tiling.unpermuteTiling.unpermuteDebugMode;
            PrintOrderedByRank(rank_id, world_size, os.str());
        }
        if (frontDebug.frontMode != 0U && (frontDebug.frontDebugMode != 0U || frontDebug.frontStopStep != 0U)) {
            const bool front_full_load = front.frontCase == 21000U;
            const bool front_postprocess =
                front.frontCase == 21000U || front.frontCase == 11000U || front.frontCase == 11010U;
            const DispatchLayoutReport front_layout_report = CheckFrontLayout(cfg, build, runtime, workspace_dev);
            stage_check_pass = stage_check_pass && front_layout_report.pass;
            PrintOrderedByRank(rank_id, world_size, BuildFrontLayoutReportText(rank_id, front_layout_report));
            if (front_full_load || (front_postprocess && frontDebug.frontStopStep == 8U)) {
                const CountExchangeReport count_exchange_report =
                    CheckCountExchange(cfg, build, runtime, workspace_dev, expert_idx);
                stage_check_pass = stage_check_pass && count_exchange_report.pass;
                PrintOrderedByRank(rank_id, world_size, BuildCountExchangeReportText(rank_id, count_exchange_report));
                const CumsumReport cumsum_report =
                    CheckCumsum(cfg, build, runtime, workspace_dev, expert_token_nums_dev, expert_idx);
                stage_check_pass = stage_check_pass && cumsum_report.pass;
                PrintOrderedByRank(rank_id, world_size, BuildCumsumReportText(rank_id, cumsum_report));
                const FrontDispatchContractReport front_dispatch_contract_report =
                    CheckFrontDispatchContract(cfg, build, runtime, workspace_dev, expert_token_nums_dev, expert_idx);
                stage_check_pass = stage_check_pass && front_dispatch_contract_report.pass;
                PrintOrderedByRank(rank_id, world_size,
                                   BuildFrontDispatchContractReportText(rank_id, front_dispatch_contract_report));
            } else if (frontDebug.frontStopStep == 8U) {
                const FrontDispatchContractReport front_dispatch_contract_report =
                    CheckFrontDispatchContract(cfg, build, runtime, workspace_dev, expert_token_nums_dev, expert_idx);
                stage_check_pass = stage_check_pass && front_dispatch_contract_report.pass;
                PrintOrderedByRank(rank_id, world_size,
                                   BuildFrontDispatchContractReportText(rank_id, front_dispatch_contract_report));
            }
            if (frontDebug.frontStopStep != 0U && dispatch.dispatchGatherStopStep == 0U) {
                return stage_check_pass;
            }
        }
        if (dispatch.dispatchGatherMode != 0U &&
            (dispatch.dispatchGatherDebugMode != 0U || dispatch.dispatchGatherStopStep != 0U)) {
            const DispatchLayoutReport dispatch_gather_layout_report =
                CheckDispatchGatherLayout(cfg, build, runtime, workspace_dev);
            stage_check_pass = stage_check_pass && dispatch_gather_layout_report.pass;
            PrintOrderedByRank(rank_id, world_size,
                               BuildDispatchGatherLayoutReportText(rank_id, dispatch_gather_layout_report));
            if (dispatch.dispatchGatherStopStep != 0U) {
                if (dispatch.dispatchGatherStopStep >= 2U) {
                    const DispatchGatherDetailReport dispatch_gather_rank_split_report =
                        CheckDispatchGatherRankSplit(cfg, build, runtime, workspace_dev, x, expert_idx);
                    stage_check_pass = stage_check_pass && dispatch_gather_rank_split_report.pass;
                    const uint32_t active_copy_cores =
                        std::min(static_cast<uint32_t>(runtime.hccl.world_size), cfg.aiv_num);
                    const uint32_t inactive_copy_cores = cfg.aiv_num - active_copy_cores;
                    PrintOrderedByRank(
                        rank_id, world_size,
                        BuildDispatchGatherRankSplitReportText(rank_id, dispatch_gather_rank_split_report,
                                                               active_copy_cores, inactive_copy_cores));
                }
                if (dispatch.dispatchGatherStopStep >= 3U && build.tiling.gmm1Tiling.gmm1DebugMode != 0U) {
                    const Gmm1LayoutReport gmm1_layout_report =
                        CheckGmm1Layout(cfg, build, workspace_dev, weight1_dev, scale1_dev, expert_token_nums_dev);
                    stage_check_pass = stage_check_pass && gmm1_layout_report.pass;
                    PrintOrderedByRank(rank_id, world_size, BuildGmm1LayoutReportText(rank_id, gmm1_layout_report));
                    const Gmm1SyncReport gmm1_sync_report =
                        CheckGmm1Sync(cfg, build, runtime, workspace_dev, expert_idx);
                    stage_check_pass = stage_check_pass && gmm1_sync_report.pass;
                    PrintOrderedByRank(rank_id, world_size, BuildGmm1SyncReportText(rank_id, gmm1_sync_report));
                    const Gmm1TaskReport gmm1_task_report =
                        CheckGmm1Task(cfg, build, runtime, workspace_dev, expert_idx);
                    stage_check_pass = stage_check_pass && gmm1_task_report.pass;
                    PrintOrderedByRank(rank_id, world_size, BuildGmm1TaskReportText(rank_id, gmm1_task_report));
                    const Gmm1DoneReport gmm1_done_report =
                        CheckGmm1Done(cfg, build, runtime, workspace_dev, expert_idx);
                    stage_check_pass = stage_check_pass && gmm1_done_report.pass;
                    PrintOrderedByRank(rank_id, world_size, BuildGmm1DoneReportText(rank_id, gmm1_done_report));
                }
                return stage_check_pass;
            }
        }
        if (frontDebug.frontCheckMode != 0U && frontDebug.frontCheckDebugMode != 0U) {
            const FrontCheckLayoutReport front_check_layout_report =
                CheckFrontCheckLayout(cfg, build, runtime, workspace_dev);
            stage_check_pass = stage_check_pass && front_check_layout_report.pass;
            PrintOrderedByRank(rank_id, world_size,
                               BuildFrontCheckLayoutReportText(rank_id, front_check_layout_report));
            if (frontDebug.frontCheckStopStep == 2U && (front.frontCase == 21000U || front.frontCase == 11000U)) {
                const FrontSortCheckReport front_sort_check_report = CheckFrontSort(build, workspace_dev, expert_idx);
                stage_check_pass = stage_check_pass && front_sort_check_report.pass;
                PrintOrderedByRank(rank_id, world_size,
                                   BuildFrontSortCheckReportText(rank_id, front_sort_check_report));
            }
            if (frontDebug.frontCheckStopStep == 2U && front.frontCase == 11010U &&
                frontDebug.frontSortCheckDebugStep == 1U) {
                const FrontRunSortReport front_run_sort_report = CheckFrontRunSort(build, workspace_dev, expert_idx);
                stage_check_pass = stage_check_pass && front_run_sort_report.pass;
                PrintOrderedByRank(rank_id, world_size, BuildFrontRunSortReportText(rank_id, front_run_sort_report));
            }
            if (frontDebug.frontCheckStopStep == 2U && front.frontCase == 11010U &&
                frontDebug.frontSortCheckDebugStep == 2U) {
                const FrontRunSortReport front_local_merge_report = CheckFrontRunSort(build, workspace_dev, expert_idx);
                stage_check_pass = stage_check_pass && front_local_merge_report.pass;
                PrintOrderedByRank(rank_id, world_size,
                                   BuildFrontLocalMergeReportText(rank_id, front_local_merge_report));
            }
            if (frontDebug.frontCheckStopStep == 2U && front.frontCase == 11010U &&
                frontDebug.frontSortCheckDebugStep == 3U) {
                const FrontRunSortReport front_middle_merge_report =
                    CheckFrontMiddleMerge(build, workspace_dev, expert_idx);
                stage_check_pass = stage_check_pass && front_middle_merge_report.pass;
                PrintOrderedByRank(rank_id, world_size,
                                   BuildFrontMiddleMergeReportText(rank_id, front_middle_merge_report));
            }
            if (frontDebug.frontCheckStopStep == 2U && front.frontCase == 11010U &&
                frontDebug.frontSortCheckDebugStep == 4U) {
                const FrontSortCheckReport front_merge_out_report = CheckFrontSort(build, workspace_dev, expert_idx);
                stage_check_pass = stage_check_pass && front_merge_out_report.pass;
                PrintOrderedByRank(rank_id, world_size, BuildFrontMergeOutReportText(rank_id, front_merge_out_report));
            }
            if (frontDebug.frontCheckStopStep >= 3U &&
                (front.frontCase == 21000U || front.frontCase == 11000U || front.frontCase == 11010U)) {
                const FrontMetadataReport front_metadata_report =
                    CheckFrontMetadata(cfg, build, workspace_dev, expert_idx);
                stage_check_pass = stage_check_pass && front_metadata_report.pass;
                PrintOrderedByRank(rank_id, world_size, BuildFrontMetadataReportText(rank_id, front_metadata_report));
            }
            if (frontDebug.frontCheckStopStep == 4U &&
                (front.frontCase == 21000U || front.frontCase == 11000U || front.frontCase == 11010U)) {
                const ScatterQuantReport front_quant_report =
                    CheckScatterQuant(cfg, build, runtime, workspace_dev, x, expert_idx);
                stage_check_pass = stage_check_pass && front_quant_report.pass;
                PrintOrderedByRank(rank_id, world_size, BuildFrontQuantReportText(rank_id, front_quant_report));
            }
            if (frontDebug.frontCheckStopStep >= 5U &&
                (front.frontCase == 21000U || front.frontCase == 11000U || front.frontCase == 11010U)) {
                const CountExchangeReport count_exchange_report =
                    CheckCountExchange(cfg, build, runtime, workspace_dev, expert_idx);
                stage_check_pass = stage_check_pass && count_exchange_report.pass;
                PrintOrderedByRank(rank_id, world_size, BuildCountExchangeReportText(rank_id, count_exchange_report));
            }
            if (frontDebug.frontCheckStopStep >= 6U &&
                (front.frontCase == 21000U || front.frontCase == 11000U || front.frontCase == 11010U)) {
                const CumsumReport cumsum_report =
                    CheckCumsum(cfg, build, runtime, workspace_dev, expert_token_nums_dev, expert_idx);
                stage_check_pass = stage_check_pass && cumsum_report.pass;
                PrintOrderedByRank(rank_id, world_size, BuildCumsumReportText(rank_id, cumsum_report));
            }
            if (frontDebug.frontCheckStopStep >= 7U &&
                (front.frontCase == 21000U || front.frontCase == 11000U || front.frontCase == 11010U)) {
                const FrontDispatchContractReport front_dispatch_contract_report =
                    CheckFrontDispatchContract(cfg, build, runtime, workspace_dev, expert_token_nums_dev, expert_idx);
                stage_check_pass = stage_check_pass && front_dispatch_contract_report.pass;
                PrintOrderedByRank(rank_id, world_size,
                                   BuildFrontDispatchContractReportText(rank_id, front_dispatch_contract_report));
                const FrontDoneReport front_done_inactive_report = CheckFrontDoneInactive(runtime);
                stage_check_pass = stage_check_pass && front_done_inactive_report.pass;
                PrintOrderedByRank(rank_id, world_size,
                                   BuildFrontDoneInactiveReportText(rank_id, front_done_inactive_report));
            }
        }
        if (frontDebug.frontCheckMode != 0U && frontDebug.frontCheckStopStep != 0U) {
            return stage_check_pass;
        }
        const bool legacy_front_debug_enabled = frontDebug.frontCheckDebugMode != 0U ||
                                                frontDebug.frontCheckStopStep != 0U ||
                                                frontDebug.smallFrontDebugMode != 0U;
        if (legacy_front_debug_enabled) {
            if (front.frontPath == 1U && frontDebug.smallFrontMode != 0U && frontDebug.smallFrontDebugMode != 0U &&
                front.stageNum >= 3U) {
                const FrontSortReport small_front_sort_report = CheckSmallFrontSort(build, workspace_dev, expert_idx);
                stage_check_pass = stage_check_pass && small_front_sort_report.pass;
                PrintOrderedByRank(rank_id, world_size, BuildFrontSortReportText(rank_id, small_front_sort_report));
                PrintOrderedByRank(rank_id, world_size, BuildFrontRouteReportText(rank_id, small_front_sort_report));
            }

            if (front.stageNum >= 3U) {
                const CoreCountReport core_count_report = CheckCoreCount(cfg, build, workspace_dev, expert_idx);
                stage_check_pass = stage_check_pass && core_count_report.pass;
                PrintOrderedByRank(rank_id, world_size, BuildCoreCountReportText(rank_id, core_count_report));
            }

            const bool base_cursor_ready = front.frontPath == 1U ? front.stageNum >= 3U : front.stageNum >= 4U;
            if (base_cursor_ready) {
                const BaseCursorReport base_cursor_report = CheckBaseCursor(cfg, build, workspace_dev, expert_idx);
                stage_check_pass = stage_check_pass && base_cursor_report.pass;
                PrintOrderedByRank(rank_id, world_size, BuildBaseCursorReportText(rank_id, base_cursor_report));
            }

            if (front.stageNum >= 5U) {
                const ScatterQuantReport scatter_quant_report =
                    CheckScatterQuant(cfg, build, runtime, workspace_dev, x, expert_idx);
                stage_check_pass = stage_check_pass && scatter_quant_report.pass;
                PrintOrderedByRank(rank_id, world_size, BuildScatterQuantReportText(rank_id, scatter_quant_report));
            }

            if (front.stageNum >= 6U) {
                const CountExchangeReport count_exchange_report =
                    CheckCountExchange(cfg, build, runtime, workspace_dev, expert_idx);
                stage_check_pass = stage_check_pass && count_exchange_report.pass;
                PrintOrderedByRank(rank_id, world_size, BuildCountExchangeReportText(rank_id, count_exchange_report));
            }

            if (front.stageNum >= 7U) {
                const CumsumReport cumsum_report =
                    CheckCumsum(cfg, build, runtime, workspace_dev, expert_token_nums_dev, expert_idx);
                stage_check_pass = stage_check_pass && cumsum_report.pass;
                PrintOrderedByRank(rank_id, world_size, BuildCumsumReportText(rank_id, cumsum_report));
            }

            if (front.stageNum >= 8U) {
                const FrontDoneReport front_done_report = CheckFrontDone(runtime);
                stage_check_pass = stage_check_pass && front_done_report.pass;
                PrintOrderedByRank(rank_id, world_size, BuildFrontDoneReportText(rank_id, front_done_report));
            }
        }
    }

    if (skip_accuracy && build.tiling.dispatchTiling.dispatchDebugMode != 0U) {
        const DispatchLayoutReport dispatch_layout_report = CheckDispatchLayout(cfg, build, runtime, workspace_dev);
        stage_check_pass = stage_check_pass && dispatch_layout_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildDispatchLayoutReportText(rank_id, dispatch_layout_report));
        const DispatchMetadataReport dispatch_metadata_report =
            CheckDispatchMetadata(cfg, build, runtime, workspace_dev, expert_idx);
        stage_check_pass = stage_check_pass && dispatch_metadata_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildDispatchMetadataReportText(rank_id, dispatch_metadata_report));
        const DispatchTaskReport dispatch_task_report =
            CheckDispatchTaskSplit(cfg, build, runtime, workspace_dev, expert_idx);
        stage_check_pass = stage_check_pass && dispatch_task_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildDispatchTaskReportText(rank_id, dispatch_task_report));
        const DispatchGatherReport dispatch_gather_report =
            CheckDispatchGather(cfg, build, runtime, workspace_dev, expert_idx);
        stage_check_pass = stage_check_pass && dispatch_gather_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildDispatchGatherReportText(rank_id, dispatch_gather_report));
        const DispatchGroupDoneReport dispatch_group_done_report =
            CheckDispatchGroupDone(cfg, build, runtime, workspace_dev, expert_idx);
        stage_check_pass = stage_check_pass && dispatch_group_done_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildDispatchGroupDoneReportText(rank_id, dispatch_group_done_report));
        const DispatchGatherDetailReport dispatch_gather_detail_report =
            CheckDispatchGatherDetail(cfg, build, runtime, workspace_dev, x, expert_idx);
        stage_check_pass = stage_check_pass && dispatch_gather_detail_report.pass;
        PrintOrderedByRank(rank_id, world_size,
                           BuildDispatchGatherDetailReportText(rank_id, dispatch_gather_detail_report));
    }

    if (skip_accuracy && build.tiling.gmm1Tiling.gmm1DebugMode != 0U) {
        const Gmm1LayoutReport gmm1_layout_report =
            CheckGmm1Layout(cfg, build, workspace_dev, weight1_dev, scale1_dev, expert_token_nums_dev);
        stage_check_pass = stage_check_pass && gmm1_layout_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildGmm1LayoutReportText(rank_id, gmm1_layout_report));
        const Gmm1SyncReport gmm1_sync_report = CheckGmm1Sync(cfg, build, runtime, workspace_dev, expert_idx);
        stage_check_pass = stage_check_pass && gmm1_sync_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildGmm1SyncReportText(rank_id, gmm1_sync_report));
        const Gmm1TaskReport gmm1_task_report = CheckGmm1Task(cfg, build, runtime, workspace_dev, expert_idx);
        stage_check_pass = stage_check_pass && gmm1_task_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildGmm1TaskReportText(rank_id, gmm1_task_report));
        const Gmm1OutputReport gmm1_output_report =
            CheckGmm1Output(cfg, build, runtime, workspace_dev, expert_idx, weight1, scale1, case_dir);
        stage_check_pass = stage_check_pass && gmm1_output_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildGmm1OutputReportText(rank_id, gmm1_output_report));
        const Gmm1SegmentPlanReport gmm1_segment_plan_report = CheckGmm1SegmentPlan(cfg, build, workspace_dev);
        stage_check_pass = stage_check_pass && gmm1_segment_plan_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildGmm1SegmentPlanReportText(rank_id, gmm1_segment_plan_report));
        const Gmm1DoneReport gmm1_done_report = CheckGmm1Done(cfg, build, runtime, workspace_dev, expert_idx);
        stage_check_pass = stage_check_pass && gmm1_done_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildGmm1DoneReportText(rank_id, gmm1_done_report));
        const Gmm1DetailReport gmm1_detail_report =
            CheckGmm1Detail(cfg, build, runtime, workspace_dev, expert_idx, weight1, scale1);
        stage_check_pass = stage_check_pass && gmm1_detail_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildGmm1DetailReportText(rank_id, gmm1_detail_report));
    }

    if (skip_accuracy && build.tiling.swigluTiling.swigluDebugMode != 0U) {
        const SwigluLayoutReport swiglu_layout_report =
            CheckSwigluLayout(cfg, build, workspace_dev, expert_token_nums_dev);
        stage_check_pass = stage_check_pass && swiglu_layout_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildSwigluLayoutReportText(rank_id, swiglu_layout_report));
        const SwigluMetadataModeReport swiglu_metadata_mode_report = CheckSwigluMetadataMode(cfg, build, workspace_dev);
        stage_check_pass = stage_check_pass && swiglu_metadata_mode_report.pass;
        PrintOrderedByRank(rank_id, world_size,
                           BuildSwigluMetadataModeReportText(rank_id, swiglu_metadata_mode_report));
        const SwigluC2VReport swiglu_c2v_report = CheckSwigluC2VOverlap(cfg, build, runtime, workspace_dev);
        stage_check_pass = stage_check_pass && swiglu_c2v_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildSwigluC2VReportText(rank_id, swiglu_c2v_report));
        const SwigluSegmentReport swiglu_segment_report =
            CheckSwigluSegment(cfg, build, runtime, workspace_dev, expert_idx);
        stage_check_pass = stage_check_pass && swiglu_segment_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildSwigluSegmentReportText(rank_id, swiglu_segment_report));
        const SwigluSegmentMetaReport swiglu_segment_meta_report =
            CheckSwigluSegmentMeta(cfg, build, runtime, workspace_dev, expert_idx);
        stage_check_pass = stage_check_pass && swiglu_segment_meta_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildSwigluSegmentMetaReportText(rank_id, swiglu_segment_meta_report));
        const SwigluTaskReport swiglu_task_report = CheckSwigluTask(cfg, build, runtime, workspace_dev, expert_idx);
        stage_check_pass = stage_check_pass && swiglu_task_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildSwigluTaskReportText(rank_id, swiglu_task_report));
        const SwigluOutputReport swiglu_output_report =
            CheckSwigluOutput(cfg, build, runtime, workspace_dev, expert_idx);
        stage_check_pass = stage_check_pass && swiglu_output_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildSwigluOutputReportText(rank_id, swiglu_output_report));
        PrintOrderedByRank(rank_id, world_size, BuildSwigluDetailReportText(rank_id, swiglu_output_report));
        const SwigluDoneReport swiglu_done_report = CheckSwigluDone(cfg, build, runtime, workspace_dev, expert_idx);
        stage_check_pass = stage_check_pass && swiglu_done_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildSwigluDoneReportText(rank_id, swiglu_done_report));
    }

    if (skip_accuracy && build.tiling.gmm2Tiling.gmm2DebugMode != 0U) {
        const Gmm2LayoutReport gmm2_layout_report =
            CheckGmm2Layout(cfg, build, workspace_dev, weight2_dev, scale2_dev, expert_token_nums_dev);
        stage_check_pass = stage_check_pass && gmm2_layout_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildGmm2LayoutReportText(rank_id, gmm2_layout_report));
        const Gmm2V2CReport gmm2_v2c_report = CheckGmm2V2COverlap(cfg, build, runtime, workspace_dev);
        stage_check_pass = stage_check_pass && gmm2_v2c_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildGmm2V2CReportText(rank_id, gmm2_v2c_report));
        if (build.tiling.swigluTiling.swigluDebugMode != 0U) {
            const SwigluFinalSyncReport swiglu_final_sync_disabled_report =
                CheckSwigluFinalSyncDisabled(build, workspace_dev, cfg.aiv_num);
            stage_check_pass = stage_check_pass && swiglu_final_sync_disabled_report.pass;
            PrintOrderedByRank(rank_id, world_size,
                               BuildSwigluFinalSyncDisabledReportText(rank_id, swiglu_final_sync_disabled_report));
        }
        const Gmm2SegmentReport gmm2_segment_report = CheckGmm2Segment(cfg, build, runtime, workspace_dev, expert_idx);
        stage_check_pass = stage_check_pass && gmm2_segment_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildGmm2SegmentReportText(rank_id, gmm2_segment_report));
        const Gmm2TaskReport gmm2_task_report = CheckGmm2Task(cfg, build, runtime, workspace_dev, expert_idx);
        stage_check_pass = stage_check_pass && gmm2_task_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildGmm2TaskReportText(rank_id, gmm2_task_report));
        const Gmm2DoneReport gmm2_done_report = CheckGmm2Done(cfg, build, runtime, workspace_dev);
        stage_check_pass = stage_check_pass && gmm2_done_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildGmm2DoneReportText(rank_id, gmm2_done_report));
        const Gmm2FinalSyncReport gmm2_final_sync_report = CheckGmm2FinalSync(cfg, build, runtime, workspace_dev);
        stage_check_pass = stage_check_pass && gmm2_final_sync_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildGmm2FinalSyncDisabledReportText(rank_id, gmm2_final_sync_report));
        const Gmm2OutputReport gmm2_output_report =
            CheckGmm2Output(cfg, build, runtime, workspace_dev, expert_idx, weight2, scale2);
        stage_check_pass = stage_check_pass && gmm2_output_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildGmm2OutputReportText(rank_id, gmm2_output_report));
        PrintOrderedByRank(rank_id, world_size, BuildGmm2DetailReportText(rank_id, gmm2_output_report));
    }

    if (skip_accuracy && build.tiling.combineTiling.combineDebugMode != 0U) {
        PrintOrderedByRank(rank_id, world_size, "rank=" + std::to_string(rank_id) + " combineCheckEnter");
        const CombineLayoutReport combine_layout_report = CheckCombineLayout(cfg, build, runtime, workspace_dev);
        stage_check_pass = stage_check_pass && combine_layout_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildCombineLayoutReportText(rank_id, combine_layout_report));
        if (build.tiling.combineTiling.combineStopStep == 1U) {
            return stage_check_pass;
        }
        if (build.tiling.combineTiling.combineStopStep == 2U) {
            const CombineTaskReport combine_task_report =
                CheckCombineTask(cfg, build, runtime, workspace_dev, expert_idx);
            stage_check_pass = stage_check_pass && combine_task_report.pass;
            PrintOrderedByRank(rank_id, world_size, BuildCombineTaskReportText(rank_id, combine_task_report));
            return stage_check_pass;
        }
        const CombineReadyReport combine_ready_report =
            CheckCombineGmm2Ready(cfg, build, runtime, workspace_dev, expert_idx);
        stage_check_pass = stage_check_pass && combine_ready_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildCombineReadyReportText(rank_id, combine_ready_report));
        if (build.tiling.combineTiling.combineStopStep == 3U) {
            return stage_check_pass;
        }
        if (build.tiling.combineTiling.combineStopStep >= 4U) {
            const CombineLoadReport combine_load_report =
                CheckCombineLoad(cfg, build, runtime, workspace_dev, expert_idx);
            stage_check_pass = stage_check_pass && combine_load_report.pass;
            PrintOrderedByRank(rank_id, world_size, BuildCombineLoadReportText(rank_id, combine_load_report));
            if (build.tiling.combineTiling.combineStopStep == 4U) {
                return stage_check_pass;
            }
            const CombineDequantReport combine_dequant_report =
                CheckCombineDequant(cfg, build, runtime, workspace_dev, expert_idx);
            stage_check_pass = stage_check_pass && combine_dequant_report.pass;
            PrintOrderedByRank(rank_id, world_size, BuildCombineDequantReportText(rank_id, combine_dequant_report));
            if (build.tiling.combineTiling.combineStopStep == 5U) {
                return stage_check_pass;
            }
            if (build.tiling.combineTiling.combineStopStep >= 6U) {
                const CombineStoreReport combine_store_report =
                    CheckCombineStore(cfg, build, runtime, workspace_dev, expert_idx);
                stage_check_pass = stage_check_pass && combine_store_report.pass;
                PrintOrderedByRank(rank_id, world_size, BuildCombineStoreReportText(rank_id, combine_store_report));
                if (build.tiling.combineTiling.combineStopStep == 6U) {
                    return stage_check_pass;
                }
                const CombineFinalizeReport combine_finalize_report =
                    CheckCombineFinalize(cfg, build, runtime, workspace_dev);
                stage_check_pass = stage_check_pass && combine_finalize_report.pass;
                PrintOrderedByRank(rank_id, world_size,
                                   BuildCombineFinalizeReportText(rank_id, combine_finalize_report));
                return stage_check_pass;
            }
        }
        if (build.tiling.combineTiling.combineStopStep == 2U) {
            return stage_check_pass;
        }
        const CombineMetaReport combine_meta_report = CheckCombineMeta(cfg, build, runtime, workspace_dev, expert_idx);
        stage_check_pass = stage_check_pass && combine_meta_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildCombineMetaReportText(rank_id, combine_meta_report));
        if (build.tiling.combineTiling.combineStopStep == 3U) {
            return stage_check_pass;
        }
        if (build.tiling.combineTiling.combineStopStep >= 4U) {
            const CombineLoadReport combine_load_report =
                CheckCombineLoad(cfg, build, runtime, workspace_dev, expert_idx);
            stage_check_pass = stage_check_pass && combine_load_report.pass;
            PrintOrderedByRank(rank_id, world_size, BuildCombineLoadReportText(rank_id, combine_load_report));
            if (build.tiling.combineTiling.combineStopStep == 4U) {
                return stage_check_pass;
            }
        }
        if (build.tiling.combineTiling.combineStopStep >= 5U) {
            const CombineDequantReport combine_dequant_report =
                CheckCombineDequant(cfg, build, runtime, workspace_dev, expert_idx);
            stage_check_pass = stage_check_pass && combine_dequant_report.pass;
            PrintOrderedByRank(rank_id, world_size, BuildCombineDequantReportText(rank_id, combine_dequant_report));
            if (build.tiling.combineTiling.combineStopStep == 5U) {
                return stage_check_pass;
            }
        }
        if (build.tiling.combineTiling.combineStopStep >= 6U) {
            const CombineStoreReport combine_store_report =
                CheckCombineStore(cfg, build, runtime, workspace_dev, expert_idx);
            stage_check_pass = stage_check_pass && combine_store_report.pass;
            PrintOrderedByRank(rank_id, world_size, BuildCombineStoreReportText(rank_id, combine_store_report));
            if (build.tiling.combineTiling.combineStopStep == 6U) {
                return stage_check_pass;
            }
        }
        if (build.tiling.combineTiling.combineStopStep >= 7U) {
            const CombineFinalizeReport combine_finalize_report =
                CheckCombineFinalize(cfg, build, runtime, workspace_dev);
            stage_check_pass = stage_check_pass && combine_finalize_report.pass;
            PrintOrderedByRank(rank_id, world_size, BuildCombineFinalizeReportText(rank_id, combine_finalize_report));
            if (build.tiling.combineTiling.combineStopStep == 7U) {
                return stage_check_pass;
            }
        }
        const CombineTaskReport combine_task_report = CheckCombineTask(cfg, build, runtime, workspace_dev, expert_idx);
        stage_check_pass = stage_check_pass && combine_task_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildCombineTaskReportText(rank_id, combine_task_report));
        const CombineDoneReport combine_done_report = CheckCombineDone(cfg, build, runtime, workspace_dev, expert_idx);
        stage_check_pass = stage_check_pass && combine_done_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildCombineDoneReportText(rank_id, combine_done_report));
        const CombineDetailReport combine_detail_report =
            CheckCombineDetail(cfg, build, runtime, workspace_dev, expert_idx);
        stage_check_pass = stage_check_pass && combine_detail_report.pass;
        PrintOrderedByRank(rank_id, world_size,
                           BuildCombineStageReportText(rank_id, "combineOutput", combine_detail_report));
        PrintOrderedByRank(rank_id, world_size,
                           BuildCombineStageReportText(rank_id, "combineWriteBack", combine_detail_report));
        PrintOrderedByRank(rank_id, world_size, BuildCombineDetailReportText(rank_id, combine_detail_report));
    }

    if (build.tiling.unpermuteTiling.unpermuteDebugMode != 0U) {
        const UnpermuteLayoutReport unpermute_layout_report =
            CheckUnpermuteLayout(cfg, build, runtime, workspace_dev, probs_dev, out_dev);
        stage_check_pass = stage_check_pass && unpermute_layout_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildUnpermuteLayoutReportText(rank_id, unpermute_layout_report));
        const UnpermuteIndexedReport unpermute_task_report = CheckUnpermuteTask(cfg, build, runtime, workspace_dev);
        stage_check_pass = stage_check_pass && unpermute_task_report.pass;
        PrintOrderedByRank(rank_id, world_size,
                           BuildUnpermuteIndexedReportText(rank_id, "unpermuteTask", unpermute_task_report));
        const UnpermuteIndexedReport unpermute_meta_report =
            CheckUnpermuteMeta(cfg, build, runtime, workspace_dev, probs);
        stage_check_pass = stage_check_pass && unpermute_meta_report.pass;
        PrintOrderedByRank(rank_id, world_size,
                           BuildUnpermuteIndexedReportText(rank_id, "unpermuteMeta", unpermute_meta_report));
        const UnpermuteIndexedReport unpermute_accum_report = CheckUnpermuteAccum(cfg, build, runtime, workspace_dev);
        stage_check_pass = stage_check_pass && unpermute_accum_report.pass;
        PrintOrderedByRank(rank_id, world_size,
                           BuildUnpermuteIndexedReportText(rank_id, "unpermuteAccum", unpermute_accum_report));
        const UnpermuteIndexedReport unpermute_output_report = CheckUnpermuteOutput(cfg, build, runtime, workspace_dev);
        stage_check_pass = stage_check_pass && unpermute_output_report.pass;
        PrintOrderedByRank(rank_id, world_size,
                           BuildUnpermuteIndexedReportText(rank_id, "unpermuteOutput", unpermute_output_report));
        const UnpermuteIndexedReport unpermute_done_report = CheckUnpermuteDone(cfg, build, runtime, workspace_dev);
        stage_check_pass = stage_check_pass && unpermute_done_report.pass;
        PrintOrderedByRank(rank_id, world_size,
                           BuildUnpermuteIndexedReportText(rank_id, "unpermuteDone", unpermute_done_report));
        const UnpermuteDetailReport unpermute_detail_report =
            CheckUnpermuteDetail(cfg, build, runtime, workspace_dev, probs, actual_out);
        stage_check_pass = stage_check_pass && unpermute_detail_report.pass;
        PrintOrderedByRank(rank_id, world_size, BuildUnpermuteDetailReportText(rank_id, unpermute_detail_report));
    }
    return stage_check_pass;
}
