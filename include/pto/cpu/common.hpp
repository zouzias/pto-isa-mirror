#ifndef COMMON_HPP
#define COMMON_HPP

#include <pto/common/type.hpp>

namespace pto {

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantMode_t GetCastPreQuantMode()
{
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    if constexpr (std::is_same<SrcType, float>::value) {
        if constexpr (std::is_same<DstType, half>::value) {
            quantPre = QuantMode_t::F322F16;
        } else if constexpr (std::is_same<DstType, bfloat16_t>::value) {
            quantPre = QuantMode_t::F322BF16;
        }
    }
    return quantPre;
}

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantMode_t GetScalarPreQuantMode()
{
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    if constexpr (std::is_same<SrcType, float>::value) {
        if constexpr ((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value)) {
            quantPre = QuantMode_t::QF322B8_PRE;
        } else if constexpr ((std::is_same<DstType, __gm__ half>::value) || (std::is_same<DstType, half>::value)) {
            quantPre = QuantMode_t::QF322F16_PRE;
        } else if constexpr (std::is_same<DstType, bfloat16_t>::value) {
            quantPre = QuantMode_t::QF322BF16_PRE;
        }
    } else if constexpr (std::is_same<SrcType, int32_t>::value) {
        if constexpr ((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value)) {
            quantPre = QuantMode_t::REQ8;
        } else if constexpr (std::is_same<DstType, half>::value) {
            quantPre = QuantMode_t::DEQF16;
        } else if constexpr (std::is_same<DstType, int16_t>::value) {
            quantPre = QuantMode_t::SHIFTS322S16;
        }
    }
    return quantPre;
}

template <typename SrcType, typename DstType>
PTO_INTERNAL constexpr QuantMode_t GetVectorPreQuantMode()
{
    QuantMode_t quantPre = QuantMode_t::NoQuant;
    if constexpr (std::is_same<SrcType, float>::value) {
        if constexpr ((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value)) {
            quantPre = QuantMode_t::VQF322B8_PRE;
        }
    } else if constexpr (std::is_same<SrcType, int32_t>::value) {
        if constexpr ((std::is_same<DstType, int8_t>::value) || (std::is_same<DstType, uint8_t>::value)) {
            quantPre = QuantMode_t::VREQ8;
        } else if constexpr (std::is_same<DstType, half>::value) {
            quantPre = QuantMode_t::VDEQF16;
        } else if constexpr (std::is_same<DstType, int16_t>::value) {
            quantPre = QuantMode_t::VSHIFTS322S16;
        }
    }
    return quantPre;
}

}

#endif