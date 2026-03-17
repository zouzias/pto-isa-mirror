template <typename T = DefaultType, typename RegT>
__simd_callee__ inline void Select(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current data type is not supported on current device!");
    vsel(dstReg, srcReg0, srcReg1, mask);
}

template <typename T = DefaultType, CMPMODE mode = CMPMODE::EQ, typename RegT>
__simd_callee__ inline void Compare(MaskReg &dstMask, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current data type is not supported on current device!");
    if constexpr (mode == CMPMODE::EQ) {
        vcmp_eq(dstMask, srcReg0, srcReg1, mask);
    } else if constexpr (mode == CMPMODE::NE) {
        vcmp_ne(dstMask, srcReg0, srcReg1, mask);
    } else if constexpr (mode == CMPMODE::GT) {
        vcmp_gt(dstMask, srcReg0, srcReg1, mask);
    } else if constexpr (mode == CMPMODE::GE) {
        vcmp_ge(dstMask, srcReg0, srcReg1, mask);
    } else if constexpr (mode == CMPMODE::LT) {
        vcmp_lt(dstMask, srcReg0, srcReg1, mask);
    } else if constexpr (mode == CMPMODE::LE) {
        vcmp_le(dstMask, srcReg0, srcReg1, mask);
    }
}

template <typename T>
__aicore__ inline void DuplicateIntrinsics(__ubuf__ T* dstLocal, uint32_t scalarValue, const uint8_t repeatTime,
    const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    vector_dup(dstLocal, scalarValue, repeatTime, dstBlockStride, 1, dstRepeatStride, 0);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void Duplicate(__ubuf__ T* dstLocal, const T& scalarValue, uint64_t mask,
    const uint8_t repeatTime, const uint16_t dstBlockStride, const uint8_t dstRepeatStride)
{
    if constexpr (isSetMask) {
        AscendCUtils::SetMask<T>(mask);
    }
    DuplicateIntrinsics(dstLocal, scalarValue, repeatTime, dstBlockStride, dstRepeatStride);
}

template <typename T, bool isSetMask = true>
__aicore__ static inline void SetMask(const uint64_t& maskHigh, const uint64_t& maskLow)
{
    if constexpr (!isSetMask) {
        return;
    }

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
    if (sizeof(T) >= sizeof(int32_t)) {
        ASCENDC_ASSERT((maskHigh == 0ULL),
                        { KERNEL_LOG(KERNEL_ERROR, "maskHigh must be 0 for type b32 and b64"); });
    }
    ASCENDC_ASSERT(((maskLow != 0ULL) || (maskHigh != 0ULL)),
                    { KERNEL_LOG(KERNEL_ERROR, "maskLow and maskHigh can not be zero at the same time"); });
#endif
#endif
    if ASCEND_IS_NOT_AIC {
        set_vector_mask(maskHigh, maskLow);
    }
}

template <typename T, bool isSetMask = true> __aicore__ static inline void SetMask(int32_t len)
{
    if constexpr (!isSetMask) {
        return;
    }

    int32_t typeLen = 0;
    if constexpr (IsSameType<T, int4b_t>::value) {
        typeLen = DEFAULT_BLOCK_SIZE * INT4_TWO;
#if (__NPU_ARCH__ == 5102)
    } else if constexpr (IsSameType<T, int2b_t>::value) {
        typeLen = DEFAULT_BLOCK_SIZE * INT2_FOUR;
    } else if constexpr (IsSameType<T, uint1b_t>::value) {
        typeLen = DEFAULT_BLOCK_SIZE * INT1_EIGHT;
#endif
    } else {
        typeLen = DEFAULT_BLOCK_SIZE / sizeof(T);
    }
    constexpr int32_t halfTypeLen = 64;  // 1 register -> 64 bits -> 64 elements
    constexpr int32_t lenCoeff = 2;      // 2 registers for masks
    if (len == halfTypeLen) {
        SetMask<T>(0, FULL_MASK);
        return;
    } else if (len == typeLen || len >= halfTypeLen * lenCoeff) { // len = max ele per repeat / len >= 128
        SetMask<T>(FULL_MASK, FULL_MASK);
        return;
    }
    SetMask<T>(static_cast<uint64_t>(
        (len > halfTypeLen) ? (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(len - halfTypeLen)) - 1) : 0),
        static_cast<uint64_t>(
        (len > halfTypeLen) ? FULL_MASK : (((static_cast<uint64_t>(1)) << static_cast<uint32_t>(len)) - 1)));
}

__simd_callee__ inline void MaskNot(MaskReg &dstMask, MaskReg &srcMask, MaskReg &mask)
{
    pnot(dstMask, srcMask, mask);
}

__simd_callee__ inline void MaskAnd(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, MaskReg &mask)
{
    pand(dstMask, srcMask0, srcMask1, mask);
}

__simd_callee__ inline void MaskOr(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, MaskReg &mask)
{
    por(dstMask, srcMask0, srcMask1, mask);
}

__simd_callee__ inline void MaskXor(MaskReg &dstMask, MaskReg &srcMask0, MaskReg &srcMask1, MaskReg &mask)
{
    pxor(dstMask, srcMask0, srcMask1, mask);
}

__simd_callee__ inline void MaskMov(MaskReg &dstMask, MaskReg &srcMask, MaskReg &mask)
{
    pmov(dstMask, srcMask, mask);
}

__simd_callee__ inline void MaskMov(MaskReg &dstMask, MaskReg &srcMask)
{
    pmov(dstMask, srcMask);
}

}