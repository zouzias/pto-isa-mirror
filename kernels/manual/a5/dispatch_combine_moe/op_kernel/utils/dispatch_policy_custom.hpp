#ifndef DISPATH_POLICY_CUSTOM_HPP
#define DISPATH_POLICY_CUSTOM_HPP

#include "moe_pto_utils.hpp"

namespace pto_ext {
namespace Gemm {

struct MmadAtlasA5 {
    using ArchTag = Arch::AtlasA5;
    static constexpr bool ASYNC = false;
};

template <uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
          uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
struct MmadAtlasA5PreloadAsync {
    using ArchTag = Arch::AtlasA5;
    static constexpr bool ASYNC = true;
    static constexpr uint32_t PRELOAD_STAGES = PRELOAD_STAGES_;
    static constexpr uint32_t L1_STAGES = L1_STAGES_;
    static constexpr uint32_t L0A_STAGES = L0A_STAGES_;
    static constexpr uint32_t L0B_STAGES = L0B_STAGES_;
    static constexpr uint32_t L0C_STAGES = L0C_STAGES_;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <bool ENABLE_UNIT_FLAG_ = false, bool ENABLE_SHUFFLE_K_ = false>
struct MmadAtlasA5PreloadFixpipeQuant : public MmadAtlasA5 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    static constexpr bool ENABLE_SHUFFLE_K = ENABLE_SHUFFLE_K_;
};

template <uint32_t PRELOAD_STAGES_, uint32_t L1_STAGES_, uint32_t L0A_STAGES_, uint32_t L0B_STAGES_,
          uint32_t L0C_STAGES_, bool ENABLE_UNIT_FLAG_, bool ENABLE_SHUFFLE_K_>
struct MmadAtlasA5PreloadAsyncFixpipe
    : public MmadAtlasA5PreloadAsync<PRELOAD_STAGES_, L1_STAGES_, L0A_STAGES_, L0B_STAGES_, L0C_STAGES_,
                                     ENABLE_UNIT_FLAG_, ENABLE_SHUFFLE_K_> {
};

} // namespace Gemm

namespace Epilogue {

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA5UnQuant {
    using ArchTag = Arch::AtlasA5;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA5PerTokenDequant {
    using ArchTag = Arch::AtlasA5;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA5PerTokenDequantSwigluQuant {
    using ArchTag = Arch::AtlasA5;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

template <uint32_t UB_STAGES_>
struct EpilogueAtlasA5PerTokenDequantV2 {
    using ArchTag = Arch::AtlasA5;
    static constexpr uint32_t UB_STAGES = UB_STAGES_;
};

} // namespace Epilogue

} // namespace pto_ext

#endif // DISPATH_POLICY_CUSTOM_HPP
