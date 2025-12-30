#ifndef PTO_COMM_BACKEND_SHMEM_BACKEND_HPP
#define PTO_COMM_BACKEND_SHMEM_BACKEND_HPP

#include <cstddef>
#include <cstdint>


#include <type_traits>

#include "shmem_api.h"
#include "pto/comm/comm_types.hpp"
#include "pto/comm/backend/shmem/copy_param_builder.hpp"
#include "pto/common/pto_tile.hpp"

namespace pto {
namespace backend {

struct ShmemInitOptions {
    int rank {0};
    int nranks {1};
    uint64_t heapBytes {0};
    const char *ipPort {nullptr};
};

// Helper trait按类型分发到具体的shmem函数
template <typename T>
struct ShmemOps;

#define DEFINE_SHMEM_OPS(TYPE, NAME)                                                                                 \
    template <>                                                                                                      \
    struct ShmemOps<TYPE> {                                                                                          \
        PTO_INST static void Put(TYPE *dst, TYPE *src, const pto::comm::Copy2DParams &params, int pe) {              \
            if (params.lenElems == 0U) {                                                                             \
                return;                                                                                              \
            }                                                                                                        \
            const uint32_t srcLd = (params.srcStrideElems == 0U) ? params.lenElems : params.srcStrideElems;          \
            const uint32_t dstLd = (params.dstStrideElems == 0U) ? params.lenElems : params.dstStrideElems;          \
            if ((params.repeat == 1U) && (srcLd == params.lenElems) && (dstLd == params.lenElems)) {                 \
                shmem_put_##NAME##_mem_nbi(dst, src, params.lenElems, pe);                                           \
                return;                                                                                              \
            }                                                                                                        \
            non_contiguous_copy_param p {params.repeat, params.lenElems, srcLd, dstLd};                              \
            shmem_put_##NAME##_mem_nbi(dst, src, p, pe);                                                             \
        }                                                                                                            \
        PTO_INST static void Get(TYPE *dst, TYPE *src, const pto::comm::Copy2DParams &params, int pe) {              \
            if (params.lenElems == 0U) {                                                                             \
                return;                                                                                              \
            }                                                                                                        \
            const uint32_t srcLd = (params.srcStrideElems == 0U) ? params.lenElems : params.srcStrideElems;          \
            const uint32_t dstLd = (params.dstStrideElems == 0U) ? params.lenElems : params.dstStrideElems;          \
            if ((params.repeat == 1U) && (srcLd == params.lenElems) && (dstLd == params.lenElems)) {                 \
                shmem_get_##NAME##_mem_nbi(dst, src, params.lenElems, pe);                                           \
                return;                                                                                              \
            }                                                                                                        \
            non_contiguous_copy_param p {params.repeat, params.lenElems, srcLd, dstLd};                              \
            shmem_get_##NAME##_mem_nbi(dst, src, p, pe);                                                             \
        }                                                                                                            \
    }


DEFINE_SHMEM_OPS(__gm__ float, float); // delete the __gm__ prefix
DEFINE_SHMEM_OPS(__gm__ double, double);
DEFINE_SHMEM_OPS(__gm__ int8_t, int8);
DEFINE_SHMEM_OPS(__gm__ int16_t, int16);
DEFINE_SHMEM_OPS(__gm__ int32_t, int32);
DEFINE_SHMEM_OPS(__gm__ int64_t, int64);    
DEFINE_SHMEM_OPS(__gm__ uint8_t, uint8);
DEFINE_SHMEM_OPS(__gm__ uint16_t, uint16);
DEFINE_SHMEM_OPS(__gm__ uint32_t, uint32);
DEFINE_SHMEM_OPS(__gm__ uint64_t, uint64);
DEFINE_SHMEM_OPS(__gm__ char, char);
DEFINE_SHMEM_OPS(__gm__ half, half);
DEFINE_SHMEM_OPS(__gm__ bfloat16_t, bfloat16);

#undef DEFINE_SHMEM_OPS

struct ShmemBackend {
    static int Init(const ShmemInitOptions &opts)
    {
        shmem_init_attr_t *attr = nullptr;
        const int ret = shmem_set_attr(opts.rank, opts.nranks, opts.heapBytes, opts.ipPort, &attr);
        if (ret != 0) {
            return ret;
        }
        return shmem_init_attr(attr);
    }


    static void Finalize()
    {
        shmem_finalize();
    }

    static void *SymmetricAlloc(std::size_t bytes)
    {
        return shmem_malloc(bytes);
    }

    static void SymmetricFree(void *ptr)
    {
        shmem_free(ptr);
    }

    template <typename GlobalSrcData, typename GlobalDstData>
    PTO_INST static void Put(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal)
    {
        static_assert(std::is_same_v<typename GlobalSrcData::DType, typename GlobalDstData::DType>,
                      "ShmemBackend::Put: src/dst element type mismatch");
        static_assert(GlobalSrcData::layout == GlobalDstData::layout,
                      "ShmemBackend::Put: src/dst layout mismatch");

        auto pe = dstGlobal.GetRank();
        auto params = BuildCopyParams(srcGlobal);
        using DType = typename GlobalSrcData::DType;
        ShmemOps<DType>::Put(dstGlobal.data(), srcGlobal.data(), params, pe);
    }

    // template <typename T>
    // PTO_INST static void Put(__gm__ T *dst, __gm__ T *src, const pto::comm::Copy2DParams &params, int pe)
    // {
    //     ShmemOps<T>::Put(dst, src, params, pe);
    // }

    // Keep the parameter order consistent with TGET_IMPL (dst first, then src).
    template <typename GlobalDstData, typename GlobalSrcData>
    PTO_INST static void Get(GlobalDstData &dstGlobal, GlobalSrcData &srcGlobal)
    {
        static_assert(std::is_same_v<typename GlobalSrcData::DType, typename GlobalDstData::DType>,
                      "ShmemBackend::Get: src/dst element type mismatch");
        static_assert(GlobalSrcData::layout == GlobalDstData::layout,
                      "ShmemBackend::Get: src/dst layout mismatch");

        auto pe = srcGlobal.GetRank();
        auto params = BuildCopyParams(srcGlobal);
        using DType = typename GlobalSrcData::DType;
        ShmemOps<DType>::Get(dstGlobal.data(), srcGlobal.data(), params, pe);
    }

    // template <typename T>
    // PTO_INST static void Get(__gm__ T *dst, __gm__ T *src, const pto::comm::Copy2DParams &params, int pe)
    // {
    //     ShmemOps<T>::Get(dst, src, params, pe);
    // }

    PTO_INST static void Barrier()
    {
        shmem_barrier_all();
    }

    PTO_INST static void Wait()
    {
        shmem_quiet();
    }

private:
    template <typename GlobalData>
    struct GlobalDataTraits {
        using DType = typename GlobalData::DType; // Element to DType
        static constexpr bool isRowMajor = (GlobalData::layout == pto::Layout::ND);
        static constexpr pto::SLayout SFractal =
            (GlobalData::layout == pto::Layout::NZ) ? pto::SLayout::RowMajor : pto::SLayout::NoneBox;
        static constexpr int SFractalSize = pto::TileConfig::fractalABSize;
    };

    template <typename GlobalData>
    PTO_INST static pto::comm::Copy2DParams BuildCopyParams(GlobalData &global) // delete the const keyword
    {
        static_assert(
            GlobalData::layout == pto::Layout::ND || GlobalData::layout == pto::Layout::DN ||
                GlobalData::layout == pto::Layout::NZ,
            "ShmemBackend: unsupported layout");

        const int gShape0 = global.GetShape(pto::GlobalTensorDim::DIM_0);
        const int gShape1 = global.GetShape(pto::GlobalTensorDim::DIM_1);
        const int gShape2 = global.GetShape(pto::GlobalTensorDim::DIM_2);
        const int gShape3 = global.GetShape(pto::GlobalTensorDim::DIM_3);
        const int gShape4 = global.GetShape(pto::GlobalTensorDim::DIM_4);

        const int gStride0 = global.GetStride(pto::GlobalTensorDim::DIM_0);
        const int gStride1 = global.GetStride(pto::GlobalTensorDim::DIM_1);
        const int gStride2 = global.GetStride(pto::GlobalTensorDim::DIM_2);
        const int gStride3 = global.GetStride(pto::GlobalTensorDim::DIM_3);
        const int gStride4 = global.GetStride(pto::GlobalTensorDim::DIM_4);

        int validRow = 0;
        int validCol = 0;
        if constexpr (GlobalData::layout == pto::Layout::ND) {
            validRow = gShape0 * gShape1 * gShape2 * gShape3;
            validCol = gShape4;
        } else if constexpr (GlobalData::layout == pto::Layout::DN) {
            validRow = gShape3;
            validCol = gShape0 * gShape1 * gShape2 * gShape4;
        } else { // NZ
            validRow = gShape2 * gShape3;
            validCol = gShape0 * gShape1 * gShape4;
        }

        return pto::comm::detail::BuildCopy2DParams<GlobalDataTraits<GlobalData>>(
            gShape0, gShape1, gShape2, gShape3, gShape4, gStride0, gStride1, gStride2, gStride3, gStride4, validRow,
            validCol);
    }
};

} // namespace backend
} // namespace pto

#endif // PTO_COMM_BACKEND_SHMEM_BACKEND_HPP
