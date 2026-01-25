#ifndef PTO_COMM_BACKEND_SHMEM_BACKEND_HPP
#define PTO_COMM_BACKEND_SHMEM_BACKEND_HPP

#include <cstddef>
#include <cstdint>


#include <type_traits>

#if defined(ASCEND_SHMEM)
#include "shmem_api.h"
#elif defined(CANN_SHMEM)
#include "shmem.h"
#endif

#include "pto/comm/comm_types.hpp"
#include "pto/comm/backend/shmem/copy_param_builder.hpp"
#include "pto/common/pto_tile.hpp"

namespace pto {
namespace backend {

#if defined(CANN_SHMEM)
#define CANN_SHMEM_SIGNAL_SET ACLSHMEM_SIGNAL_SET
#define CANN_SHMEM_CMP_EQ ACLSHMEM_CMP_EQ
#elif defined(ASCEND_SHMEM)
#define CANN_SHMEM_SIGNAL_SET SHMEM_SIGNAL_SET
#define CANN_SHMEM_CMP_EQ SHMEM_CMP_EQ
#endif


struct ShmemInitOptions {
    int rank {0};
    int nranks {1};
    uint64_t heapBytes {0};
    const char *ipPort {nullptr};
};

// Helper trait to dispatch to specific shmem functions by type
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
#if defined(ASCEND_SHMEM)
DEFINE_SHMEM_OPS(__gm__ half, half);
DEFINE_SHMEM_OPS(__gm__ bfloat16_t, bfloat16);
#endif

#undef DEFINE_SHMEM_OPS

struct ShmemBackend {
    static int Init(const ShmemInitOptions &opts)
    {
#if defined(ASCEND_SHMEM)
        shmem_init_attr_t *attr = nullptr;
        const int ret = shmem_set_attr(opts.rank, opts.nranks, opts.heapBytes, opts.ipPort, &attr);
        if (ret != 0) {
            return ret;
        }
        return shmem_init_attr(attr);
#elif defined(CANN_SHMEM)
        aclshmemx_init_attr_t attributes;
        
        attributes.my_pe = opts.rank;
        attributes.n_pes = opts.nranks;
        attributes.local_mem_size = opts.heapBytes;
        
        size_t ipLen = 0;
        if (opts.ipPort != nullptr) {
            for (; ipLen < ACLSHMEM_MAX_IP_PORT_LEN - 1 && opts.ipPort[ipLen] != '\0'; ++ipLen) {
                attributes.ip_port[ipLen] = opts.ipPort[ipLen];
            }
        }
        attributes.ip_port[ipLen] = '\0';
        
        constexpr int attrVersion = (1 << 16) + sizeof(aclshmemx_init_attr_t);
        attributes.option_attr = {attrVersion, ACLSHMEM_DATA_OP_MTE, 
                                  DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, DEFAULT_TIMEOUT, -1};
        
        aclshmemx_uniqueid_t defaultUid = ACLSHMEM_UNIQUEID_INITIALIZER;
        attributes.comm_args = reinterpret_cast<void *>(&defaultUid);
        
        return aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
#endif
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


    // ========================================================================
    // AllReduce - Naive implementation (preserved for reference)
    // Uses direct remote memory access via shmem_ptr, slow due to per-element access
    // ========================================================================
    template <typename ParallelGroup, typename GlobalDstData>
    PTO_INST static void AllReduceNaive(ParallelGroup &pg, GlobalDstData &dstGlobal)
    {
        using GlobalData = typename pto::comm::ParallelGroupTraits<ParallelGroup>::GlobalDataType;
        using DType = typename GlobalData::DType;

        const int my_rank = pg.GetRank();
        const int nranks = pg.GetSize();

        if (nranks <= 0) return;

        // Local source tensor
        auto &srcGlobal = pg[my_rank];
        auto srcParams = BuildCopyParams(srcGlobal);

        if (nranks == 1) {
            // Copy local source to local destination
            Get(dstGlobal, srcGlobal);
            return;
        }

        // Initialize dstGlobal with local source data
        ShmemOps<DType>::Get(dstGlobal.data(), srcGlobal.data(), srcParams, my_rank);

        Quiet();
        // shmem_quiet();


        const uint32_t totalElems = srcParams.repeat * ((srcParams.srcStrideElems == 0) ? srcParams.lenElems : srcParams.srcStrideElems);
        DType *dstPtr = dstGlobal.data();

        for (int teamRank = 0; teamRank < nranks; ++teamRank) {
            const int pe = pg[teamRank].GetRank();
            if(pe == my_rank) continue;

            DType *remoteSrcPtr = reinterpret_cast<DType *>(pg[teamRank].data());

            if (remoteSrcPtr != nullptr) {
                for (uint32_t i = 0; i < totalElems; ++i) {
                    dstPtr[i] += remoteSrcPtr[i];
                }
            }
        }

        // shmem_quiet();
        // shmem_barrier_all();
        Quiet();
        Barrier();
    }

    // ========================================================================
    // AllReduce - Bulk Optimized implementation
    // Uses bulk shmem_get to fetch remote data, then performs local addition
    // ========================================================================
    template <typename ParallelGroup, typename GlobalDstData, typename TileData>
    PTO_INST static void AllReduceBulk(ParallelGroup &pg, GlobalDstData &dstGlobal, 
        TileData &src0BufTile, TileData &src1BufTile)
    {
        using GlobalData = typename pto::comm::ParallelGroupTraits<ParallelGroup>::GlobalDataType;
        using DType = typename GlobalData::DType;

        const int my_rank = pg.GetRank();
        const int nranks = pg.GetSize();

        if (nranks <= 0) return;

        // Local source tensor (in shmem)
        auto &srcGlobal = pg[my_rank];
        auto srcParams = BuildCopyParams(srcGlobal);

        if (nranks == 1) {
            // Single rank: just copy local data to output
            TLOAD(src0BufTile, srcGlobal);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstGlobal, src0BufTile);
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            return;
        }

        TLOAD(src0BufTile, srcGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        int cnt = 0;
        for (int iters = 0; iters < nranks; ++iters) {
            auto & remoteGlobal = pg[iters];
            int pe = remoteGlobal.GetRank();
            if (pe == my_rank) continue;
            cnt++;
            TLOAD(src1BufTile, remoteGlobal);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            TADD(src0BufTile, src0BufTile, src1BufTile);
            if(cnt < (nranks - 1)){
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            }else{
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);                
            }
        }

        // Step 3: Store final result to output
        TSTORE(dstGlobal, src0BufTile);
    }

    // ========================================================================
    // AllReduce - Ping-Pong optimized implementation
    // Overlaps data transfer with computation using double buffering
    // ===================================================
    template <typename ParallelGroup, typename GlobalDstData, typename TileData>
    PTO_INST static void AllReducePingPong(ParallelGroup &pg, GlobalDstData &dstGlobal, 
        TileData &accTile, TileData &pingTile, TileData &pongTile)
    {
        using GlobalData = typename pto::comm::ParallelGroupTraits<ParallelGroup>::GlobalDataType;
        using DType = typename GlobalData::DType;

        const int my_rank = pg.GetRank();
        const int nranks = pg.GetSize();

        if (nranks <= 0) return;

        auto &srcGlobal = pg[my_rank];

        if (nranks == 1) {
            TLOAD(accTile, srcGlobal);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstGlobal, accTile);
            return;
        }

        // Build list of remote rank indices (skip self)
        int remoteIdx[16];
        int numRemote = 0;
        for (int i = 0; i < nranks; ++i) {
            if (pg[i].GetRank() != my_rank) {
                remoteIdx[numRemote++] = i;
            }
        }

        // Step 1: Load local data into accumulator
        TLOAD(accTile, srcGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // Step 2: Start prefetching first remote data into pingTile
        TLOAD(pingTile, pg[remoteIdx[0]]);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

        // Wait for local data ready
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // Ping-pong processing
        for (int i = 0; i < numRemote; ++i) {
            const bool hasNext = (i + 1 < numRemote);
            const bool usePing = (i % 2 == 0);
            
            TileData &currentTile = usePing ? pingTile : pongTile;
            TileData &nextTile = usePing ? pongTile : pingTile;
            const auto currentEvent = usePing ? EVENT_ID1 : EVENT_ID2;
            const auto nextEvent = usePing ? EVENT_ID2 : EVENT_ID1;

            // Start prefetch of next remote data (overlapped with current TADD)
            if (hasNext) {
                TLOAD(nextTile, pg[remoteIdx[i + 1]]);
                set_flag(PIPE_MTE2, PIPE_V, nextEvent);
            }

            // Wait for current remote data ready
            wait_flag(PIPE_MTE2, PIPE_V, currentEvent);

            // Add current remote data to accumulator
            TADD(accTile, accTile, currentTile);

            // Sync based on next operation
            if (hasNext) {
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            } else {
                // Last iteration: prepare for TSTORE
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            }
        }

        // Step 3: Store final result
        TSTORE(dstGlobal, accTile);
    }


    // ========================================================================
    // AllReduce - Ring algorithm with Ping-Pong prefetch
    // Each rank fetches data from other ranks in ring order and accumulates
    // Uses double buffering to overlap data transfer with computation
    // ========================================================================
    template <typename ParallelGroup, typename GlobalDstData, typename TileData>
    PTO_INST static void AllReduceRing(ParallelGroup &pg, GlobalDstData &dstGlobal, 
        TileData &sendTile, TileData &recvTile, TileData &accTile)
    {
        using GlobalData = typename pto::comm::ParallelGroupTraits<ParallelGroup>::GlobalDataType;
        using DType = typename GlobalData::DType;

        const int my_rank = pg.GetRank();
        const int nranks = pg.GetSize();

        if (nranks <= 0) return;

        auto &srcGlobal = pg[my_rank];

        // 单 rank 场景：直接拷贝
        if (nranks == 1) {
            TLOAD(accTile, srcGlobal);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstGlobal, accTile);
            return;
        }

        // Step 1: 加载本地数据到累加器
        TLOAD(accTile, srcGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // Step 2: 预取第一个远程数据到 sendTile
        int firstRemoteIdx = (my_rank - 1 + nranks) % nranks;
        TLOAD(sendTile, pg[firstRemoteIdx]);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

        // 等待本地数据就绪
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // Step 3: Ping-pong 环形累加
        for (int step = 0; step < nranks - 1; ++step) {
            const bool hasNext = (step + 1 < nranks - 1);
            const bool useSendTile = (step % 2 == 0);
            
            TileData &currentTile = useSendTile ? sendTile : recvTile;
            TileData &nextTile = useSendTile ? recvTile : sendTile;
            const auto currentEvent = useSendTile ? EVENT_ID1 : EVENT_ID2;
            const auto nextEvent = useSendTile ? EVENT_ID2 : EVENT_ID1;

            // 预取下一轮数据 (与当前 TADD 重叠)
            if (hasNext) {
                int nextSrcIdx = (my_rank - step - 2 + nranks) % nranks;
                TLOAD(nextTile, pg[nextSrcIdx]);
                set_flag(PIPE_MTE2, PIPE_V, nextEvent);
            }

            // 等待当前数据就绪
            wait_flag(PIPE_MTE2, PIPE_V, currentEvent);

            // 累加
            TADD(accTile, accTile, currentTile);

            // 同步
            if (hasNext) {
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            } else {
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            }
        }

        // Step 3: 存储最终结果
        TSTORE(dstGlobal, accTile);
    }

    template <typename ParallelGroup, typename GlobalDstData, typename TileData>
    PTO_INST static void AllReduce(ParallelGroup &pg, GlobalDstData &dstGlobal, 
        TileData &tile0, TileData &tile1, TileData &tile2)
    {
        // AllReduceNaive(pg, dstGlobal);
        // AllReduceBulk(pg, dstGlobal, tile0, tile1);
        AllReducePingPong(pg, dstGlobal, tile0, tile1, tile2);
        // AllReduceRing(pg, dstGlobal, tile0, tile1, tile2);
    }

    template <typename ParallelGroup, typename GlobalDstData>
    PTO_INST static void AllGather(ParallelGroup &pg, GlobalDstData &dstGlobal)
    {
        using GlobalSrcData = typename pto::comm::ParallelGroupTraits<ParallelGroup>::GlobalDataType;
        using DType = typename GlobalSrcData::DType;

        const int my_rank = pg.GetRank();
        const int nranks = pg.GetSize();

        // Local source tensor
        auto &srcGlobal = pg[my_rank];
        auto srcParams = BuildCopyParams(srcGlobal);
        
        if (nranks <= 1) {
            // Copy local source to local destination if needed, or just return
            // For now, assume it's already there or handled.
            return;
        }

        const uint32_t srcLd = (srcParams.srcStrideElems == 0U) ? srcParams.lenElems : srcParams.srcStrideElems;
        const uint32_t srcSize = srcParams.repeat * srcLd;

        for (int teamRank = 0; teamRank < nranks; ++teamRank) {
            const int pe = pg[teamRank].GetRank();
            
            // We want to put our local data into the pe-th rank's dstGlobal at the offset for my_rank.
            // Since dstGlobal is likely symmetric, we just need to target the correct PE.
            
            DType *remoteDstPtr = dstGlobal.data() + my_rank * srcSize;
            ShmemOps<DType>::Put(remoteDstPtr, srcGlobal.data(), srcParams, pe);
        }

        // shmem_quiet();
        // shmem_barrier_all();
        Quiet();
        Barrier();
    }


    template <typename ParallelGroup, typename GlobalSrcData>
    PTO_INST static void BroadCast(ParallelGroup &pg, GlobalSrcData &srcGlobal, int root)
    {
        using GlobalDstData = typename pto::comm::ParallelGroupTraits<ParallelGroup>::GlobalDataType;
        using DType = typename GlobalSrcData::DType;

        const int my_rank = pg.GetRank();
        const int nranks = pg.GetSize();

        if (nranks <= 0) return;

        if (my_rank == root) {
            auto srcParams = BuildCopyParams(srcGlobal);
            for (int r = 0; r < nranks; ++r) {
                const int pe = pg[r].GetRank();
                
                // Put root's source data into PE r's destination tensor in ParallelGroup
                ShmemOps<DType>::Put(pg[r].data(), srcGlobal.data(), srcParams, pe);
            }
            // shmem_quiet();
            Quiet();
        }

        // shmem_barrier_all();
        Barrier();
    }


    PTO_INST static void Barrier()
    {
        // shmem_barrier_all();
        #if defined(ASCEND_SHMEM)
            shmem_barrier_all();
        #elif defined(CANN_SHMEM)
            aclshmem_barrier_all();
        #endif
    }

    PTO_INST static void Quiet()
    {
        // shmem_quiet();
        #if defined(ASCEND_SHMEM)
            shmem_quiet();
        #elif defined(CANN_SHMEM)
            aclshmem_quiet();
        #endif
    }

    // ========================================================================
    // Notify: Send flag notification to remote PE
    // Target address and PE info are obtained from GlobalSignalData's data() and GetRank()
    // Uses int32_t type, compatible with shmem signal API
    // ========================================================================

    // Notify: Compile-time specified op (recommended, zero overhead)
    template <pto::comm::NotifyOp op, typename GlobalSignalData>
    PTO_INST static void Notify(GlobalSignalData &dstSignal, int32_t value)
    {
        // Type check via sizeof to ensure 32-bit type
        static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
                      "ShmemBackend::Notify: signal type must be 32-bit (int32_t)");

        const int pe = dstSignal.GetRank();
        auto dstPtr = dstSignal.data();

        if constexpr (op == pto::comm::NotifyOp::AtomicAdd) {
            #if defined(ASCEND_SHMEM)
                shmem_int32_atomic_add(dstPtr, value, pe);
            #elif defined(CANN_SHMEM)
                aclshmem_int32_atomic_add(dstPtr, value, pe);
            #endif
        } else {
            // Set mode uses signal_op
            #if defined(ASCEND_SHMEM)
                shmemx_signal_op(dstPtr, value, CANN_SHMEM_SIGNAL_SET, pe);
            #elif defined(CANN_SHMEM)
                aclshmemx_signal_op(dstPtr, value, CANN_SHMEM_SIGNAL_SET, pe);
            #endif
        }
    }

    // Notify: Runtime specified op
    template <typename GlobalSignalData>
    PTO_INST static void Notify(GlobalSignalData &dstSignal, int32_t value, pto::comm::NotifyOp op)
    {
        static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
                      "ShmemBackend::Notify: signal type must be 32-bit (int32_t)");

        const int pe = dstSignal.GetRank();
        auto dstPtr = dstSignal.data();

        if (op == pto::comm::NotifyOp::AtomicAdd) {
            #if defined(ASCEND_SHMEM)
                shmem_int32_atomic_add(dstPtr, value, pe);
            #elif defined(CANN_SHMEM)
                aclshmem_int32_atomic_add(dstPtr, value, pe);
            #endif
        } else {
            #if defined(ASCEND_SHMEM)
                shmemx_signal_op(dstPtr, value, CANN_SHMEM_SIGNAL_SET, pe);
            #elif defined(CANN_SHMEM)
                aclshmemx_signal_op(dstPtr, value, CANN_SHMEM_SIGNAL_SET, pe);
            #endif
        }
    }

    // ========================================================================
    // SignalWait: Wait until signal meets the comparison condition
    // Reads signal from GlobalSignalData and blocks until condition is satisfied
    // Uses int32_t type, compatible with shmem signal API
    // ========================================================================

    // Convert WaitCmp enum to shmem comparison constant
    PTO_INST static int WaitCmpToShmem(pto::comm::WaitCmp cmp)
    {
        #if defined(ASCEND_SHMEM)
            switch (cmp) {
                case pto::comm::WaitCmp::EQ: return SHMEM_CMP_EQ;
                case pto::comm::WaitCmp::NE: return SHMEM_CMP_NE;
                case pto::comm::WaitCmp::GT: return SHMEM_CMP_GT;
                case pto::comm::WaitCmp::GE: return SHMEM_CMP_GE;
                case pto::comm::WaitCmp::LT: return SHMEM_CMP_LT;
                case pto::comm::WaitCmp::LE: return SHMEM_CMP_LE;
                default: return SHMEM_CMP_EQ;
            }
        #elif defined(CANN_SHMEM)
            switch (cmp) {
                case pto::comm::WaitCmp::EQ: return ACLSHMEM_CMP_EQ;
                case pto::comm::WaitCmp::NE: return ACLSHMEM_CMP_NE;
                case pto::comm::WaitCmp::GT: return ACLSHMEM_CMP_GT;
                case pto::comm::WaitCmp::GE: return ACLSHMEM_CMP_GE;
                case pto::comm::WaitCmp::LT: return ACLSHMEM_CMP_LT;
                case pto::comm::WaitCmp::LE: return ACLSHMEM_CMP_LE;
                default: return ACLSHMEM_CMP_EQ;
            }
        #endif
    }

    // SignalWait: Compile-time specified comparison (recommended, zero overhead)
    template <pto::comm::WaitCmp cmp, typename GlobalSignalData>
    PTO_INST static void SignalWait(GlobalSignalData &signal, int32_t cmpValue)
    {
        static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
                      "ShmemBackend::SignalWait: signal type must be 32-bit (int32_t)");

        auto sigPtr = signal.data();
        const int shmemCmp = WaitCmpToShmem(cmp);

        #if defined(ASCEND_SHMEM)
            shmem_int32_wait_until(sigPtr, shmemCmp, cmpValue);
        #elif defined(CANN_SHMEM)
            aclshmem_signal_wait_until(sigPtr, shmemCmp, cmpValue);
        #endif
    }

    // SignalWait: Runtime specified comparison
    template <typename GlobalSignalData>
    PTO_INST static void SignalWait(GlobalSignalData &signal, pto::comm::WaitCmp cmp, int32_t cmpValue)
    {
        static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
                      "ShmemBackend::SignalWait: signal type must be 32-bit (int32_t)");

        auto sigPtr = signal.data();
        const int shmemCmp = WaitCmpToShmem(cmp);

        #if defined(ASCEND_SHMEM)
            shmem_int32_wait_until(sigPtr, shmemCmp, cmpValue);
        #elif defined(CANN_SHMEM)
            aclshmem_signal_wait_until(sigPtr, shmemCmp, cmpValue);
        #endif
    }

    // SignalWaitAll: Wait until all signals in array meet the comparison condition
    template <pto::comm::WaitCmp cmp, typename GlobalSignalData>
    PTO_INST static void SignalWaitAll(GlobalSignalData *signals, int count, int32_t cmpValue)
    {
        static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
                      "ShmemBackend::SignalWaitAll: signal type must be 32-bit (int32_t)");

        for (int i = 0; i < count; ++i) {
            SignalWait<cmp>(signals[i], cmpValue);
        }
    }

    // SignalWaitAll: Runtime specified comparison
    template <typename GlobalSignalData>
    PTO_INST static void SignalWaitAll(GlobalSignalData *signals, int count, pto::comm::WaitCmp cmp, int32_t cmpValue)
    {
        static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
                      "ShmemBackend::SignalWaitAll: signal type must be 32-bit (int32_t)");

        for (int i = 0; i < count; ++i) {
            SignalWait(signals[i], cmp, cmpValue);
        }
    }

    // ========================================================================
    // SignalTest: Non-blocking test if signal meets the comparison condition
    // Returns true if condition is satisfied, false otherwise
    // Uses int32_t type, compatible with shmem signal API
    // ========================================================================

    // SignalTest: Compile-time specified comparison (recommended, zero overhead)
    template <pto::comm::WaitCmp cmp, typename GlobalSignalData>
    PTO_INST static bool SignalTest(GlobalSignalData &signal, int32_t cmpValue)
    {
        static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
                      "ShmemBackend::SignalTest: signal type must be 32-bit (int32_t)");

        auto sigPtr = signal.data();
        const int shmemCmp = WaitCmpToShmem(cmp);

        #if defined(ASCEND_SHMEM)
            return shmem_int32_test(sigPtr, shmemCmp, cmpValue) != 0;
        #elif defined(CANN_SHMEM)
            return aclshmem_int32_test(sigPtr, shmemCmp, cmpValue) != 0;
        #endif
    }

    // SignalTest: Runtime specified comparison
    template <typename GlobalSignalData>
    PTO_INST static bool SignalTest(GlobalSignalData &signal, pto::comm::WaitCmp cmp, int32_t cmpValue)
    {
        static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
                      "ShmemBackend::SignalTest: signal type must be 32-bit (int32_t)");

        auto sigPtr = signal.data();
        const int shmemCmp = WaitCmpToShmem(cmp);

        #if defined(ASCEND_SHMEM)
            return shmem_int32_test(sigPtr, shmemCmp, cmpValue) != 0;
        #elif defined(CANN_SHMEM)
            return aclshmem_int32_test(sigPtr, shmemCmp, cmpValue) != 0;
        #endif
    }

    // SignalTestAll: Non-blocking test if all signals in array meet the comparison condition
    // Returns true only if ALL signals satisfy the condition
    template <pto::comm::WaitCmp cmp, typename GlobalSignalData>
    PTO_INST static bool SignalTestAll(GlobalSignalData *signals, int count, int32_t cmpValue)
    {
        static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
                      "ShmemBackend::SignalTestAll: signal type must be 32-bit (int32_t)");

        for (int i = 0; i < count; ++i) {
            if (!SignalTest<cmp>(signals[i], cmpValue)) {
                return false;
            }
        }
        return true;
    }

    // SignalTestAll: Runtime specified comparison
    template <typename GlobalSignalData>
    PTO_INST static bool SignalTestAll(GlobalSignalData *signals, int count, pto::comm::WaitCmp cmp, int32_t cmpValue)
    {
        static_assert(sizeof(typename GlobalSignalData::DType) == sizeof(int32_t),
                      "ShmemBackend::SignalTestAll: signal type must be 32-bit (int32_t)");

        for (int i = 0; i < count; ++i) {
            if (!SignalTest(signals[i], cmp, cmpValue)) {
                return false;
            }
        }
        return true;
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
