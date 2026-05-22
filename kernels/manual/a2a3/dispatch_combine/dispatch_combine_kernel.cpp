#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>
#include "kernel_launchers.h"

using namespace pto;

constexpr uint64_t kXUb = 0x00000;
constexpr uint64_t kPackedUb = 0x04000;
constexpr uint64_t kDispatchUb = 0x0C000;
constexpr uint64_t kReturnUb = 0x1C000;
constexpr uint64_t kOutUb = 0x24000;
constexpr int kPtoVectorTileElems = 1024;

using PtoShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
using PtoStrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;

template <typename Element>
using PtoGlobalNd = GlobalTensor<Element, PtoShapeDyn, PtoStrideDyn, Layout::ND>;

template <typename Element, int TileElems = kPtoVectorTileElems>
using PtoVecTile = Tile<TileType::Vec, Element, 1, TileElems, BLayout::RowMajor, -1, -1>;

template <typename Element>
PTO_INTERNAL PtoGlobalNd<Element> MakeContiguousGlobalFromPtr(__gm__ Element *ptr, uint32_t elemNum)
{
    PtoShapeDyn shape(1, 1, 1, 1, elemNum);
    PtoStrideDyn stride(elemNum, elemNum, elemNum, elemNum, 1);
    return PtoGlobalNd<Element>(ptr, shape, stride);
}

template <typename Element>
PTO_INTERNAL uint64_t PtoElemOffsetBytes(uint32_t elemOffset)
{
    return static_cast<uint64_t>(elemOffset) * sizeof(Element);
}

template <typename TileData, typename Element>
PTO_INTERNAL void PtoAssignUbTile(TileData &tile, uint64_t baseOffsetBytes, uint32_t elemOffset)
{
    TASSIGN(tile, baseOffsetBytes + PtoElemOffsetBytes<Element>(elemOffset));
}

template <typename Element, int TileElems = kPtoVectorTileElems>
PTO_INTERNAL void PtoLoadVector(uint64_t dstUbOffsetBytes, __gm__ Element *src, uint32_t elemNum)
{
    using TileData = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto src_global = MakeContiguousGlobalFromPtr<Element>(src + offset, cur);
        TileData tile(1, cur);
        PtoAssignUbTile<TileData, Element>(tile, dstUbOffsetBytes, offset);
        TLOAD(tile, src_global);
    }
    pipe_barrier(PIPE_ALL);
}

template <typename Element, int TileElems = kPtoVectorTileElems>
PTO_INTERNAL void PtoStoreVector(__gm__ Element *dst, uint64_t srcUbOffsetBytes, uint32_t elemNum)
{
    using TileData = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        auto dst_global = MakeContiguousGlobalFromPtr<Element>(dst + offset, cur);
        TileData tile(1, cur);
        PtoAssignUbTile<TileData, Element>(tile, srcUbOffsetBytes, offset);
        TSTORE(dst_global, tile);
    }
    pipe_barrier(PIPE_ALL);
}

template <typename Element, int TileElems = kPtoVectorTileElems>
PTO_INTERNAL Element PtoGetValue(uint64_t ubOffsetBytes, uint32_t elemOffset)
{
    using TileData = PtoVecTile<Element, TileElems>;
    uint64_t tileOffsetBytes = ubOffsetBytes +
                               static_cast<uint64_t>(elemOffset / TileElems) * TileElems * sizeof(Element);
    TileData tile(1, TileElems);
    TASSIGN(tile, tileOffsetBytes);
    return tile.GetValue(elemOffset % TileElems);
}

template <typename Element, int TileElems = kPtoVectorTileElems>
PTO_INTERNAL void PtoSetValue(uint64_t ubOffsetBytes, uint32_t elemOffset, Element value)
{
    using TileData = PtoVecTile<Element, TileElems>;
    uint64_t tileOffsetBytes = ubOffsetBytes +
                               static_cast<uint64_t>(elemOffset / TileElems) * TileElems * sizeof(Element);
    TileData tile(1, TileElems);
    TASSIGN(tile, tileOffsetBytes);
    tile.SetValue(elemOffset % TileElems, value);
}

template <typename Element, int TileElems = kPtoVectorTileElems>
PTO_INTERNAL void PtoFillVector(uint64_t dstUbOffsetBytes, Element scalar, uint32_t elemNum)
{
    using TileData = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elemNum; offset += TileElems) {
        uint32_t cur = (elemNum - offset > TileElems) ? TileElems : (elemNum - offset);
        TileData tile(1, cur);
        PtoAssignUbTile<TileData, Element>(tile, dstUbOffsetBytes, offset);
        TEXPANDS(tile, scalar);
    }
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL int FlatTokenSlot(int token, int slot, int topk)
{
    return token * topk + slot;
}

PTO_INTERNAL int XOffset(int rank, int token, int h, int tokens, int hidden)
{
    return (rank * tokens + token) * hidden + h;
}

PTO_INTERNAL int RouteOffset(int rank, int token, int slot, int tokens, int topk)
{
    return (rank * tokens + token) * topk + slot;
}

PTO_INTERNAL int CountOffset(int rank, int expert, int global_experts)
{
    return rank * global_experts + expert;
}

PTO_INTERNAL int PackedOffset(int rank, int row, int h, int max_rows, int hidden)
{
    return (rank * max_rows + row) * hidden + h;
}

PTO_INTERNAL int DispatchOffset(int dst, int row, int h, int max_dispatch_rows, int hidden)
{
    return (dst * max_dispatch_rows + row) * hidden + h;
}

template <typename T>
PTO_INTERNAL void ClearVector(__gm__ T __out__ *ptr, int elems, T value)
{
    for (int i = 0; i < elems; ++i) {
        ptr[i] = value;
    }
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL int IsValidReplica(__gm__ int32_t *expert_idx, __gm__ int32_t *active, int src, int token, int slot,
                                int tokens, int topk, int global_experts)
{
    if (active[src * tokens + token] == 0) {
        return 0;
    }
    int g = expert_idx[RouteOffset(src, token, slot, tokens, topk)];
    return (g >= 0 && g < global_experts) ? 1 : 0;
}

template <int Hidden>
PTO_INTERNAL void DispatchPhaseImpl(__gm__ float __in__ *x, __gm__ int32_t __in__ *expert_idx,
                                    __gm__ int32_t __in__ *active, __gm__ int32_t __out__ *count,
                                    __gm__ int32_t __out__ *src_expert_offset,
                                    __gm__ int32_t __out__ *expanded_row_idx, __gm__ float __out__ *src_packed_x,
                                    __gm__ float __out__ *dispatch_x, int ranks, int experts_per_rank, int tokens,
                                    int topk)
{
    if (get_block_idx() != 0 || get_subblockid() != 0) {
        return;
    }

    int hidden = Hidden;
    int global_experts = ranks * experts_per_rank;
    int max_rows = tokens * topk;
    int max_dispatch_rows = ranks * max_rows;
    int route_elems = ranks * max_rows;
    int count_elems = ranks * global_experts;
    int packed_elems = ranks * max_rows * hidden;
    int dispatch_elems = ranks * max_dispatch_rows * hidden;

    ClearVector<int32_t>(count, count_elems, 0);
    ClearVector<int32_t>(src_expert_offset, count_elems, 0);
    ClearVector<int32_t>(expanded_row_idx, route_elems, -1);
    ClearVector<float>(src_packed_x, packed_elems, 0.0f);
    ClearVector<float>(dispatch_x, dispatch_elems, 0.0f);

    for (int src = 0; src < ranks; ++src) {
        for (int token = 0; token < tokens; ++token) {
            for (int slot = 0; slot < topk; ++slot) {
                if (IsValidReplica(expert_idx, active, src, token, slot, tokens, topk, global_experts) == 0) {
                    continue;
                }
                int g = expert_idx[RouteOffset(src, token, slot, tokens, topk)];
                int count_offset = CountOffset(src, g, global_experts);
                count[count_offset] = count[count_offset] + 1;
            }
        }
    }
    pipe_barrier(PIPE_ALL);

    for (int src = 0; src < ranks; ++src) {
        int running = 0;
        for (int g = 0; g < global_experts; ++g) {
            int idx = CountOffset(src, g, global_experts);
            src_expert_offset[idx] = running;
            running += count[idx];
        }
    }
    pipe_barrier(PIPE_ALL);

    PtoLoadVector<float>(kXUb, x, static_cast<uint32_t>(ranks * tokens * hidden));
    PtoFillVector<float>(kPackedUb, 0.0f, static_cast<uint32_t>(packed_elems));
    PtoFillVector<float>(kDispatchUb, 0.0f, static_cast<uint32_t>(dispatch_elems));

    for (int src = 0; src < ranks; ++src) {
        for (int token = 0; token < tokens; ++token) {
            for (int slot = 0; slot < topk; ++slot) {
                int flat = FlatTokenSlot(token, slot, topk);
                if (IsValidReplica(expert_idx, active, src, token, slot, tokens, topk, global_experts) == 0) {
                    expanded_row_idx[src * max_rows + flat] = -1;
                    continue;
                }
                int g = expert_idx[RouteOffset(src, token, slot, tokens, topk)];
                int count_offset = CountOffset(src, g, global_experts);
                int row = src_expert_offset[count_offset];
                for (int prev_token = 0; prev_token <= token; ++prev_token) {
                    int slot_limit = (prev_token == token) ? slot : topk;
                    for (int prev_slot = 0; prev_slot < slot_limit; ++prev_slot) {
                        if (IsValidReplica(expert_idx, active, src, prev_token, prev_slot, tokens, topk,
                                           global_experts) == 0) {
                            continue;
                        }
                        if (expert_idx[RouteOffset(src, prev_token, prev_slot, tokens, topk)] == g) {
                            row += 1;
                        }
                    }
                }
                expanded_row_idx[src * max_rows + flat] = row;
                for (int h = 0; h < Hidden; ++h) {
                    float value = PtoGetValue<float>(kXUb, static_cast<uint32_t>(XOffset(src, token, h, tokens, hidden)));
                    PtoSetValue<float>(kPackedUb, static_cast<uint32_t>(PackedOffset(src, row, h, max_rows, hidden)),
                                       value);
                }
            }
        }
    }
    pipe_barrier(PIPE_ALL);

    for (int dst = 0; dst < ranks; ++dst) {
        int dispatch_cursor = 0;
        for (int e = 0; e < experts_per_rank; ++e) {
            int g = dst * experts_per_rank + e;
            for (int src = 0; src < ranks; ++src) {
                int rows = count[CountOffset(src, g, global_experts)];
                int read_base = src_expert_offset[CountOffset(src, g, global_experts)];
                for (int row = 0; row < rows; ++row) {
                    for (int h = 0; h < Hidden; ++h) {
                        float value = PtoGetValue<float>(
                            kPackedUb, static_cast<uint32_t>(PackedOffset(src, read_base + row, h, max_rows, hidden)));
                        PtoSetValue<float>(
                            kDispatchUb,
                            static_cast<uint32_t>(DispatchOffset(dst, dispatch_cursor + row, h, max_dispatch_rows,
                                                                 hidden)),
                            value);
                    }
                }
                dispatch_cursor += rows;
            }
        }
    }
    pipe_barrier(PIPE_ALL);
    PtoStoreVector<float>(src_packed_x, kPackedUb, static_cast<uint32_t>(packed_elems));
    PtoStoreVector<float>(dispatch_x, kDispatchUb, static_cast<uint32_t>(dispatch_elems));
}

template <int Hidden>
PTO_INTERNAL void CombineRestoreImpl(__gm__ float __out__ *out, __gm__ float __in__ *probs,
                                     __gm__ int32_t __in__ *count, __gm__ int32_t __in__ *src_expert_offset,
                                     __gm__ int32_t __in__ *expanded_row_idx, __gm__ float __in__ *dispatch_x,
                                     __gm__ float __out__ *return_y, int ranks, int experts_per_rank, int tokens,
                                     int topk)
{
    if (get_block_idx() != 0 || get_subblockid() != 0) {
        return;
    }

    int hidden = Hidden;
    int global_experts = ranks * experts_per_rank;
    int max_rows = tokens * topk;
    int max_dispatch_rows = ranks * max_rows;
    int packed_elems = ranks * max_rows * hidden;
    int out_elems = ranks * tokens * hidden;

    int dispatch_elems = ranks * max_dispatch_rows * hidden;

    PtoLoadVector<float>(kDispatchUb, dispatch_x, static_cast<uint32_t>(dispatch_elems));
    PtoFillVector<float>(kReturnUb, 0.0f, static_cast<uint32_t>(packed_elems));
    PtoFillVector<float>(kOutUb, 0.0f, static_cast<uint32_t>(out_elems));

    for (int dst = 0; dst < ranks; ++dst) {
        int dispatch_cursor = 0;
        for (int e = 0; e < experts_per_rank; ++e) {
            int g = dst * experts_per_rank + e;
            for (int src = 0; src < ranks; ++src) {
                int rows = count[CountOffset(src, g, global_experts)];
                int write_base = src_expert_offset[CountOffset(src, g, global_experts)];
                for (int row = 0; row < rows; ++row) {
                    for (int h = 0; h < Hidden; ++h) {
                        float value = PtoGetValue<float>(
                            kDispatchUb,
                            static_cast<uint32_t>(DispatchOffset(dst, dispatch_cursor + row, h, max_dispatch_rows,
                                                                 hidden)));
                        PtoSetValue<float>(
                            kReturnUb, static_cast<uint32_t>(PackedOffset(src, write_base + row, h, max_rows, hidden)),
                            value);
                    }
                }
                dispatch_cursor += rows;
            }
        }
    }
    pipe_barrier(PIPE_ALL);

    for (int src = 0; src < ranks; ++src) {
        for (int token = 0; token < tokens; ++token) {
            for (int slot = 0; slot < topk; ++slot) {
                int flat = FlatTokenSlot(token, slot, topk);
                int row = expanded_row_idx[src * max_rows + flat];
                if (row < 0) {
                    continue;
                }
                float scale = probs[RouteOffset(src, token, slot, tokens, topk)];
                for (int h = 0; h < Hidden; ++h) {
                    uint32_t out_offset = static_cast<uint32_t>(XOffset(src, token, h, tokens, hidden));
                    float acc = PtoGetValue<float>(kOutUb, out_offset);
                    float value = PtoGetValue<float>(
                        kReturnUb, static_cast<uint32_t>(PackedOffset(src, row, h, max_rows, hidden)));
                    PtoSetValue<float>(kOutUb, out_offset, acc + scale * value);
                }
            }
        }
    }
    pipe_barrier(PIPE_ALL);
    PtoStoreVector<float>(return_y, kReturnUb, static_cast<uint32_t>(packed_elems));
    PtoStoreVector<float>(out, kOutUb, static_cast<uint32_t>(out_elems));
}

template <int Hidden>
__global__ AICORE void DispatchPhaseKernel(__gm__ float __in__ *x, __gm__ int32_t __in__ *expert_idx,
                                           __gm__ int32_t __in__ *active, __gm__ int32_t __out__ *count,
                                           __gm__ int32_t __out__ *src_expert_offset,
                                           __gm__ int32_t __out__ *expanded_row_idx,
                                           __gm__ float __out__ *src_packed_x, __gm__ float __out__ *dispatch_x,
                                           int ranks, int experts_per_rank, int tokens, int topk)
{
    DispatchPhaseImpl<Hidden>(x, expert_idx, active, count, src_expert_offset, expanded_row_idx, src_packed_x,
                              dispatch_x, ranks, experts_per_rank, tokens, topk);
}

template <int Hidden>
__global__ AICORE void CombineRestoreKernel(__gm__ float __out__ *out, __gm__ float __in__ *probs,
                                            __gm__ int32_t __in__ *count,
                                            __gm__ int32_t __in__ *src_expert_offset,
                                            __gm__ int32_t __in__ *expanded_row_idx,
                                            __gm__ float __in__ *dispatch_x, __gm__ float __out__ *return_y, int ranks,
                                            int experts_per_rank, int tokens, int topk)
{
    CombineRestoreImpl<Hidden>(out, probs, count, src_expert_offset, expanded_row_idx, dispatch_x, return_y, ranks,
                               experts_per_rank, tokens, topk);
}

template <int Hidden>
void LaunchPhases(float *out, float *x, int32_t *expert_idx, float *probs, int32_t *active, int32_t *count,
                  int32_t *src_expert_offset, int32_t *expanded_row_idx, float *src_packed_x, float *dispatch_x,
                  float *return_y, int ranks, int experts_per_rank, int tokens, int topk, void *stream)
{
    DispatchPhaseKernel<Hidden><<<1, nullptr, stream>>>(x, expert_idx, active, count, src_expert_offset,
                                                        expanded_row_idx, src_packed_x, dispatch_x, ranks,
                                                        experts_per_rank, tokens, topk);
    CombineRestoreKernel<Hidden><<<1, nullptr, stream>>>(out, probs, count, src_expert_offset, expanded_row_idx,
                                                         dispatch_x, return_y, ranks, experts_per_rank, tokens, topk);
}

void launchDispatchCombine(float *out, float *x, int32_t *expert_idx, float *probs, int32_t *active, int32_t *count,
                           int32_t *src_expert_offset, int32_t *expanded_row_idx, float *src_packed_x,
                           float *dispatch_x, float *return_y, int ranks, int experts_per_rank, int tokens, int hidden,
                           int topk, void *stream)
{
    switch (hidden) {
        case 1:
            LaunchPhases<1>(out, x, expert_idx, probs, active, count, src_expert_offset, expanded_row_idx, src_packed_x,
                            dispatch_x, return_y, ranks, experts_per_rank, tokens, topk, stream);
            break;
        case 4:
            LaunchPhases<4>(out, x, expert_idx, probs, active, count, src_expert_offset, expanded_row_idx, src_packed_x,
                            dispatch_x, return_y, ranks, experts_per_rank, tokens, topk, stream);
            break;
        case 8:
            LaunchPhases<8>(out, x, expert_idx, probs, active, count, src_expert_offset, expanded_row_idx, src_packed_x,
                            dispatch_x, return_y, ranks, experts_per_rank, tokens, topk, stream);
            break;
        case 16:
            LaunchPhases<16>(out, x, expert_idx, probs, active, count, src_expert_offset, expanded_row_idx, src_packed_x,
                             dispatch_x, return_y, ranks, experts_per_rank, tokens, topk, stream);
            break;
        case 32:
            LaunchPhases<32>(out, x, expert_idx, probs, active, count, src_expert_offset, expanded_row_idx, src_packed_x,
                             dispatch_x, return_y, ranks, experts_per_rank, tokens, topk, stream);
            break;
        case 64:
            LaunchPhases<64>(out, x, expert_idx, probs, active, count, src_expert_offset, expanded_row_idx, src_packed_x,
                             dispatch_x, return_y, ranks, experts_per_rank, tokens, topk, stream);
            break;
        default:
            break;
    }
}
