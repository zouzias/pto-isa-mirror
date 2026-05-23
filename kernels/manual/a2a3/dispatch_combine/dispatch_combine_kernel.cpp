#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>
#include <pto/pto-inst.hpp>
#include "kernel_launchers.h"

using namespace pto;

constexpr uint64_t kVecPingUb = 0x00000;
constexpr uint64_t kVecPongUb = 0x04000;
constexpr uint64_t kVecUb = kVecPingUb;
constexpr uint64_t kAccUb = 0x08000;
constexpr int kPtoVectorTileElems = 1024;
constexpr int kSegmentFieldCount = 7;

using PtoShapeDyn = Shape<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;
using PtoStrideDyn = Stride<DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC, DYNAMIC>;

template <typename Element>
using PtoGlobalNd = GlobalTensor<Element, PtoShapeDyn, PtoStrideDyn, Layout::ND>;

template <typename Element, int TileElems = kPtoVectorTileElems>
using PtoVecTile = Tile<TileType::Vec, Element, 1, TileElems, BLayout::RowMajor, -1, -1>;

struct DispatchShape {
    int ranks;
    int experts_per_rank;
    int tokens;
    int hidden;
    int topk;
    int global_experts;
    int max_rows;
    int max_dispatch_rows;
    int route_elems;
    int count_elems;
    int packed_elems;
    int dispatch_elems;
    int out_elems;
    int segment_count;
    int token_blocks;

    PTO_INTERNAL DispatchShape(int ranks_in, int experts_per_rank_in, int tokens_in, int hidden_in, int topk_in)
        : ranks(ranks_in),
          experts_per_rank(experts_per_rank_in),
          tokens(tokens_in),
          hidden(hidden_in),
          topk(topk_in),
          global_experts(ranks_in * experts_per_rank_in),
          max_rows(tokens_in * topk_in),
          max_dispatch_rows(ranks_in * tokens_in * topk_in),
          route_elems(ranks_in * tokens_in * topk_in),
          count_elems(ranks_in * ranks_in * experts_per_rank_in),
          packed_elems(ranks_in * tokens_in * topk_in * hidden_in),
          dispatch_elems(ranks_in * ranks_in * tokens_in * topk_in * hidden_in),
          out_elems(ranks_in * tokens_in * hidden_in),
          segment_count(ranks_in * experts_per_rank_in * ranks_in),
          token_blocks(ranks_in * tokens_in)
    {}
};

struct SegmentDesc {
    int32_t src;
    int32_t dst;
    int32_t local_expert;
    int32_t global_expert;
    int32_t src_row_base;
    int32_t dispatch_row_base;
    int32_t rows;
};

struct XRows {
    __gm__ float *base;
    PTO_INTERNAL __gm__ float *Row(const DispatchShape &shape, int rank, int token) const
    {
        return base + (rank * shape.tokens + token) * shape.hidden;
    }
};

struct ProbRows {
    __gm__ float *base;
    PTO_INTERNAL float At(const DispatchShape &shape, int rank, int token, int slot) const
    {
        return base[(rank * shape.tokens + token) * shape.topk + slot];
    }
};

struct ExpertIdxRows {
    __gm__ int32_t *base;
    PTO_INTERNAL int At(const DispatchShape &shape, int rank, int token, int slot) const
    {
        return base[(rank * shape.tokens + token) * shape.topk + slot];
    }
};

struct ActiveRows {
    __gm__ int32_t *base;
    PTO_INTERNAL int At(const DispatchShape &shape, int rank, int token) const
    {
        return base[rank * shape.tokens + token];
    }
};

struct CountRows {
    __gm__ int32_t *base;
    PTO_INTERNAL __gm__ int32_t *Cell(const DispatchShape &shape, int rank, int expert) const
    {
        return base + rank * shape.global_experts + expert;
    }
    PTO_INTERNAL int At(const DispatchShape &shape, int rank, int expert) const
    {
        return *Cell(shape, rank, expert);
    }
    PTO_INTERNAL void Set(const DispatchShape &shape, int rank, int expert, int value) const
    {
        *Cell(shape, rank, expert) = value;
    }
    PTO_INTERNAL void AddOne(const DispatchShape &shape, int rank, int expert) const
    {
        __gm__ int32_t *ptr = Cell(shape, rank, expert);
        *ptr = *ptr + 1;
    }
};

struct ExpandedRows {
    __gm__ int32_t *base;
    PTO_INTERNAL int Flat(const DispatchShape &shape, int token, int slot) const
    {
        return token * shape.topk + slot;
    }
    PTO_INTERNAL __gm__ int32_t *Cell(const DispatchShape &shape, int rank, int token, int slot) const
    {
        return base + rank * shape.max_rows + Flat(shape, token, slot);
    }
    PTO_INTERNAL int AtFlat(const DispatchShape &shape, int rank, int flat) const
    {
        return base[rank * shape.max_rows + flat];
    }
    PTO_INTERNAL void Set(const DispatchShape &shape, int rank, int token, int slot, int value) const
    {
        *Cell(shape, rank, token, slot) = value;
    }
};

struct PackedRows {
    __gm__ float *base;
    PTO_INTERNAL __gm__ float *Row(const DispatchShape &shape, int rank, int row) const
    {
        return base + (rank * shape.max_rows + row) * shape.hidden;
    }
};

struct DispatchRows {
    __gm__ float *base;
    PTO_INTERNAL __gm__ float *Row(const DispatchShape &shape, int dst, int row) const
    {
        return base + (dst * shape.max_dispatch_rows + row) * shape.hidden;
    }
};

struct ReturnRows {
    __gm__ float *base;
    PTO_INTERNAL __gm__ float *Row(const DispatchShape &shape, int rank, int row) const
    {
        return base + (rank * shape.max_rows + row) * shape.hidden;
    }
};

struct OutRows {
    __gm__ float *base;
    PTO_INTERNAL __gm__ float *Row(const DispatchShape &shape, int rank, int token) const
    {
        return base + (rank * shape.tokens + token) * shape.hidden;
    }
};

struct SegmentRows {
    __gm__ int32_t *base;
    PTO_INTERNAL int Id(const DispatchShape &shape, int dst, int local_expert, int src) const
    {
        return (dst * shape.experts_per_rank + local_expert) * shape.ranks + src;
    }
    PTO_INTERNAL __gm__ int32_t *Row(int segment_id) const
    {
        return base + segment_id * kSegmentFieldCount;
    }
    PTO_INTERNAL void Set(int segment_id, const SegmentDesc &desc) const
    {
        __gm__ int32_t *row = Row(segment_id);
        row[0] = desc.src;
        row[1] = desc.dst;
        row[2] = desc.local_expert;
        row[3] = desc.global_expert;
        row[4] = desc.src_row_base;
        row[5] = desc.dispatch_row_base;
        row[6] = desc.rows;
    }
    PTO_INTERNAL SegmentDesc Load(int segment_id) const
    {
        __gm__ int32_t *row = Row(segment_id);
        SegmentDesc desc{row[0], row[1], row[2], row[3], row[4], row[5], row[6]};
        return desc;
    }
};

struct ReadyRows {
    __gm__ int32_t *base;
    PTO_INTERNAL void Set(int index, int value) const
    {
        volatile __gm__ int32_t *ptr = reinterpret_cast<volatile __gm__ int32_t *>(base + index);
        *ptr = value;
        dcci((__gm__ void *)(base + index), SINGLE_CACHE_LINE);
        __asm__ __volatile__("");
    }
};

struct DispatchViews {
    XRows x;
    ExpertIdxRows expert_idx;
    ActiveRows active;
    CountRows count;
    CountRows src_expert_offset;
    ExpandedRows expanded_row_idx;
    PackedRows packed;
    DispatchRows dispatch;
    ReturnRows ret;
    SegmentRows segments;
    ReadyRows segment_ready;
    ReadyRows return_ready;
};

struct SegmentDispatchViews {
    PackedRows packed;
    DispatchRows dispatch;
    SegmentRows segments;
    ReadyRows segment_ready;
};

struct SegmentReturnViews {
    DispatchRows dispatch;
    ReturnRows ret;
    SegmentRows segments;
    ReadyRows return_ready;
};

struct RestoreViews {
    OutRows out;
    ProbRows probs;
    ExpandedRows expanded_row_idx;
    ReturnRows ret;
};

template <typename Element>
PTO_INTERNAL PtoGlobalNd<Element> MakeContiguousGlobalFromPtr(__gm__ Element *ptr, uint32_t elem_num)
{
    PtoShapeDyn shape(1, 1, 1, 1, elem_num);
    PtoStrideDyn stride(elem_num, elem_num, elem_num, elem_num, 1);
    return PtoGlobalNd<Element>(ptr, shape, stride);
}

template <typename Element>
PTO_INTERNAL uint64_t PtoElemOffsetBytes(uint32_t elem_offset)
{
    return static_cast<uint64_t>(elem_offset) * sizeof(Element);
}

template <typename TileData, typename Element>
PTO_INTERNAL void AssignUbTile(TileData &tile, uint64_t base_offset_bytes, uint32_t elem_offset)
{
    TASSIGN(tile, base_offset_bytes + PtoElemOffsetBytes<Element>(elem_offset));
}

template <typename Element, int TileElems = kPtoVectorTileElems>
PTO_INTERNAL void LoadVec(uint64_t dst_ub_offset_bytes, __gm__ Element *src, uint32_t elem_num)
{
    using TileData = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elem_num; offset += TileElems) {
        uint32_t cur = (elem_num - offset > TileElems) ? TileElems : (elem_num - offset);
        auto src_global = MakeContiguousGlobalFromPtr<Element>(src + offset, cur);
        TileData tile(1, cur);
        AssignUbTile<TileData, Element>(tile, dst_ub_offset_bytes, offset);
        TLOAD(tile, src_global);
    }
    pipe_barrier(PIPE_ALL);
}

template <typename Element, int TileElems = kPtoVectorTileElems>
PTO_INTERNAL void StoreVec(__gm__ Element *dst, uint64_t src_ub_offset_bytes, uint32_t elem_num)
{
    using TileData = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elem_num; offset += TileElems) {
        uint32_t cur = (elem_num - offset > TileElems) ? TileElems : (elem_num - offset);
        auto dst_global = MakeContiguousGlobalFromPtr<Element>(dst + offset, cur);
        TileData tile(1, cur);
        AssignUbTile<TileData, Element>(tile, src_ub_offset_bytes, offset);
        TSTORE(dst_global, tile);
    }
    pipe_barrier(PIPE_ALL);
}

template <typename Element, int TileElems = kPtoVectorTileElems>
PTO_INTERNAL void FillVecTile(uint64_t dst_ub_offset_bytes, Element scalar, uint32_t elem_num)
{
    using TileData = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elem_num; offset += TileElems) {
        uint32_t cur = (elem_num - offset > TileElems) ? TileElems : (elem_num - offset);
        TileData tile(1, cur);
        AssignUbTile<TileData, Element>(tile, dst_ub_offset_bytes, offset);
        TEXPANDS(tile, scalar);
    }
    pipe_barrier(PIPE_ALL);
}

template <typename Element, int TileElems = kPtoVectorTileElems>
PTO_INTERNAL void MulVecTile(uint64_t dst_ub_offset_bytes, uint64_t src_ub_offset_bytes, uint32_t elem_num,
                             Element scalar)
{
    using TileData = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elem_num; offset += TileElems) {
        uint32_t cur = (elem_num - offset > TileElems) ? TileElems : (elem_num - offset);
        TileData dst_tile(1, cur);
        TileData src_tile(1, cur);
        AssignUbTile<TileData, Element>(dst_tile, dst_ub_offset_bytes, offset);
        AssignUbTile<TileData, Element>(src_tile, src_ub_offset_bytes, offset);
        TMULS(dst_tile, src_tile, scalar);
    }
    pipe_barrier(PIPE_V);
}

template <typename Element, int TileElems = kPtoVectorTileElems>
PTO_INTERNAL void AddVecTile(uint64_t dst_ub_offset_bytes, uint64_t src0_ub_offset_bytes, uint64_t src1_ub_offset_bytes,
                             uint32_t elem_num)
{
    using TileData = PtoVecTile<Element, TileElems>;
    for (uint32_t offset = 0; offset < elem_num; offset += TileElems) {
        uint32_t cur = (elem_num - offset > TileElems) ? TileElems : (elem_num - offset);
        TileData dst_tile(1, cur);
        TileData src0_tile(1, cur);
        TileData src1_tile(1, cur);
        AssignUbTile<TileData, Element>(dst_tile, dst_ub_offset_bytes, offset);
        AssignUbTile<TileData, Element>(src0_tile, src0_ub_offset_bytes, offset);
        AssignUbTile<TileData, Element>(src1_tile, src1_ub_offset_bytes, offset);
        TADD(dst_tile, src0_tile, src1_tile);
    }
    pipe_barrier(PIPE_V);
}

PTO_INTERNAL void CopyFloatRow(__gm__ float *dst_row, __gm__ float *src_row, int hidden)
{
    int chunk_id = 0;
    for (int h0 = 0; h0 < hidden; h0 += kPtoVectorTileElems) {
        int cur = (hidden - h0 > kPtoVectorTileElems) ? kPtoVectorTileElems : (hidden - h0);
        uint64_t ub = ((chunk_id & 1) == 0) ? kVecPingUb : kVecPongUb;
        LoadVec<float>(ub, src_row + h0, static_cast<uint32_t>(cur));
        StoreVec<float>(dst_row + h0, ub, static_cast<uint32_t>(cur));
        ++chunk_id;
    }
}

PTO_INTERNAL void FillFloatRows(__gm__ float *dst, int elems, float value)
{
    for (int offset = 0; offset < elems; offset += kPtoVectorTileElems) {
        int cur = (elems - offset > kPtoVectorTileElems) ? kPtoVectorTileElems : (elems - offset);
        FillVecTile<float>(kVecUb, value, static_cast<uint32_t>(cur));
        StoreVec<float>(dst + offset, kVecUb, static_cast<uint32_t>(cur));
    }
}

PTO_INTERNAL void ScaleAddFloatRow(__gm__ float *dst_row, __gm__ float *src_row, int hidden, float scale)
{
    for (int h0 = 0; h0 < hidden; h0 += kPtoVectorTileElems) {
        int cur = (hidden - h0 > kPtoVectorTileElems) ? kPtoVectorTileElems : (hidden - h0);
        LoadVec<float>(kVecUb, src_row + h0, static_cast<uint32_t>(cur));
        LoadVec<float>(kAccUb, dst_row + h0, static_cast<uint32_t>(cur));
        MulVecTile<float>(kVecUb, kVecUb, static_cast<uint32_t>(cur), scale);
        AddVecTile<float>(kAccUb, kAccUb, kVecUb, static_cast<uint32_t>(cur));
        StoreVec<float>(dst_row + h0, kAccUb, static_cast<uint32_t>(cur));
    }
}

template <typename T>
PTO_INTERNAL void ClearMetadata(__gm__ T __out__ *ptr, int elems, T value)
{
    for (int i = 0; i < elems; ++i) {
        ptr[i] = value;
    }
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL int IsValidReplica(const DispatchShape &shape, const ExpertIdxRows &expert_idx, const ActiveRows &active,
                                int src, int token, int slot)
{
    if (active.At(shape, src, token) == 0) {
        return 0;
    }
    int expert = expert_idx.At(shape, src, token, slot);
    return (expert >= 0 && expert < shape.global_experts) ? 1 : 0;
}

PTO_INTERNAL int CountEarlierReplicasForExpert(const DispatchShape &shape, const ExpertIdxRows &expert_idx,
                                               const ActiveRows &active, int src, int token, int slot, int expert)
{
    int earlier = 0;
    for (int prev_token = 0; prev_token <= token; ++prev_token) {
        int slot_limit = (prev_token == token) ? slot : shape.topk;
        for (int prev_slot = 0; prev_slot < slot_limit; ++prev_slot) {
            if (IsValidReplica(shape, expert_idx, active, src, prev_token, prev_slot) == 0) {
                continue;
            }
            if (expert_idx.At(shape, src, prev_token, prev_slot) == expert) {
                earlier += 1;
            }
        }
    }
    return earlier;
}

PTO_INTERNAL void ClearDispatchMetadata(const DispatchShape &shape, const DispatchViews &views)
{
    ClearMetadata<int32_t>(views.count.base, shape.count_elems, 0);
    ClearMetadata<int32_t>(views.src_expert_offset.base, shape.count_elems, 0);
    ClearMetadata<int32_t>(views.expanded_row_idx.base, shape.route_elems, -1);
    ClearMetadata<int32_t>(views.segments.base, shape.segment_count * kSegmentFieldCount, 0);
    ClearMetadata<int32_t>(views.segment_ready.base, shape.segment_count, 0);
    ClearMetadata<int32_t>(views.return_ready.base, shape.segment_count, 0);
}

PTO_INTERNAL void CountReplicas(const DispatchShape &shape, const DispatchViews &views)
{
    for (int src = 0; src < shape.ranks; ++src) {
        for (int token = 0; token < shape.tokens; ++token) {
            for (int slot = 0; slot < shape.topk; ++slot) {
                if (IsValidReplica(shape, views.expert_idx, views.active, src, token, slot) == 0) {
                    continue;
                }
                int expert = views.expert_idx.At(shape, src, token, slot);
                views.count.AddOne(shape, src, expert);
            }
        }
    }
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL void BuildSourceExpertPrefix(const DispatchShape &shape, const DispatchViews &views)
{
    for (int src = 0; src < shape.ranks; ++src) {
        int running = 0;
        for (int expert = 0; expert < shape.global_experts; ++expert) {
            views.src_expert_offset.Set(shape, src, expert, running);
            running += views.count.At(shape, src, expert);
        }
    }
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL void BuildSegmentDescriptors(const DispatchShape &shape, const DispatchViews &views)
{
    for (int dst = 0; dst < shape.ranks; ++dst) {
        int dispatch_cursor = 0;
        for (int local_expert = 0; local_expert < shape.experts_per_rank; ++local_expert) {
            int expert = dst * shape.experts_per_rank + local_expert;
            for (int src = 0; src < shape.ranks; ++src) {
                int rows = views.count.At(shape, src, expert);
                int src_row_base = views.src_expert_offset.At(shape, src, expert);
                int segment_id = views.segments.Id(shape, dst, local_expert, src);
                SegmentDesc desc{src, dst, local_expert, expert, src_row_base, dispatch_cursor, rows};
                views.segments.Set(segment_id, desc);
                dispatch_cursor += rows;
            }
        }
    }
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL void ClearDispatchData(const DispatchShape &shape, const DispatchViews &views)
{
    FillFloatRows(views.packed.base, shape.packed_elems, 0.0f);
    FillFloatRows(views.dispatch.base, shape.dispatch_elems, 0.0f);
    FillFloatRows(views.ret.base, shape.packed_elems, 0.0f);
}

PTO_INTERNAL void PackSourceRows(const DispatchShape &shape, const DispatchViews &views)
{
    for (int src = 0; src < shape.ranks; ++src) {
        for (int token = 0; token < shape.tokens; ++token) {
            for (int slot = 0; slot < shape.topk; ++slot) {
                if (IsValidReplica(shape, views.expert_idx, views.active, src, token, slot) == 0) {
                    views.expanded_row_idx.Set(shape, src, token, slot, -1);
                    continue;
                }
                int expert = views.expert_idx.At(shape, src, token, slot);
                int row = views.src_expert_offset.At(shape, src, expert) +
                          CountEarlierReplicasForExpert(shape, views.expert_idx, views.active, src, token, slot,
                                                        expert);
                views.expanded_row_idx.Set(shape, src, token, slot, row);
                CopyFloatRow(views.packed.Row(shape, src, row), views.x.Row(shape, src, token), shape.hidden);
            }
        }
    }
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL void DispatchSegmentLocal(const DispatchShape &shape, const SegmentDesc &desc,
                                       const SegmentDispatchViews &views)
{
    for (int row = 0; row < desc.rows; ++row) {
        CopyFloatRow(views.dispatch.Row(shape, desc.dst, desc.dispatch_row_base + row),
                     views.packed.Row(shape, desc.src, desc.src_row_base + row), shape.hidden);
    }
}

PTO_INTERNAL void ExpertIdentitySegmentLocal(const DispatchShape &, const SegmentDesc &, const DispatchRows &)
{}

PTO_INTERNAL void ReturnSegmentLocal(const DispatchShape &shape, const SegmentDesc &desc, const SegmentReturnViews &views)
{
    for (int row = 0; row < desc.rows; ++row) {
        CopyFloatRow(views.ret.Row(shape, desc.src, desc.src_row_base + row),
                     views.dispatch.Row(shape, desc.dst, desc.dispatch_row_base + row), shape.hidden);
    }
}

PTO_INTERNAL void RestoreTokenOutput(const DispatchShape &shape, const RestoreViews &views, int src, int token)
{
    FillFloatRows(views.out.Row(shape, src, token), shape.hidden, 0.0f);
    for (int slot = 0; slot < shape.topk; ++slot) {
        int flat = views.expanded_row_idx.Flat(shape, token, slot);
        int row = views.expanded_row_idx.AtFlat(shape, src, flat);
        if (row < 0) {
            continue;
        }
        float scale = views.probs.At(shape, src, token, slot);
        ScaleAddFloatRow(views.out.Row(shape, src, token), views.ret.Row(shape, src, row), shape.hidden, scale);
    }
    pipe_barrier(PIPE_ALL);
}

PTO_INTERNAL void MetadataPackImpl(__gm__ float __in__ *x, __gm__ int32_t __in__ *expert_idx,
                                   __gm__ int32_t __in__ *active, __gm__ int32_t __out__ *count,
                                   __gm__ int32_t __out__ *src_expert_offset,
                                   __gm__ int32_t __out__ *expanded_row_idx, __gm__ float __out__ *src_packed_x,
                                   __gm__ float __out__ *dispatch_x, __gm__ float __out__ *return_y,
                                   __gm__ int32_t __out__ *segment_desc, __gm__ int32_t __out__ *segment_ready,
                                   __gm__ int32_t __out__ *return_ready, int ranks, int experts_per_rank, int tokens,
                                   int hidden, int topk)
{
    if (get_block_idx() != 0 || get_subblockid() != 0) {
        return;
    }
    DispatchShape shape(ranks, experts_per_rank, tokens, hidden, topk);
    DispatchViews views{XRows{x},
                        ExpertIdxRows{expert_idx},
                        ActiveRows{active},
                        CountRows{count},
                        CountRows{src_expert_offset},
                        ExpandedRows{expanded_row_idx},
                        PackedRows{src_packed_x},
                        DispatchRows{dispatch_x},
                        ReturnRows{return_y},
                        SegmentRows{segment_desc},
                        ReadyRows{segment_ready},
                        ReadyRows{return_ready}};

    ClearDispatchMetadata(shape, views);
    CountReplicas(shape, views);
    BuildSourceExpertPrefix(shape, views);
    BuildSegmentDescriptors(shape, views);
    ClearDispatchData(shape, views);
    PackSourceRows(shape, views);
}

PTO_INTERNAL void DispatchSegmentImpl(__gm__ float __in__ *src_packed_x, __gm__ float __out__ *dispatch_x,
                                      __gm__ int32_t __in__ *segment_desc,
                                      __gm__ int32_t __out__ *segment_ready, int ranks, int experts_per_rank,
                                      int tokens, int hidden, int topk)
{
    int segment_id = get_block_idx();
    DispatchShape shape(ranks, experts_per_rank, tokens, hidden, topk);
    if (segment_id >= shape.segment_count || get_subblockid() != 0) {
        return;
    }
    SegmentDispatchViews views{PackedRows{src_packed_x}, DispatchRows{dispatch_x}, SegmentRows{segment_desc},
                               ReadyRows{segment_ready}};
    SegmentDesc desc = views.segments.Load(segment_id);
    DispatchSegmentLocal(shape, desc, views);
    pipe_barrier(PIPE_ALL);
    views.segment_ready.Set(segment_id, 1);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

PTO_INTERNAL void ReturnSegmentImpl(__gm__ float __in__ *dispatch_x, __gm__ float __out__ *return_y,
                                    __gm__ int32_t __in__ *segment_desc, __gm__ int32_t __out__ *return_ready,
                                    int ranks, int experts_per_rank, int tokens, int hidden, int topk)
{
    int segment_id = get_block_idx();
    DispatchShape shape(ranks, experts_per_rank, tokens, hidden, topk);
    if (segment_id >= shape.segment_count || get_subblockid() != 0) {
        return;
    }
    SegmentReturnViews views{DispatchRows{dispatch_x}, ReturnRows{return_y}, SegmentRows{segment_desc},
                             ReadyRows{return_ready}};
    SegmentDesc desc = views.segments.Load(segment_id);
    ExpertIdentitySegmentLocal(shape, desc, views.dispatch);
    ReturnSegmentLocal(shape, desc, views);
    pipe_barrier(PIPE_ALL);
    views.return_ready.Set(segment_id, 1);
    pipe_barrier(PIPE_ALL);
    dsb(DSB_DDR);
}

PTO_INTERNAL void MarkReadyImpl(__gm__ int32_t __out__ *ready, int elems)
{
    if (get_block_idx() != 0 || get_subblockid() != 0) {
        return;
    }
    ClearMetadata<int32_t>(ready, elems, 1);
    dsb(DSB_DDR);
}

PTO_INTERNAL void RestoreTokenImpl(__gm__ float __out__ *out, __gm__ float __in__ *probs,
                                   __gm__ int32_t __in__ *expanded_row_idx, __gm__ float __in__ *return_y, int ranks,
                                   int experts_per_rank, int tokens, int hidden, int topk)
{
    int token_block = get_block_idx();
    DispatchShape shape(ranks, experts_per_rank, tokens, hidden, topk);
    if (token_block >= shape.token_blocks || get_subblockid() != 0) {
        return;
    }
    int src = token_block / shape.tokens;
    int token = token_block % shape.tokens;
    RestoreViews views{OutRows{out}, ProbRows{probs}, ExpandedRows{expanded_row_idx}, ReturnRows{return_y}};
    RestoreTokenOutput(shape, views, src, token);
}

__global__ AICORE void MetadataPackKernel(__gm__ float __in__ *x, __gm__ int32_t __in__ *expert_idx,
                                          __gm__ int32_t __in__ *active, __gm__ int32_t __out__ *count,
                                          __gm__ int32_t __out__ *src_expert_offset,
                                          __gm__ int32_t __out__ *expanded_row_idx,
                                          __gm__ float __out__ *src_packed_x, __gm__ float __out__ *dispatch_x,
                                          __gm__ float __out__ *return_y, __gm__ int32_t __out__ *segment_desc,
                                          __gm__ int32_t __out__ *segment_ready,
                                          __gm__ int32_t __out__ *return_ready, int ranks, int experts_per_rank,
                                          int tokens, int hidden, int topk)
{
    MetadataPackImpl(x, expert_idx, active, count, src_expert_offset, expanded_row_idx, src_packed_x, dispatch_x,
                     return_y, segment_desc, segment_ready, return_ready, ranks, experts_per_rank, tokens, hidden,
                     topk);
}

__global__ AICORE void DispatchSegmentsKernel(__gm__ float __in__ *src_packed_x, __gm__ float __out__ *dispatch_x,
                                              __gm__ int32_t __in__ *segment_desc,
                                              __gm__ int32_t __out__ *segment_ready, int ranks, int experts_per_rank,
                                              int tokens, int hidden, int topk)
{
    DispatchSegmentImpl(src_packed_x, dispatch_x, segment_desc, segment_ready, ranks, experts_per_rank, tokens, hidden,
                        topk);
}

__global__ AICORE void ReturnSegmentsKernel(__gm__ float __in__ *dispatch_x, __gm__ float __out__ *return_y,
                                            __gm__ int32_t __in__ *segment_desc,
                                            __gm__ int32_t __out__ *return_ready, int ranks, int experts_per_rank,
                                            int tokens, int hidden, int topk)
{
    ReturnSegmentImpl(dispatch_x, return_y, segment_desc, return_ready, ranks, experts_per_rank, tokens, hidden, topk);
}

__global__ AICORE void RestoreTokensKernel(__gm__ float __out__ *out, __gm__ float __in__ *probs,
                                           __gm__ int32_t __in__ *expanded_row_idx, __gm__ float __in__ *return_y,
                                           int ranks, int experts_per_rank, int tokens, int hidden, int topk)
{
    RestoreTokenImpl(out, probs, expanded_row_idx, return_y, ranks, experts_per_rank, tokens, hidden, topk);
}

__global__ AICORE void MarkReadyKernel(__gm__ int32_t __out__ *ready, int elems)
{
    MarkReadyImpl(ready, elems);
}

void LaunchPhases(float *out, float *x, int32_t *expert_idx, float *probs, int32_t *active, int32_t *count,
                  int32_t *src_expert_offset, int32_t *expanded_row_idx, float *src_packed_x, float *dispatch_x,
                  float *return_y, int32_t *segment_desc, int32_t *segment_ready, int32_t *return_ready, int ranks,
                  int experts_per_rank, int tokens, int hidden, int topk, void *stream)
{
    int segment_count = ranks * experts_per_rank * ranks;
    int token_blocks = ranks * tokens;
    MetadataPackKernel<<<1, nullptr, stream>>>(x, expert_idx, active, count, src_expert_offset, expanded_row_idx,
                                               src_packed_x, dispatch_x, return_y, segment_desc, segment_ready,
                                               return_ready, ranks, experts_per_rank, tokens, hidden, topk);
    DispatchSegmentsKernel<<<segment_count, nullptr, stream>>>(src_packed_x, dispatch_x, segment_desc, segment_ready,
                                                               ranks, experts_per_rank, tokens, hidden, topk);
    ReturnSegmentsKernel<<<segment_count, nullptr, stream>>>(dispatch_x, return_y, segment_desc, return_ready, ranks,
                                                             experts_per_rank, tokens, hidden, topk);
    MarkReadyKernel<<<1, nullptr, stream>>>(segment_ready, segment_count);
    MarkReadyKernel<<<1, nullptr, stream>>>(return_ready, segment_count);
    RestoreTokensKernel<<<token_blocks, nullptr, stream>>>(out, probs, expanded_row_idx, return_y, ranks,
                                                           experts_per_rank, tokens, hidden, topk);
}

void launchDispatchCombine(float *out, float *x, int32_t *expert_idx, float *probs, int32_t *active, int32_t *count,
                           int32_t *src_expert_offset, int32_t *expanded_row_idx, float *src_packed_x,
                           float *dispatch_x, float *return_y, int32_t *segment_desc, int32_t *segment_ready,
                           int32_t *return_ready, int ranks, int experts_per_rank, int tokens, int hidden, int topk,
                           void *stream)
{
    LaunchPhases(out, x, expert_idx, probs, active, count, src_expert_offset, expanded_row_idx, src_packed_x,
                 dispatch_x, return_y, segment_desc, segment_ready, return_ready, ranks, experts_per_rank, tokens,
                 hidden, topk, stream);
}
