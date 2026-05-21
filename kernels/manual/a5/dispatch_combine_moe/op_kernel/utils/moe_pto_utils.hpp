#ifndef MOE_PTO_UTILS_HPP
#define MOE_PTO_UTILS_HPP

#include "kernel_operator.h"

#include <pto/common/pto_tile.hpp>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "pto_vector_ops.hpp"

namespace pto_ext {

template <class...>
inline constexpr bool DEPENDENT_FALSE = false;

constexpr uint32_t BYTE_PER_C0 = 32;
constexpr uint32_t BYTE_PER_C2 = 64;
constexpr uint32_t C0_NUM_PER_FRACTAL = 16;
constexpr uint32_t BYTE_PER_FRACTAL = BYTE_PER_C0 * C0_NUM_PER_FRACTAL;
constexpr uint32_t BYTE_PER_BLK = 32;
constexpr uint32_t BLK_NUM_PER_VECTOR_FRACTAL = 8;
constexpr uint32_t BYTE_PER_VECTOR_FRACTAL = BYTE_PER_BLK * BLK_NUM_PER_VECTOR_FRACTAL;
constexpr uint64_t L2_OFFSET = 0;
constexpr uint32_t STRIDE_LIMIT = 65536;
constexpr uint32_t BYTE_PER_BLK_FP = 128;


template <auto Pipe>
__forceinline__ __aicore__ void PtoPipeBarrier()
{
    AscendC::PipeBarrier<Pipe>();
}

template <AscendC::HardEvent Event>
__forceinline__ __aicore__ void PtoSetFlag(int32_t eventId)
{
    AscendC::SetFlag<Event>(eventId);
}

template <AscendC::HardEvent Event>
__forceinline__ __aicore__ void PtoWaitFlag(int32_t eventId)
{
    AscendC::WaitFlag<Event>(eventId);
}

template <AscendC::HardEvent Event>
__forceinline__ __aicore__ void PtoSetWaitFlag(AscendC::HardEvent eventId)
{
    event_t pipeEventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(eventId));
    AscendC::SetFlag<Event>(pipeEventId);
    AscendC::WaitFlag<Event>(pipeEventId);
}

template <bool NeedWait>
__forceinline__ __aicore__ void PtoSyncAll()
{
    AscendC::SyncAll<NeedWait>();
}

__forceinline__ __aicore__ void PtoSyncAll()
{
    AscendC::SyncAll();
}

using PtoShape1D = pto::Shape<pto::DYNAMIC, 1, 1, 1, 1>;
using PtoShape2D = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, 1, 1, 1>;
using PtoShape3D = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1, 1>;
using PtoShape4D = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
using PtoStride1D = pto::Stride<pto::DYNAMIC, 1, 1, 1, 1>;
using PtoStride2D = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, 1, 1, 1>;
using PtoStride4D = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
using PtoCoord1D = PtoShape1D;
using PtoCoord2D = PtoShape2D;
using PtoCoord3D = PtoShape3D;

template <uint32_t Align, class T>
__forceinline__[host, aicore] constexpr T CeilDiv(T value)
{
    return (value + static_cast<T>(Align) - 1) / static_cast<T>(Align);
}

template <class T, class U>
__forceinline__[host, aicore] constexpr auto CeilDiv(T lhs, U rhs)
{
    using Common = std::common_type_t<T, U>;
    Common lhsValue = static_cast<Common>(lhs);
    Common rhsValue = static_cast<Common>(rhs);
    return (lhsValue + rhsValue - 1) / rhsValue;
}

template <uint32_t Align, class T>
__forceinline__[host, aicore] constexpr T RoundUp(T value)
{
    return CeilDiv<Align>(value) * static_cast<T>(Align);
}

template <class T, class U>
__forceinline__[host, aicore] constexpr auto AlignUp(T value, U align)
{
    using Common = std::common_type_t<T, U>;
    Common alignValue = static_cast<Common>(align);
    return CeilDiv(static_cast<Common>(value), alignValue) * alignValue;
}

template <int RANK_, class Index_ = uint32_t, class LongIndex_ = int64_t>
struct Coord {
    static constexpr int RANK = RANK_;
    using Index = Index_;
    using LongIndex = LongIndex_;

    Index idx[RANK]{};

    __forceinline__[host, aicore] constexpr explicit Coord(Index value = Index(0))
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] = value;
        }
    }

    __forceinline__[host, aicore] constexpr Coord(Index const (&idx_)[RANK])
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] = idx_[i];
        }
    }

    __forceinline__[host, aicore] explicit operator bool() const
    {
        for (int i = 0; i < RANK; ++i) {
            if (idx[i] != 0) {
                return true;
            }
        }
        return false;
    }

    __forceinline__[host, aicore] bool operator!() const
    {
        return !static_cast<bool>(*this);
    }

    __forceinline__[host, aicore] Coord operator+(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c[i] = idx[i] + b[i];
        }
        return c;
    }

    __forceinline__[host, aicore] Coord operator-(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c[i] = idx[i] - b[i];
        }
        return c;
    }

    __forceinline__[host, aicore] Coord operator*(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c[i] = idx[i] * b[i];
        }
        return c;
    }

    __forceinline__[host, aicore] Coord operator/(Coord const &b) const
    {
        Coord c;
        for (int i = 0; i < RANK; ++i) {
            c[i] = idx[i] / b[i];
        }
        return c;
    }

    __forceinline__[host, aicore] Coord &operator+=(Coord const &b)
    {
        for (int i = 0; i < RANK; ++i) {
            idx[i] += b[i];
        }
        return *this;
    }

    __forceinline__[host, aicore] bool operator==(Coord const &b) const
    {
        for (int i = 0; i < RANK; ++i) {
            if (idx[i] != b[i]) {
                return false;
            }
        }
        return true;
    }

    __forceinline__[host, aicore] Index &operator[](int dim)
    {
        return idx[dim];
    }

    __forceinline__[host, aicore] Index const &operator[](int dim) const
    {
        return idx[dim];
    }

    template <int DIM>
    __forceinline__[host, aicore] Index &At()
    {
        return idx[DIM];
    }

    __forceinline__[host, aicore] Index &At(int dim)
    {
        return idx[dim];
    }

    template <int DIM>
    __forceinline__[host, aicore] Index const &At() const
    {
        return idx[DIM];
    }

    __forceinline__[host, aicore] Index const &At(int dim) const
    {
        return idx[dim];
    }

    template <int... Is>
    __forceinline__[host, aicore] auto GetCoordByAxis() const
    {
        Index values[sizeof...(Is)]{idx[Is]...};
        return Coord<sizeof...(Is), Index, LongIndex>{values};
    }
};

template <class... Ts>
__forceinline__[host, aicore] constexpr auto MakeCoord(Ts... values)
{
    using Index = std::common_type_t<Ts...>;
    Index data[sizeof...(Ts)]{static_cast<Index>(values)...};
    return Coord<sizeof...(Ts), Index>{data};
}

template <int RANK, class Index, class LongIndex>
__forceinline__[host, aicore] Coord<RANK, Index, LongIndex> CeilDiv(Coord<RANK, Index, LongIndex> const &lhs,
                                                      Coord<RANK, Index, LongIndex> const &rhs)
{
    Coord<RANK, Index, LongIndex> out;
    for (int i = 0; i < RANK; ++i) {
        out[i] = pto_ext::CeilDiv(lhs[i], rhs[i]);
    }
    return out;
}

template <uint32_t ROW_ = 1, uint32_t COLUMN_ = 1>
struct MatrixShape {
    static constexpr uint32_t ROW = ROW_;
    static constexpr uint32_t COLUMN = COLUMN_;
    static constexpr int64_t COUNT = ROW * COLUMN;

    __forceinline__[host, aicore] static Coord<2> ToCoord()
    {
        return MakeCoord(ROW, COLUMN);
    }

    __forceinline__[host, aicore] static PtoShape2D ToPtoShape()
    {
        return PtoShape2D(ROW, COLUMN);
    }
};

struct MatrixCoord : public Coord<2, uint32_t> {
    using Index = uint32_t;
    using Base = Coord<2, Index>;
    using LongIndex = typename Base::LongIndex;

    static constexpr uint32_t ROW_INDEX = 0;
    static constexpr uint32_t COLUMN_INDEX = 1;

    __forceinline__[host, aicore] MatrixCoord() = default;
    __forceinline__[host, aicore] MatrixCoord(Coord<2, Index> const &coord) : Base(coord)
    {}
    __forceinline__[host, aicore] MatrixCoord(Index row, Index column) : Base(MakeCoord(row, column))
    {}
    __forceinline__[host, aicore] MatrixCoord(LongIndex row, LongIndex column) : Base(MakeCoord(Index(row), Index(column)))
    {}

    __forceinline__[host, aicore] Index const &row() const
    {
        return this->At(ROW_INDEX);
    }
    __forceinline__[host, aicore] Index &row()
    {
        return this->At(ROW_INDEX);
    }
    __forceinline__[host, aicore] Index const &column() const
    {
        return this->At(COLUMN_INDEX);
    }
    __forceinline__[host, aicore] Index &column()
    {
        return this->At(COLUMN_INDEX);
    }

    __forceinline__[host, aicore] MatrixCoord operator+(Base const &b) const
    {
        return MatrixCoord(Base::operator+(b));
    }

    __forceinline__[host, aicore] MatrixCoord &operator+=(Base const &b)
    {
        Base::operator+=(b);
        return *this;
    }

    __forceinline__[host, aicore] PtoShape2D ToPtoShape() const
    {
        return PtoShape2D(row(), column());
    }
};

__forceinline__[host, aicore] MatrixCoord CeilDiv(MatrixCoord const &lhs, MatrixCoord const &rhs)
{
    return MatrixCoord(pto_ext::CeilDiv(lhs.row(), rhs.row()), pto_ext::CeilDiv(lhs.column(), rhs.column()));
}

__forceinline__[host, aicore] PtoShape2D CeilDiv(PtoShape2D const &lhs, PtoShape2D const &rhs)
{
    return PtoShape2D(pto_ext::CeilDiv(lhs.shape[0], rhs.shape[0]), pto_ext::CeilDiv(lhs.shape[1], rhs.shape[1]));
}

template <uint32_t M_ = 1, uint32_t N_ = 1, uint32_t K_ = 1>
struct GemmShape {
    static constexpr uint32_t M = M_;
    static constexpr uint32_t N = N_;
    static constexpr uint32_t K = K_;
    static constexpr int64_t MN = M * N;
    static constexpr int64_t MK = M * K;
    static constexpr int64_t KN = N * K;
    static constexpr int64_t MNK = M * N * K;
    static constexpr int64_t COUNT = MNK;

    __forceinline__[host, aicore] static Coord<3> ToCoord()
    {
        return MakeCoord(M, N, K);
    }
    __forceinline__[host, aicore] static Coord<2> ToCoordMN()
    {
        return MakeCoord(M, N);
    }
    __forceinline__[host, aicore] static Coord<2> ToCoordMK()
    {
        return MakeCoord(M, K);
    }
    __forceinline__[host, aicore] static Coord<2> ToCoordKN()
    {
        return MakeCoord(K, N);
    }
    __forceinline__[host, aicore] static PtoShape3D ToPtoShape()
    {
        return PtoShape3D(M, N, K);
    }
    __forceinline__[host, aicore] static PtoShape2D ToPtoShapeMN()
    {
        return PtoShape2D(M, N);
    }
    __forceinline__[host, aicore] static PtoShape2D ToPtoShapeMK()
    {
        return PtoShape2D(M, K);
    }
    __forceinline__[host, aicore] static PtoShape2D ToPtoShapeKN()
    {
        return PtoShape2D(K, N);
    }
};


} // namespace pto_ext


namespace pto_ext {
namespace layout {

struct ND {
    static constexpr int RANK = 2;
    static constexpr pto::Layout kPtoLayout = pto::Layout::ND;
    static constexpr pto::TileLayoutCustom kTileLayout = pto::TileLayoutCustom::ND;
    static constexpr pto::BLayout kBLayout = pto::BLayout::RowMajor;
    static constexpr pto::SLayout kSLayout = pto::SLayout::NoneBox;
    using Index = uint32_t;
    using LongIndex = int64_t;
    using Shape = PtoShape2D;
    using Stride = PtoStride2D;

    Shape shape_{};
    Stride stride_{};

    __forceinline__[host, aicore] ND(Index rows = 0, Index cols = 0) : shape_(rows, cols), stride_(LongIndex(cols), LongIndex(1))
    {}

    __forceinline__[host, aicore] ND(Index rows, Index cols, LongIndex ldm) : shape_(rows, cols), stride_(ldm, LongIndex(1))
    {}

    __forceinline__[host, aicore] ND(Shape shape, Stride stride) : shape_(shape), stride_(stride)
    {}

    template <class Element>
    __forceinline__[host, aicore] static ND MakeLayout(Index rows, Index cols)
    {
        return ND(rows, cols);
    }

    __forceinline__[host, aicore] LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) * stride_.stride[0] + LongIndex(coord.column());
    }

    __forceinline__[host, aicore] LongIndex GetOffset(PtoCoord2D const &coord) const
    {
        return LongIndex(coord.shape[0]) * stride_.stride[0] + LongIndex(coord.shape[1]);
    }

    __forceinline__[host, aicore] ND GetTileLayout(MatrixCoord const &tileShape) const
    {
        return ND(tileShape.ToPtoShape(), stride());
    }

    __forceinline__[host, aicore] ND GetTileLayout(PtoShape2D const &tileShape) const
    {
        return ND(tileShape, stride());
    }

    __forceinline__[host, aicore] Shape shape() const
    {
        return shape_;
    }
    __forceinline__[host, aicore] Shape &shape()
    {
        return shape_;
    }
    __forceinline__[host, aicore] int64_t shape(int idx) const
    {
        return shape_.shape[idx];
    }
    __forceinline__[host, aicore] int64_t &shape(int idx)
    {
        return shape_.shape[idx];
    }
    __forceinline__[host, aicore] Stride stride() const
    {
        return stride_;
    }
    __forceinline__[host, aicore] Stride &stride()
    {
        return stride_;
    }
    __forceinline__[host, aicore] int64_t stride(int idx) const
    {
        return stride_.stride[idx];
    }
    __forceinline__[host, aicore] int64_t &stride(int idx)
    {
        return stride_.stride[idx];
    }
};

struct DN {
    static constexpr int RANK = 2;
    static constexpr pto::Layout kPtoLayout = pto::Layout::DN;
    static constexpr pto::TileLayoutCustom kTileLayout = pto::TileLayoutCustom::DN;
    static constexpr pto::BLayout kBLayout = pto::BLayout::ColMajor;
    static constexpr pto::SLayout kSLayout = pto::SLayout::NoneBox;
    using Index = uint32_t;
    using LongIndex = int64_t;
    using Shape = PtoShape2D;
    using Stride = PtoStride2D;

    Shape shape_{};
    Stride stride_{};

    __forceinline__[host, aicore] DN(Index rows = 0, Index cols = 0) : shape_(rows, cols), stride_(LongIndex(1), LongIndex(rows))
    {}

    __forceinline__[host, aicore] DN(Index rows, Index cols, LongIndex ldm) : shape_(rows, cols), stride_(LongIndex(1), ldm)
    {}

    __forceinline__[host, aicore] DN(Shape shape, Stride stride) : shape_(shape), stride_(stride)
    {}

    template <class Element>
    __forceinline__[host, aicore] static DN MakeLayout(Index rows, Index cols)
    {
        return DN(rows, cols);
    }

    __forceinline__[host, aicore] LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) + LongIndex(coord.column()) * stride_.stride[1];
    }

    __forceinline__[host, aicore] LongIndex GetOffset(PtoCoord2D const &coord) const
    {
        return LongIndex(coord.shape[0]) + LongIndex(coord.shape[1]) * stride_.stride[1];
    }

    __forceinline__[host, aicore] DN GetTileLayout(MatrixCoord const &tileShape) const
    {
        return DN(tileShape.ToPtoShape(), stride());
    }

    __forceinline__[host, aicore] DN GetTileLayout(PtoShape2D const &tileShape) const
    {
        return DN(tileShape, stride());
    }

    __forceinline__[host, aicore] Shape shape() const
    {
        return shape_;
    }
    __forceinline__[host, aicore] Shape &shape()
    {
        return shape_;
    }
    __forceinline__[host, aicore] int64_t shape(int idx) const
    {
        return shape_.shape[idx];
    }
    __forceinline__[host, aicore] int64_t &shape(int idx)
    {
        return shape_.shape[idx];
    }
    __forceinline__[host, aicore] Stride stride() const
    {
        return stride_;
    }
    __forceinline__[host, aicore] Stride &stride()
    {
        return stride_;
    }
    __forceinline__[host, aicore] int64_t stride(int idx) const
    {
        return stride_.stride[idx];
    }
    __forceinline__[host, aicore] int64_t &stride(int idx)
    {
        return stride_.stride[idx];
    }
};

struct VectorLayout {
    static constexpr int RANK = 1;
    static constexpr pto::Layout kPtoLayout = pto::Layout::SCALE;
    static constexpr pto::TileLayoutCustom kTileLayout = pto::TileLayoutCustom::NONE;
    static constexpr pto::BLayout kBLayout = pto::BLayout::RowMajor;
    static constexpr pto::SLayout kSLayout = pto::SLayout::NoneBox;
    using Index = uint32_t;
    using LongIndex = int64_t;
    using Shape = PtoShape1D;
    using Stride = PtoStride1D;
    using TensorCoord = Coord<RANK, Index>;

    Shape shape_{};
    Stride stride_{};

    __forceinline__[host, aicore] VectorLayout(Index size = 0) : shape_(size), stride_(LongIndex(1))
    {}

    __forceinline__[host, aicore] VectorLayout(Shape shape, Stride stride) : shape_(shape), stride_(stride)
    {}

    __forceinline__[host, aicore] LongIndex GetOffset(TensorCoord const &coord) const
    {
        return stride_.stride[0] * coord[0];
    }

    __forceinline__[host, aicore] LongIndex GetOffset(PtoCoord1D const &coord) const
    {
        return stride_.stride[0] * coord.shape[0];
    }

    __forceinline__[host, aicore] VectorLayout GetTileLayout(TensorCoord const &tileShape) const
    {
        return VectorLayout(Shape(tileShape[0]), stride());
    }

    __forceinline__[host, aicore] VectorLayout GetTileLayout(PtoShape1D const &tileShape) const
    {
        return VectorLayout(tileShape, stride());
    }

    __forceinline__[host, aicore] Shape shape() const
    {
        return shape_;
    }
    __forceinline__[host, aicore] Shape &shape()
    {
        return shape_;
    }
    __forceinline__[host, aicore] int64_t shape(int idx) const
    {
        return shape_.shape[idx];
    }
    __forceinline__[host, aicore] int64_t &shape(int idx)
    {
        return shape_.shape[idx];
    }
    __forceinline__[host, aicore] Stride stride() const
    {
        return stride_;
    }
    __forceinline__[host, aicore] Stride &stride()
    {
        return stride_;
    }
    __forceinline__[host, aicore] int64_t stride(int idx) const
    {
        return stride_.stride[idx];
    }
    __forceinline__[host, aicore] int64_t &stride(int idx)
    {
        return stride_.stride[idx];
    }
};

struct Nz {
    static constexpr int RANK = 4;
    static constexpr pto::TileLayoutCustom kTileLayout = pto::TileLayoutCustom::NZ;
    static constexpr pto::BLayout kBLayout = pto::BLayout::ColMajor;
    static constexpr pto::SLayout kSLayout = pto::SLayout::RowMajor;
    using Index = uint32_t;
    using LongIndex = int64_t;
    static constexpr int ORG_SHAPE_RANK = 2;
    using OrgShape = PtoShape2D;
    using Shape = PtoShape4D;
    using Stride = PtoStride4D;

    OrgShape orgShape_{};
    Shape shape_{};
    Stride stride_{};

    __forceinline__[host, aicore] Nz(Index orgRows = 0, Index orgCols = 0, Index rowsInFractal = 0, Index rowsByFractal = 0,
                       Index colsInFractal = 0, Index colsByFractal = 0, LongIndex strideRowsInFractal = 0,
                       LongIndex strideRowsByFractal = 0, LongIndex strideColsInFractal = 0,
                       LongIndex strideColsByFractal = 0)
        : orgShape_(orgRows, orgCols),
          shape_(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal),
          stride_(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal)
    {}

    __forceinline__[host, aicore] Nz(OrgShape orgShape, Shape shape, Stride stride)
        : orgShape_(orgShape), shape_(shape), stride_(stride)
    {}

    template <class Element>
    __forceinline__[host, aicore] static Nz MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        Index rowsRound = RoundUp<ELE_NUM_PER_C0>(orgRows);
        Index colsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgCols);
        return Nz(orgRows, orgCols, ELE_NUM_PER_C0, rowsRound / ELE_NUM_PER_C0, C0_NUM_PER_FRACTAL,
                  colsRound / C0_NUM_PER_FRACTAL, 1, colsRound * ELE_NUM_PER_C0, ELE_NUM_PER_C0,
                  BYTE_PER_FRACTAL / sizeof(Element));
    }

    __forceinline__[host, aicore] LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) / shape_.shape[0] * stride_.stride[1] +
               LongIndex(coord.column()) / shape_.shape[2] * stride_.stride[3] +
               (LongIndex(coord.row()) % shape_.shape[0]) * stride_.stride[0] +
               (LongIndex(coord.column()) % shape_.shape[2]) * stride_.stride[2];
    }

    __forceinline__[host, aicore] LongIndex GetOffset(PtoCoord2D const &coord) const
    {
        return LongIndex(coord.shape[0]) / shape_.shape[0] * stride_.stride[1] +
               LongIndex(coord.shape[1]) / shape_.shape[2] * stride_.stride[3] +
               (LongIndex(coord.shape[0]) % shape_.shape[0]) * stride_.stride[0] +
               (LongIndex(coord.shape[1]) % shape_.shape[2]) * stride_.stride[2];
    }

    __forceinline__[host, aicore] Nz GetTileLayout(MatrixCoord const &tileOriShape) const
    {
        Shape tileShape(shape(0), CeilDiv(tileOriShape.row(), shape(0)), shape(2),
                        CeilDiv(tileOriShape.column(), shape(2)));
        return Nz(tileOriShape.ToPtoShape(), tileShape, stride());
    }

    __forceinline__[host, aicore] Nz GetTileLayout(PtoShape2D const &tileOriShape) const
    {
        Shape tileShape(shape(0), CeilDiv(tileOriShape.shape[0], shape(0)), shape(2),
                        CeilDiv(tileOriShape.shape[1], shape(2)));
        return Nz(tileOriShape, tileShape, stride());
    }

    __forceinline__[host, aicore] static Nz MakeLayoutInL0C(MatrixCoord const &shape)
    {
        return MakeLayoutInL0C(shape.ToPtoShape());
    }

    __forceinline__[host, aicore] static Nz MakeLayoutInL0C(PtoShape2D const &shape)
    {
        return Nz(shape.shape[0], shape.shape[1], C0_NUM_PER_FRACTAL, CeilDiv<C0_NUM_PER_FRACTAL>(shape.shape[0]),
                  C0_NUM_PER_FRACTAL, CeilDiv<C0_NUM_PER_FRACTAL>(shape.shape[1]), C0_NUM_PER_FRACTAL,
                  C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL, 1,
                  RoundUp<C0_NUM_PER_FRACTAL>(shape.shape[0]) * C0_NUM_PER_FRACTAL);
    }

    __forceinline__[host, aicore] int64_t orgShape(int idx) const
    {
        return orgShape_.shape[idx];
    }
    __forceinline__[host, aicore] int64_t &orgShape(int idx)
    {
        return orgShape_.shape[idx];
    }
    __forceinline__[host, aicore] Shape shape() const
    {
        return shape_;
    }
    __forceinline__[host, aicore] Shape &shape()
    {
        return shape_;
    }
    __forceinline__[host, aicore] int64_t shape(int idx) const
    {
        return shape_.shape[idx];
    }
    __forceinline__[host, aicore] int64_t &shape(int idx)
    {
        return shape_.shape[idx];
    }
    __forceinline__[host, aicore] Stride stride() const
    {
        return stride_;
    }
    __forceinline__[host, aicore] Stride &stride()
    {
        return stride_;
    }
    __forceinline__[host, aicore] int64_t stride(int idx) const
    {
        return stride_.stride[idx];
    }
    __forceinline__[host, aicore] int64_t &stride(int idx)
    {
        return stride_.stride[idx];
    }
};

struct Zn {
    static constexpr int RANK = 4;
    static constexpr pto::TileLayoutCustom kTileLayout = pto::TileLayoutCustom::ZN;
    static constexpr pto::BLayout kBLayout = pto::BLayout::RowMajor;
    static constexpr pto::SLayout kSLayout = pto::SLayout::ColMajor;
    using Index = uint32_t;
    using LongIndex = int64_t;
    static constexpr int ORG_SHAPE_RANK = 2;
    using OrgShape = PtoShape2D;
    using Shape = PtoShape4D;
    using Stride = PtoStride4D;

    OrgShape orgShape_{};
    Shape shape_{};
    Stride stride_{};

    __forceinline__[host, aicore] Zn(Index orgRows = 0, Index orgCols = 0, Index rowsInFractal = 0, Index rowsByFractal = 0,
                       Index colsInFractal = 0, Index colsByFractal = 0, LongIndex strideRowsInFractal = 0,
                       LongIndex strideRowsByFractal = 0, LongIndex strideColsInFractal = 0,
                       LongIndex strideColsByFractal = 0)
        : orgShape_(orgRows, orgCols),
          shape_(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal),
          stride_(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal)
    {}

    __forceinline__[host, aicore] Zn(OrgShape orgShape, Shape shape, Stride stride)
        : orgShape_(orgShape), shape_(shape), stride_(stride)
    {}

    template <class Element>
    __forceinline__[host, aicore] static Zn MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
        Index colsRound = RoundUp<ELE_NUM_PER_C0>(orgCols);
        return Zn(orgRows, orgCols, C0_NUM_PER_FRACTAL, rowsRound / C0_NUM_PER_FRACTAL, ELE_NUM_PER_C0,
                  colsRound / ELE_NUM_PER_C0, ELE_NUM_PER_C0, BYTE_PER_FRACTAL / sizeof(Element), 1,
                  rowsRound * ELE_NUM_PER_C0);
    }

    __forceinline__[host, aicore] static Zn MakeLayoutInL0C(MatrixCoord const &shape)
    {
        return MakeLayoutInL0C(shape.ToPtoShape());
    }

    __forceinline__[host, aicore] static Zn MakeLayoutInL0C(PtoShape2D const &shape)
    {
        return Zn(shape.shape[0], shape.shape[1], C0_NUM_PER_FRACTAL, CeilDiv<C0_NUM_PER_FRACTAL>(shape.shape[0]),
                  C0_NUM_PER_FRACTAL, CeilDiv<C0_NUM_PER_FRACTAL>(shape.shape[1]), C0_NUM_PER_FRACTAL,
                  C0_NUM_PER_FRACTAL * C0_NUM_PER_FRACTAL, 1,
                  RoundUp<C0_NUM_PER_FRACTAL>(shape.shape[0]) * C0_NUM_PER_FRACTAL);
    }

    __forceinline__[host, aicore] LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) / shape_.shape[0] * stride_.stride[1] +
               LongIndex(coord.column()) / shape_.shape[2] * stride_.stride[3] +
               (LongIndex(coord.row()) % shape_.shape[0]) * stride_.stride[0] +
               (LongIndex(coord.column()) % shape_.shape[2]) * stride_.stride[2];
    }

    __forceinline__[host, aicore] LongIndex GetOffset(PtoCoord2D const &coord) const
    {
        return LongIndex(coord.shape[0]) / shape_.shape[0] * stride_.stride[1] +
               LongIndex(coord.shape[1]) / shape_.shape[2] * stride_.stride[3] +
               (LongIndex(coord.shape[0]) % shape_.shape[0]) * stride_.stride[0] +
               (LongIndex(coord.shape[1]) % shape_.shape[2]) * stride_.stride[2];
    }

    __forceinline__[host, aicore] Zn GetTileLayout(MatrixCoord const &tileOriShape) const
    {
        Shape tileShape(shape(0), CeilDiv(tileOriShape.row(), shape(0)), shape(2),
                        CeilDiv(tileOriShape.column(), shape(2)));
        return Zn(tileOriShape.ToPtoShape(), tileShape, stride());
    }

    __forceinline__[host, aicore] Zn GetTileLayout(PtoShape2D const &tileOriShape) const
    {
        Shape tileShape(shape(0), CeilDiv(tileOriShape.shape[0], shape(0)), shape(2),
                        CeilDiv(tileOriShape.shape[1], shape(2)));
        return Zn(tileOriShape, tileShape, stride());
    }

    __forceinline__[host, aicore] int64_t orgShape(int idx) const
    {
        return orgShape_.shape[idx];
    }
    __forceinline__[host, aicore] int64_t &orgShape(int idx)
    {
        return orgShape_.shape[idx];
    }
    __forceinline__[host, aicore] Shape shape() const
    {
        return shape_;
    }
    __forceinline__[host, aicore] Shape &shape()
    {
        return shape_;
    }
    __forceinline__[host, aicore] int64_t shape(int idx) const
    {
        return shape_.shape[idx];
    }
    __forceinline__[host, aicore] int64_t &shape(int idx)
    {
        return shape_.shape[idx];
    }
    __forceinline__[host, aicore] Stride stride() const
    {
        return stride_;
    }
    __forceinline__[host, aicore] Stride &stride()
    {
        return stride_;
    }
    __forceinline__[host, aicore] int64_t stride(int idx) const
    {
        return stride_.stride[idx];
    }
    __forceinline__[host, aicore] int64_t &stride(int idx)
    {
        return stride_.stride[idx];
    }
};

struct Zz {
    static constexpr int RANK = 4;
    static constexpr pto::TileLayoutCustom kTileLayout = pto::TileLayoutCustom::ZZ;
    static constexpr pto::BLayout kBLayout = pto::BLayout::RowMajor;
    static constexpr pto::SLayout kSLayout = pto::SLayout::RowMajor;
    using Index = uint32_t;
    using LongIndex = int64_t;
    static constexpr int ORG_SHAPE_RANK = 2;
    using OrgShape = PtoShape2D;
    using Shape = PtoShape4D;
    using Stride = PtoStride4D;

    OrgShape orgShape_{};
    Shape shape_{};
    Stride stride_{};

    __forceinline__[host, aicore] Zz(Index orgRows = 0, Index orgCols = 0, Index rowsInFractal = 0, Index rowsByFractal = 0,
                       Index colsInFractal = 0, Index colsByFractal = 0, LongIndex strideRowsInFractal = 0,
                       LongIndex strideRowsByFractal = 0, LongIndex strideColsInFractal = 0,
                       LongIndex strideColsByFractal = 0)
        : orgShape_(orgRows, orgCols),
          shape_(rowsInFractal, rowsByFractal, colsInFractal, colsByFractal),
          stride_(strideRowsInFractal, strideRowsByFractal, strideColsInFractal, strideColsByFractal)
    {}

    __forceinline__[host, aicore] Zz(OrgShape orgShape, Shape shape, Stride stride)
        : orgShape_(orgShape), shape_(shape), stride_(stride)
    {}

    template <class Element>
    __forceinline__[host, aicore] static Zz MakeLayout(Index orgRows, Index orgCols)
    {
        constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
        Index rowsRound = RoundUp<C0_NUM_PER_FRACTAL>(orgRows);
        Index colsRound = RoundUp<ELE_NUM_PER_C0>(orgCols);
        return Zz(orgRows, orgCols, C0_NUM_PER_FRACTAL, rowsRound / C0_NUM_PER_FRACTAL, ELE_NUM_PER_C0,
                  colsRound / ELE_NUM_PER_C0, ELE_NUM_PER_C0, colsRound * C0_NUM_PER_FRACTAL, 1,
                  BYTE_PER_FRACTAL / sizeof(Element));
    }

    __forceinline__[host, aicore] LongIndex GetOffset(MatrixCoord const &coord) const
    {
        return LongIndex(coord.row()) / shape_.shape[0] * stride_.stride[1] +
               LongIndex(coord.column()) / shape_.shape[2] * stride_.stride[3];
    }

    __forceinline__[host, aicore] LongIndex GetOffset(PtoCoord2D const &coord) const
    {
        return LongIndex(coord.shape[0]) / shape_.shape[0] * stride_.stride[1] +
               LongIndex(coord.shape[1]) / shape_.shape[2] * stride_.stride[3];
    }

    __forceinline__[host, aicore] int64_t orgShape(int idx) const
    {
        return orgShape_.shape[idx];
    }
    __forceinline__[host, aicore] int64_t &orgShape(int idx)
    {
        return orgShape_.shape[idx];
    }
    __forceinline__[host, aicore] Shape shape() const
    {
        return shape_;
    }
    __forceinline__[host, aicore] Shape &shape()
    {
        return shape_;
    }
    __forceinline__[host, aicore] int64_t shape(int idx) const
    {
        return shape_.shape[idx];
    }
    __forceinline__[host, aicore] int64_t &shape(int idx)
    {
        return shape_.shape[idx];
    }
    __forceinline__[host, aicore] Stride stride() const
    {
        return stride_;
    }
    __forceinline__[host, aicore] Stride &stride()
    {
        return stride_;
    }
    __forceinline__[host, aicore] int64_t stride(int idx) const
    {
        return stride_.stride[idx];
    }
    __forceinline__[host, aicore] int64_t &stride(int idx)
    {
        return stride_.stride[idx];
    }
};

} // namespace layout

namespace Arch {

struct AtlasA2 {
    static constexpr uint32_t BIAS_SIZE = 1024;
    static constexpr uint32_t FIXBUF_SIZE = 7 * 1024;
    static constexpr uint32_t UB_SIZE = 192 * 1024;
    static constexpr uint32_t L1_SIZE = 512 * 1024;
    static constexpr uint32_t L0A_SIZE = 64 * 1024;
    static constexpr uint32_t L0B_SIZE = 64 * 1024;
    static constexpr uint32_t L0C_SIZE = 128 * 1024;
};

struct AtlasA5 {
    static constexpr uint32_t BIAS_SIZE = 4 * 1024;
    static constexpr uint32_t FIXBUF_SIZE = 4 * 1024;
    static constexpr uint32_t UB_SIZE = 256 * 1024;
    static constexpr uint32_t L1_SIZE = 512 * 1024;
    static constexpr uint32_t L0A_SIZE = 64 * 1024;
    static constexpr uint32_t L0B_SIZE = 64 * 1024;
    static constexpr uint32_t L0C_SIZE = 256 * 1024;
};

struct PtoTileBufferBase {
    __forceinline__ __aicore__ uint64_t GetBufferAddrByByte(const uint32_t offset) const
    {
        return offset;
    }

protected:
    __forceinline__ __aicore__ PtoTileBufferBase() = default;
};

template <pto::TileType TileType>
struct PtoTileBuffer : PtoTileBufferBase {
    static constexpr pto::TileType TILE_TYPE = TileType;
};

template <class ArchTag>
struct Resource {
    PtoTileBuffer<pto::TileType::Mat> l1Buf;
    PtoTileBuffer<pto::TileType::Left> l0ABuf;
    PtoTileBuffer<pto::TileType::Right> l0BBuf;
    PtoTileBuffer<pto::TileType::Bias> btBuf;
    PtoTileBuffer<pto::TileType::Acc> l0CBuf;
    PtoTileBuffer<pto::TileType::Vec> ubBuf;
    PtoTileBuffer<pto::TileType::Scaling> fpBuf;

    __forceinline__ __aicore__ Resource() = default;
};

} // namespace Arch

namespace Gemm {

template <class Element_, class Layout_, pto::TileType TileType_ = pto::TileType::Vec>
struct GemmType {
    using Element = Element_;
    using Layout = Layout_;
    static constexpr pto::TileType TileType = TileType_;
};

namespace helper {

template <class ElementA, class ElementB>
struct ElementAccumulatorSelector {
    static_assert(DEPENDENT_FALSE<ElementA>, "Unsupported accumulator selector");
};

template <>
struct ElementAccumulatorSelector<int8_t, int8_t> {
    using ElementAccumulator = int32_t;
};

template <>
struct ElementAccumulatorSelector<half, half> {
    using ElementAccumulator = float;
};

template <>
struct ElementAccumulatorSelector<float, float> {
    using ElementAccumulator = float;
};

template <>
struct ElementAccumulatorSelector<bfloat16_t, bfloat16_t> {
    using ElementAccumulator = float;
};

template <class Element, class Layout>
struct L1AlignHelper {
    static_assert(DEPENDENT_FALSE<Element>, "Unsupported L1 alignment helper");
};

template <class Element>
struct L1AlignHelper<Element, layout::ND> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template <class Element>
struct L1AlignHelper<Element, layout::Zn> {
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t M_ALIGNED = C0_NUM_PER_FRACTAL;
    static constexpr uint32_t K_ALIGNED = ELE_NUM_PER_C0;
    static constexpr uint32_t N_ALIGNED = ELE_NUM_PER_C0;
};

template <class GmAType>
struct L1ATypeSelector {
    static_assert(DEPENDENT_FALSE<GmAType>, "Unsupported L1A type selector");
};

template <class Element>
struct L1ATypeSelector<GemmType<Element, layout::VectorLayout>> {
    using L1AType = GemmType<Element, layout::VectorLayout, pto::TileType::Mat>;
};

template <class Element>
struct L1ATypeSelector<GemmType<Element, layout::ND>> {
    using L1AType = GemmType<Element, layout::Zn, pto::TileType::Mat>;
};

template <class Element>
struct L1ATypeSelector<GemmType<Element, layout::Zn>> {
    using L1AType = GemmType<Element, layout::Zn, pto::TileType::Mat>;
};

template <class GmBType>
struct L1BTypeSelector {
    static_assert(DEPENDENT_FALSE<GmBType>, "Unsupported L1B type selector");
};

template <class Element>
struct L1BTypeSelector<GemmType<Element, layout::Zn>> {
    using L1BType = GemmType<Element, layout::Zn, pto::TileType::Mat>;
};

} // namespace helper

namespace Tile {

enum class ScaleGranularity
{
    UNDEFINED = -1,
    NO_QUANT = 0,
    PER_TENSOR,
    PER_CHANNEL,
    PER_GROUP
};

template <class ArchTag, class ElementSrc, class ElementDst,
          ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT>
struct CopyL0CToGmQuantMode {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c quant mode");
};

template <>
struct CopyL0CToGmQuantMode<Arch::AtlasA2, int32_t, half, ScaleGranularity::PER_TENSOR> {
    static constexpr auto VALUE = QuantMode_t::DEQF16;
};

template <>
struct CopyL0CToGmQuantMode<Arch::AtlasA2, int32_t, half, ScaleGranularity::PER_CHANNEL> {
    static constexpr auto VALUE = QuantMode_t::VDEQF16;
};

template <>
struct CopyL0CToGmQuantMode<Arch::AtlasA2, int32_t, int8_t, ScaleGranularity::PER_TENSOR> {
    static constexpr auto VALUE = QuantMode_t::REQ8;
};

template <>
struct CopyL0CToGmQuantMode<Arch::AtlasA2, int32_t, int8_t, ScaleGranularity::PER_CHANNEL> {
    static constexpr auto VALUE = QuantMode_t::VREQ8;
};

template <>
struct CopyL0CToGmQuantMode<Arch::AtlasA2, int32_t, uint8_t, ScaleGranularity::PER_TENSOR> {
    static constexpr auto VALUE = QuantMode_t::REQ8;
};

template <>
struct CopyL0CToGmQuantMode<Arch::AtlasA2, int32_t, uint8_t, ScaleGranularity::PER_CHANNEL> {
    static constexpr auto VALUE = QuantMode_t::VREQ8;
};

template <>
struct CopyL0CToGmQuantMode<Arch::AtlasA2, float, half, ScaleGranularity::NO_QUANT> {
    static constexpr auto VALUE = QuantMode_t::F322F16;
};

template <>
struct CopyL0CToGmQuantMode<Arch::AtlasA2, float, bfloat16_t, ScaleGranularity::NO_QUANT> {
    static constexpr auto VALUE = QuantMode_t::F322BF16;
};

template <class ElementSrc, class ElementDst, ScaleGranularity Granularity>
struct CopyL0CToGmQuantMode<Arch::AtlasA5, ElementSrc, ElementDst, Granularity>
    : CopyL0CToGmQuantMode<Arch::AtlasA2, ElementSrc, ElementDst, Granularity> {
};

template <class ArchTag, class GmType, class L1Type = typename helper::L1ATypeSelector<GmType>::L1AType>
struct CopyGmToL1Traits {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy gm to l1");
};

template <class Element>
struct CopyGmToL1Traits<Arch::AtlasA2, GemmType<Element, layout::ND>, GemmType<Element, layout::Zn, pto::TileType::Mat>> {
    using LayoutDst = layout::Zn;
    using LayoutSrc = layout::ND;
};

template <class Element>
struct CopyGmToL1Traits<Arch::AtlasA5, GemmType<Element, layout::ND>, GemmType<Element, layout::Zn, pto::TileType::Mat>>
    : CopyGmToL1Traits<Arch::AtlasA2, GemmType<Element, layout::ND>, GemmType<Element, layout::Zn, pto::TileType::Mat>> {
};

template <class ArchTag, class Element>
struct CopyGmToL1Traits<ArchTag, GemmType<Element, layout::Zn>, GemmType<Element, layout::Zn, pto::TileType::Mat>> {
    using LayoutDst = layout::Zn;
    using LayoutSrc = layout::Zn;
};

template <class ArchTag, class Element>
struct CopyGmToL1Traits<ArchTag, GemmType<Element, layout::VectorLayout>,
                         GemmType<Element, layout::VectorLayout, pto::TileType::Mat>> {
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
};

template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToL0ATraits {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0a");
};

template <class ArchTag, class Element>
struct CopyL1ToL0ATraits<ArchTag, GemmType<Element, layout::Zn, pto::TileType::Mat>,
                          GemmType<Element, layout::Zz, pto::TileType::Left>> {
    using LayoutDst = layout::Zz;
    using LayoutSrc = layout::Zn;
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
};

template <class ArchTag, class Element>
struct CopyL1ToL0ATraits<ArchTag, GemmType<Element, layout::Zn, pto::TileType::Mat>>
    : CopyL1ToL0ATraits<ArchTag, GemmType<Element, layout::Zn, pto::TileType::Mat>,
                          GemmType<Element, layout::Zz, pto::TileType::Left>> {
};

template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToL0BTraits {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to l0b");
};

template <class ArchTag>
struct CopyL1ToL0BTraits<ArchTag, GemmType<int8_t, layout::Zn, pto::TileType::Mat>,
                          GemmType<int8_t, layout::Nz, pto::TileType::Right>> {
    using Element = int8_t;
    using LayoutDst = layout::Nz;
    using LayoutSrc = layout::Zn;
    static constexpr uint32_t ELE_NUM_PER_C0 = BYTE_PER_C0 / sizeof(Element);
    static constexpr uint32_t ELE_NUM_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(Element);
};

template <class ArchTag>
struct CopyL1ToL0BTraits<ArchTag, GemmType<int8_t, layout::Zn, pto::TileType::Mat>>
    : CopyL1ToL0BTraits<ArchTag, GemmType<int8_t, layout::Zn, pto::TileType::Mat>,
                          GemmType<int8_t, layout::Nz, pto::TileType::Right>> {
};

template <class ArchTag, class L1Type, class L0Type = void>
struct CopyL1ToFPTraits {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l1 to fixpipe buffer");
};

template <class ArchTag, class ElementSrc, class ElementDst>
struct CopyL1ToFPTraits<ArchTag, GemmType<ElementSrc, layout::VectorLayout, pto::TileType::Mat>,
                        GemmType<ElementDst, layout::VectorLayout, pto::TileType::Scaling>> {
    using LayoutDst = layout::VectorLayout;
    using LayoutSrc = layout::VectorLayout;
    static constexpr uint32_t ELE_NUM_PER_FP = BYTE_PER_BLK_FP / sizeof(ElementSrc);
};

template <class ArchTag, class ElementAccumulator, class GmType,
          ScaleGranularity DEQUANT_GRANULARITY = ScaleGranularity::NO_QUANT, bool ReluEnable = false>
struct CopyL0CToGmTraits {
    static_assert(DEPENDENT_FALSE<ArchTag>, "Unsupported copy l0c to gm");
};

template <class ElementAccumulator_, class ElementDst_, ScaleGranularity Granularity_, bool ReluEnable_>
struct CopyL0CToGmTraits<Arch::AtlasA2, ElementAccumulator_, GemmType<ElementDst_, layout::ND>, Granularity_, ReluEnable_> {
    using ArchTag = Arch::AtlasA2;
    using ElementDst = ElementDst_;
    using ElementSrc = ElementAccumulator_;
    using LayoutSrc = layout::Zn;
    using LayoutDst = layout::ND;
    static constexpr auto quantPre = CopyL0CToGmQuantMode<ArchTag, ElementSrc, ElementDst, Granularity_>::VALUE;
    static constexpr auto reluEn = ReluEnable_;

    struct Params {
        float scale = 1.0f;
        __forceinline__[host, aicore] Params() = default;
        __forceinline__[host, aicore] explicit Params(float scale_) : scale(scale_)
        {}
    };

    Params params;

    __forceinline__ __aicore__ CopyL0CToGmTraits() = default;
    __forceinline__ __aicore__ CopyL0CToGmTraits(Params const &params_) : params(params_)
    {}
};

template <class ElementAccumulator_, class ElementDst_, ScaleGranularity Granularity_, bool ReluEnable_>
struct CopyL0CToGmTraits<Arch::AtlasA5, ElementAccumulator_, GemmType<ElementDst_, layout::ND>, Granularity_, ReluEnable_>
    : CopyL0CToGmTraits<Arch::AtlasA2, ElementAccumulator_, GemmType<ElementDst_, layout::ND>, Granularity_, ReluEnable_> {
    using ArchTag = Arch::AtlasA5;
    using Base =
        CopyL0CToGmTraits<Arch::AtlasA2, ElementAccumulator_, GemmType<ElementDst_, layout::ND>, Granularity_, ReluEnable_>;
    using Base::Base;
};

template <class ArchTag, class AType, class BType, class CType, class BiasType = void>
struct TileCopyGemm {
    using ElementA = typename AType::Element;
    using ElementB = typename BType::Element;
    using ElementAccumulator = typename helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;
    using CopyGmToL1TraitsA = Tile::CopyGmToL1Traits<ArchTag, AType>;
    using CopyGmToL1TraitsB = Tile::CopyGmToL1Traits<ArchTag, BType>;
    using CopyL1ToL0ATraits = Tile::CopyL1ToL0ATraits<ArchTag, typename helper::L1ATypeSelector<AType>::L1AType>;
    using CopyL1ToL0BTraits = Tile::CopyL1ToL0BTraits<ArchTag, typename helper::L1BTypeSelector<BType>::L1BType>;
    using CopyL0CToGmTraits = Tile::CopyL0CToGmTraits<ArchTag, ElementAccumulator, CType>;
};

template <class ArchTag, class AType, class BType, class CType, class BiasType = void,
          ScaleGranularity SCALE_GRANU = ScaleGranularity::PER_TENSOR>
struct QuantTileCopy : public TileCopyGemm<ArchTag, AType, BType, CType, BiasType> {
    using Base = TileCopyGemm<ArchTag, AType, BType, CType, BiasType>;
    using ElementAccumulator = typename Base::ElementAccumulator;
    using CopyL0CToGmTraits = Tile::CopyL0CToGmTraits<ArchTag, ElementAccumulator, CType, SCALE_GRANU, false>;
    using CopyL1ToFPTraits = Tile::CopyL1ToFPTraits<ArchTag, GemmType<uint64_t, layout::VectorLayout, pto::TileType::Mat>,
                                                   GemmType<uint64_t, layout::VectorLayout, pto::TileType::Scaling>>;
};

template <class ArchTag, class AType, class BType, class BiasType = void>
struct TileMmad {
};

} // namespace Tile

namespace Block {

template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct GemmIdentityBlockSwizzle {
    PtoShape3D problemShape;
    PtoShape2D tileMN;
    PtoShape2D loopsMN;

    __forceinline__ __aicore__ GemmIdentityBlockSwizzle() = default;

    __forceinline__ __aicore__ GemmIdentityBlockSwizzle(PtoShape3D const &problemShape_, PtoShape2D const &tileMN_)
        : problemShape(problemShape_), tileMN(tileMN_)
    {
        loopsMN = pto_ext::CeilDiv(PtoShape2D(problemShape.shape[0], problemShape.shape[1]), tileMN);
    }

    __forceinline__ __aicore__ void Update(PtoShape3D const &problemShape_, PtoShape2D const &tileMN_)
    {
        problemShape = problemShape_;
        tileMN = tileMN_;
        loopsMN = pto_ext::CeilDiv(PtoShape2D(problemShape.shape[0], problemShape.shape[1]), tileMN);
    }

    __forceinline__ __aicore__ uint32_t GetCoreLoops() const
    {
        return static_cast<uint32_t>(loopsMN.shape[0] * loopsMN.shape[1]);
    }

    __forceinline__ __aicore__ PtoCoord2D GetBlockCoordMN(uint32_t taskIdx)
    {
        uint32_t innerIdx = taskIdx % GetCoreLoops();
        uint32_t loopM = static_cast<uint32_t>(loopsMN.shape[0]);
        uint32_t loopN = static_cast<uint32_t>(loopsMN.shape[1]);
        uint32_t mIdx = 0;
        uint32_t nIdx = 0;
        if constexpr (SwizzleDirection == 0) {
            uint32_t tileBlockLoop = pto_ext::CeilDiv(loopM, SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopN);
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopN);
            uint32_t nRow = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = loopM - SwizzleOffset * tileBlockIdx;
            }
            mIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nRow;
            nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 == 1) {
                nIdx = loopN - nIdx - 1;
            }
        } else {
            uint32_t tileBlockLoop = pto_ext::CeilDiv(loopN, SwizzleOffset);
            uint32_t tileBlockIdx = innerIdx / (SwizzleOffset * loopM);
            uint32_t inTileBlockIdx = innerIdx % (SwizzleOffset * loopM);
            uint32_t nCol = SwizzleOffset;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = loopN - SwizzleOffset * tileBlockIdx;
            }
            mIdx = inTileBlockIdx / nCol;
            nIdx = tileBlockIdx * SwizzleOffset + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 == 1) {
                mIdx = loopM - mIdx - 1;
            }
        }
        return PtoCoord2D(mIdx, nIdx);
    }

    __forceinline__ __aicore__ PtoShape2D GetActualBlockShapeMN(PtoCoord2D const &blockCoord)
    {
        uint32_t tileM = static_cast<uint32_t>(tileMN.shape[0]);
        uint32_t tileN = static_cast<uint32_t>(tileMN.shape[1]);
        uint32_t loopM = static_cast<uint32_t>(loopsMN.shape[0]);
        uint32_t loopN = static_cast<uint32_t>(loopsMN.shape[1]);
        uint32_t blockM = static_cast<uint32_t>(blockCoord.shape[0]);
        uint32_t blockN = static_cast<uint32_t>(blockCoord.shape[1]);
        uint32_t mActual = (blockM == (loopM - 1)) ? (static_cast<uint32_t>(problemShape.shape[0]) - blockM * tileM) : tileM;
        uint32_t nActual = (blockN == (loopN - 1)) ? (static_cast<uint32_t>(problemShape.shape[1]) - blockN * tileN) : tileN;
        return PtoShape2D(mActual, nActual);
    }
};

template <class DispatchPolicy, class L1TileShape, class L0TileShape, class AType, class BType, class CType,
          class BiasType = void,
          class TileCopy = Gemm::Tile::TileCopyGemm<typename DispatchPolicy::ArchTag, AType, BType, CType, BiasType>,
          class TileMmad = Gemm::Tile::TileMmad<typename DispatchPolicy::ArchTag, AType, BType, BiasType>>
struct BlockMmad {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "BlockMmad is not implemented for this DispatchPolicy");
};

} // namespace Block

} // namespace Gemm

namespace Epilogue {
namespace Tile {

template <class ArchTag, class CType, class ScaleType, class PerTokenScaleType, class DType>
struct TileCopy {
};

template <class ArchTag, class ElementMulType, int Dummy = 0>
struct TileElemWiseMuls {
};

} // namespace Tile

namespace Block {

template <class DispatchPolicy, class... Args>
class BlockEpilogue {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "Could not find an epilogue specialization");
};

} // namespace Block

} // namespace Epilogue

} // namespace pto_ext

namespace pto_ext::support {

constexpr uint64_t kL2Offset = pto_ext::L2_OFFSET;

struct NoopCallback {
    __forceinline__ __aicore__ void operator()() const
    {}
};

} // namespace pto_ext::support

namespace pto_ext::dispatch_combine_moe::pto_detail {

using pto_ext::dispatch_combine_moe::pto_bridge::PtoAbsVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoAddScalarVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoAddVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoCastVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoDivVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoFillVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoGetValue;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoLoadVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoMoveVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoMulElementwiseVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoMulVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoReduceMaxVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoSetValue;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoStoreAtomicAddVector;
using pto_ext::dispatch_combine_moe::pto_bridge::PtoStoreVector;

using pto_ext::PtoPipeBarrier;
using pto_ext::PtoSetFlag;
using pto_ext::PtoSetWaitFlag;
using pto_ext::PtoSyncAll;
using pto_ext::PtoWaitFlag;

} // namespace pto_ext::dispatch_combine_moe::pto_detail

#endif // MOE_PTO_UTILS_HPP
