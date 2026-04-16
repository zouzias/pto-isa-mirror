// a2a3/cce_stub_reference.hpp
#pragma once

// Documentation-only reference stub.
// This file mirrors the A2/A3 callable surface used by include/pto/npu/a2a3
// and annotates parameters for latency modeling.
//
// It is intentionally not included by the live costmodel path.
// Do not include it together with a2a3/cce_stub.hpp: the symbols intentionally overlap.
//
// Parameter names are normalized across instruction families when semantics are clear.
// If a meaning is still uncertain, the comment keeps a '?' marker.

// Copy / move helpers
inline void copy_cbuf_to_bt(
    auto dst /* bias-buffer dst base */,
    auto src /* L1/cbuf src base */,
    auto convControl /* half->float convert control? */,
    auto nBurst /* burst count */,
    auto lenBurst /* burst length in 64B units */,
    auto srcStride /* src stride between bursts? */,
    auto dstStride /* dst stride between bursts? */)
{}

inline void copy_cbuf_to_fbuf(
    auto dst /* scaling/fbuf dst base */,
    auto src /* L1/cbuf src base */,
    auto nBurst /* burst count */,
    auto lenBurst /* burst length in 128B units */,
    auto srcStride /* src stride between bursts? */,
    auto dstStride /* dst stride between bursts? */)
{}

inline void copy_cbuf_to_gm(
    auto dst /* GM dst base */,
    auto src /* L1/cbuf src base */,
    auto sid /* stream id? */,
    auto nBurst /* burst count */,
    auto lenBurst /* burst length in 32B blocks */,
    auto srcStride /* cbuf stride in 32B blocks */,
    auto dstStride /* GM stride in 32B blocks */)
{}

inline void copy_gm_to_cbuf(
    auto dst /* L1/cbuf dst base */,
    auto src /* GM src base */,
    auto sid /* stream id? */,
    auto nBurst /* burst count */,
    auto lenBurst /* burst length in 32B blocks */,
    auto gmGap /* GM gap in 32B blocks */,
    auto l1Gap /* L1 gap in 32B blocks */,
    auto pad /* pad mode/value? */)
{}

inline void copy_gm_to_cbuf_multi_nd2nz_b8(
    auto dst /* NZ-formatted L1 dst base */,
    auto src /* ND GM src base */,
    auto sid /* stream id? */,
    auto ndNum /* number of ND matrices */,
    auto nValue /* N dimension length */,
    auto dValue /* D/C0-related length */,
    auto srcNdMatrixStride /* ND matrix stride */,
    auto srcDValue /* source D stride/value */,
    auto dstNzC0Stride /* NZ C0 stride */,
    auto dstNzNStride /* NZ N stride */,
    auto dstNzMatrixStride /* NZ matrix stride */)
{}

inline void copy_gm_to_cbuf_multi_nd2nz_b16(
    auto dst /* NZ-formatted L1 dst base */,
    auto src /* ND GM src base */,
    auto sid /* stream id? */,
    auto ndNum /* number of ND matrices */,
    auto nValue /* N dimension length */,
    auto dValue /* D/C0-related length */,
    auto srcNdMatrixStride /* ND matrix stride */,
    auto srcDValue /* source D stride/value */,
    auto dstNzC0Stride /* NZ C0 stride */,
    auto dstNzNStride /* NZ N stride */,
    auto dstNzMatrixStride /* NZ matrix stride */)
{}

inline void copy_gm_to_cbuf_multi_nd2nz_b32s(
    auto dst /* NZ-formatted L1 dst base */,
    auto src /* ND GM src base */,
    auto sid /* stream id? */,
    auto ndNum /* number of ND matrices */,
    auto nValue /* N dimension length */,
    auto dValue /* D/C0-related length */,
    auto srcNdMatrixStride /* ND matrix stride */,
    auto srcDValue /* source D stride/value */,
    auto dstNzC0Stride /* NZ C0 stride */,
    auto dstNzNStride /* NZ N stride */,
    auto dstNzMatrixStride /* NZ matrix stride */)
{}

inline void copy_gm_to_ubuf_align_b8(
    auto dst /* UB dst base */,
    auto src /* GM src base */,
    auto sid /* stream id? */,
    auto nBurst /* burst count */,
    auto lenBurst /* burst length in bytes */,
    auto leftPadding /* left padding elements? */,
    auto rightPadding /* right padding elements? */,
    auto gmGap /* GM gap in bytes */,
    auto ubGap /* UB gap in 32B blocks */)
{}

inline void copy_gm_to_ubuf_align_b16(
    auto dst /* UB dst base */,
    auto src /* GM src base */,
    auto sid /* stream id? */,
    auto nBurst /* burst count */,
    auto lenBurst /* burst length in bytes */,
    auto leftPadding /* left padding elements? */,
    auto rightPadding /* right padding elements? */,
    auto gmGap /* GM gap in bytes */,
    auto ubGap /* UB gap in 32B blocks */)
{}

inline void copy_gm_to_ubuf_align_b32(
    auto dst /* UB dst base */,
    auto src /* GM src base */,
    auto sid /* stream id? */,
    auto nBurst /* burst count */,
    auto lenBurst /* burst length in bytes */,
    auto leftPadding /* left padding elements? */,
    auto rightPadding /* right padding elements? */,
    auto gmGap /* GM gap in bytes */,
    auto ubGap /* UB gap in 32B blocks */)
{}

inline void copy_matrix_cc_to_cbuf(
    auto dst /* L1/cbuf dst base */,
    auto src /* L0C/cc src base */,
    auto sid /* stream id? */,
    auto nSize /* stored N/column span, usually c0-aligned */,
    auto mSize /* stored M/row span */,
    auto dstStrideD /* dst D/leading stride? */,
    auto srcStride /* src leading stride? */,
    auto reserved /* reserved/config? */,
    auto quantPre /* pre-quant mode */,
    auto reluMode /* relu/pre-activation mode */,
    auto flag0 /* ? */,
    auto flag1 /* ? */)
{}

inline void copy_matrix_cc_to_gm(
    auto dst /* GM dst base */,
    auto src /* L0C/cc src base */,
    auto xmReg /* matrix-store shape/config register */,
    auto xtReg /* matrix-store extra config register? */)
{}

inline void copy_ubuf_to_gm_align_b8(
    auto dst /* GM dst base */,
    auto src /* UB src base */,
    auto sid /* stream id? */,
    auto nBurst /* burst count */,
    auto lenBurst /* burst length in bytes */,
    auto leftPadding /* left padding elements? */,
    auto rightPadding /* right padding elements? */,
    auto ubGap /* UB gap in 32B blocks */,
    auto gmGap /* GM gap in bytes */)
{}

inline void copy_ubuf_to_gm_align_b16(
    auto dst /* GM dst base */,
    auto src /* UB src base */,
    auto sid /* stream id? */,
    auto nBurst /* burst count */,
    auto lenBurst /* burst length in bytes */,
    auto leftPadding /* left padding elements? */,
    auto rightPadding /* right padding elements? */,
    auto ubGap /* UB gap in 32B blocks */,
    auto gmGap /* GM gap in bytes */)
{}

inline void copy_ubuf_to_gm_align_b32(
    auto dst /* GM dst base */,
    auto src /* UB src base */,
    auto sid /* stream id? */,
    auto nBurst /* burst count */,
    auto lenBurst /* burst length in bytes */,
    auto leftPadding /* left padding elements? */,
    auto rightPadding /* right padding elements? */,
    auto ubGap /* UB gap in 32B blocks */,
    auto gmGap /* GM gap in bytes */)
{}

inline void copy_ubuf_to_ubuf(
    auto dst /* UB dst base */,
    auto src /* UB src base */,
    auto sid /* stream id? */,
    auto nBurst /* burst count */,
    auto lenBurst /* burst length in 32B blocks */,
    auto srcGap /* src gap in 32B blocks */,
    auto dstGap /* dst gap in 32B blocks */)
{}

// Matrix / load helpers
inline void create_cbuf_matrix(
    auto dst /* cbuf matrix dst base */,
    auto repeatConfig /* packed repeat/block-gap config */,
    auto value /* fill value or packed fill bits */)
{}

inline void create_cbuf_matrix_bf16(
    auto dst /* cbuf bf16 matrix dst base */,
    auto repeatConfig /* packed repeat/block-gap config */,
    auto value /* bf16 fill value */)
{}

inline void dsb(auto barrierType /* DSB type */) {}

inline void ffts_cross_core_sync(
    auto srcPipe /* source pipe for cross-core sync */,
    auto msg /* FFTS message/config word */)
{}

inline void img2colv2_cbuf_to_ca(
    auto dst /* L0A/ca dst base */,
    auto src /* L1/cbuf src base */,
    auto stepK /* K span / dst cols */,
    auto stepM /* M span / dst rows */,
    auto posK /* K/index col position */,
    auto posM /* M/index row position */,
    auto strideW /* stride W */,
    auto strideH /* stride H */,
    auto filterW /* low 8 bits of filter W or full filter W when small */,
    auto filterH /* low 8 bits of filter H or full filter H when small */,
    auto dilationW /* dilation W */,
    auto dilationH /* dilation H */,
    auto highFilterW /* high-filter-W flag */,
    auto highFilterH /* high-filter-H flag */,
    auto transpose /* transpose enable */,
    auto fmatrixCtrl /* use/set fmatrix B-mode? */,
    auto channelSize /* channel size / src leading span */)
{}

inline void img2colv2_cbuf_to_cb(
    auto dst /* L0B/cb dst base */,
    auto src /* L1/cbuf src base */,
    auto stepK /* K span / dst cols */,
    auto stepM /* M span / dst rows */,
    auto posK /* K/index col position */,
    auto posM /* M/index row position */,
    auto strideW /* stride W */,
    auto strideH /* stride H */,
    auto filterW /* low 8 bits of filter W or full filter W when small */,
    auto filterH /* low 8 bits of filter H or full filter H when small */,
    auto dilationW /* dilation W */,
    auto dilationH /* dilation H */,
    auto highFilterW /* high-filter-W flag */,
    auto highFilterH /* high-filter-H flag */,
    auto transpose /* transpose enable */,
    auto fmatrixCtrl /* use/set fmatrix B-mode? */,
    auto channelSize /* channel size / src leading span */)
{}

inline void load_cbuf_to_ca(
    auto dst /* L0A/ca dst base */,
    auto src /* L1/cbuf src base */,
    auto baseIdx /* source block/fractal index */,
    auto repeat /* repeat count */,
    auto srcStride /* source stride between repeats */,
    auto sid /* stream id? */,
    auto transpose /* transpose/addr-calc flag? */)
{}

inline void load_cbuf_to_ca_transpose(
    auto dst /* L0A/ca dst base */,
    auto src /* L1/cbuf src base */,
    auto baseIdx /* source block/fractal index */,
    auto repeat /* repeat count */,
    auto srcStride /* source stride between repeats */,
    auto dstStride /* dst stride */,
    auto addrCalMode /* address calculation mode? */,
    auto dstFracStride /* dst fractal stride */)
{}

inline void load_cbuf_to_cb(
    auto dst /* L0B/cb dst base */,
    auto src /* L1/cbuf src base */,
    auto baseIdx /* source block/fractal index */,
    auto repeat /* repeat count */,
    auto srcStride /* source stride between repeats */,
    auto dstStride /* dst stride */,
    auto sid /* stream id? */,
    auto transpose /* transpose flag */,
    auto addrCalMode /* address calculation mode? */)
{}

inline void load_cbuf_to_cb_transpose(
    auto dst /* L0B/cb dst base */,
    auto src /* L1/cbuf src base */,
    auto baseIdx /* source block/fractal index */,
    auto repeat /* repeat count */,
    auto srcStride /* source stride between repeats */,
    auto dstStride /* dst stride */,
    auto addrCalMode /* address calculation mode? */,
    auto dstFracStride /* dst fractal stride */)
{}

// Cube / sync / setup helpers
inline void mad(
    auto c /* L0C/cc dst/accumulator base */,
    auto a /* L0A/ca left operand base */,
    auto b /* L0B/cb right operand base */,
    auto m /* M size */,
    auto k /* K size */,
    auto n /* N size */,
    auto phase /* accumulation phase */,
    auto kDirectionAlign /* K-direction alignment flag */,
    auto cmatrixSource /* use existing C matrix as source */,
    auto cmatrixInitVal /* initialize C matrix */)
{}

inline void pipe_barrier(auto pipe /* pipe selector */) {}

inline void scatter_vnchwconv_b8(
    auto dst /* dst VA register group */,
    auto src /* src VA register group */,
    auto repeat /* repeat count */,
    auto dstStride /* dst stride */,
    auto srcStride /* src stride */,
    auto dstHighHalf /* use high half of dst VA group */,
    auto srcHighHalf /* use high half of src VA group */)
{}

inline void scatter_vnchwconv_b16(
    auto dst /* dst VA register group */,
    auto src /* src VA register group */,
    auto repeat /* repeat count */,
    auto dstStride /* dst stride */,
    auto srcStride /* src stride */)
{}

inline void scatter_vnchwconv_b32(
    auto dst /* dst VA register group */,
    auto src /* src VA register group */,
    auto repeat /* repeat count */,
    auto dstStride /* dst stride */,
    auto srcStride /* src stride */)
{}

inline void set_atomic_add() {}
inline void set_atomic_bf16() {}
inline void set_atomic_f16() {}
inline void set_atomic_f32() {}
inline void set_atomic_none() {}
inline void set_atomic_s16() {}
inline void set_atomic_s32() {}
inline void set_atomic_s8() {}

inline void set_cmpmask(auto cmpMaskPtr /* compare-mask buffer base */) {}
inline void set_ctrl(auto ctrl /* control register value */) {}
inline void set_deqscale(auto scale /* dequant scale */) {}
inline void set_ffts_base_addr(auto fftsAddr /* FFTS base address */) {}

inline void set_flag(
    auto srcPipe /* source pipe */,
    auto dstPipe /* destination pipe */,
    auto token /* event token */)
{}

inline void set_fmatrix(auto regFmatrix /* packed FMATRIX register */) {}
inline void set_fmatrix_b(auto regFmatrix /* packed FMATRIX_B register */) {}
inline void set_fpc(auto deqTensorAddr /* packed FPC/dequant address */) {}
inline void set_l3d_rpt(auto rptConfig /* packed L3D repeat config */) {}
inline void set_mask_count() {}
inline void set_mask_norm() {}
inline void set_mov_pad_val(auto value /* UB move padding value */) {}
inline void set_nd_para(auto ndParaSPR /* packed ND parameter register */) {}
inline void set_padding(auto paddingValue /* padding value */) {}
inline void set_quant_pre(auto preQuantScalar /* pre-quant scalar/config */) {}

inline void set_va_reg_sb(
    auto vaReg /* VA register id */,
    auto addrArray /* VA address table base */)
{}

inline void set_vector_mask(
    auto mask0 /* low mask/config word */,
    auto mask1 /* high mask/config word or element count */)
{}

inline void wait_flag(
    auto srcPipe /* source pipe */,
    auto dstPipe /* destination pipe */,
    auto token /* event token */)
{}

inline void wait_flag_dev(auto flagId /* device/cross-core flag id */) {}

// Shared vector signatures
#define PTO_REF_VECTOR_BINARY(name)                                                          \
    inline void name(                                                                        \
        auto dst /* dst vector base */,                                                      \
        auto src0 /* src0 vector base */,                                                    \
        auto src1 /* src1 vector base */,                                                    \
        auto repeat /* repeat count */,                                                      \
        auto dstBlockStride /* dst block stride */,                                          \
        auto src0BlockStride /* src0 block stride */,                                        \
        auto src1BlockStride /* src1 block stride */,                                        \
        auto dstRepeatStride /* dst repeat stride */,                                        \
        auto src0RepeatStride /* src0 repeat stride */,                                      \
        auto src1RepeatStride /* src1 repeat stride */)                                      \
    {}

#define PTO_REF_VECTOR_BINARY_SCALAR(name)                                                   \
    inline void name(                                                                        \
        auto dst /* dst vector base */,                                                      \
        auto src0 /* src0 vector base */,                                                    \
        auto src1 /* scalar/immediate or scalar-like source */,                              \
        auto repeat /* repeat count */,                                                      \
        auto dstBlockStride /* dst block stride */,                                          \
        auto src0BlockStride /* src0 block stride */,                                        \
        auto dstRepeatStride /* dst repeat stride */,                                        \
        auto src0RepeatStride /* src0 repeat stride */)                                      \
    {}

#define PTO_REF_VECTOR_UNARY(name)                                                           \
    inline void name(                                                                        \
        auto dst /* dst vector base */,                                                      \
        auto src /* src vector base */,                                                      \
        auto repeat /* repeat count */,                                                      \
        auto dstBlockStride /* dst block stride */,                                          \
        auto srcBlockStride /* src block stride */,                                          \
        auto dstRepeatStride /* dst repeat stride */,                                        \
        auto srcRepeatStride /* src repeat stride */)                                        \
    {}

#define PTO_REF_VCMPV(name)                                                                  \
    inline void name(                                                                        \
        auto dst /* compare result / mask dst base */,                                       \
        auto src0 /* src0 vector base */,                                                    \
        auto src1 /* src1 vector base */,                                                    \
        auto repeat /* repeat count */,                                                      \
        auto dstBlockStride /* dst block stride */,                                          \
        auto src0BlockStride /* src0 block stride */,                                        \
        auto src1BlockStride /* src1 block stride */,                                        \
        auto dstRepeatStride /* dst repeat stride */,                                        \
        auto src0RepeatStride /* src0 repeat stride */,                                      \
        auto src1RepeatStride /* src1 repeat stride */)                                      \
    {}

#define PTO_REF_VCMPVS(name)                                                                 \
    inline void name(                                                                        \
        auto dst /* compare result / mask dst base */,                                       \
        auto src0 /* src0 vector base */,                                                    \
        auto src1 /* scalar compare value */,                                                \
        auto repeat /* repeat count */,                                                      \
        auto dstBlockStride /* dst block stride */,                                          \
        auto src0BlockStride /* src0 block stride */,                                        \
        auto dstRepeatStride /* dst repeat stride */,                                        \
        auto srcRepeatStride /* src repeat stride */)                                        \
    {}

#define PTO_REF_VCONV(name)                                                                  \
    inline void name(                                                                        \
        auto dst /* dst vector base */,                                                      \
        auto src /* src vector base */,                                                      \
        auto repeat /* repeat count */,                                                      \
        auto dstBlockStride /* dst block stride */,                                          \
        auto srcBlockStride /* src block stride */,                                          \
        auto dstRepeatStride /* dst repeat stride */,                                        \
        auto srcRepeatStride /* src repeat stride */)                                        \
    {}

// Vector arithmetic / compare families
PTO_REF_VECTOR_UNARY(vabs)
PTO_REF_VECTOR_BINARY(vadd)
PTO_REF_VECTOR_BINARY_SCALAR(vadds)
PTO_REF_VECTOR_BINARY(vand)
PTO_REF_VECTOR_BINARY_SCALAR(vaxpy)

inline void vbitsort(
    auto dst /* sorted values dst base */,
    auto src /* unsorted values src base */,
    auto idx /* index/output index base */,
    auto repeat /* repeat count */)
{}

inline void vbrcb(
    auto dst /* broadcast-expanded dst base */,
    auto src /* compact src base */,
    auto dstBlockStride /* dst block stride? */,
    auto dstRepeatStride /* dst repeat stride? */,
    auto repeat /* repeat count */)
{}

inline void vcadd(
    auto dst /* reduced dst base */,
    auto src /* src vector base */,
    auto repeat /* repeat count */,
    auto dstRepeatStride /* dst repeat stride */,
    auto srcBlockStride /* src block stride */,
    auto srcRepeatStride /* src repeat stride */,
    auto mode /* mode flag? */)
{}

inline void vcgadd(
    auto dst /* grouped-reduce dst base */,
    auto src /* src vector base */,
    auto repeat /* repeat count */,
    auto dstRepeatStride /* dst repeat stride */,
    auto src0RepeatStride /* first src repeat stride */,
    auto src1RepeatStride /* second src repeat stride */)
{}

inline void vcgmax(
    auto dst /* grouped-reduce dst base */,
    auto src /* src vector base */,
    auto repeat /* repeat count */,
    auto dstRepeatStride /* dst repeat stride */,
    auto src0RepeatStride /* first src repeat stride */,
    auto src1RepeatStride /* second src repeat stride */)
{}

inline void vcgmin(
    auto dst /* grouped-reduce dst base */,
    auto src /* src vector base */,
    auto repeat /* repeat count */,
    auto dstRepeatStride /* dst repeat stride */,
    auto src0RepeatStride /* first src repeat stride */,
    auto src1RepeatStride /* second src repeat stride */)
{}

inline void vcmax(
    auto dst /* reduce dst base */,
    auto src /* src vector base */,
    auto repeat /* repeat count */,
    auto dstRepeatStride /* dst repeat stride */,
    auto srcBlockStride /* src block stride */,
    auto srcRepeatStride /* src repeat stride */,
    auto mode /* ONLY_VALUE / ONLY_INDEX / VALUE_INDEX */)
{}

inline void vcmin(
    auto dst /* reduce dst base */,
    auto src /* src vector base */,
    auto repeat /* repeat count */,
    auto dstRepeatStride /* dst repeat stride */,
    auto srcBlockStride /* src block stride */,
    auto srcRepeatStride /* src repeat stride */,
    auto mode /* ONLY_VALUE / ONLY_INDEX / VALUE_INDEX */)
{}

PTO_REF_VCMPV(vcmpv_eq)
PTO_REF_VCMPV(vcmpv_ge)
PTO_REF_VCMPV(vcmpv_gt)
PTO_REF_VCMPV(vcmpv_le)
PTO_REF_VCMPV(vcmpv_lt)
PTO_REF_VCMPV(vcmpv_ne)

PTO_REF_VCMPVS(vcmpvs_eq)
PTO_REF_VCMPVS(vcmpvs_ge)
PTO_REF_VCMPVS(vcmpvs_gt)
PTO_REF_VCMPVS(vcmpvs_le)
PTO_REF_VCMPVS(vcmpvs_lt)
PTO_REF_VCMPVS(vcmpvs_ne)

PTO_REF_VCONV(vconv_bf162f32)
PTO_REF_VCONV(vconv_bf162s32a)
PTO_REF_VCONV(vconv_bf162s32c)
PTO_REF_VCONV(vconv_bf162s32f)
PTO_REF_VCONV(vconv_bf162s32r)
PTO_REF_VCONV(vconv_bf162s32z)
PTO_REF_VCONV(vconv_deq)
PTO_REF_VCONV(vconv_f162f32)
PTO_REF_VCONV(vconv_f162s16a)
PTO_REF_VCONV(vconv_f162s16c)
PTO_REF_VCONV(vconv_f162s16f)
PTO_REF_VCONV(vconv_f162s16r)
PTO_REF_VCONV(vconv_f162s16z)
PTO_REF_VCONV(vconv_f162s32a)
PTO_REF_VCONV(vconv_f162s32c)
PTO_REF_VCONV(vconv_f162s32f)
PTO_REF_VCONV(vconv_f162s32r)
PTO_REF_VCONV(vconv_f162s32z)
PTO_REF_VCONV(vconv_f162s8a)
PTO_REF_VCONV(vconv_f162s8c)
PTO_REF_VCONV(vconv_f162s8f)
PTO_REF_VCONV(vconv_f162s8r)
PTO_REF_VCONV(vconv_f162s8z)
PTO_REF_VCONV(vconv_f162u8a)
PTO_REF_VCONV(vconv_f162u8c)
PTO_REF_VCONV(vconv_f162u8f)
PTO_REF_VCONV(vconv_f162u8r)
PTO_REF_VCONV(vconv_f162u8z)
PTO_REF_VCONV(vconv_f322bf16a)
PTO_REF_VCONV(vconv_f322bf16c)
PTO_REF_VCONV(vconv_f322bf16f)
PTO_REF_VCONV(vconv_f322bf16r)
PTO_REF_VCONV(vconv_f322bf16z)
PTO_REF_VCONV(vconv_f322f16a)
PTO_REF_VCONV(vconv_f322f16c)
PTO_REF_VCONV(vconv_f322f16f)
PTO_REF_VCONV(vconv_f322f16o)
PTO_REF_VCONV(vconv_f322f16r)
PTO_REF_VCONV(vconv_f322f16z)
PTO_REF_VCONV(vconv_f322f32a)
PTO_REF_VCONV(vconv_f322f32c)
PTO_REF_VCONV(vconv_f322f32r)
PTO_REF_VCONV(vconv_f322s16a)
PTO_REF_VCONV(vconv_f322s16c)
PTO_REF_VCONV(vconv_f322s16f)
PTO_REF_VCONV(vconv_f322s16r)
PTO_REF_VCONV(vconv_f322s16z)
PTO_REF_VCONV(vconv_f322s32a)
PTO_REF_VCONV(vconv_f322s32c)
PTO_REF_VCONV(vconv_f322s32f)
PTO_REF_VCONV(vconv_f322s32z)
PTO_REF_VCONV(vconv_f322s64a)
PTO_REF_VCONV(vconv_f322s64c)
PTO_REF_VCONV(vconv_f322s64f)
PTO_REF_VCONV(vconv_f322s64r)
PTO_REF_VCONV(vconv_f322s64z)
PTO_REF_VCONV(vconv_s162f16)
PTO_REF_VCONV(vconv_s162f16a)
PTO_REF_VCONV(vconv_s162f16c)
PTO_REF_VCONV(vconv_s162f16f)
PTO_REF_VCONV(vconv_s162f16r)
PTO_REF_VCONV(vconv_s162f16z)
PTO_REF_VCONV(vconv_s162f32)
PTO_REF_VCONV(vconv_s322f32a)
PTO_REF_VCONV(vconv_s322f32c)
PTO_REF_VCONV(vconv_s322f32f)
PTO_REF_VCONV(vconv_s322f32r)
PTO_REF_VCONV(vconv_s322f32z)
PTO_REF_VCONV(vconv_s322s16)
PTO_REF_VCONV(vconv_s322s64)
PTO_REF_VCONV(vconv_s642f32a)
PTO_REF_VCONV(vconv_s642f32c)
PTO_REF_VCONV(vconv_s642f32f)
PTO_REF_VCONV(vconv_s642f32r)
PTO_REF_VCONV(vconv_s642f32z)
PTO_REF_VCONV(vconv_s642s32)
PTO_REF_VCONV(vconv_s82f16)
PTO_REF_VCONV(vconv_u82f16)

PTO_REF_VECTOR_UNARY(vcopy)
PTO_REF_VECTOR_BINARY(vdiv)
PTO_REF_VECTOR_UNARY(vector_dup)
PTO_REF_VECTOR_UNARY(vexp)

inline void vgather(
    auto dst /* gathered dst base */,
    auto offset /* byte-offset/index vector base */,
    auto srcBaseAddr /* GM/UB source base address */,
    auto dstRepeatStride /* dst repeat stride? */,
    auto repeat /* repeat count */)
{}

inline void vgatherb(
    auto dst /* gathered dst base */,
    auto offset /* offset vector base */,
    auto srcBaseAddr /* source base address */,
    auto dstRepeatStride /* dst repeat stride */,
    auto dstBlockStride /* dst block stride? */,
    auto repeat /* repeat count */)
{}

PTO_REF_VECTOR_UNARY(vln)
PTO_REF_VECTOR_BINARY_SCALAR(vlrelu)
PTO_REF_VECTOR_BINARY(vmax)
PTO_REF_VECTOR_BINARY_SCALAR(vmaxs)
PTO_REF_VECTOR_BINARY(vmin)
PTO_REF_VECTOR_BINARY_SCALAR(vmins)

inline void vmrgsort4(
    auto dst /* merged/sorted dst base */,
    auto addrArray /* array of input list base addresses */,
    auto count /* packed per-list element counts */,
    auto config /* packed merge-sort config */)
{}

PTO_REF_VECTOR_BINARY(vmul)
PTO_REF_VECTOR_BINARY_SCALAR(vmuls)
PTO_REF_VECTOR_UNARY(vnot)
PTO_REF_VECTOR_BINARY(vor)

inline void vreducev2(
    auto dst /* reduced dst base */,
    auto src0 /* src/value-or-index base */,
    auto src1 /* mask/index/helper base */,
    auto repeat /* repeat count */,
    auto src0BlockStride /* src0 block stride? */,
    auto modeOrMaskPattern /* mask pattern / mode */,
    auto src0RepeatStride /* src0 repeat stride */,
    auto src1RepeatStride /* src1/dst repeat stride? */)
{}

PTO_REF_VECTOR_UNARY(vrelu)
PTO_REF_VECTOR_UNARY(vrsqrt)

inline void vsel(
    auto dst /* dst vector base */,
    auto src0 /* selected-when-true source */,
    auto src1 /* selected-when-false source */,
    auto repeat /* repeat count */,
    auto dstBlockStride /* dst block stride */,
    auto src0BlockStride /* src0 block stride */,
    auto src1BlockStride /* src1 block stride */,
    auto dstRepeatStride /* dst repeat stride */,
    auto src0RepeatStride /* src0 repeat stride */,
    auto src1RepeatStride /* src1 repeat stride */,
    auto mode /* select mode, e.g. tensor/tensor vs cmpmask */)
{}

PTO_REF_VECTOR_BINARY_SCALAR(vshl)

inline void vshr(
    auto dst /* dst vector base */,
    auto src0 /* src0 vector base */,
    auto src1 /* scalar/immediate shift value */,
    auto repeat /* repeat count */,
    auto dstBlockStride /* dst block stride */,
    auto src0BlockStride /* src0 block stride */,
    auto dstRepeatStride /* dst repeat stride */,
    auto src0RepeatStride /* src0 repeat stride */,
    auto isArithmetic /* arithmetic shift enable */)
{}

PTO_REF_VECTOR_UNARY(vsqrt)
PTO_REF_VECTOR_BINARY(vsub)

#undef PTO_REF_VECTOR_BINARY
#undef PTO_REF_VECTOR_BINARY_SCALAR
#undef PTO_REF_VECTOR_UNARY
#undef PTO_REF_VCMPV
#undef PTO_REF_VCMPVS
#undef PTO_REF_VCONV

// Optional debug-print surface used by TPrint.hpp
namespace cce {
template <typename Fmt, typename... Args>
inline void printf(Fmt fmt /* format string? */, Args... args /* print payload? */)
{}
} // namespace cce
