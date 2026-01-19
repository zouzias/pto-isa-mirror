#include <pypto/tileop/tileop_common.h>
#include <pypto/tileop/arch32/vector.h>
#include <pypto/tileop/arch32/vector_bin.h>
#include <pypto/tileop/utils/layout.h>
#include <pypto/tileop/utils/common_type.h>
#include <pypto/tileop/utils/tile_tensor.h>
#include <pypto/tileop/vector/binary_scalar.h>
#include <pypto/tileop/vector/binary.h>
#include <pypto/tileop/vector/expand.h>
#include <pypto/tileop/vector/cast.h>
#include <pypto/tileop/vector/reduce.h>
#include <pypto/tileop/vector/mte.h>

#include "acl/acl.h"

// funcHash: 11462971439825151006

#define TILEOP_IMPL

struct ShapeInfo{
    uint32_t shape0;
    uint32_t shape1;
    uint32_t shape2;
    uint32_t shape3;
};

template <typename T, int S0, int S1, int S2, int S3>
__global__ __aicore__ void TDeepseek_hc_host(__gm__ float *output, __gm__ float *input0, __gm__ float* input1, __gm__ ShapeInfo *param) { 
    float __ubuf__ *UB_S163840_E164352 = (float __ubuf__ *)get_imm(0x28000); // size: 0x200
    float *UB_S163840_E164352_T = (float *)get_imm(0x28000); // size: 0x200
    bfloat16_t __ubuf__ *UB_S213504_E217600 = (bfloat16_t __ubuf__ *)get_imm(0x34200); // size: 0x1000
    bfloat16_t *UB_S213504_E217600_T = (bfloat16_t *)get_imm(0x34200); // size: 0x1000
    float __ubuf__ *UB_S225792_E225920 = (float __ubuf__ *)get_imm(0x37200); // size: 0x80
    float *UB_S225792_E225920_T = (float *)get_imm(0x37200); // size: 0x80
    float __ubuf__ *UB_S0_E32768 = (float __ubuf__ *)get_imm(0x0); // size: 0x8000
    float *UB_S0_E32768_T = (float *)get_imm(0x0); // size: 0x8000
    float __ubuf__ *UB_S32768_E163840 = (float __ubuf__ *)get_imm(0x8000); // size: 0x20000
    float *UB_S32768_E163840_T = (float *)get_imm(0x8000); // size: 0x20000
    float __ubuf__ *UB_S164352_E197120 = (float __ubuf__ *)get_imm(0x28200); // size: 0x8000
    float *UB_S164352_E197120_T = (float *)get_imm(0x28200); // size: 0x8000
    float __ubuf__ *UB_S197120_E213504 = (float __ubuf__ *)get_imm(0x30200); // size: 0x4000
    float *UB_S197120_E213504_T = (float *)get_imm(0x30200); // size: 0x4000
    float __ubuf__ *UB_S217600_E225792 = (float __ubuf__ *)get_imm(0x35200); // size: 0x2000
    float *UB_S217600_E225792_T = (float *)get_imm(0x35200); // size: 0x2000
    bfloat16_t __ubuf__ *UB_S225920_E242304 = (bfloat16_t __ubuf__ *)get_imm(0x37280); // size: 0x4000
    bfloat16_t *UB_S225920_E242304_T = (bfloat16_t *)get_imm(0x37280); // size: 0x4000

    // uint64_t sym_0_dim_0 = param->shape0;
    // uint64_t sym_0_dim_1 = param->shape1;
    // uint64_t sym_0_dim_2 = param->shape2;
    // uint64_t sym_0_dim_3 = param->shape3;
    // uint64_t sym_1_dim_0 = param->shape0; 
    // uint64_t sym_1_dim_1 = param->shape1;
    // uint64_t sym_1_dim_2 = param->shape2;
    // uint64_t sym_1_dim_3 = param->shape3;

    
    // uint64_t sym_121_dim_1 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE_MAYBE_CONST(1, 4, 3, 61, 1)); 
    // uint64_t sym_121_dim_2 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE_MAYBE_CONST(1, 2048, 3, 61, 2));
    // uint64_t sym_2681_dim_0 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(3, 35, 0)); 
    // uint64_t sym_2681_dim_1 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE_MAYBE_CONST(1, 1, 3, 35, 1)); 
    // uint64_t sym_2681_dim_2 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE_MAYBE_CONST(1, 2048, 3, 35, 2)); 
    // uint64_t sym_2808_dim_2 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE_MAYBE_CONST(1, 2048, 3, 185, 2)); 
    // uint64_t sym_3065_dim_0 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(3, 48, 0)); 
    // uint64_t sym_3255_dim_3 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE_MAYBE_CONST(1, 2048, 4, 108, 3));
    // uint64_t sym_3257_dim_0 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(4, 1, 0)); 
    // uint64_t sym_3257_dim_3 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE_MAYBE_CONST(1, 2048, 4, 1, 3)); 
    // uint64_t sym_3513_dim_0 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE(4, 18, 0));
    // uint64_t sym_6881_dim_2 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE_MAYBE_CONST(1, 2048, 3, 211, 2));
    // uint64_t sym_2808_dim_0 = sym_2681_dim_1;
    // uint64_t sym_2808_dim_1 = sym_121_dim_1;
    // uint64_t sym_3065_dim_1 = sym_121_dim_1;
    // uint64_t sym_3065_dim_2 = sym_2681_dim_1;
    // uint64_t sym_3255_dim_0 = sym_2681_dim_1;
    // uint64_t sym_3255_dim_1 = sym_121_dim_1;
    // uint64_t sym_3255_dim_2 = sym_121_dim_1;
    // uint64_t sym_3257_dim_1 = sym_121_dim_1;
    // uint64_t sym_3257_dim_2 = sym_2681_dim_1;
    // uint64_t sym_3513_dim_1 = sym_121_dim_1;
    // uint64_t sym_3513_dim_2 = sym_121_dim_1;
    // uint64_t sym_3513_dim_3 = sym_2681_dim_1;
    // uint64_t sym_6881_dim_0 = sym_2681_dim_1;
    // uint64_t sym_6881_dim_1 = sym_121_dim_1;
    
    using UBTileTensorFP32Dim4_15 = TileTensor<float, LocalLayout4Dim<1, 4, 4, 8>, Hardware::UB>;
    using UBTileTensorFP32Dim4_21 = TileTensor<float, LocalLayout4Dim<1, 4, 1, 2048>, Hardware::UB>;
    using GMTileTensorFP32Dim4_16 = TileTensor<__gm__ float, DynLayout4Dim, Hardware::GM>;
    using UBTileTensorBF16Dim3_28 = TileTensor<bfloat16_t, LocalLayout3Dim<1, 4, 2048>, Hardware::UB>;
    using UBTileTensorFP32Dim3_26 = TileTensor<float, LocalLayout3Dim<1, 1, 2048>, Hardware::UB>;
    using UBTileTensorBF16Dim3_17 = TileTensor<bfloat16_t, LocalLayout3Dim<1, 1, 2048>, Hardware::UB>;
    using UBTileTensorFP32Dim3_19 = TileTensor<float, LocalLayout3Dim<1, 4, 8>, Hardware::UB>;
    using GMTileTensorBF16Dim3_18 = TileTensor<__gm__ bfloat16_t, DynLayout3Dim, Hardware::GM>;
    using GMTileTensorFP32Dim3_20 = TileTensor<__gm__ float, DynLayout3Dim, Hardware::GM>;
    using GMTileTensorFP32Dim4_22 = TileTensor<__gm__ float, DynLayout4Dim, Hardware::GM>;
    using UBTileTensorFP32Dim4_23 = TileTensor<float, LocalLayout4Dim<1, 4, 4, 2048>, Hardware::UB>;
    using UBTileTensorFP32Dim4_24 = TileTensor<float, LocalLayout4Dim<1, 1, 4, 2048>, Hardware::UB>;
    using UBTileTensorFP32Dim2_25 = TileTensor<float, LocalLayout2Dim<2, 2048>, Hardware::UB>;
    using GMTileTensorBF16Dim3_29 = TileTensor<__gm__ bfloat16_t, DynLayout3Dim, Hardware::GM>;
    using UBTileTensorFP32Dim3_27 = TileTensor<float, LocalLayout3Dim<1, 4, 2048>, Hardware::UB>;

    GMTileTensorBF16Dim3_29 gmTensor_58((__gm__ bfloat16_t*)0x44000, DynLayout3Dim(Shape3Dim(1, 4, 2048), Stride3Dim(1, 4, 2048)));
    UBTileTensorBF16Dim3_28 ubTensor_56((uint64_t)UB_S225920_E242304_T, (Shape3Dim(1, 4, 2048)));
    UBTileTensorFP32Dim3_27 ubTensor_55((uint64_t)UB_S164352_E197120_T, (Shape3Dim(1, 4, 2048)));
    UBTileTensorFP32Dim4_15 ubTensor_30((uint64_t)UB_S163840_E164352_T, (Shape4Dim(1, 4, 4, 1)));
    GMTileTensorFP32Dim4_16 gmTensor_31((__gm__ float*)input0, DynLayout4Dim(Shape4Dim(1, 4, 4, 1), Stride4Dim(1, 4, 4, 1)));
    UBTileTensorBF16Dim3_17 ubTensor_32((uint64_t)UB_S213504_E217600_T, (Shape3Dim(1, 1, 2048)));
    GMTileTensorBF16Dim3_18 gmTensor_33((__gm__ bfloat16_t*)0x58000, DynLayout3Dim(Shape3Dim(1, 1, 2048), Stride3Dim(1, 1, 2048)));
    GMTileTensorFP32Dim3_20 gmTensor_35((__gm__ float*)0x65000, DynLayout3Dim(Shape3Dim(1, 4, 1), Stride3Dim(1, 4, 1)));
    UBTileTensorFP32Dim4_24 ubTensor_43((uint64_t)UB_S164352_E197120_T, (Shape4Dim(1, 1, 4, 2048)));
    UBTileTensorFP32Dim3_19 ubTensor_34((uint64_t)UB_S225792_E225920_T, (Shape3Dim(1, 4, 1)));
    UBTileTensorFP32Dim2_25 ubTensor_44((uint64_t)UB_S197120_E213504_T, (Shape2Dim(2, 2048)));
    UBTileTensorFP32Dim4_21 ubTensor_36((uint64_t)UB_S0_E32768_T, (Shape4Dim(1, 4, 1, 2048)));
    GMTileTensorFP32Dim4_22 gmTensor_37((__gm__ float*)0x670000, DynLayout4Dim(Shape4Dim(1, 4, 1, 2048), Stride4Dim(1, 4, 1, 2048)));
    UBTileTensorFP32Dim3_27 ubTensor_48((uint64_t)UB_S0_E32768_T, (Shape3Dim(1, 4, 2048)));
    UBTileTensorFP32Dim4_23 ubTensor_38((uint64_t)UB_S32768_E163840_T, (Shape4Dim(1, 4, 4, 2048)));
    UBTileTensorFP32Dim3_26 ubTensor_46((uint64_t)UB_S217600_E225792_T, (Shape3Dim(1, 1, 2048)));
    SUBKERNEL_PHASE1
    TLoad(ubTensor_30, gmTensor_31, Coord4Dim(1, 4, 4, 1));
    TLoad(ubTensor_32, gmTensor_33, Coord3Dim(1, 1, 2048));
    SUBKERNEL_PHASE2
    TLoad(ubTensor_34, gmTensor_35, Coord3Dim(1, 4, 1));
    TLoad(ubTensor_36, gmTensor_37, Coord4Dim(1, 4, 1, 2048));
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TExpand<2>(ubTensor_38, ubTensor_36);
    TMul<TileOp::BroadcastOperand::RIGHT_OPERAND>(ubTensor_38, ubTensor_38, ubTensor_30);
    TRowSumLine<2>(ubTensor_43, ubTensor_38, ubTensor_44);
    TCast<0>(ubTensor_46, ubTensor_32);
    TExpand<2>(ubTensor_48, ubTensor_46);
    TMul<TileOp::BroadcastOperand::LEFT_OPERAND>(ubTensor_48, ubTensor_34, ubTensor_48);
    TAdd(ubTensor_48, ubTensor_48, ubTensor_55);
    TCast<0>(ubTensor_56, ubTensor_48);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TStore(gmTensor_58, ubTensor_56, Coord3Dim(1, 4, 2048));
}


template <typename T, int S0, int S1, int S2, int S3>
void LaunchDeepseek_hc_post(T *out, T *src0, T *src1, ShapeInfo* param, void *stream)
{
    if constexpr ( std::is_same_v<T, aclFloat16> )
        TDeepseek_hc_host<half, S0, S1, S2, S3><<<1, nullptr, stream>>>((half*)(out), (half*)(src0), (half*)(src1), &param);
    else
        TDeepseek_hc_host<T, S0, S1, S2, S3><<<1, nullptr, stream>>>(out, src0, src1, param);
}


template void LaunchDeepseek_hc_post<float, 3, 33, 1, 8>(float *out, float *src0, float *src1, ShapeInfo *param, void *stream);
