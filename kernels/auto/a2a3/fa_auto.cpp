#include <pto/pto-inst.hpp>
using namespace pto;

const int TOTAL_M = 2048;   // Large Query Length
const int TOTAL_N = 2048;   // Large Key/Value Length
const int HEAD_DIM = 128;
const int BLOCK_M = 32;     // Tile size for Q (new)
const int BLOCK_N = 32;     // Tile size for K/V

// Helper for alignment
const int ALIGNED_COL_1 = 8; 

// BLOCK_M, TOTAL_N, HEAD_DIM, BLOCK_N
template <typename T, int kBlockM, int kN, int kD, int kBlockN>
AICORE PTO_INLINE void runFlashAttention(__gm__ T *out,
                                         __gm__ half *q,
                                         __gm__ half *k,
                                         __gm__ half *v,
                                         __gm__ uint8_t *ffts_addr,
                                         T scale) {
    using T_Val = T;      
    using T_Cube = half; 

#ifdef __DAV_C220_VEC__
    set_atomic_none();
    set_mask_norm();
    set_ffts_base_addr((uint64_t)ffts_addr);
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    using TileRowStat = Tile<TileType::Vec, T_Val, kBlockM, ALIGNED_COL_1>;
    using TileAccVec  = Tile<TileType::Vec, T_Val, kBlockM, kD>; 

    TileRowStat mTile;      
    TileRowStat lTile;      
    TileAccVec  oAccTile;   

    using TileScoreVec = Tile<TileType::Vec, T_Val, kBlockM, kBlockN>;
    using TileProbVec  = Tile<TileType::Vec, T_Val, kBlockM, kBlockN>;
    using TileProbHalf = Tile<TileType::Vec, T_Cube, kBlockM, kBlockN>;
    using TileRowTmp   = Tile<TileType::Vec, T_Val, kBlockM, ALIGNED_COL_1>;
    using TileBcastN   = Tile<TileType::Vec, T_Val, kBlockM, kBlockN>;
    using TileBcastD   = Tile<TileType::Vec, T_Val, kBlockM, kD>;

    TileScoreVec scoreVec;
    TileProbVec  probVec;
    TileProbHalf probVecH;
    TileRowTmp   tmpRowVec; 
    TileRowTmp   localMax;
    TileRowTmp   localSum;
    TileRowTmp   mPrev;     
    TileRowTmp   scaleVec;  
    TileBcastN   bcastN; 
    TileBcastD   bcastD;

    using TilePartOut = Tile<TileType::Vec, T_Val, kBlockM, kD>;
    TilePartOut partOutVec;
#endif

#ifdef __DAV_C220_CUBE__
    set_ffts_base_addr((uint64_t)ffts_addr);

    // Q Definitions (Size is kBlockM)
    using TileMatQ_H = Tile<TileType::Mat, T_Cube, kBlockM, kD, BLayout::ColMajor, kBlockM, kD, SLayout::RowMajor, 512>;
    using TileLeftQ_H = Tile<TileType::Left, T_Cube, kBlockM, kD, BLayout::RowMajor, kBlockM, kD, SLayout::RowMajor, 512>;
    
    TileMatQ_H  qMatTile;
    TileLeftQ_H qLeft;

    // K Definitions
    using TileMatK_H = Tile<TileType::Mat, T_Cube, kD, kBlockN, BLayout::ColMajor, kD, kBlockN, SLayout::RowMajor, 512>;
    using TileRightK_H = Tile<TileType::Right, T_Cube, kD, kBlockN, BLayout::RowMajor, kD, kBlockN, SLayout::ColMajor, 512>;
    
    using TileAccScore = pto::TileAcc<T_Val, kBlockM, kBlockN>;
    
    // Probs Input
    using TileMatProb_H = Tile<TileType::Mat, T_Cube, kBlockM, kBlockN, BLayout::ColMajor, kBlockM, kBlockN, SLayout::RowMajor, 512>;
    using TileLeftProb_H = Tile<TileType::Left, T_Cube, kBlockM, kBlockN, BLayout::RowMajor, kBlockM, kBlockN, SLayout::RowMajor, 512>;

    // V Definitions
    using TileMatV_H = Tile<TileType::Mat, T_Cube, kBlockN, kD, BLayout::ColMajor, kBlockN, kD, SLayout::RowMajor, 512>;
    using TileRightV_H = Tile<TileType::Right, T_Cube, kBlockN, kD, BLayout::RowMajor, kBlockN, kD, SLayout::ColMajor, 512>;
    
    using TileAccOut = pto::TileAcc<T_Val, kBlockM, kD>;

    TileMatK_H     kMatTile;
    TileRightK_H   kRight;
    TileAccScore   scoreAcc;
    TileMatProb_H  probMatTile;
    TileLeftProb_H probLeft;
    TileMatV_H     vMatTile;
    TileRightV_H   vRight;
    TileAccOut     outAcc;
#endif

    const int num_m_blocks = TOTAL_M / kBlockM;
    const int num_n_blocks = kN / kBlockN;
    #pragma clang loop unroll_count(1)
    for (int m_idx = 0; m_idx < num_m_blocks; ++m_idx) {
        __gm__ half* curr_q_ptr   = q + m_idx * kBlockM * kD;
        __gm__ float* curr_out_ptr = out + m_idx * kBlockM * kD;
        __gm__ float* scratch_ptr  = curr_out_ptr; 
        __gm__ half* probs_ptr    = curr_q_ptr;

#ifdef __DAV_C220_VEC__
        TEXPANDS(mTile, -65504.0f);
        TEXPANDS(lTile, 0.0f);    
        TEXPANDS(oAccTile, 0.0f);

        uint64_t config = 1 | (2 << 4) | (1 << 8); 
        ffts_cross_core_sync(PIPE_V, config);
#endif

#ifdef __DAV_C220_CUBE__
        wait_flag_dev(1);
        GlobalTensor<half, Shape<1, 1, 1, kBlockM, kD>, Stride<1, 1, 1, kD, 1>> qGlobal(curr_q_ptr);
        TLOAD(qMatTile, qGlobal);
#endif

        #pragma clang loop unroll_count(1)
        for (int i = 0; i < num_n_blocks; ++i) {
            
#ifdef __DAV_C220_CUBE__
            GlobalTensor<half, Shape<1, 1, 1, kD, kBlockN>, Stride<1, 1, 1, kN, 1>> kBlockGlobal(k + i * kBlockN);
            
            TLOAD(kMatTile, kBlockGlobal);
            TEXTRACT(kRight, kMatTile);
            TEXTRACT(qLeft, qMatTile);
            TMATMUL(scoreAcc, qLeft, kRight);
            
            GlobalTensor<float, Shape<1, 1, 1, kBlockM, kBlockN>, Stride<1, 1, 1, kBlockN, 1>> scoreGlobalScratch(scratch_ptr);
            TSTORE(scoreGlobalScratch, scoreAcc);

            uint64_t config = 1 | (2 << 4) | (2 << 8); 
            ffts_cross_core_sync(PIPE_FIX, config);
#endif

#ifdef __DAV_C220_VEC__
            wait_flag_dev(2);

            GlobalTensor<float, Shape<1, 1, 1, kBlockM, kBlockN>, Stride<1, 1, 1, kBlockN, 1>> vecScoreGlobal(scratch_ptr);
            TLOAD(scoreVec, vecScoreGlobal);

            TMULS(scoreVec, scoreVec, scale);
            TROWMAX(localMax, scoreVec, tmpRowVec);

            TMOV(mPrev, mTile); 
            TMAX(mTile, mTile, localMax);

            TSUB(scaleVec, mPrev, mTile);
            TEXP(scaleVec, scaleVec);

            TROWEXPAND(bcastN, mTile);
            TSUB(probVec, scoreVec, bcastN);
            TEXP(probVec, probVec);

            TROWSUM(localSum, probVec, tmpRowVec);
            TMUL(lTile, lTile, scaleVec); 
            TADD(lTile, lTile, localSum);

            TROWEXPAND(bcastD, scaleVec);
            TMUL(oAccTile, oAccTile, bcastD);

            TCVT(probVecH, probVec, RoundMode::CAST_RINT);
            GlobalTensor<half, Shape<1, 1, 1, kBlockM, kBlockN>, Stride<1, 1, 1, kBlockN, 1>> vecProbGlobal(probs_ptr);
            TSTORE(vecProbGlobal, probVecH);

            uint64_t config = 1 | (2 << 4) | (3 << 8);
            ffts_cross_core_sync(PIPE_MTE3, config);
#endif

#ifdef __DAV_C220_CUBE__
            wait_flag_dev(3);

            GlobalTensor<half, Shape<1, 1, 1, kBlockM, kBlockN>, Stride<1, 1, 1, kBlockN, 1>> cubeProbGlobal(probs_ptr);
            TLOAD(probMatTile, cubeProbGlobal);

            GlobalTensor<half, Shape<1, 1, 1, kBlockN, kD>, Stride<1, 1, 1, kD, 1>> vBlockGlobal(v + i * kBlockN * kD);
            TLOAD(vMatTile, vBlockGlobal);

            TEXTRACT(probLeft, probMatTile);
            TEXTRACT(vRight, vMatTile);
            
            TMATMUL(outAcc, probLeft, vRight);

            GlobalTensor<float, Shape<1, 1, 1, kBlockM, kD>, Stride<1, 1, 1, kD, 1>> outPartGlobal(scratch_ptr);
            TSTORE(outPartGlobal, outAcc);

            uint64_t config = 1 | (2 << 4) | (4 << 8); 
            ffts_cross_core_sync(PIPE_FIX, config);
#endif

#ifdef __DAV_C220_VEC__
            wait_flag_dev(4);
            GlobalTensor<float, Shape<1, 1, 1, kBlockM, kD>, Stride<1, 1, 1, kD, 1>> vecPartGlobal(scratch_ptr);
            TLOAD(partOutVec, vecPartGlobal);

            TADD(oAccTile, oAccTile, partOutVec);

            uint64_t config = 1 | (2 << 4) | (5 << 8);
            ffts_cross_core_sync(PIPE_MTE2, config);

            GlobalTensor<float, Shape<1, 1, 1, kBlockM, kD>, Stride<1, 1, 1, kD, 1>> finalOutGlobal(curr_out_ptr);
            TSTORE(finalOutGlobal, oAccTile);
#endif

#ifdef __DAV_C220_CUBE__
            wait_flag_dev(5);
#endif
        }

#ifdef __DAV_C220_VEC__
        if(m_idx == num_m_blocks)
            continue;

        TROWEXPAND(bcastD, lTile);
        TDIV(oAccTile, oAccTile, bcastD);

        GlobalTensor<float, Shape<1, 1, 1, kBlockM, kD>, Stride<1, 1, 1, kD, 1>> finalOutGlobal(curr_out_ptr);
        TSTORE(finalOutGlobal, oAccTile);
#endif

    }
}

extern "C" __global__ AICORE
void attention(__gm__ float *out,
               __gm__ half *q,
               __gm__ half *k,
               __gm__ half *v,
               __gm__ uint8_t *ffts_addr,
               float scale) {
    runFlashAttention<float, BLOCK_M, TOTAL_N, HEAD_DIM, BLOCK_N>(
        out, q, k, v, ffts_addr, scale
    );
}