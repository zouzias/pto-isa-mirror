/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file moe_token_unpermute.h
 * \brief
 */

#ifndef MOE_TOKEN_UNPERMUTE
#define MOE_TOKEN_UNPERMUTE

#include "kernel_operator.h"
#include "moe_token_unpermute_tiling.h"
#include "../moe_init_routing_quant_v2/moe_v2_pto_sort.h"
using namespace AscendC;
using pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoFillVector;
using pto_ext::dispatch_ffn_combine_v3::pto_bridge::PtoGetValue;

template <typename T1, typename T2, typename T3, bool PROBS>
class KernelMoeTokenUnpermute {
public:
    __aicore__ inline KernelMoeTokenUnpermute()
    {}

    __aicore__ inline void Init(GM_ADDR permuted_tokens, GM_ADDR sorted_indices, GM_ADDR probs,
                                GM_ADDR unpermuted_tokens, const MoeTokenUnpermuteTilingData *__restrict tiling_data);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void CalMultiOutToken(const int64_t out_offset, const int64_t out_tokens_number);
    __aicore__ inline void CalSingleOutToken(const int64_t start_token, const int64_t out_token_idx);
    __aicore__ inline void CalPartOutToken(const int64_t start_token, const int64_t h_index, const int64_t h_length,
                                           const int64_t out_token_index);
    __aicore__ inline void LoadTokenSlice(uint64_t tokensUb, const int64_t offset, const int64_t h_length);
    __aicore__ inline void StoreTokenSlice(const int64_t offset, uint64_t tokensUb, const int64_t h_length);
    __aicore__ inline void CopyTokenIn(const T2 in_token_index, const int64_t h_index, const int64_t h_length);
    __aicore__ inline void CalFirstToken(const float prob_value, const int64_t h_length);
    __aicore__ inline void CalToken(const float prob_value, const int64_t h_length);
    __aicore__ inline void CopyOut(const int64_t out_token_index, const int64_t h_index, const int64_t h_length);

    uint64_t tokensUb;
    uint64_t indicesUb;
    uint64_t probsInputUb;
    uint64_t tokenTensor0Ub;
    uint64_t tokenTensor1Ub;
    uint64_t probsUb;
    uint64_t outUb;
    __gm__ T1 *tokensGM;
    __gm__ T1 *outGM;
    __gm__ T2 *indicesGM;
    __gm__ T3 *probsGM;
    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static uint32_t ALIGN_512 = 512;

    int64_t hidden_size;
    int64_t top_k;
    int64_t num_out_tokens;
    int64_t hidden_splited_length;
    int64_t hidden_splited_num;
    int64_t hidden_splited_remain;
    int64_t tokens_core_length;
    int64_t tokens_core_remain;
    int64_t tokens_splited_length;
    int64_t tokens_splited_num;
    int64_t tokens_splited_remain;
    int32_t blockIdx;
    int32_t blockNum;
};

template <typename T1, typename T2, typename T3, bool PROBS>
__aicore__ inline void KernelMoeTokenUnpermute<T1, T2, T3, PROBS>::Init(
    GM_ADDR permuted_tokens, GM_ADDR sorted_indices, GM_ADDR probs, GM_ADDR unpermuted_tokens,
    const MoeTokenUnpermuteTilingData *__restrict tiling_data)
{
    this->blockIdx = get_block_idx() + get_subblockid() * get_block_num();
    this->blockNum = get_block_num() * get_subblockdim();
    if (blockIdx >= blockNum) {
        return;
    }
    ASSERT(blockNum != 0 && "block dim can not be zero!");
    // row_input
    this->hidden_size = tiling_data->hidden_size;
    this->top_k = tiling_data->top_k;
    this->num_out_tokens = tiling_data->num_out_tokens;
    // hidden_tiling
    this->hidden_splited_length = tiling_data->hidden_splited_length;
    this->hidden_splited_num = tiling_data->hidden_splited_num;
    this->hidden_splited_remain = tiling_data->hidden_splited_remain;
    // token_tiling
    this->tokens_core_length = tiling_data->tokens_core_length;
    this->tokens_core_remain = tiling_data->tokens_core_remain;
    this->tokens_splited_length = tiling_data->tokens_splited_length;
    this->tokens_splited_num = tiling_data->tokens_splited_num;
    this->tokens_splited_remain = tiling_data->tokens_splited_remain;

    // Handle the tail block for token_by_core
    if (this->tokens_core_remain > 0 && blockIdx < this->tokens_core_remain) {
        this->tokens_core_length += 1;
        this->tokens_splited_remain += 1;
    }

    int64_t hidden_splited_length_align512 = (this->hidden_splited_length + ALIGN_512 - 1) & ~(ALIGN_512 - 1);

    int64_t block_length = this->tokens_core_length * this->top_k;
    int64_t block_splited_length = this->tokens_splited_length * this->top_k;

    int64_t block_offset;
    if (this->tokens_core_remain > 0) {
        if (blockIdx < this->tokens_core_remain) {
            block_offset = block_length * blockIdx;
        } else {
            block_offset = (block_length + this->top_k) * this->tokens_core_remain +
                           block_length * (blockIdx - this->tokens_core_remain);
        }
    } else {
        block_offset = block_length * blockIdx;
    }

    this->tokensGM = (__gm__ T1 *)permuted_tokens;
    this->indicesGM = (__gm__ T2 *)sorted_indices + block_offset;

    int64_t out_block_offset;
    if (this->tokens_core_remain > 0) {
        if (blockIdx < this->tokens_core_remain) {
            out_block_offset = this->tokens_core_length * blockIdx * hidden_size;
        } else {
            out_block_offset = (this->tokens_core_length + 1) * this->tokens_core_remain +
                               this->tokens_core_length * (blockIdx - this->tokens_core_remain);
            out_block_offset *= this->hidden_size;
        }
    } else {
        out_block_offset = this->tokens_core_length * blockIdx * hidden_size;
    }

    this->outGM = (__gm__ T1 *)unpermuted_tokens + out_block_offset;

    this->indicesUb = 0;
    this->probsInputUb = this->indicesUb + AlignBytes(block_splited_length, sizeof(T2));
    this->probsUb = this->probsInputUb + AlignBytes(block_splited_length, sizeof(T3));
    this->tokensUb = this->probsUb + AlignBytes(block_splited_length, sizeof(float));
    this->tokenTensor0Ub = this->tokensUb + AlignBytes(hidden_splited_length_align512, sizeof(T1));
    this->tokenTensor1Ub = this->tokenTensor0Ub + AlignBytes(hidden_splited_length_align512, sizeof(float)) + 256;
    this->outUb = this->tokenTensor1Ub + AlignBytes(hidden_splited_length_align512, sizeof(float));

    if constexpr (PROBS) {
        this->probsGM = (__gm__ T3 *)probs + block_offset;
    }
};

template <typename T1, typename T2, typename T3, bool PROBS>
__aicore__ inline void KernelMoeTokenUnpermute<T1, T2, T3, PROBS>::Process()
{
    if (blockIdx >= blockNum) {
        return;
    }
    for (int64_t i = 0; i < this->tokens_splited_num; ++i) {
        CalMultiOutToken(i * this->tokens_splited_length, this->tokens_splited_length);
    }
    // Handle the tail block when tokens_num is not evenly divisible by core count
    if (this->tokens_splited_remain > 0) {
        CalMultiOutToken(this->tokens_splited_num * this->tokens_splited_length, this->tokens_splited_remain);
    }
}

template <typename T1, typename T2, typename T3, bool PROBS>
__aicore__ inline void KernelMoeTokenUnpermute<T1, T2, T3, PROBS>::CalMultiOutToken(const int64_t out_offset,
                                                                                    const int64_t out_tokens_number)
{
    int64_t in_offset = out_offset * this->top_k;
    MoeInitRoutingQuantV2::pto_detail::PtoLoadVector<T2>(this->indicesUb, this->indicesGM + in_offset,
                                                         out_tokens_number * this->top_k);

    if constexpr (PROBS) {
        MoeInitRoutingQuantV2::pto_detail::PtoLoadVector<T3>(this->probsInputUb, this->probsGM + in_offset,
                                                             out_tokens_number * this->top_k);
        pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        if constexpr (!IsSameType<T3, float>::value) {
            MoeInitRoutingQuantV2::pto_detail::PtoCastVector<float, T3>(
                this->probsUb, this->probsInputUb, out_tokens_number * this->top_k, pto::RoundMode::CAST_NONE);
            MoeInitRoutingQuantV2::pto_detail::PtoPipeBarrier<PIPE_V>();
        } else {
            MoeInitRoutingQuantV2::pto_detail::PtoMoveVector<float>(this->probsUb, this->probsInputUb,
                                                                    out_tokens_number * this->top_k);
            MoeInitRoutingQuantV2::pto_detail::PtoPipeBarrier<PIPE_V>();
        }
    } else {
        pto_detail::PtoSetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    }

    for (int64_t out_token_idx = 0; out_token_idx < out_tokens_number; ++out_token_idx) {
        CalSingleOutToken(out_token_idx * this->top_k, out_offset + out_token_idx);
    }
}

template <typename T1, typename T2, typename T3, bool PROBS>
__aicore__ inline void KernelMoeTokenUnpermute<T1, T2, T3, PROBS>::CalSingleOutToken(const int64_t start_token,
                                                                                     const int64_t out_token_idx)
{
    for (int64_t h_index = 0; h_index < this->hidden_splited_num; ++h_index) {
        CalPartOutToken(start_token, h_index, this->hidden_splited_length, out_token_idx);
    }
    // Handle the tail block when a full hidden_size does not fit in one pass
    if (this->hidden_splited_remain > 0) {
        CalPartOutToken(start_token, this->hidden_splited_num, this->hidden_splited_remain, out_token_idx);
    }
}

template <typename T1, typename T2, typename T3, bool PROBS>
__aicore__ inline void KernelMoeTokenUnpermute<T1, T2, T3, PROBS>::CalPartOutToken(const int64_t start_token,
                                                                                   const int64_t h_index,
                                                                                   const int64_t h_length,
                                                                                   const int64_t out_token_index)
{
    int64_t end_token = start_token + this->top_k;
    T2 cal_token_idx = PtoGetValue<T2>(this->indicesUb, start_token);

    // Handle the first token
    if (cal_token_idx < this->num_out_tokens) {
        float probsValue = 0;
        if constexpr (PROBS) {
            probsValue = PtoGetValue<float>(this->probsUb, start_token);
        }

        CopyTokenIn(cal_token_idx, h_index, h_length);
        MoeInitRoutingQuantV2::pto_detail::PtoPipeBarrier<PIPE_V>();
        CalFirstToken(probsValue, h_length);
    } else {
        MoeInitRoutingQuantV2::pto_detail::PtoPipeBarrier<PIPE_V>();
        PtoFillVector<float>(this->tokenTensor0Ub, static_cast<float>(0), h_length);
    }

    // Handle the remaining tokens
    for (int64_t token_index = start_token + 1; token_index < end_token; ++token_index) {
        cal_token_idx = PtoGetValue<T2>(this->indicesUb, token_index);
        if (cal_token_idx < this->num_out_tokens) {
            float probsValue = 0;
            if constexpr (PROBS) {
                probsValue = PtoGetValue<float>(this->probsUb, token_index);
            }

            CopyTokenIn(cal_token_idx, h_index, h_length);
            MoeInitRoutingQuantV2::pto_detail::PtoPipeBarrier<PIPE_V>();
            CalToken(probsValue, h_length);
        }
    }

    // Write out the computed result
    CopyOut(out_token_index, h_index, h_length);
}

template <typename T1, typename T2, typename T3, bool PROBS>
__aicore__ inline void KernelMoeTokenUnpermute<T1, T2, T3, PROBS>::LoadTokenSlice(uint64_t tokensUb,
                                                                                  const int64_t offset,
                                                                                  const int64_t h_length)
{
    MoeInitRoutingQuantV2::pto_detail::PtoLoadVector(tokensUb, this->tokensGM + offset, h_length);
}

template <typename T1, typename T2, typename T3, bool PROBS>
__aicore__ inline void KernelMoeTokenUnpermute<T1, T2, T3, PROBS>::StoreTokenSlice(const int64_t offset,
                                                                                   uint64_t tokensUb,
                                                                                   const int64_t h_length)
{
    MoeInitRoutingQuantV2::pto_detail::PtoStoreVector(this->outGM + offset, tokensUb, h_length);
}

template <typename T1, typename T2, typename T3, bool PROBS>
__aicore__ inline void KernelMoeTokenUnpermute<T1, T2, T3, PROBS>::CopyTokenIn(const T2 in_token_index,
                                                                               const int64_t h_index,
                                                                               const int64_t h_length)
{
    int64_t offset = in_token_index * this->hidden_size + h_index * this->hidden_splited_length;
    LoadTokenSlice(this->tokensUb, offset, h_length);
    pto_detail::PtoSetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
}

template <typename T1, typename T2, typename T3, bool PROBS>
__aicore__ inline void KernelMoeTokenUnpermute<T1, T2, T3, PROBS>::CalFirstToken(const float prob_value,
                                                                                 const int64_t h_length)
{
    if constexpr (!IsSameType<T1, float>::value) {
        MoeInitRoutingQuantV2::pto_detail::PtoCastVector<float, T1>(this->tokenTensor0Ub, this->tokensUb, h_length,
                                                                    pto::RoundMode::CAST_NONE);
    } else {
        MoeInitRoutingQuantV2::pto_detail::PtoMoveVector<T1>(this->tokenTensor0Ub, this->tokensUb, h_length);
    }

    if constexpr (PROBS) {
        MoeInitRoutingQuantV2::pto_detail::PtoPipeBarrier<PIPE_V>();
        MoeInitRoutingQuantV2::pto_detail::PtoMulVector<float>(this->tokenTensor0Ub, this->tokenTensor0Ub, h_length,
                                                               prob_value);
    }
}

template <typename T1, typename T2, typename T3, bool PROBS>
__aicore__ inline void KernelMoeTokenUnpermute<T1, T2, T3, PROBS>::CalToken(const float prob_value,
                                                                            const int64_t h_length)
{
    if constexpr (!IsSameType<T1, float>::value) {
        MoeInitRoutingQuantV2::pto_detail::PtoCastVector<float, T1>(this->tokenTensor1Ub, this->tokensUb, h_length,
                                                                    pto::RoundMode::CAST_NONE);
        if constexpr (PROBS) {
            MoeInitRoutingQuantV2::pto_detail::PtoPipeBarrier<PIPE_V>();
            MoeInitRoutingQuantV2::pto_detail::PtoMulVector<float>(this->tokenTensor1Ub, this->tokenTensor1Ub, h_length,
                                                                   prob_value);
        }
        MoeInitRoutingQuantV2::pto_detail::PtoPipeBarrier<PIPE_V>();
        MoeInitRoutingQuantV2::pto_detail::PtoAddVector<float>(this->tokenTensor0Ub, this->tokenTensor0Ub,
                                                               this->tokenTensor1Ub, h_length);
    } else {
        if constexpr (PROBS) {
            MoeInitRoutingQuantV2::pto_detail::PtoMulVector<T1>(this->tokensUb, this->tokensUb, h_length, prob_value);
            MoeInitRoutingQuantV2::pto_detail::PtoPipeBarrier<PIPE_V>();
        }
        MoeInitRoutingQuantV2::pto_detail::PtoAddVector<T1>(this->tokenTensor0Ub, this->tokenTensor0Ub, this->tokensUb,
                                                            h_length);
    }
}

template <typename T1, typename T2, typename T3, bool PROBS>
__aicore__ inline void KernelMoeTokenUnpermute<T1, T2, T3, PROBS>::CopyOut(const int64_t out_token_index,
                                                                           const int64_t h_index,
                                                                           const int64_t h_length)
{
    uint64_t outUb = this->tokenTensor0Ub;
    if constexpr (!IsSameType<T1, float>::value) {
        MoeInitRoutingQuantV2::pto_detail::PtoPipeBarrier<PIPE_V>();
        MoeInitRoutingQuantV2::pto_detail::PtoCastVector<T1, float>(this->outUb, this->tokenTensor0Ub, h_length,
                                                                    pto::RoundMode::CAST_RINT);
        outUb = this->outUb;
    }

    MoeInitRoutingQuantV2::pto_detail::PtoSetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    int64_t offset = out_token_index * this->hidden_size + h_index * this->hidden_splited_length;
    StoreTokenSlice(offset, outUb, h_length);
    MoeInitRoutingQuantV2::pto_detail::PtoSetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
}
#endif // MOE_TOKEN_UNPERMUTE