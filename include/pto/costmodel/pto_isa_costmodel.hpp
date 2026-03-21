/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef PTO_ISA_COST_MODEL_HPP
#define PTO_ISA_COST_MODEL_HPP

#include <string>
#include <unordered_map>
#include <utility>
#include <cstdint>
#include <type_traits>
#include <pto/common/pto_tile.hpp>
#include "pto/costmodel/op_struct.hpp"
#include "pto/costmodel/a2a3/TBinOp.hpp"
#include "pto/costmodel/a2a3/TBinSOp.hpp"
#include "pto/costmodel/a2a3/TUnaryOp.hpp"
#include "pto/costmodel/a2a3/TColReduceOp.hpp"
#include "pto/costmodel/a2a3/TRowReduceOp.hpp"
#include "pto/costmodel/a2a3/TRowExpand.hpp"

namespace pto {

enum class DataType
{
    FP16,
    FP32,
    INT8,
    INT16,
    UINT8,
    INT32,
    BF16
};

template <typename Element_, const int Rows_, const int Cols_, const int RowValid_ = Rows_, const int ColValid_ = Cols_>
struct TileInfo {
public:
    using DType = Element_;

    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;

    static constexpr int ValidRow = RowValid_;
    static constexpr int ValidCol = ColValid_;

    static constexpr int GetValidRow()
    {
        return ValidRow;
    }

    static constexpr int GetValidCol()
    {
        return ValidCol;
    }

    static constexpr int RowStride = Cols;
    static constexpr int ColStride = Rows;

    float cycle;
    void SetCycle(const float cycle_)
    {
        cycle = cycle_;
    }

    float GetCycle()
    {
        return cycle;
    }
};

struct InstrTypeHash {
    size_t operator()(const std::pair<std::string, DataType> &key) const
    {
        auto hash_instr = std::hash<std::string>()(key.first);
        auto hash_dtype = std::hash<int>()(static_cast<int>(key.second));
        return hash_instr ^ (hash_dtype << 1);
    }
};

class CostModel {
public:
    static CostModel &GetInstance()
    {
        static CostModel instance;
        return instance;
    }

    CostModel(const CostModel &) = delete;
    CostModel &operator=(const CostModel &) = delete;

    void InitDefaultParams()
    {
        SetParam("PIPE_V", DataType::INT16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        SetParam("PIPE_V", DataType::INT32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        SetParam("PIPE_V", DataType::FP16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        SetParam("PIPE_V", DataType::FP32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        SetParam("vector_dup", DataType::INT16, 14.0, 14.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vector_dup", DataType::INT32, 14.0, 14.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vector_dup", DataType::FP16, 14.0, 14.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vector_dup", DataType::FP32, 14.0, 14.0, 1.0, 18.0, 1.0, 0.0);

        // TADD
        SetParam("vadd", DataType::INT16, 14.0, 17.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vadd", DataType::INT32, 14.0, 17.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vadd", DataType::FP16, 14.0, 19.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vadd", DataType::FP32, 14.0, 19.0, 2.0, 18.0, 1.0, 0.0);

        // TMUL
        SetParam("vmul", DataType::INT16, 14.0, 18.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vmul", DataType::INT32, 14.0, 18.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vmul", DataType::FP16, 14.0, 20.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vmul", DataType::FP32, 14.0, 20.0, 2.0, 18.0, 1.0, 0.0);

        // TSUB
        SetParam("vsub", DataType::INT16, 14.0, 17.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vsub", DataType::INT32, 14.0, 17.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vsub", DataType::FP16, 14.0, 19.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vsub", DataType::FP32, 14.0, 19.0, 2.0, 18.0, 1.0, 0.0);

        // TEXP
        SetParam("vexp", DataType::FP16, 13.0, 28.0, 4.0, 18.0, 1.0, 0.0);
        SetParam("vexp", DataType::FP32, 13.0, 26.0, 2.0, 18.0, 1.0, 0.0);

        // TSQRT
        SetParam("vsqrt", DataType::FP16, 13.0, 29.0, 4.0, 18.0, 1.0, 0.0);
        SetParam("vsqrt", DataType::FP32, 13.0, 27.0, 2.0, 18.0, 1.0, 0.0);

        // TADDS
        SetParam("vadds", DataType::INT16, 14.0, 17.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vadds", DataType::INT32, 14.0, 17.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vadds", DataType::FP16, 14.0, 19.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vadds", DataType::FP32, 14.0, 19.0, 1.0, 18.0, 1.0, 0.0);

        // TABS
        SetParam("vabs", DataType::INT16, 13.0, 19.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vabs", DataType::INT32, 13.0, 19.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vabs", DataType::FP16, 13.0, 19.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vabs", DataType::FP32, 13.0, 19.0, 1.0, 18.0, 1.0, 0.0);

        // TMINS
        SetParam("vmins", DataType::INT16, 14.0, 17.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vmins", DataType::INT32, 14.0, 17.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vmins", DataType::FP16, 14.0, 17.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vmins", DataType::FP32, 14.0, 17.0, 1.0, 18.0, 1.0, 0.0);

        // TMULS
        SetParam("vmuls", DataType::INT16, 14.0, 18.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vmuls", DataType::INT32, 14.0, 18.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vmuls", DataType::FP16, 14.0, 20.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vmuls", DataType::FP32, 14.0, 20.0, 1.0, 18.0, 1.0, 0.0);

        // TDIVS
        SetParam("vdivs", DataType::INT16, 14.0, 18.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vdivs", DataType::INT32, 14.0, 18.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vdivs", DataType::FP16, 14.0, 20.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vdivs", DataType::FP32, 14.0, 20.0, 1.0, 18.0, 1.0, 0.0);

        // vmax
        SetParam("vmax", DataType::INT16, 14.0, 17.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vmax", DataType::INT32, 14.0, 17.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vmax", DataType::FP16, 14.0, 17.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vmax", DataType::FP32, 14.0, 17.0, 2.0, 18.0, 1.0, 0.0);

        // vmin
        SetParam("vmin", DataType::INT16, 14.0, 17.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vmin", DataType::INT32, 14.0, 17.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vmin", DataType::FP16, 14.0, 17.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vmin", DataType::FP32, 14.0, 17.0, 2.0, 18.0, 1.0, 0.0);

        // vcgmax
        SetParam("vcgmax", DataType::INT16, 13.0, 17.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vcgmax", DataType::INT32, 13.0, 19.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vcgmax", DataType::FP16, 13.0, 21.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vcgmax", DataType::FP32, 13.0, 19.0, 2.0, 18.0, 1.0, 0.0);

        // vcgmin
        SetParam("vcgmin", DataType::INT16, 13.0, 17.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vcgmin", DataType::INT32, 13.0, 19.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vcgmin", DataType::FP16, 13.0, 21.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vcgmin", DataType::FP32, 13.0, 19.0, 2.0, 18.0, 1.0, 0.0);

        // vcgadd
        SetParam("vcgadd", DataType::INT16, 13.0, 17.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vcgadd", DataType::INT32, 13.0, 19.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vcgadd", DataType::FP16, 13.0, 21.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vcgadd", DataType::FP32, 13.0, 19.0, 2.0, 18.0, 1.0, 0.0);

        // vcmax
        SetParam("vcmax", DataType::INT16, 13.0, 17.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vcmax", DataType::INT32, 13.0, 19.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vcmax", DataType::FP16, 13.0, 21.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vcmax", DataType::FP32, 13.0, 19.0, 2.0, 18.0, 1.0, 0.0);

        // vcmin
        SetParam("vcmin", DataType::INT16, 13.0, 17.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vcmin", DataType::INT32, 13.0, 19.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vcmin", DataType::FP16, 13.0, 21.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vcmin", DataType::FP32, 13.0, 19.0, 2.0, 18.0, 1.0, 0.0);

        // vcadd
        SetParam("vcadd", DataType::INT16, 13.0, 17.0, 1.0, 18.0, 1.0, 0.0);
        SetParam("vcadd", DataType::INT32, 13.0, 19.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vcadd", DataType::FP16, 13.0, 21.0, 2.0, 18.0, 1.0, 0.0);
        SetParam("vcadd", DataType::FP32, 13.0, 19.0, 2.0, 18.0, 1.0, 0.0);

        // copy_ubuf_to_ubuf (memory copy, 0-cycle placeholder)
        SetParam("copy_ubuf_to_ubuf", DataType::INT16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        SetParam("copy_ubuf_to_ubuf", DataType::INT32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        SetParam("copy_ubuf_to_ubuf", DataType::FP16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        SetParam("copy_ubuf_to_ubuf", DataType::FP32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

        // mask (mask set instruction, 0-cycle placeholder)
        SetParam("mask", DataType::INT16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        SetParam("mask", DataType::INT32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        SetParam("mask", DataType::FP16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        SetParam("mask", DataType::FP32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    // TBinOp
    template <typename Op, typename TileDataDst, typename TileDataSrc0, typename TileDataSrc1>
    void BinOpPredictCycle(const std::string &instr_name, TileDataDst &dst, TileDataSrc0 &src0, TileDataSrc1 &src1)
    {
        using T = typename TileDataDst::DType;
        std::vector<CostModelStats> stats = runBinaryOp<Op>(dst, src0, src1);
        float totalCycles = PredictCycle<T>(stats);
        dst.SetCycle(totalCycles);
        std::cout << "Instr: " << instr_name << " Cycles: " << totalCycles << std::endl;
    }

    // TBinSOp
    template <typename Op, typename TileDataDst, typename TileDataSrc>
    void BinSOpPredictCycle(const std::string &instr_name, TileDataDst &dst, TileDataSrc &src)
    {
        using T = typename TileDataDst::DType;
        std::vector<CostModelStats> stats = runBinaryScalarOp<Op>(dst, src);
        float totalCycles = PredictCycle<T>(stats);
        dst.SetCycle(totalCycles);
        std::cout << "Instr: " << instr_name << " Cycles: " << totalCycles << std::endl;
    }

    // TUnaryOp
    template <typename Op, typename TileDataDst, typename TileDataSrc>
    void UnaryOpPredictCycle(const std::string &instr_name, TileDataDst &dst, TileDataSrc &src)
    {
        using T = typename TileDataDst::DType;
        std::vector<CostModelStats> stats = runUnaryOp<Op>(dst, src);
        float totalCycles = PredictCycle<T>(stats);
        dst.SetCycle(totalCycles);
        std::cout << "Instr: " << instr_name << " Cycles: " << totalCycles << std::endl;
    }

    // TColMax / TColMin
    template <typename TileDataOut, typename TileDataIn>
    void ColReducePredictCycle(const std::string &instr_name, TileDataOut &dst, TileDataIn &src)
    {
        using T = typename TileDataIn::DType;
        CostModelStats stats = runColReduceOp(dst, src);
        float resultCycles = PredictCycle<T>(stats);
        dst.SetCycle(resultCycles);
    }

    // TRowMax / TRowMin
    template <typename TileDataOut, typename TileDataIn>
    void RowReducePredictCycle(const std::string &instr_name, TileDataOut &dst, TileDataIn &src)
    {
        using T = typename TileDataIn::DType;
        CostModelStats stats = runRowReduceOp(dst, src);
        float resultCycles = PredictCycle<T>(stats);
        dst.SetCycle(resultCycles);
    }

    // TRowReduceOpPredict
    template <typename Op, typename TileDataOut, typename TileDataIn, typename TileDataTmp>
    void RowReduceOpPredictCycle(const std::string& instr_name, TileDataOut &dst, TileDataIn &src, TileDataTmp &tmp)
    {
        using T = typename TileDataIn::DType;
        std::vector<CostModelStats> stats = runRowReduceOps<T, Op, TileDataOut, TileDataIn, TileDataTmp>(instr_name, dst, src, tmp);
        float totalCycles = PredictCycle<T>(stats);
        dst.SetCycle(totalCycles);
        std::cout << "Instr: " << instr_name << " Cycles: " << totalCycles << std::endl;
    }

    // TColMax / TColMin
    template <typename Op, typename TileDataOut, typename TileDataIn>
    void ColReduceOpPredictCycle(const std::string& instr_name, TileDataOut &dst, TileDataIn &src)
    {
        using T = typename TileDataIn::DType;
        std::vector<CostModelStats> stats = runColReduceOps<T, Op, TileDataOut, TileDataIn>(dst, src);
        float totalCycles = PredictCycle<T>(stats);
        dst.SetCycle(totalCycles);
        std::cout << "Instr: " << instr_name << " Cycles: " << totalCycles << std::endl;
    }

    // TColSum
    template <typename TileDataDst, typename TileDataSrc, typename TileDataTmp>
    void ColSumOpPredictCycle(const std::string& instr_name, TileDataDst &dst, TileDataSrc &src, TileDataTmp &tmp,
                              bool IsBinary)
    {
        using T = typename TileDataSrc::DType;
        std::vector<CostModelStats> stats = runColSumOp<T, TileDataDst, TileDataSrc, TileDataTmp>(dst, src, tmp,
                                                                                                  IsBinary);
        float totalCycles = PredictCycle<T>(stats);
        dst.SetCycle(totalCycles);
        std::cout << "Instr: " << instr_name << " Cycles: " << totalCycles << std::endl;
    }

    // TRowExpand
    template <typename TileDataDst, typename TileDataSrc>
    void RowExpandPredictCycle(const std::string &instr_name, TileDataDst &dst, TileDataSrc &src)
    {
        using T = typename TileDataDst::DType;
        std::vector<CostModelStats> stats = runRowExpandOp(dst, src);
        float totalCycles = PredictCycle<T>(stats);
        dst.SetCycle(totalCycles);
        std::cout << "Instr: " << instr_name << " Cycles: " << totalCycles << std::endl;
    }

    template <typename T>
    float PredictCycle(const std::vector<CostModelStats> stats)
    {

        float total_cycles = 0.0f;
        bool pipe = true;
        bool first = true;
        for (auto &stat : stats) {
            std::string instr_name = stat.cceInstName;
            DataType dtype = GetDataTypeEnum<T>();
            auto key = std::make_pair(instr_name, dtype);

            if (!CheckParamExist(key)) {
                fprintf(stderr, "[CostModel] Error: <%s> <%d> \n", instr_name.c_str(), static_cast<int>(dtype));
                return 0.0f;
            }

            const CostModelParams &params = params_map_.at(key);
            float masked_repeat_penalty = params.per_repeat_cycles * (params.mask_effect - 1.0f);
            int effective_repeats = stat.total_repeats > 0 ? stat.total_repeats - 1 : 0;

            if (first){
                total_cycles += params.startup_cycles;
            }

            if (pipe){
                total_cycles += params.completion_cycles;
            }

            total_cycles += effective_repeats * params.per_repeat_cycles + stat.masked_repeats * masked_repeat_penalty + params.bank_conflict_cycles;

            fprintf(stdout, "[CostModel] Instr: %s Cycles: %f\n", instr_name.c_str(), total_cycles);
        }

        return total_cycles;
    }

private:
    CostModel()
    {
        InitDefaultParams();
    }

    void SetParam(const std::string &instr_name, DataType dtype, double head, double complete, double computing,
                  double interval, double mask, double bank_conflict)
    {
        auto key = std::make_pair(instr_name, dtype);
        params_map_[key] = CostModelParams{
            static_cast<float>(head),     static_cast<float>(complete), static_cast<float>(computing),
            static_cast<float>(interval), static_cast<float>(mask),     static_cast<float>(bank_conflict),
        };
    }

    bool CheckParamExist(const std::pair<std::string, DataType> &key)
    {
        return params_map_.count(key) > 0;
    }

    template <typename T>
    DataType GetDataTypeEnum()
    {
        if constexpr (std::is_same_v<T, __bf16>) {
            return DataType::BF16;
        } else if constexpr (std::is_same_v<T, half>) {
            return DataType::FP16;
        } else if constexpr (std::is_same_v<T, float>) {
            return DataType::FP32;
        } else if constexpr (std::is_same_v<T, int16_t>) {
            return DataType::INT16;
        } else if constexpr (std::is_same_v<T, int32_t>) {
            return DataType::INT32;
        } else if constexpr (std::is_same_v<T, int8_t>) {
            return DataType::INT8;
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            return DataType::UINT8;
        } else {
            fprintf(stderr, "[CostModel] Warning: unknow data type, use FP16 instead.\n");
            return DataType::FP16;
        }
    }

    std::unordered_map<std::pair<std::string, DataType>, CostModelParams, InstrTypeHash> params_map_;
};

} // namespace pto

#endif
