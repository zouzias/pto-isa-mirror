/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#include <cmath>

#include <gtest/gtest.h>
#include <pto/costmodel/arch_config.hpp>
#include <pto/costmodel/lightweight_costmodel.hpp>

TEST(TTime, compute_time_uses_frequency)
{
    ::pto::mocker::lightweight::CostModelInput input{
        .op = ::pto::mocker::lightweight::PtoOpcode::TADDS,
        .dtype = ::pto::mocker::lightweight::DType::Float,
        .rows = 32,
        .cols = 64,
    };
    ::pto::mocker::lightweight::CostModelResult result{};
    ::pto::mocker::lightweight::PredictRuntimeConfig config{
        .frequency_mhz = 925.0L,
        .bandwidth_bytes_per_us = ::pto::mocker::evaluator::GetDefaultArchConfig().bandwidth,
    };
    ASSERT_TRUE(::pto::mocker::lightweight::EstimateCycles(input, config, result));
    constexpr long double kExpectedUs = 57.0L / 925.0L;
    const long double expectedUs = kExpectedUs;
    EXPECT_NEAR(static_cast<double>(result.latency_us), static_cast<double>(expectedUs), 1e-6);
}

TEST(TTime, compute_time_scales_with_frequency_in_fit_backend)
{
    ::pto::mocker::lightweight::CostModelInput input{
        .op = ::pto::mocker::lightweight::PtoOpcode::TADDS,
        .dtype = ::pto::mocker::lightweight::DType::Float,
        .rows = 32,
        .cols = 64,
    };
    ::pto::mocker::lightweight::CostModelResult result{};
    ::pto::mocker::lightweight::PredictRuntimeConfig low_freq_config{
        .frequency_mhz = 925.0L,
        .bandwidth_bytes_per_us = ::pto::mocker::evaluator::GetDefaultArchConfig().bandwidth,
    };
    ASSERT_TRUE(::pto::mocker::lightweight::EstimateCycles(input, low_freq_config, result));
    const long double base_us = result.latency_us;

    ::pto::mocker::lightweight::PredictRuntimeConfig high_freq_config{
        .frequency_mhz = 1850.0L,
        .bandwidth_bytes_per_us = ::pto::mocker::evaluator::GetDefaultArchConfig().bandwidth,
    };
    ASSERT_TRUE(::pto::mocker::lightweight::EstimateCycles(input, high_freq_config, result));
    const long double scaled_us = result.latency_us;
    EXPECT_NEAR(static_cast<double>(scaled_us * 2.0L), static_cast<double>(base_us), 1e-6);
}

TEST(TTime, compute_time_uses_default_frequency_when_runtime_config_is_not_provided)
{
    ::pto::mocker::lightweight::CostModelInput input{
        .op = ::pto::mocker::lightweight::PtoOpcode::TADDS,
        .dtype = ::pto::mocker::lightweight::DType::Float,
        .rows = 32,
        .cols = 64,
    };
    ::pto::mocker::lightweight::CostModelResult result{};
    ASSERT_TRUE(::pto::mocker::lightweight::EstimateCycles(input, result));
    const long double expected_us = 57.0L / 1850.0L;
    EXPECT_NEAR(static_cast<double>(result.latency_us), static_cast<double>(expected_us), 1e-6);
}

TEST(TTime, fallback_to_zero_cycles_and_time_for_unsupported_input)
{
    ::pto::mocker::lightweight::CostModelInput unsupportedDType{
        .op = ::pto::mocker::lightweight::PtoOpcode::TADDS,
        .dtype = ::pto::mocker::lightweight::DType::Int8,
        .rows = 32,
        .cols = 64,
    };
    ::pto::mocker::lightweight::CostModelResult result{};
    const ::pto::mocker::lightweight::PredictRuntimeConfig config{
        .frequency_mhz = 925.0L,
        .bandwidth_bytes_per_us = ::pto::mocker::evaluator::GetDefaultArchConfig().bandwidth,
    };
    ASSERT_TRUE(::pto::mocker::lightweight::EstimateCycles(unsupportedDType, config, result));
    EXPECT_DOUBLE_EQ(result.cycles, 0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(result.latency_us), 0.0);

    ::pto::mocker::lightweight::CostModelInput unsupportedCols{
        .op = ::pto::mocker::lightweight::PtoOpcode::TADDS,
        .dtype = ::pto::mocker::lightweight::DType::Float,
        .rows = 32,
        .cols = 70000,
    };
    ASSERT_TRUE(::pto::mocker::lightweight::EstimateCycles(unsupportedCols, config, result));
    EXPECT_DOUBLE_EQ(result.cycles, 0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(result.latency_us), 0.0);

    ::pto::mocker::lightweight::CostModelInput unsupportedRows{
        .op = ::pto::mocker::lightweight::PtoOpcode::TADDS,
        .dtype = ::pto::mocker::lightweight::DType::Float,
        .rows = 0,
        .cols = 64,
    };
    ASSERT_TRUE(::pto::mocker::lightweight::EstimateCycles(unsupportedRows, config, result));
    EXPECT_DOUBLE_EQ(result.cycles, 0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(result.latency_us), 0.0);
}
