// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/cum_sum.hpp"

using LayerTestsDefinitions::CumSumLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> inputShapes = {{3, 6, 8, 7}, {2, 5, 5, 4}};

INSTANTIATE_TEST_CASE_P(smoke, CumSumLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<std::size_t>({4, 3, 3, 6})),  // input shape
                                           ::testing::Values(InferenceEngine::Precision::FP32),        // precision
                                           ::testing::Values(1),                                       // axis
                                           ::testing::Values(false),                                   // Exclusive
                                           ::testing::Values(false),                                   // Reverse
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),        // Target Device
                        CumSumLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(CumSum_Test, CumSumLayerTest,
                        ::testing::Combine(::testing::ValuesIn(inputShapes),                     // input shape
                                           ::testing::ValuesIn(netPrecisions),                   // precision
                                           ::testing::ValuesIn(std::vector<int64_t>{0, 1, 2}),   // axis
                                           ::testing::ValuesIn({false, true}),                   // Exclusive
                                           ::testing::ValuesIn({false, true}),                   // Reverse
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  // Target Device
                        CumSumLayerTest::getTestCaseName);

}  // namespace
