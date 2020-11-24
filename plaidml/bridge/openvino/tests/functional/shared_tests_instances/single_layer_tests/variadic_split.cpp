// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/variadic_split.hpp"

using LayerTestsDefinitions::VariadicSplitLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16
};

INSTANTIATE_TEST_CASE_P(smoke, VariadicSplitLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<int32_t>{2, 3, 1, -1}),   //
                                           ::testing::Values(0, 1, 2, 3),                          //
                                           ::testing::ValuesIn(netPrecisions),                     //
                                           ::testing::Values(std::vector<size_t>({7, 8, 11, 7})),  //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),    //
                        VariadicSplitLayerTest::getTestCaseName);

// Sum of elements numSplits = inputShapes[Axis]
const std::vector<std::vector<int32_t>> numSplits = {
    {1, 16, 5, 8},  //
    {2, 19, 5, 4},  //
    {7, 13, 2, 8},  //
    {5, 8, 12, 5},  //
    {4, 11, -1, 9}  //
};

INSTANTIATE_TEST_CASE_P(NumSplitsCheck, VariadicSplitLayerTest,
                        ::testing::Combine(::testing::ValuesIn(numSplits),                            //
                                           ::testing::Values(0, 1, 2, 3),                             //
                                           ::testing::ValuesIn(netPrecisions),                        //
                                           ::testing::Values(std::vector<size_t>({30, 30, 30, 30})),  //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),       //
                        VariadicSplitLayerTest::getTestCaseName);
}  // namespace
