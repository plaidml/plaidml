// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/variadic_split.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

// Sum of elements numSplits = inputShapes[Axis]
const std::vector<std::vector<int32_t>> numSplits = {
    {1, 16, 5, 8},   //
    {2, 19, 5, 4},   //
    {7, 13, 2, 8},   //
    {5, 8, 12, 5},   //
    {4, 11, -1, 9},  //
};

INSTANTIATE_TEST_CASE_P(NumSplitsCheck, VariadicSplitLayerTest,
                        ::testing::Combine(                                              //
                            ::testing::ValuesIn(numSplits),                              //
                            ::testing::Values(0, 1, 2, 3),                               //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(std::vector<size_t>({30, 30, 30, 30})),    //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        VariadicSplitLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, VariadicSplitLayerTest,
                        ::testing::Combine(                                              //
                            ::testing::Values(std::vector<int32_t>({4, 11, -1, 9})),     //
                            ::testing::Values(0, 1, 2, 3),                               //
                            ::testing::Values(InferenceEngine::Precision::FP32),         //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(std::vector<size_t>({30, 30, 30, 30})),    //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        VariadicSplitLayerTest::getTestCaseName);

}  // namespace
