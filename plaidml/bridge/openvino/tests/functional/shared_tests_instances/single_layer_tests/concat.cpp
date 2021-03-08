// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/concat.hpp"

using namespace LayerTestsDefinitions;

namespace {

std::vector<size_t> axes = {0, 1, 2, 3};
std::vector<std::vector<std::vector<size_t>>> inShapes = {
    {{10, 10, 10, 10}},
    {{10, 10, 10, 10}, {10, 10, 10, 10}},
    {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
    {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
    {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

INSTANTIATE_TEST_CASE_P(NoReshape, ConcatLayerTest,
                        ::testing::Combine(                                              //
                            ::testing::ValuesIn(axes),                                   //
                            ::testing::ValuesIn(inShapes),                               //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
                        ConcatLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, ConcatLayerTest,
                        ::testing::Combine(                                                                     //
                            ::testing::Values(2),                                                               //
                            ::testing::Values(std::vector<std::vector<size_t>>({{4, 8, 4, 2}, {4, 8, 3, 2}})),  //
                            ::testing::ValuesIn(netPrecisions),                                                 //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),                         //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),                         //
                            ::testing::Values(InferenceEngine::Layout::ANY),                                    //
                            ::testing::Values(InferenceEngine::Layout::ANY),                                    //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),                                //
                        ConcatLayerTest::getTestCaseName);

}  // namespace
