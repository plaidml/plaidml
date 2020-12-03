// Copyright (C) 2019 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/roi_pooling.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using LayerTestsDefinitions::ROIPoolingLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32, InferenceEngine::Precision::I32};

const std::vector<std::vector<size_t>> inputShape = {
    {1, 3, 12, 12},
    //{2, 3, 24, 24},
};
const std::vector<std::vector<float>> coords = {
    {0, 0, 0, 8, 8},  //
    {0, 1, 0, 6, 5}   //
};

const auto SpecificParams = ::testing::Combine(::testing::ValuesIn(std::vector<std::string>{"max", "bilinear"}),  //
                                                         ::testing::ValuesIn(std::vector<size_t>{3, 4}),                    //
                                                         ::testing::ValuesIn(std::vector<size_t>{3, 4}),                    //
                                                         ::testing::ValuesIn(std::vector<float>{1.0, 2.0}));

INSTANTIATE_TEST_CASE_P(ROIPooling_smokeTest, ROIPoolingLayerTest,
                        ::testing::Combine(SpecificParams,                             //
                                           ::testing::ValuesIn(netPrecisions),                   //
                                           ::testing::ValuesIn(inputShape),                      //
                                           ::testing::Values(coords),                          //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        ROIPoolingLayerTest::getTestCaseName);

}  // namespace
