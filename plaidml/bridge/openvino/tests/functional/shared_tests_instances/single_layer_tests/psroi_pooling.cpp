// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/psroi_pooling.hpp"

using LayerTestsDefinitions::PSROIPoolingLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16  // TODO: Not yet working
};
const std::vector<std::vector<size_t>> coords = {{1, 5}, {2, 5}};
const std::vector<float> spatial_scale = {1.0, 0.5};
const std::vector<std::string> mode = {"average", "bilinear"};

INSTANTIATE_TEST_CASE_P(smoke_PSROIPooling, PSROIPoolingLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<size_t>({1, 8, 10, 10})),  //
                                           ::testing::ValuesIn(coords),                             //
                                           ::testing::Values(2),                                    //
                                           ::testing::Values(2),                                    //
                                           ::testing::ValuesIn(spatial_scale),                      //
                                           ::testing::Values(2),                                    //
                                           ::testing::Values(2),                                    //
                                           ::testing::ValuesIn(mode),                               //
                                           ::testing::ValuesIn(netPrecisions),                      //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),     //
                        PSROIPoolingLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(PSROIPooling, PSROIPoolingLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<size_t>({1, 18, 10, 10})),  //
                                           ::testing::ValuesIn(coords),                              //
                                           ::testing::Values(2),                                     //
                                           ::testing::Values(3),                                     //
                                           ::testing::ValuesIn(spatial_scale),                       //
                                           ::testing::Values(3),                                     //
                                           ::testing::Values(3),                                     //
                                           ::testing::ValuesIn(mode),                                //
                                           ::testing::ValuesIn(netPrecisions),                       //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),      //
                        PSROIPoolingLayerTest::getTestCaseName);

}  // namespace
