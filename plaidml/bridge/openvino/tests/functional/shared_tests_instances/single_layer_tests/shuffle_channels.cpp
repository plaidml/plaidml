// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/shuffle_channels.hpp"
using LayerTestsDefinitions::ShuffleChannelsLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> inputShapes = {
    // tensor order 'NCHW'
    {5, 12, 10, 10},
    {1, 18, 16, 16}};

const std::vector<shuffleChannelsSpecificParams> shuffleChannelOneParams = {{1, 2},  //
                                                                            {1, 6},  //
                                                                            {-3, 3}};

INSTANTIATE_TEST_CASE_P(smokeChannelOne, ShuffleChannelsLayerTest,
                        ::testing::Combine(::testing::ValuesIn(shuffleChannelOneParams),         //
                                           ::testing::ValuesIn(netPrecisions),                   //
                                           ::testing::ValuesIn(inputShapes),                     //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        ShuffleChannelsLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> diffOrderInputShape = {
    // tensor order 'NHWC'
    {5, 10, 10, 6},
    {1, 16, 16, 12}};

const std::vector<shuffleChannelsSpecificParams> shuffleChannelThirdParams = {{3, 2},  //
                                                                              {3, 6},  //
                                                                              {-1, 3}};

INSTANTIATE_TEST_CASE_P(smokeChannelThree, ShuffleChannelsLayerTest,
                        ::testing::Combine(::testing::ValuesIn(shuffleChannelThirdParams),       //
                                           ::testing::ValuesIn(netPrecisions),                   //
                                           ::testing::ValuesIn(diffOrderInputShape),             //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        ShuffleChannelsLayerTest::getTestCaseName);
}  // namespace
