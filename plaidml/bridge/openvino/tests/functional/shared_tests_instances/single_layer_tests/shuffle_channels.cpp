// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/shuffle_channels.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const std::vector<int> axes = {0, 1, 2, 3};
const std::vector<int> negativeAxes = {-4, -3, -2, -1};
const std::vector<int> groups = {1, 2, 3};

const auto shuffleChannelsParams4D = ::testing::Combine(  //
    ::testing::ValuesIn(axes),                            //
    ::testing::ValuesIn(groups)                           //
);

const auto shuffleChannelsParamsNegativeAxis4D = ::testing::Combine(  //
    ::testing::ValuesIn(negativeAxes),                                //
    ::testing::ValuesIn(groups)                                       //
);

INSTANTIATE_TEST_CASE_P(smoke_ShuffleChannels4D, ShuffleChannelsLayerTest,
                        ::testing::Combine(                                              //
                            shuffleChannelsParams4D,                                     //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(std::vector<size_t>({6, 6, 6, 6})),        //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
                        ShuffleChannelsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ShuffleChannelsNegativeAxis4D, ShuffleChannelsLayerTest,
                        ::testing::Combine(                                              //
                            shuffleChannelsParamsNegativeAxis4D,                         //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(std::vector<size_t>({6, 6, 6, 6})),        //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
                        ShuffleChannelsLayerTest::getTestCaseName);

}  // namespace
