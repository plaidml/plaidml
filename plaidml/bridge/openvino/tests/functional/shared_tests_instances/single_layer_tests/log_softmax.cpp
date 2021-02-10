// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/log_softmax.hpp"

using namespace LayerTestsDefinitions;

namespace {
std::vector<std::vector<size_t>> input_sizes = {{3, 3}, {256, 56, 16}};
std::vector<int64_t> axes = {0, 1, -1};
std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

INSTANTIATE_TEST_CASE_P(LogSoftmax, LogSoftmaxLayerTest,
                        ::testing::Combine(                                                    //
                            ::testing::ValuesIn(netPrecisions),                                //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),        //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),        //
                            ::testing::Values(InferenceEngine::Layout::ANY),                   //
                            ::testing::Values(InferenceEngine::Layout::ANY),                   //
                            ::testing::ValuesIn(input_sizes),                                  //
                            ::testing::ValuesIn(axes),                                         //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),                //
                            ::testing::Values(std::map<std::string, std::string>{{"", ""}})),  //
                        LogSoftmaxLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, LogSoftmaxLayerTest,
                        ::testing::Combine(                                                    //
                            ::testing::Values(netPrecisions[0]),                               //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),        //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),        //
                            ::testing::Values(InferenceEngine::Layout::ANY),                   //
                            ::testing::Values(InferenceEngine::Layout::ANY),                   //
                            ::testing::Values(std::vector<size_t>{3, 3}),                      //
                            ::testing::Values(1),                                              //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),                //
                            ::testing::Values(std::map<std::string, std::string>{{"", ""}})),  //
                        LogSoftmaxLayerTest::getTestCaseName);
}  // namespace
