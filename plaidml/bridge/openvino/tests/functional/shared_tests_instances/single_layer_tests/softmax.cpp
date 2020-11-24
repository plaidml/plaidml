// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/softmax.hpp"

using LayerTestsDefinitions::SoftMaxLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Layout> inputLayouts2D = {
    InferenceEngine::Layout::NC,
};

const std::vector<InferenceEngine::SizeVector> inputShapes2D = {
    InferenceEngine::SizeVector{1, 100},
    InferenceEngine::SizeVector{100, 1},
    InferenceEngine::SizeVector{10, 10},
};

const std::vector<size_t> axis2D = {0, 1};

const auto params2D = testing::Combine(testing::ValuesIn(netPrecisions),                      //
                                       testing::ValuesIn(inputLayouts2D),                     //
                                       testing::ValuesIn(inputShapes2D),                      //
                                       testing::ValuesIn(axis2D),                             //
                                       testing::Values(CommonTestUtils::DEVICE_PLAIDML),      //
                                       testing::Values(std::map<std::string, std::string>())  //
);

INSTANTIATE_TEST_CASE_P(smokeSoftMax2D, SoftMaxLayerTest, params2D, SoftMaxLayerTest::getTestCaseName);

const std::vector<InferenceEngine::SizeVector> inputShapes4D = {
    InferenceEngine::SizeVector{1, 100, 1, 1},
    InferenceEngine::SizeVector{1, 3, 4, 3},
    InferenceEngine::SizeVector{2, 3, 4, 5},
};

const std::vector<size_t> axis4D = {0, 1, 2, 3};

const auto params4D = testing::Combine(testing::ValuesIn(netPrecisions),                      //
                                       testing::Values(InferenceEngine::Layout::NCHW),        //
                                       testing::ValuesIn(inputShapes4D),                      //
                                       testing::ValuesIn(axis4D),                             //
                                       testing::Values(CommonTestUtils::DEVICE_PLAIDML),      //
                                       testing::Values(std::map<std::string, std::string>())  //
);

const auto params_smoke4D = testing::Combine(testing::ValuesIn(netPrecisions),                          //
                                             testing::Values(InferenceEngine::Layout::NCHW),            //
                                             testing::Values(InferenceEngine::SizeVector{3, 4, 5, 2}),  //
                                             testing::Values(2),                                        //
                                             testing::Values(CommonTestUtils::DEVICE_PLAIDML),          //
                                             testing::Values(std::map<std::string, std::string>())      //
);

INSTANTIATE_TEST_CASE_P(SoftMax4D, SoftMaxLayerTest, params4D, SoftMaxLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smokeSoftMax4D, SoftMaxLayerTest, params_smoke4D, SoftMaxLayerTest::getTestCaseName);

}  // namespace
