// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/one_hot.hpp"

#include <limits>
#include <vector>

#include "common_test_utils/test_constants.hpp"

using LayerTestsDefinitions::OnehotLayerTest;

namespace {
std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::I32,
};

const std::vector<int64_t> axis = {-1, 1, 0};
const std::vector<size_t> depthes = {2, 5};
const std::vector<float> onValues = {1, 5};
const std::vector<float> offValues = {0, 10};
const std::vector<std::vector<size_t>> inputShapes = {{3}, {2, 3}};

INSTANTIATE_TEST_CASE_P(OneHot, OnehotLayerTest,
                        ::testing::Combine(::testing::ValuesIn(axis),                            //
                                           ::testing::ValuesIn(depthes),                         //
                                           ::testing::ValuesIn(onValues),                        //
                                           ::testing::ValuesIn(offValues),                       //
                                           ::testing::ValuesIn(inputShapes),                     //
                                           ::testing::ValuesIn(netPrecisions),                   //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        OnehotLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, OnehotLayerTest,
                        ::testing::Combine(::testing::Values(axis[0]),                           //
                                           ::testing::Values(depthes[0]),                        //
                                           ::testing::Values(offValues[0]),                      //
                                           ::testing::Values(onValues[0]),                       //
                                           ::testing::Values(inputShapes[0]),                    //
                                           ::testing::ValuesIn(netPrecisions),                   //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        OnehotLayerTest::getTestCaseName);
}  // namespace
