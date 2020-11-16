// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/rnn_cell.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using LayerTestsDefinitions::RNNCellTest;

namespace {
std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const bool shouldDecompose = true;  // false will let all test cases fail
const std::vector<size_t> batches = {1, 4};
const std::vector<size_t> hiddenSizes = {128, 256};
const std::vector<size_t> inputSizes = {16, 64};
const std::vector<std::vector<std::string>> activations = {{"tanh"}, {"sigmoid"}, {"relu"}};
const std::vector<float> clips = {std::numeric_limits<float>::infinity(), 1.0f};

INSTANTIATE_TEST_CASE_P(RNNCell, RNNCellTest,
                        ::testing::Combine(::testing::Values(shouldDecompose),                   //
                                           ::testing::ValuesIn(batches),                         //
                                           ::testing::ValuesIn(hiddenSizes),                     //
                                           ::testing::ValuesIn(inputSizes),                      //
                                           ::testing::ValuesIn(activations),                     //
                                           ::testing::ValuesIn(clips),                           //
                                           ::testing::ValuesIn(netPrecisions),                   //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        RNNCellTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, RNNCellTest,
                        ::testing::Combine(::testing::Values(shouldDecompose),                   //
                                           ::testing::Values(3),                                 //
                                           ::testing::Values(64),                                //
                                           ::testing::Values(32),                                //
                                           ::testing::ValuesIn(activations),                     //
                                           ::testing::ValuesIn(clips),                           //
                                           ::testing::ValuesIn(netPrecisions),                   //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        RNNCellTest::getTestCaseName);
}  // namespace
