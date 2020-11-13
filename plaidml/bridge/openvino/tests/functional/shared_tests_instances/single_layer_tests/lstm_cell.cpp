// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/lstm_cell.hpp"

#include <limits>
#include <vector>

#include "common_test_utils/test_constants.hpp"

using LayerTestsDefinitions::LSTMCellTest;

namespace {
std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const bool shouldDecompose = true;  // false will let all test cases fail
const std::vector<size_t> batches = {1, 4};
const std::vector<size_t> hiddenSizes = {128, 512};
const std::vector<size_t> inputSizes = {16, 64};
const std::vector<std::vector<std::string>> activations = {
    {"sigmoid", "tanh", "tanh"}, {"sigmoid", "sigmoid", "sigmoid"}, {"sigmoid", "relu", "relu"}};
const std::vector<float> clips = {std::numeric_limits<float>::infinity(), 1.0f};

INSTANTIATE_TEST_CASE_P(LSTMCell_default, LSTMCellTest,
                        ::testing::Combine(::testing::Values(shouldDecompose),                   //
                                           ::testing::ValuesIn(batches),                         //
                                           ::testing::ValuesIn(hiddenSizes),                     //
                                           ::testing::ValuesIn(inputSizes),                      //
                                           ::testing::ValuesIn(activations),                     //
                                           ::testing::ValuesIn(clips),                           //
                                           ::testing::ValuesIn(netPrecisions),                   //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        LSTMCellTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, LSTMCellTest,
                        ::testing::Combine(::testing::Values(shouldDecompose),                         //
                                           ::testing::Values(3),                                       //
                                           ::testing::Values(64),                                      //
                                           ::testing::Values(32),                                      //
                                           ::testing::ValuesIn(activations),                           //
                                           ::testing::Values(std::numeric_limits<float>::infinity()),  //
                                           ::testing::ValuesIn(netPrecisions),                         //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),        //
                        LSTMCellTest::getTestCaseName);
}  // namespace
