// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/gru_cell.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using LayerTestsDefinitions::GRUCellTest;

namespace {
std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const bool shouldDecompose = true;  // false will let all test cases fail
const std::vector<size_t> batches = {1, 4};
const std::vector<size_t> hiddenSizes = {64, 128};
const std::vector<size_t> inputSizes = {16, 32};
const std::vector<std::vector<std::string>> activations = {{"sigmoid", "tanh"}, {"sigmoid", "relu"}};
const std::vector<float> clips = {std::numeric_limits<float>::infinity(), 1.0f};
const std::vector<bool> linearBeforeResets = {false, true};

INSTANTIATE_TEST_CASE_P(GRU, GRUCellTest,
                        ::testing::Combine(::testing::Values(shouldDecompose),                   //
                                           ::testing::ValuesIn(batches),                         //
                                           ::testing::ValuesIn(hiddenSizes),                     //
                                           ::testing::ValuesIn(inputSizes),                      //
                                           ::testing::ValuesIn(activations),                     //
                                           ::testing::ValuesIn(clips),                           //
                                           ::testing::ValuesIn(linearBeforeResets),              //
                                           ::testing::ValuesIn(netPrecisions),                   //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        GRUCellTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, GRUCellTest,
                        ::testing::Combine(::testing::Values(shouldDecompose),                   //
                                           ::testing::Values(3),                                 //
                                           ::testing::Values(32),                                //
                                           ::testing::Values(16),                                //
                                           ::testing::ValuesIn(activations),                     //
                                           ::testing::Values(1.0f),                              //
                                           ::testing::ValuesIn(linearBeforeResets),              //
                                           ::testing::ValuesIn(netPrecisions),                   //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        GRUCellTest::getTestCaseName);
}  // namespace
