// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/reverse_sequence.hpp"

#include <vector>

using LayerTestsDefinitions::ReverseSequenceLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::FP32,
};

INSTANTIATE_TEST_CASE_P(ReverseSequenceLayerTest_smoke, ReverseSequenceLayerTest,
                        ::testing::Combine(::testing::Values(0),                                   //
                                           ::testing::Values(1),                                   //
                                           ::testing::Values(std::vector<size_t>{2, 10, 10, 10}),  //
                                           ::testing::Values(std::vector<size_t>{2, 1}),           //
                                           ::testing::Values(0),                                   //
                                           ::testing::ValuesIn(netPrecisions),                     //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),    //
                        ReverseSequenceLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> seq_length = {
    {1, 2, 3, 1},  //
    {2, 2, 2, 2}   //
};

INSTANTIATE_TEST_CASE_P(ReverseSequenceLayer_test, ReverseSequenceLayerTest,
                        ::testing::Combine(::testing::ValuesIn(std::vector<int64_t>{0, 1}),      //
                                           ::testing::ValuesIn(std::vector<int64_t>{2, 3}),      //
                                           ::testing::Values(std::vector<size_t>{4, 4, 4, 4}),   //
                                           ::testing::ValuesIn(seq_length),                      //
                                           ::testing::Values(0),                                 //
                                           ::testing::ValuesIn(netPrecisions),                   //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        ReverseSequenceLayerTest::getTestCaseName);

}  // namespace
