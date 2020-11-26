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

const std::vector<std::vector<size_t>> kernels = {{3, 3}, {5, 5}};
const std::vector<std::vector<size_t>> strides = {{5, 5}, {1, 1}};
const std::vector<std::vector<size_t>> rates = {{1, 1}, {2, 2}};
const std::vector<ngraph::op::PadType> padTypes = {
    ngraph::op::PadType::VALID,       //
    ngraph::op::PadType::SAME_UPPER,  //
    ngraph::op::PadType::SAME_LOWER,  //
};

INSTANTIATE_TEST_CASE_P(ReverseSequenceLayerTest_smoke, ReverseSequenceLayerTest,
                        ::testing::Combine(::testing::Values(0),                                   //
                                           ::testing::Values(1),                                   //
                                           ::testing::Values(std::vector<size_t>{2, 10, 10, 10}),  //
                                           ::testing::Values(std::vector<size_t>{4, 5}),           //
                                           ::testing::Values(0),                                   //
                                           ::testing::ValuesIn(netPrecisions),                     //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),    //
                        ReverseSequenceLayerTest::getTestCaseName);

}  // namespace
