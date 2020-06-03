// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/activation.hpp"
#include "common_test_utils/test_constants.hpp"
#include <vector>

using LayerTestsDefinitions::ActivationLayerTest;
using ngraph::helpers::ActivationTypes;
using ngraph::helpers::Relu;

namespace {
// Common params
const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<ActivationTypes> activationTypes = {
    // Sigmoid,
    // Tanh,
    Relu,
    // Exp,
    // Log,
    // Sign,
    // Abs,
    // Gelu
};

const auto basicCases = ::testing::Combine(
    ::testing::ValuesIn(activationTypes), ::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(netPrecisions),
    ::testing::Values(std::vector<size_t>({1, 50}), std::vector<size_t>({1, 128})),
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_CASE_P(Activation_Basic, ActivationLayerTest, basicCases, ActivationLayerTest::getTestCaseName);

}  // namespace
