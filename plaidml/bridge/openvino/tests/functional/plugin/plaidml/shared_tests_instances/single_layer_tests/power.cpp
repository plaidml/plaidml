// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <map>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/power.hpp"

using LayerTestsDefinitions::PowerLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<std::size_t>> inputShapes = {
    {std::vector<std::size_t>({1, 30}), std::vector<std::size_t>({1, 30})}};

INSTANTIATE_TEST_SUITE_P(PowerCheck, PowerLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputPrecisions), ::testing::Values(inputShapes),
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                         PowerLayerTest::getTestCaseName);

}  // namespace
