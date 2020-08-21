// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <map>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/logical_or.hpp"

using LayerTestsDefinitions::LogicalOrLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<std::size_t>> inputShapes = {
    {std::vector<std::size_t>({1, 1}), std::vector<std::size_t>({1, 1})}};

INSTANTIATE_TEST_SUITE_P(CompareWithRefs, LogicalOrLayerTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions), ::testing::Values(inputShapes),
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                         LogicalOrLayerTest::getTestCaseName);

}  // namespace
