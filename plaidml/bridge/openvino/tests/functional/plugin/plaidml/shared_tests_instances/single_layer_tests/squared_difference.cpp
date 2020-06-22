// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <map>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/squared_difference.hpp"

using LayerTestsDefinitions::SquaredDifferenceLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<std::size_t>> inputShapes = {
    {std::vector<std::size_t>({1, 30}), std::vector<std::size_t>({1, 30})}};

INSTANTIATE_TEST_CASE_P(CompareWithRefs, SquaredDifferenceLayerTest,
                        ::testing::Combine(::testing::ValuesIn(inputPrecisions), ::testing::Values(inputShapes),
                                           // ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        // ::testing::ValuesIn(std::vector<int>{1, 2}),  // TODO
                        SquaredDifferenceLayerTest::getTestCaseName);

}  // namespace
