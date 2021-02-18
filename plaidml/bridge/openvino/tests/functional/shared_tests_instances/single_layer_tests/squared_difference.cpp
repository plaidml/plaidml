// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <map>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/squared_difference.hpp"

using LayerTestsDefinitions::SquaredDifferenceLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const std::vector<std::vector<std::size_t>> inputShapes = {
    {1, 30},
    {1, 10, 30},
};

INSTANTIATE_TEST_CASE_P(smoke, SquaredDifferenceLayerTest,
                        ::testing::Combine(                                       //
                            ::testing::ValuesIn(netPrecisions),                   //
                            ::testing::Values(inputShapes),                       //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        SquaredDifferenceLayerTest::getTestCaseName);

}  // namespace
