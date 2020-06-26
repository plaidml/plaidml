// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <map>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/prelu.hpp"

using LayerTestsDefinitions::PReLULayerTest;

namespace {
const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<std::size_t>> inputShapes = {
    {std::vector<std::size_t>({1, 30}), std::vector<std::size_t>({1, 30})}};

INSTANTIATE_TEST_CASE_P(PReLUCheck, PReLULayerTest,
                        ::testing::Combine(::testing::ValuesIn(inputPrecisions), ::testing::Values(inputShapes),
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        PReLULayerTest::getTestCaseName);

}  // namespace
