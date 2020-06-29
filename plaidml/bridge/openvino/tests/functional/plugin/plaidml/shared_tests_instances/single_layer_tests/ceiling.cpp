// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <map>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/ceiling.hpp"

using LayerTestsDefinitions::CeilingLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

std::vector<std::vector<size_t>> inputShapes = {{2},                    //
                                                {1, 1, 1, 3},           //
                                                {1, 2, 4},              //
                                                {1, 4, 4},              //
                                                {1, 4, 4, 1},           //
                                                {1, 1, 1, 1, 1, 1, 3},  //
                                                {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

INSTANTIATE_TEST_CASE_P(ceiling, CeilingLayerTest,
                        ::testing::Combine(::testing::ValuesIn(netPrecisions), ::testing::ValuesIn(inputShapes),
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        CeilingLayerTest::getTestCaseName);

}  // namespace
