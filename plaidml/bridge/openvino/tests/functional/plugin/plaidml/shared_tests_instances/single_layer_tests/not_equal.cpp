// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <map>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/not_equal.hpp"

using LayerTestsDefinitions::NotEqualLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

std::vector<std::vector<std::vector<size_t>>> inputShapes = {
    {{1}, {1}},                         //
    {{8}, {8}},                         //
    {{4, 5}, {4, 5}},                   //
    {{3, 4, 5}, {3, 4, 5}},             //
    {{2, 3, 4, 5}, {2, 3, 4, 5}},       //
    {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}  //
};

const auto cases = ::testing::Combine(::testing::ValuesIn(inputShapes),    //
                                      ::testing::ValuesIn(netPrecisions),  //
                                      ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_SUITE_P(CompareWithRefs, NotEqualLayerTest, cases, NotEqualLayerTest::getTestCaseName);
}  // namespace
