// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <map>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/equal.hpp"

using LayerTestsDefinitions::EqualLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> inPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Precision> outPrecisions = {
    InferenceEngine::Precision::BOOL,
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
                                      ::testing::ValuesIn(inPrecisions),   //
                                      ::testing::ValuesIn(outPrecisions),  //
                                      ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_SUITE_P(CompareWithRefs, EqualLayerTest, cases, EqualLayerTest::getTestCaseName);
}  // namespace
