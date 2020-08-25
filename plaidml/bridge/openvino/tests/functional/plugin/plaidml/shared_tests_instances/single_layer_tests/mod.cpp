// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <map>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/mod.hpp"

using LayerTestsDefinitions::ModLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<std::vector<std::size_t>>> inputShapes = {
    {{1, 30}, {1, 30}},       //
    {{3, 5, 10}, {1, 5, 1}},  //
};

INSTANTIATE_TEST_SUITE_P(CompareWithRefs, ModLayerTest,
                         ::testing::Combine(                                       //
                             ::testing::ValuesIn(netPrecisions),                   //
                             ::testing::ValuesIn(inputShapes),                     //
                             ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                         ModLayerTest::getTestCaseName);

}  // namespace
