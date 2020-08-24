// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/floor_mod.hpp"
#include "common_test_utils/test_constants.hpp"
#include <vector>

using LayerTestsDefinitions::FloorModLayerTest;

namespace {

const std::vector<std::vector<std::size_t>> inputShapes = {
    {std::vector<std::size_t>({10, 30}), std::vector<std::size_t>({1, 30})}};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

INSTANTIATE_TEST_SUITE_P(FloorMod, FloorModLayerTest,
                         ::testing::Combine(                                       //
                             ::testing::ValuesIn(netPrecisions),                   //
                             ::testing::Values(inputShapes),                       //
                             ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                         FloorModLayerTest::getTestCaseName);
}  // namespace
