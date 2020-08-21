// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <map>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/greater.hpp"

using LayerTestsDefinitions::GreaterLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> inPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Precision> outPrecisions = {
    InferenceEngine::Precision::BOOL,
};

const std::vector<std::vector<std::size_t>> inputShapes = {
    {std::vector<std::size_t>({40, 30}), std::vector<std::size_t>({1, 30})}};

const auto cases = ::testing::Combine(::testing::Values(inputShapes),      //
                                      ::testing::ValuesIn(inPrecisions),   //
                                      ::testing::ValuesIn(outPrecisions),  //
                                      ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_SUITE_P(CompareWithRefs, GreaterLayerTest, cases, GreaterLayerTest::getTestCaseName);

}  // namespace
