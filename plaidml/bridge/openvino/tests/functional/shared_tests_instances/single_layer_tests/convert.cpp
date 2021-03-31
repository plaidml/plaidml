// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/convert.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine;

namespace {
const std::vector<std::vector<size_t>> inShape = {
    {1, 2, 3, 4},
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    Precision::FP32,
};

INSTANTIATE_TEST_CASE_P(smoke, ConvertLayerTest,
                        ::testing::Combine(                                       //
                            ::testing::Values(inShape),                           //
                            ::testing::ValuesIn(netPrecisions),                   //
                            ::testing::ValuesIn(netPrecisions),                   //
                            ::testing::Values(InferenceEngine::Layout::ANY),      //
                            ::testing::Values(InferenceEngine::Layout::ANY),      //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        ConvertLayerTest::getTestCaseName);

}  // namespace
