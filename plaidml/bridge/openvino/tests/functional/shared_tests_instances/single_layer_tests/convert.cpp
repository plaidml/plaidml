// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/convert.hpp"

using LayerTestsDefinitions::ConvertLayerTest;

namespace {
const std::vector<std::vector<size_t>> inShape = {{1, 2, 3, 4}};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16,
    // InferenceEngine::Precision::U8,
    // InferenceEngine::Precision::I8,
};

INSTANTIATE_TEST_CASE_P(NoReshape, ConvertLayerTest,
                        ::testing::Combine(::testing::Values(inShape),          //
                                           ::testing::ValuesIn(netPrecisions),  //
                                           ::testing::ValuesIn(netPrecisions),  //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        ConvertLayerTest::getTestCaseName);

}  // namespace
