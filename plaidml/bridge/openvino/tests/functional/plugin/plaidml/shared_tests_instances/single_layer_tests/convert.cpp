// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/convert.hpp"

using LayerTestsDefinitions::ConvertLayerTest;

namespace {
const std::vector<std::vector<size_t>> inShape = {{1, 2, 3, 4}};

// Non-FP32 targets aren't currently testing correctly; it appears to be an issue with the interpreter casting
// everything to float (which is true in general but obviously matters more on this test). However, this has persisted
// from OV 2020.2, which is making me wonder if the error is instead on the PlaidML side.
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::U8,
    InferenceEngine::Precision::I8,
};

INSTANTIATE_TEST_SUITE_P(NoReshape, ConvertLayerTest,
                         ::testing::Combine(::testing::Values(inShape),          //
                                            ::testing::ValuesIn(netPrecisions),  //
                                            ::testing::ValuesIn(netPrecisions),  //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                         ConvertLayerTest::getTestCaseName);

}  // namespace
