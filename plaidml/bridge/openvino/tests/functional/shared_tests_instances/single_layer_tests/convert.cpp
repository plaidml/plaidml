// Copyright (C) 2019 Intel Corporation
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
    // Precision::U8,
    // Precision::I8,
    // Precision::U16,
    // Precision::I16,
    // Precision::I32,
    // Precision::U64,
    // Precision::I64,
    Precision::FP32,
};

INSTANTIATE_TEST_CASE_P(smoke_NoReshape, ConvertLayerTest,
                        ::testing::Combine(                                       //
                            ::testing::Values(inShape),                           //
                            ::testing::ValuesIn(netPrecisions),                   //
                            ::testing::ValuesIn(netPrecisions),                   //
                            ::testing::Values(InferenceEngine::Layout::ANY),      //
                            ::testing::Values(InferenceEngine::Layout::ANY),      //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        ConvertLayerTest::getTestCaseName);

}  // namespace
