// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/split.hpp"

using LayerTestsDefinitions::SplitLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16
};

INSTANTIATE_TEST_CASE_P(NumSplitsCheck, SplitLayerTest,
                        ::testing::Combine(::testing::Values(1, 2),                                   //
                                           ::testing::Values(0, 1, 2, 3),                             //
                                           ::testing::ValuesIn(netPrecisions),                        //
                                           ::testing::Values(std::vector<size_t>({30, 30, 30, 30})),  //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),       //
                        SplitLayerTest::getTestCaseName);
}  // namespace
