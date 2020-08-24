// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/squeeze.hpp"

using LayerTestsDefinitions::SqueezeLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

INSTANTIATE_TEST_SUITE_P(SqueezeCheck, SqueezeLayerTest,
                         ::testing::Combine(::testing::Values(std::vector<int64_t>({1})),         //
                                            ::testing::ValuesIn(netPrecisions),                   //
                                            ::testing::Values(std::vector<size_t>({2, 1, 4})),    //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                         SqueezeLayerTest::getTestCaseName);
}  // namespace
