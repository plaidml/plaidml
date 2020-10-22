// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/unsqueeze.hpp"

using LayerTestsDefinitions::UnsqueezeLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16   // TODO: Not working yet
};

INSTANTIATE_TEST_SUITE_P(UnsqueezeCheck, UnsqueezeLayerTest,
                         ::testing::Combine(::testing::Values(std::vector<int64_t>({0}),
                                                              std::vector<int64_t>({0, 2})),      //
                                            ::testing::ValuesIn(netPrecisions),                   //
                                            ::testing::Values(std::vector<size_t>({3, 4})),       //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                         UnsqueezeLayerTest::getTestCaseName);
}  // namespace
