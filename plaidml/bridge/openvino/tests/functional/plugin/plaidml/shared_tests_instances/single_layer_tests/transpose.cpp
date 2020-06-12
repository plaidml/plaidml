// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/transpose.hpp"

using LayerTestsDefinitions::TransposeLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

INSTANTIATE_TEST_CASE_P(UnsqueezeCheck, TransposeLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<int64_t>({2, 0, 1}),
                                                             std::vector<int64_t>({0, 1, 2})),   //
                                           ::testing::ValuesIn(netPrecisions),                   //
                                           ::testing::Values(std::vector<size_t>({2, 3, 4})),    //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        TransposeLayerTest::getTestCaseName);
}  // namespace
