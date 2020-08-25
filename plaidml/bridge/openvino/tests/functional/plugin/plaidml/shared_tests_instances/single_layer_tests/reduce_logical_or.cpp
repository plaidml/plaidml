// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/reduce_logical_or.hpp"

using LayerTestsDefinitions::ReduceLogicalOrLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

INSTANTIATE_TEST_SUITE_P(ReduceLogicalOrCheck, ReduceLogicalOrLayerTest,
                         ::testing::Combine(::testing::Values(std::vector<int64_t>({1})),         //
                                            ::testing::Values(true),                              //
                                            ::testing::ValuesIn(netPrecisions),                   //
                                            ::testing::Values(std::vector<size_t>({3, 2, 2})),    //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                         ReduceLogicalOrLayerTest::getTestCaseName);
}  // namespace
