// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/one_hot.hpp"

using LayerTestsDefinitions::OneHotLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::I32,  //
    InferenceEngine::Precision::I64,  //
};

const int64_t axis = 1;
const size_t depth = 10;
const float on_value = 3;
const float off_value = 1;

INSTANTIATE_TEST_CASE_P(OneHotCheck, OneHotLayerTest,
                        ::testing::Combine(::testing::Values(axis),                               //
                                           ::testing::Values(depth),                              //
                                           ::testing::Values(on_value),                           //
                                           ::testing::Values(off_value),                          //
                                           ::testing::ValuesIn(netPrecisions),                    //
                                           ::testing::Values(std::vector<size_t>({1, 9, 2, 4})),  //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),   //
                        OneHotLayerTest::getTestCaseName);
}  // namespace
