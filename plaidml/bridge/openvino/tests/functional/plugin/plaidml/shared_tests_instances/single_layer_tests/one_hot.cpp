// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/one_hot.hpp"

using LayerTestsDefinitions::OneHotLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,  //
                                       // InferenceEngine::Precision::FP16   //
};

INSTANTIATE_TEST_CASE_P(NumSplitsCheck, OneHotLayerTest,
                        ::testing::Combine(::testing::Values(1),                                      // axis
                                           ::testing::Values(30),                                     // depth
                                           ::testing::Values(0.0),                                    // on_value
                                           ::testing::Values(4.0),                                    // off_value
                                           ::testing::ValuesIn(netPrecisions),                        //
                                           ::testing::Values(std::vector<size_t>({30, 30, 30, 30})),  //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),       //
                        OneHotLayerTest::getTestCaseName);
}  // namespace
