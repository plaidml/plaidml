// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/tan.hpp"
#include "common_test_utils/test_constants.hpp"
#include <vector>

using LayerTestsDefinitions::TanLayerTest;

namespace {

std::vector<std::vector<size_t>> inShapes = {{2},                    //
                                             {1, 1, 1, 3},           //
                                             {1, 2, 4},              //
                                             {1, 4, 4},              //
                                             {1, 4, 4, 1},           //
                                             {1, 1, 1, 1, 1, 1, 3},  //
                                             {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

INSTANTIATE_TEST_CASE_P(tan, TanLayerTest,
                        ::testing::Combine(                                       //
                            ::testing::ValuesIn(netPrecisions),                   //
                            ::testing::ValuesIn(inShapes),                        //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        TanLayerTest::getTestCaseName);
}  // namespace
