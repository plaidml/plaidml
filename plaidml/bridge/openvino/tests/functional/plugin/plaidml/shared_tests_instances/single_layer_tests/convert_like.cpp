// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/convert_like.hpp"
#include "common_test_utils/test_constants.hpp"
#include <vector>

using LayerTestsDefinitions::ConvertLikeLayerTest;

namespace {

std::vector<std::vector<size_t>> inShapes = {{2},                    //
                                             {1, 1, 1, 3},           //
                                             {1, 2, 4},              //
                                             {1, 4, 4},              //
                                             {1, 4, 4, 1},           //
                                             {1, 1, 1, 1, 1, 1, 3},  //
                                             {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16,  //
    InferenceEngine::Precision::I16,   //
    InferenceEngine::Precision::U8,    //
};

// Non-FP32 targets aren't currently testing correctly; it appears to be an issue with the interpreter casting
// everything to float (which is true in general but obviously matters more on this test)
std::vector<InferenceEngine::Precision> targetPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::I16,
    // InferenceEngine::Precision::BOOL,
};

INSTANTIATE_TEST_SUITE_P(ConvertLike, ConvertLikeLayerTest,
                         ::testing::Combine(                                       //
                             ::testing::ValuesIn(targetPrecisions),                //
                             ::testing::ValuesIn(netPrecisions),                   //
                             ::testing::ValuesIn(inShapes),                        //
                             ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                         ConvertLikeLayerTest::getTestCaseName);
}  // namespace
