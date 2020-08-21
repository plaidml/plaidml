// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/concat.hpp"

using LayerTestsDefinitions::ConcatLayerTest;

namespace {

std::vector<size_t> axes = {0, 1, 2, 3};
std::vector<std::vector<std::vector<size_t>>> inShapes = {
    {{10, 10, 10, 10}},
    {{10, 10, 10, 10}, {10, 10, 10, 10}},
    {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
    {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
    {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}}};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16  // TODO: Not yet supported
};

INSTANTIATE_TEST_SUITE_P(NoReshape, ConcatLayerTest,
                         ::testing::Combine(::testing::ValuesIn(axes),           //
                                            ::testing::ValuesIn(inShapes),       //
                                            ::testing::ValuesIn(netPrecisions),  //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                         ConcatLayerTest::getTestCaseName);

}  // namespace
