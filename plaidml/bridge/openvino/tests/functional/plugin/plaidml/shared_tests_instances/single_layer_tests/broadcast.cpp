// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/broadcast.hpp"

using LayerTestsDefinitions::BroadcastLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32};
const std::vector<ngraph::op::AutoBroadcastType> modes = {
    ngraph::op::AutoBroadcastType::EXPLICIT,  //
    ngraph::op::AutoBroadcastType::NUMPY,     //
};
const std::vector<std::vector<int64_t>> target_shapes = {
    {3, 1, 1, 2, 3, 2},  //
    {3, 3, 1, 2, 3, 2},  //
    {3, 1, 2, 2, 3, 2},  //
    {3, 3, 2, 2, 3, 2},  //
};
const std::vector<std::vector<int64_t>> axes_mapping = {
    {0, 3},  //
    {0, 5},  //
    {4, 5},  //
};
// Input shapes limited as OpenVINO 2020.2 uses a version of nGraph that did not support broadcasting from 1 -> n in
// EXPLICIT mode.
const std::vector<std::vector<size_t>> input_shapes = {
    // {1, 1}, //
    // {3, 1}, //
    // {1, 2}, //
    {3, 2},  //
};

INSTANTIATE_TEST_SUITE_P(Broadcast, BroadcastLayerTest,
                         ::testing::Combine(::testing::ValuesIn(modes),                           //
                                            ::testing::ValuesIn(target_shapes),                   //
                                            ::testing::ValuesIn(axes_mapping),                    //
                                            ::testing::ValuesIn(netPrecisions),                   //
                                            ::testing::ValuesIn(input_shapes),                    //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                         BroadcastLayerTest::getTestCaseName);

}  // namespace
