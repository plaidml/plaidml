// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/reorg_yolo.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using LayerTestsDefinitions::reorgYoloLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> inputShapes = {
    {10, 10, 10, 10},  //
    {1, 3, 30, 30}     //
};

INSTANTIATE_TEST_CASE_P(smoke, reorgYoloLayerTest,
                        ::testing::Combine(::testing::ValuesIn(inputShapes),                     //
                                           ::testing::ValuesIn(netPrecisions),                   //
                                           ::testing::ValuesIn(std::vector<size_t>({1, 2, 5})),  //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        reorgYoloLayerTest::getTestCaseName);
}  // namespace
