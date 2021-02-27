// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/gather.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const std::vector<std::vector<size_t>> inputShapes = {
    std::vector<size_t>{10, 20, 30, 40},
};

const std::vector<std::vector<int>> indices = {
    std::vector<int>{0, 3, 2, 1},
};
const std::vector<std::vector<size_t>> indicesShapes = {
    std::vector<size_t>{4},
    std::vector<size_t>{2, 2},
};

const std::vector<int> axes = {
    0, 1, 2, 3, -1,
};

const auto params = testing::Combine(                          //
    testing::ValuesIn(indices),                                //
    testing::ValuesIn(indicesShapes),                          //
    testing::ValuesIn(axes),                                   //
    testing::ValuesIn(inputShapes),                            //
    testing::ValuesIn(netPrecisions),                          //
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
    testing::Values(InferenceEngine::Layout::ANY),             //
    testing::Values(InferenceEngine::Layout::ANY),             //
    testing::Values(CommonTestUtils::DEVICE_PLAIDML)           //
);

INSTANTIATE_TEST_CASE_P(smoke, GatherLayerTest, params, GatherLayerTest::getTestCaseName);

}  // namespace
