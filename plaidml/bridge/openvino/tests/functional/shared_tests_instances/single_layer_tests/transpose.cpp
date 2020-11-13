// Copyright (C) 2019 Intel Corporation
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

const std::vector<std::vector<size_t>> inputShapes = {
    std::vector<size_t>{1, 3, 100, 100},
};

const std::vector<std::vector<size_t>> inputOrder = {
    std::vector<size_t>{0, 3, 2, 1},
    std::vector<size_t>{},
};

const auto params = testing::Combine(testing::ValuesIn(inputOrder),                    //
                                     testing::ValuesIn(netPrecisions),                 //
                                     testing::ValuesIn(inputShapes),                   //
                                     testing::Values(CommonTestUtils::DEVICE_PLAIDML)  //
);

INSTANTIATE_TEST_CASE_P(smoke, TransposeLayerTest, params, TransposeLayerTest::getTestCaseName);

}  // namespace
