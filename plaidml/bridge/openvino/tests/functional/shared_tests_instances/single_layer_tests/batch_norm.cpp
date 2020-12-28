// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/batch_norm.hpp"

using LayerTestsDefinitions::BatchNormLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<double> epsilon = {1e-6, 1e-5, 1e-4};
const std::vector<std::vector<size_t>> inputShapes = {
    {1, 3},        //
    {2, 5},        //
    {1, 3, 10},    //
    {1, 3, 1, 1},  //
    {2, 5, 4, 4},  //
};

const auto batchNormParams = testing::Combine(testing::ValuesIn(epsilon),        //
                                              testing::ValuesIn(netPrecisions),  //
                                              testing::ValuesIn(inputShapes),    //
                                              testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_CASE_P(smoke,                               //
                        BatchNormLayerTest,                  //
                        batchNormParams,                     //
                        BatchNormLayerTest::getTestCaseName  //
);

}  // namespace
