// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/batch_norm_inference.hpp"

using LayerTestsDefinitions::BatchNormInferenceLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::SizeVector> inputShapes = {
    InferenceEngine::SizeVector{4, 6, 2, 3, 2},
    InferenceEngine::SizeVector{6},
    InferenceEngine::SizeVector{6},
    InferenceEngine::SizeVector{6},
    InferenceEngine::SizeVector{6},
};

const std::vector<double> epsilon = {0.01, 0.0000001};

const auto params = testing::Combine(testing::ValuesIn(epsilon),                         //
                                     testing::ValuesIn(netPrecisions),                   //
                                     testing::Values(inputShapes),                       //
                                     testing::Values(CommonTestUtils::DEVICE_PLAIDML));  //

INSTANTIATE_TEST_SUITE_P(BatchNorm,                                    //
                         BatchNormInferenceLayerTest,                  //
                         params,                                       //
                         BatchNormInferenceLayerTest::getTestCaseName  //
);

}  // namespace
