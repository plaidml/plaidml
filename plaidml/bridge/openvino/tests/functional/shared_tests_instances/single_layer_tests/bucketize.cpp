// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/bucketize.hpp"

using LayerTestsDefinitions::BucketizeLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I8,
};

std::vector<std::vector<size_t>> inputShapes{{2, 2}};
std::vector<std::vector<size_t>> bucketsShapes{{3}};

// Output_type shall support i64 && i32, close i64 for openvino limitation
std::vector<InferenceEngine::Precision> outputPrecision = {
    InferenceEngine::Precision::I32,
    // InferenceEngine::Precision::I64,
};
std::vector<bool> withRightBound{true, false};

INSTANTIATE_TEST_CASE_P(smoke, BucketizeLayerTest,
                        ::testing::Combine(::testing::ValuesIn(netPrecisions),                          //
                                           ::testing::ValuesIn(inputShapes),                            //
                                           ::testing::ValuesIn(bucketsShapes),                          //
                                           ::testing::ValuesIn(outputPrecision),                        //
                                           ::testing::ValuesIn(withRightBound),                         //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),          //
                                           ::testing::Values(std::map<std::string, std::string>({}))),  //
                        BucketizeLayerTest::getTestCaseName);
}  // namespace
