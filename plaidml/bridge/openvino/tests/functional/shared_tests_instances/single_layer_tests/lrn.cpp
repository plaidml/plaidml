// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/lrn.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using LayerTestsDefinitions::LrnLayerTest;

namespace {
// Common params

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16,
};

const double alpha = 9.9e-05;
const double beta = 2;
const double bias = 1.0;
const size_t size = 5;

INSTANTIATE_TEST_CASE_P(smoke, LrnLayerTest,
                        ::testing::Combine(::testing::Values(alpha),                                //
                                           ::testing::Values(beta),                                 //
                                           ::testing::Values(bias),                                 //
                                           ::testing::Values(size),                                 //
                                           ::testing::Values(std::vector<int64_t>({1})),            //
                                           ::testing::ValuesIn(netPrecisions),                      //
                                           ::testing::Values(std::vector<size_t>({10, 10, 3, 2})),  //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        LrnLayerTest::getTestCaseName);

}  // namespace
