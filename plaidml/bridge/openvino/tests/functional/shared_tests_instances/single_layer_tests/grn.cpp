// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <map>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/grn.hpp"

using LayerTestsDefinitions::GrnLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

INSTANTIATE_TEST_CASE_P(smoke, GrnLayerTest,
                        ::testing::Combine(                                              //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(std::vector<std::size_t>({4, 3, 3, 6})),   //
                            ::testing::Values(0.01),                                     //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
                        GrnLayerTest::getTestCaseName);

}  // namespace
