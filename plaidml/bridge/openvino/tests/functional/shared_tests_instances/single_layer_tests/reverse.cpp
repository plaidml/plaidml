// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/reverse.hpp"

using LayerTestsDefinitions::ReverseLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16,
    // InferenceEngine::Precision::I64
};

std::vector<std::vector<size_t>> shape_index{{4, 6, 5}, {3, 9, 2}, {1, 4, 2, 1, 1, 3}};
std::vector<std::vector<size_t>> axes_index{{0}, {1}, {0, 2}};

std::vector<std::vector<size_t>> shape_mask{{4, 6, 5}, {3, 9, 2}, {1, 4, 2}};
std::vector<std::vector<size_t>> axes_mask{{1, 0, 0}, {0, 1, 0}, {1, 0, 1}};

INSTANTIATE_TEST_CASE_P(smoke_ReverseCheckIndex, ReverseLayerTest,
                        ::testing::Combine(::testing::ValuesIn(netPrecisions),                          //
                                           ::testing::ValuesIn(shape_index),                            //
                                           ::testing::ValuesIn(axes_index),                             //
                                           ::testing::Values("index"),                                  //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),          //
                                           ::testing::Values(std::map<std::string, std::string>({}))),  //
                        ReverseLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ReverseCheckMask, ReverseLayerTest,
                        ::testing::Combine(::testing::ValuesIn(netPrecisions),                          //
                                           ::testing::ValuesIn(shape_mask),                             //
                                           ::testing::ValuesIn(axes_mask),                              //
                                           ::testing::Values("mask"),                                   //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),          //
                                           ::testing::Values(std::map<std::string, std::string>({}))),  //
                        ReverseLayerTest::getTestCaseName);
}  // namespace
