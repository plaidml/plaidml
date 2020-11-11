// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/embedding_bag_offsets_sum.hpp"

using LayerTestsDefinitions::EmbeddingBagOffsetsSumLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::I32,
};

const std::vector<InferenceEngine::Precision> indPrecisions = {InferenceEngine::Precision::I32};

const std::vector<std::vector<size_t>> embTableShape = {{5, 6}, {10, 35}, {5, 4, 16}};
const std::vector<std::vector<size_t>> indices = {{0, 1, 2, 2, 3}, {4, 4, 3, 1, 0}, {1, 2, 1, 2, 1, 2, 1, 2, 1, 2}};
const std::vector<std::vector<size_t>> offsets = {{0, 2}, {0, 0, 2, 2}, {2, 4}};
const std::vector<size_t> defaultIndex = {0, 4};
const std::vector<bool> withWeights = {true, false};
const std::vector<bool> withDefaultIndex = {true, false};

const auto embBagOffsetSumArgSet = ::testing::Combine(::testing::ValuesIn(embTableShape),  //
                                                      ::testing::ValuesIn(indices),        //
                                                      ::testing::ValuesIn(offsets),        //
                                                      ::testing::ValuesIn(defaultIndex),   //
                                                      ::testing::ValuesIn(withWeights),    //
                                                      ::testing::ValuesIn(withDefaultIndex));

INSTANTIATE_TEST_CASE_P(smoke, EmbeddingBagOffsetsSumLayerTest,
                        ::testing::Combine(embBagOffsetSumArgSet,               //
                                           ::testing::ValuesIn(netPrecisions),  //
                                           ::testing::ValuesIn(indPrecisions),  //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        EmbeddingBagOffsetsSumLayerTest::getTestCaseName);
}  // namespace
