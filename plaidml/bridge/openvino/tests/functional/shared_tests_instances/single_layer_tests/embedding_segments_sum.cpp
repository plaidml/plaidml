// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/embedding_segments_sum.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I32,
};

const std::vector<InferenceEngine::Precision> indPrecisions = {
    InferenceEngine::Precision::I32,
};

const std::vector<std::vector<size_t>> embTableShape = {
    {5, 6},
    {10, 35},
    {5, 4, 16},
};
const std::vector<std::vector<size_t>> indices = {
    {0, 1, 2, 2, 3},
    {4, 4, 3, 1, 2},
};
const std::vector<std::vector<size_t>> segmentIds = {
    {0, 1, 2, 3, 4},
    {0, 0, 2, 2, 4},
};
const std::vector<size_t> numSegments = {
    5,
    7,
};
const std::vector<size_t> defaultIndex = {
    0,
    4,
};
const std::vector<bool> withWeights = {
    false,
    true,
};
const std::vector<bool> withDefaultIndex = {
    false,
    true,
};

const auto embSegmentsSumArgSet = ::testing::Combine(  //
    ::testing::ValuesIn(embTableShape),                //
    ::testing::ValuesIn(indices),                      //
    ::testing::ValuesIn(segmentIds),                   //
    ::testing::ValuesIn(numSegments),                  //
    ::testing::ValuesIn(defaultIndex),                 //
    ::testing::ValuesIn(withWeights),                  //
    ::testing::ValuesIn(withDefaultIndex)              //
);

const auto smokeArgSet = ::testing::Combine(                  //
    ::testing::Values(std::vector<size_t>({5, 6})),           //
    ::testing::Values(std::vector<size_t>({0, 1, 2, 2, 3})),  //
    ::testing::Values(std::vector<size_t>({0, 0, 2, 2, 4})),  //
    ::testing::Values(7),                                     //
    ::testing::Values(4),                                     //
    ::testing::ValuesIn(withWeights),                         //
    ::testing::ValuesIn(withDefaultIndex)                     //
);

INSTANTIATE_TEST_CASE_P(EmbeddingSegmentsSum, EmbeddingSegmentsSumLayerTest,
                        ::testing::Combine(                                       //
                            embSegmentsSumArgSet,                                 //
                            ::testing::ValuesIn(netPrecisions),                   //
                            ::testing::ValuesIn(indPrecisions),                   //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        EmbeddingSegmentsSumLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, EmbeddingSegmentsSumLayerTest,
                        ::testing::Combine(                                       //
                            smokeArgSet,                                          //
                            ::testing::ValuesIn(netPrecisions),                   //
                            ::testing::ValuesIn(indPrecisions),                   //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        EmbeddingSegmentsSumLayerTest::getTestCaseName);
}  // namespace
