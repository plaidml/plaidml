// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/scatter_update.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I32,
};

const std::vector<InferenceEngine::Precision> idxPrecisions = {
    InferenceEngine::Precision::I32,
};

// map<inputShape, map<indicesShape, axis>>
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> axesShapeInShape{
    {{10, 16, 12, 15}, {{{2, 4}, {0, 1, 2, 3}}, {{8}, {-1, -2, -3, -4}}}},
    {{10, 9, 10, 9, 10}, {{{8}, {-3, -1, 0, 2, 4}}, {{4, 2}, {-2, 2}}}},
};
// indices should not be random value
const std::vector<std::vector<int64_t>> idxValue = {
    {0, 2, 4, 6, 1, 3, 5, 7},
};

const auto ScatterUpdateArgSet = ::testing::Combine(                               //
    ::testing::ValuesIn(ScatterUpdateLayerTest::combineShapes(axesShapeInShape)),  //
    ::testing::ValuesIn(idxValue),                                                 //
    ::testing::ValuesIn(inputPrecisions),                                          //
    ::testing::ValuesIn(idxPrecisions),                                            //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)                             //
);

INSTANTIATE_TEST_CASE_P(ScatterUpdate, ScatterUpdateLayerTest, ScatterUpdateArgSet,
                        ScatterUpdateLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, ScatterUpdateLayerTest,
                        ::testing::Combine(                                                                //
                            ::testing::ValuesIn(ScatterUpdateLayerTest::combineShapes(axesShapeInShape)),  //
                            ::testing::ValuesIn(idxValue),                                                 //
                            ::testing::Values(InferenceEngine::Precision::FP32),                           //
                            ::testing::Values(InferenceEngine::Precision::I32),                            //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),                           //
                        ScatterUpdateLayerTest::getTestCaseName);

}  // namespace
