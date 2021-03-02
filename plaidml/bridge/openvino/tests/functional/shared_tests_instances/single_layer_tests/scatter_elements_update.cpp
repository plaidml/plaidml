// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/scatter_elements_update.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "ngraph/opsets/opset3.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::opset3;

namespace {
// map<inputShape, map<indicesShape, axis>>
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> axesShapeInShape{
    {{10, 12, 15}, {{{1, 2, 4}, {0, 1, 2}}, {{2, 2, 2}, {-1, -2, -3}}}},
    {{15, 9, 8, 12}, {{{1, 2, 2, 2}, {0, 1, 2, 3}}, {{1, 2, 1, 4}, {-1, -2, -3, -4}}}},
    {{9, 9, 8, 8, 11, 10}, {{{1, 2, 1, 2, 1, 2}, {5, -3}}}},
};
// index value should not be random data
const std::vector<std::vector<size_t>> idxValue = {
    {1, 0, 4, 6, 2, 3, 7, 5},
};

const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I32,
};

const std::vector<InferenceEngine::Precision> idxPrecisions = {
    InferenceEngine::Precision::I32,
};

const auto ScatterEltUpdateCases = ::testing::Combine(                                     //
    ::testing::ValuesIn(ScatterElementsUpdateLayerTest::combineShapes(axesShapeInShape)),  //
    ::testing::ValuesIn(idxValue),                                                         //
    ::testing::ValuesIn(inputPrecisions),                                                  //
    ::testing::ValuesIn(idxPrecisions),                                                    //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)                                     //
);

INSTANTIATE_TEST_CASE_P(ScatterEltsUpdate, ScatterElementsUpdateLayerTest, ScatterEltUpdateCases,
                        ScatterElementsUpdateLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, ScatterElementsUpdateLayerTest,
                        ::testing::Combine(                                                                        //
                            ::testing::ValuesIn(ScatterElementsUpdateLayerTest::combineShapes(axesShapeInShape)),  //
                            ::testing::ValuesIn(idxValue),                                                         //
                            ::testing::Values(InferenceEngine::Precision::FP32),                                   //
                            ::testing::Values(InferenceEngine::Precision::I32),                                    //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        ScatterElementsUpdateLayerTest::getTestCaseName);

}  // namespace
