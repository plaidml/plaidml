// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/scatter_ND_update.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "ngraph/opsets/opset3.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::opset3;

namespace {
const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I32,
};

const std::vector<InferenceEngine::Precision> idxPrecisions = {
    InferenceEngine::Precision::I32,
};
// map<inputShape map<indicesShape, indicesValue>>
// updateShape is gotten from inputShape and indicesShape
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<size_t>>> sliceSelectInShape{
    {{10, 9, 9, 11}, {{{4, 1}, {1, 3, 5, 7}}, {{1, 2}, {4, 6}}, {{2, 3}, {0, 1, 1, 2, 2, 2}}, {{1, 4}, {5, 5, 4, 9}}}},
    {{10, 9, 10, 9, 10}, {{{2, 2, 1}, {5, 6, 2, 8}}, {{2, 3}, {0, 4, 6, 5, 7, 1}}}},
};

const auto ScatterNDUpdateCases = ::testing::Combine(                                  //
    ::testing::ValuesIn(ScatterNDUpdateLayerTest::combineShapes(sliceSelectInShape)),  //
    ::testing::ValuesIn(inputPrecisions),                                              //
    ::testing::ValuesIn(idxPrecisions),                                                //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)                                 //
);

INSTANTIATE_TEST_CASE_P(ScatterNDUpdate, ScatterNDUpdateLayerTest, ScatterNDUpdateCases,
                        ScatterNDUpdateLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, ScatterNDUpdateLayerTest,
                        ::testing::Combine(                                                                    //
                            ::testing::ValuesIn(ScatterNDUpdateLayerTest::combineShapes(sliceSelectInShape)),  //
                            ::testing::Values(InferenceEngine::Precision::FP32),                               //
                            ::testing::Values(InferenceEngine::Precision::I32),                                //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        ScatterNDUpdateLayerTest::getTestCaseName);
}  // namespace
