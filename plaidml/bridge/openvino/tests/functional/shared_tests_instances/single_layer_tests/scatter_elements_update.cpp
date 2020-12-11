// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/scatter_elements_update.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::I32,
};

const std::vector<InferenceEngine::Precision> idxPrecisions = {
    InferenceEngine::Precision::I32,
};

// map<inputShape, map<indicesShape, axis>>
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> axesShapeInShape{
    {{10, 12, 15}, {{{1, 2, 4}, {0, 1, 2}}, {{2, 2, 2}, {-1, -2, -3}}}},
    {{15, 9, 8, 12}, {{{1, 2, 2, 2}, {0, 1, 2, 3}}, {{1, 2, 1, 4}, {-1, -2, -3, -4}}}},
    {{9, 9, 8, 8, 11, 10}, {{{1, 2, 1, 2, 1, 2}, {5, -3}}}},
};
// indices should not be random value
const std::vector<std::vector<size_t>> idxValue = {{1, 0, 4, 6, 2, 3, 7, 5}};

const auto ScatterEltUpdateArgSet =
    ::testing::Combine(::testing::ValuesIn(ScatterElementsUpdateLayerTest::combineShapes(axesShapeInShape)),
                       ::testing::ValuesIn(idxValue), ::testing::ValuesIn(inputPrecisions),
                       ::testing::ValuesIn(idxPrecisions), ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_CASE_P(ScatterEltsUpdate, ScatterElementsUpdateLayerTest, ScatterEltUpdateArgSet,
                        ScatterElementsUpdateLayerTest::getTestCaseName);

}  // namespace
