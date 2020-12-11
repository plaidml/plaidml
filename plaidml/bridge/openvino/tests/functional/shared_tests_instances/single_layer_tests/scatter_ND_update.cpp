// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/scatter_ND_update.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
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
// indices should not be random value
const std::vector<std::vector<size_t>> idxValue = {{0, 2, 4, 6, 1, 3, 5, 7}};

const auto ScatterNDUpdateArgSet =
    ::testing::Combine(::testing::ValuesIn(ScatterNDUpdateLayerTest::combineShapes(sliceSelectInShape)),
                       ::testing::ValuesIn(inputPrecisions), ::testing::ValuesIn(idxPrecisions),
                       ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_CASE_P(ScatterNDUpdate, ScatterNDUpdateLayerTest, ScatterNDUpdateArgSet,
                        ScatterNDUpdateLayerTest::getTestCaseName);

}  // namespace
