// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/space_to_batch.hpp"

using LayerTestsDefinitions::SpaceToBatchLayerTest;
using LayerTestsDefinitions::spaceToBatchParamsTuple;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::I32,
};

const std::vector<std::vector<size_t>> blockShapes = {{1, 2, 2, 1}};

const std::vector<std::vector<size_t>> pads_begins = {{0, 0, 0, 0}, {0, 2, 2, 0}, {0, 4, 0, 0}};

const std::vector<std::vector<size_t>> pads_ends = {
    {0, 0, 0, 0},
    {0, 2, 2, 0},
    {0, 0, 4, 0},
};

const std::vector<std::vector<size_t>> inputShapes = {
    {1, 8, 8, 1},  //
    {3, 4, 4, 1},  //
};

INSTANTIATE_TEST_CASE_P(smoke_SpaceToBatchCheck, SpaceToBatchLayerTest,
                        ::testing::Combine(::testing::ValuesIn(blockShapes),                    //
                                           ::testing::Values(std::vector<size_t>{0, 2, 2, 0}),  //
                                           ::testing::Values(std::vector<size_t>{0, 0, 4, 0}),  //
                                           ::testing::Values(std::vector<size_t>{3, 4, 4, 1}),  //
                                           ::testing::ValuesIn(inputPrecisions),                //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        SpaceToBatchLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(SpaceToBatchCoreCheck, SpaceToBatchLayerTest,
                        ::testing::Combine(::testing::ValuesIn(blockShapes),      //
                                           ::testing::ValuesIn(pads_begins),      //
                                           ::testing::ValuesIn(pads_ends),        //
                                           ::testing::ValuesIn(inputShapes),      //
                                           ::testing::ValuesIn(inputPrecisions),  //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        SpaceToBatchLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smokeSpaceToBatchTfCaseCheck, SpaceToBatchLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<size_t>({1, 2, 2, 1})),
                                           ::testing::Values(std::vector<size_t>({0, 1, 1, 0})),
                                           ::testing::Values(std::vector<size_t>({0, 1, 1, 0})),
                                           ::testing::Values(std::vector<size_t>({2, 2, 4, 1})),
                                           ::testing::ValuesIn(inputPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        SpaceToBatchLayerTest::getTestCaseName);
}  // namespace
