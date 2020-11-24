// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/tile.hpp"

using LayerTestsDefinitions::TileLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32};

const std::vector<std::vector<size_t>> repeats = {
    {1, 2, 3},
    {2, 1, 1},
    {2, 3, 1},
    {2, 2, 2},
};

INSTANTIATE_TEST_CASE_P(smoke, TileLayerTest,
                        ::testing::Combine(::testing::ValuesIn(repeats),                         //
                                           ::testing::ValuesIn(netPrecisions),                   //
                                           ::testing::Values(std::vector<size_t>({2, 3, 4})),    //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        TileLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Tile6d, TileLayerTest,
                        ::testing::Combine(::testing::Values(std::vector<size_t>({1, 1, 1, 2, 1, 2})),  //
                                           ::testing::ValuesIn(netPrecisions),                          //
                                           ::testing::Values(std::vector<size_t>({1, 4, 3, 1, 3, 1})),  //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
                        TileLayerTest::getTestCaseName);

}  // namespace
