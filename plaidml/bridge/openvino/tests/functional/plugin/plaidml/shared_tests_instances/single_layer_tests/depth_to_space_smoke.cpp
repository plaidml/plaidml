// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset3.hpp>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/depth_to_space.hpp"

using LayerTestsDefinitions::DepthToSpaceLayerTest;
using ngraph::opset3::DepthToSpace;

namespace {
const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::U8,
    // InferenceEngine::Precision::I16,
};

const std::vector<DepthToSpace::DepthToSpaceMode> modes = {DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,
                                                           DepthToSpace::DepthToSpaceMode::DEPTH_FIRST};

const std::vector<std::vector<size_t>> inputShapes = {{1, 9, 1, 1},  //
                                                      {1, 9, 2, 2},  //
                                                      {2, 36, 3, 3}};

const auto DepthToSpace = ::testing::Combine(::testing::ValuesIn(inputShapes),      //
                                             ::testing::ValuesIn(inputPrecisions),  //
                                             ::testing::ValuesIn(modes),            //
                                             ::testing::Values(3),                  //
                                             ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_SUITE_P(DepthToSpaceSmoke, DepthToSpaceLayerTest, DepthToSpace,
                         DepthToSpaceLayerTest::getTestCaseName);

}  // namespace
