// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/space_to_depth.hpp"

using LayerTestsDefinitions::SpaceToDepthLayerTest;

namespace {

using spaceToDepthParamsTuple = typename std::tuple<std::vector<size_t>,                             // Input shape
                                                    InferenceEngine::Precision,                      // Input precision
                                                    ngraph::opset1::SpaceToDepth::SpaceToDepthMode,  // Mode
                                                    std::size_t,                                     // Block size
                                                    std::string>;                                    // Device name>

spaceToDepthParamsTuple s2d_only_test_cases[] = {
    spaceToDepthParamsTuple({3, 7, 10, 10},                                                 //
                            InferenceEngine::Precision::FP32,                              //
                            ngraph::opset1::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST,  //
                            2,                                                             //
                            CommonTestUtils::DEVICE_PLAIDML),
    spaceToDepthParamsTuple({3, 7, 10, 10},  //
                            InferenceEngine::Precision::FP32,
                            ngraph::opset1::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST,  //
                            2,                                                            //
                            CommonTestUtils::DEVICE_PLAIDML),
};

INSTANTIATE_TEST_CASE_P(Smoke, SpaceToDepthLayerTest, ::testing::ValuesIn(s2d_only_test_cases),
                        SpaceToDepthLayerTest::getTestCaseName);

}  // namespace
