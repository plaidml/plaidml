// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/depth_to_space.hpp"

using LayerTestsDefinitions::DepthToSpaceLayerTest;

namespace {

using depthToSpaceParamsTuple = typename std::tuple<std::vector<size_t>,                             // Input shape
                                                    InferenceEngine::Precision,                      // Input precision
                                                    ngraph::opset1::DepthToSpace::DepthToSpaceMode,  // Mode
                                                    std::size_t,                                     // Block size
                                                    std::string>;                                    // Device name>

depthToSpaceParamsTuple dts_only_test_cases[] = {
    depthToSpaceParamsTuple({3, 28, 5, 5},                                                 //
                            InferenceEngine::Precision::FP32,                              //
                            ngraph::opset1::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST,  //
                            2,                                                             //
                            CommonTestUtils::DEVICE_PLAIDML),
    depthToSpaceParamsTuple({3, 28, 5, 5},  //
                            InferenceEngine::Precision::FP32,
                            ngraph::opset1::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST,  //
                            2,                                                            //
                            CommonTestUtils::DEVICE_PLAIDML),
};

INSTANTIATE_TEST_CASE_P(Smoke, DepthToSpaceLayerTest, ::testing::ValuesIn(dts_only_test_cases),
                        DepthToSpaceLayerTest::getTestCaseName);

}  // namespace
