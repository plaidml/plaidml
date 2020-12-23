// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/roi_align.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using LayerTestsDefinitions::ROIAlignLayerTest;

namespace {
std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> inputShapes = {{6, 3, 128, 128}, {7, 256, 200, 200}};
const std::vector<std::vector<float>> rois = {{{0.1, 0.2, 0.4, 0.3},  //
                                               {0.3, 0.4, 0.2, 0.7},  //
                                               {0.6, 0.5, 0.8, 0.8}}};
const std::vector<size_t> batchIndices = {2, 3, 5};
const std::vector<size_t> numROIs = {3};
const std::vector<size_t> pooledHs = {6};
const std::vector<size_t> pooledWs = {6};
const std::vector<size_t> samplingRatios = {0};
const std::vector<float> spatialScale = {16.0, 20.0};
const std::vector<std::string> modes = {"avg", "max"};

const auto roiAlignParams = ::testing::Combine(::testing::ValuesIn(inputShapes),     //
                                               ::testing::Values(rois),              //
                                               ::testing::Values(batchIndices),      //
                                               ::testing::ValuesIn(numROIs),         //
                                               ::testing::ValuesIn(pooledHs),        //
                                               ::testing::ValuesIn(pooledWs),        //
                                               ::testing::ValuesIn(samplingRatios),  //
                                               ::testing::ValuesIn(spatialScale),    //
                                               ::testing::ValuesIn(modes)            //
);

INSTANTIATE_TEST_CASE_P(ROIAlign, ROIAlignLayerTest,
                        ::testing::Combine(roiAlignParams,                                       //
                                           ::testing::ValuesIn(netPrecisions),                   //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        ROIAlignLayerTest::getTestCaseName);

const auto smokeRoiAlignParams = ::testing::Combine(::testing::Values(inputShapes[0]),  //
                                                    ::testing::Values(rois),            //
                                                    ::testing::Values(batchIndices),    //
                                                    ::testing::Values(2),               //
                                                    ::testing::Values(3),               //
                                                    ::testing::Values(3),               //
                                                    ::testing::Values(0),               //
                                                    ::testing::Values(20.0f),           //
                                                    ::testing::Values("max")            //
);

INSTANTIATE_TEST_CASE_P(smoke, ROIAlignLayerTest,
                        ::testing::Combine(smokeRoiAlignParams,                                  //
                                           ::testing::ValuesIn(netPrecisions),                   //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        ROIAlignLayerTest::getTestCaseName);
}  // namespace
