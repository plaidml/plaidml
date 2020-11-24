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

std::vector<InferenceEngine::Precision> indPrecisions = {
    InferenceEngine::Precision::I32,
};

const std::vector<InferenceEngine::SizeVector> inputShapes = {{7, 256, 200, 200}};
const std::vector<std::vector<std::vector<int64_t>>> rois = {{{2, 2, 4, 4}, {10, 11, 11, 12}}};
const std::vector<std::vector<int64_t>> batchIndices = {{2, 5}};
const std::vector<size_t> numROIs = {2};
const std::vector<size_t> pooledHs = {6};
const std::vector<size_t> pooledWs = {6};
const std::vector<size_t> samplingRatios = {2, 0};
const std::vector<float> spatialScale = {16.0};
const std::vector<std::string> modes = {"avg", "max"};

const auto roiAlignArgSet = ::testing::Combine(::testing::ValuesIn(inputShapes),     //
                                               ::testing::ValuesIn(rois),            //
                                               ::testing::ValuesIn(batchIndices),    //
                                               ::testing::ValuesIn(numROIs),         //
                                               ::testing::ValuesIn(pooledHs),        //
                                               ::testing::ValuesIn(pooledWs),        //
                                               ::testing::ValuesIn(samplingRatios),  //
                                               ::testing::ValuesIn(spatialScale),    //
                                               ::testing::ValuesIn(modes)            //
);

INSTANTIATE_TEST_CASE_P(ROIAlign, ROIAlignLayerTest,
                        ::testing::Combine(roiAlignArgSet,                                       //
                                           ::testing::ValuesIn(netPrecisions),                   //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        ROIAlignLayerTest::getTestCaseName);
}  // namespace
