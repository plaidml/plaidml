// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/roi_pooling.hpp"

using namespace LayerTestsDefinitions;

const std::vector<std::vector<size_t>> inShapes = {
    {1, 3, 8, 8},
    {3, 4, 50, 50},
};

const std::vector<std::vector<size_t>> pooledShapes_max = {
    {1, 1},
    {2, 2},
    {3, 3},
};

const std::vector<std::vector<size_t>> pooledShapes_bilinear = {
    {2, 2},
    {3, 3},
    {6, 6},
};

const std::vector<std::vector<size_t>> coordShapes = {
    {1, 5},
    {3, 5},
};

const std::vector<InferenceEngine::Precision> netPRCs = {
    InferenceEngine::Precision::FP32,
};

const std::vector<float> spatial_scales = {0.625f, 1.f};

const std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
    ngraph::helpers::InputLayerType::CONSTANT,
    //     ngraph::helpers::InputLayerType::PARAMETER,
};

const auto test_ROIPooling_max = ::testing::Combine(               //
    ::testing::ValuesIn(inShapes),                                 //
    ::testing::ValuesIn(coordShapes),                              //
    ::testing::ValuesIn(pooledShapes_max),                         //
    ::testing::ValuesIn(spatial_scales),                           //
    ::testing::Values(ngraph::helpers::ROIPoolingTypes::ROI_MAX),  //
    ::testing::ValuesIn(netPRCs),                                  //
    ::testing::ValuesIn(secondaryInputTypes),                      //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)             //
);

const auto test_ROIPooling_bilinear = ::testing::Combine(               //
    ::testing::ValuesIn(inShapes),                                      //
    ::testing::ValuesIn(coordShapes),                                   //
    ::testing::ValuesIn(pooledShapes_bilinear),                         //
    ::testing::Values(spatial_scales[1]),                               //
    ::testing::Values(ngraph::helpers::ROIPoolingTypes::ROI_BILINEAR),  //
    ::testing::ValuesIn(netPRCs),                                       //
    ::testing::ValuesIn(secondaryInputTypes),                           //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)                  //
);

const auto smoke_args = ::testing::Combine(                             //
    ::testing::Values(inShapes[0]),                                     //
    ::testing::Values(coordShapes[0]),                                  //
    ::testing::Values(pooledShapes_bilinear[0]),                        //
    ::testing::Values(spatial_scales[1]),                               //
    ::testing::Values(ngraph::helpers::ROIPoolingTypes::ROI_BILINEAR),  //
    ::testing::ValuesIn(netPRCs),                                       //
    ::testing::ValuesIn(secondaryInputTypes),                           //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)                  //
);

INSTANTIATE_TEST_CASE_P(ROIPooling_max, ROIPoolingLayerTest, test_ROIPooling_max, ROIPoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(ROIPooling_bilinear, ROIPoolingLayerTest, test_ROIPooling_bilinear,
                        ROIPoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(smoke, ROIPoolingLayerTest, smoke_args, ROIPoolingLayerTest::getTestCaseName);
