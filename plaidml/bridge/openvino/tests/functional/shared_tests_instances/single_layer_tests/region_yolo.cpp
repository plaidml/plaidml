// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/region_yolo.hpp"

using namespace LayerTestsDefinitions;

const std::vector<ngraph::Shape> inShapes_caffe = {
    {1, 125, 13, 13},
};

const std::vector<ngraph::Shape> inShapes_mxnet = {
    {1, 75, 52, 52},  //
    {1, 75, 32, 32},  //
    {1, 75, 26, 26},  //
    {1, 75, 16, 16},  //
    {1, 75, 13, 13},  //
    {1, 75, 8, 8},    //
};

const std::vector<ngraph::Shape> inShapes_v3 = {
    {1, 255, 52, 52},
    {1, 255, 26, 26},
    {1, 255, 13, 13},
};

const std::vector<std::vector<int64_t>> masks = {
    {0, 1, 2},
    {3, 4, 5},
    {6, 7, 8},
};

const std::vector<bool> do_softmax = {
    true,
    false,
};

const std::vector<size_t> classes = {
    80,
    20,
};

const std::vector<size_t> num_regions = {
    5,
    9,
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const size_t coords = 4;
const int start_axis = 1;
const int end_axis = 3;

INSTANTIATE_TEST_CASE_P(TestsRegionYolov3, RegionYoloLayerTest,
                        ::testing::Combine(                                       //
                            ::testing::ValuesIn(inShapes_v3),                     //
                            ::testing::Values(classes[0]),                        //
                            ::testing::Values(coords),                            //
                            ::testing::Values(num_regions[1]),                    //
                            ::testing::Values(do_softmax[1]),                     //
                            ::testing::Values(masks[2]),                          //
                            ::testing::Values(start_axis),                        //
                            ::testing::Values(end_axis),                          //
                            ::testing::ValuesIn(netPrecisions),                   //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        RegionYoloLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(TestsRegionYoloMxnet, RegionYoloLayerTest,
                        ::testing::Combine(                                       //
                            ::testing::ValuesIn(inShapes_mxnet),                  //
                            ::testing::Values(classes[1]),                        //
                            ::testing::Values(coords),                            //
                            ::testing::Values(num_regions[1]),                    //
                            ::testing::Values(do_softmax[1]),                     //
                            ::testing::Values(masks[1]),                          //
                            ::testing::Values(start_axis),                        //
                            ::testing::Values(end_axis),                          //
                            ::testing::ValuesIn(netPrecisions),                   //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        RegionYoloLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(TestsRegionYoloCaffe, RegionYoloLayerTest,
                        ::testing::Combine(                                       //
                            ::testing::ValuesIn(inShapes_caffe),                  //
                            ::testing::Values(classes[1]),                        //
                            ::testing::Values(coords),                            //
                            ::testing::Values(num_regions[0]),                    //
                            ::testing::Values(do_softmax[0]),                     //
                            ::testing::Values(masks[0]),                          //
                            ::testing::Values(start_axis),                        //
                            ::testing::Values(end_axis),                          //
                            ::testing::ValuesIn(netPrecisions),                   //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        RegionYoloLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, RegionYoloLayerTest,
                        ::testing::Combine(                                       //
                            ::testing::ValuesIn(inShapes_caffe),                  //
                            ::testing::Values(classes[1]),                        //
                            ::testing::Values(coords),                            //
                            ::testing::Values(num_regions[0]),                    //
                            ::testing::Values(do_softmax[0]),                     //
                            ::testing::Values(masks[0]),                          //
                            ::testing::Values(start_axis),                        //
                            ::testing::Values(end_axis),                          //
                            ::testing::ValuesIn(netPrecisions),                   //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        RegionYoloLayerTest::getTestCaseName);
