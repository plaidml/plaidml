// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/reorg_yolo.hpp"

using LayerTestsDefinitions::ReorgYoloLayerTest;

namespace {
const std::vector<ngraph::Shape> inShapes_caffe_yolov2 = {
    {1, 64, 26, 26},
};

const std::vector<ngraph::Shape> inShapes = {
    {1, 4, 4, 4},     //
    {1, 8, 4, 4},     //
    {1, 9, 3, 3},     //
    {1, 24, 34, 62},  //
    {2, 8, 4, 4},     //
};

const std::vector<size_t> strides = {
    2,
    3,
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const auto testCase_caffe_yolov2 = ::testing::Combine(  //
    ::testing::ValuesIn(inShapes_caffe_yolov2),         //
    ::testing::Values(strides[0]),                      //
    ::testing::ValuesIn(netPrecisions),                 //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)  //
);

const auto testCase_smallest = ::testing::Combine(      //
    ::testing::Values(inShapes[0]),                     //
    ::testing::Values(strides[0]),                      //
    ::testing::ValuesIn(netPrecisions),                 //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)  //
);

const auto testCase_stride_2 = ::testing::Combine(      //
    ::testing::Values(inShapes[1]),                     //
    ::testing::Values(strides[0]),                      //
    ::testing::ValuesIn(netPrecisions),                 //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)  //
);

const auto testCase_stride_3 = ::testing::Combine(      //
    ::testing::Values(inShapes[2]),                     //
    ::testing::Values(strides[1]),                      //
    ::testing::ValuesIn(netPrecisions),                 //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)  //
);

const auto testCase_smaller_h = ::testing::Combine(     //
    ::testing::Values(inShapes[4]),                     //
    ::testing::Values(strides[0]),                      //
    ::testing::ValuesIn(netPrecisions),                 //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)  //
);

const auto testCase_batch_2 = ::testing::Combine(       //
    ::testing::Values(inShapes[3]),                     //
    ::testing::Values(strides[0]),                      //
    ::testing::ValuesIn(netPrecisions),                 //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)  //
);

INSTANTIATE_TEST_CASE_P(TestsReorgYolo_caffe_YoloV2, ReorgYoloLayerTest, testCase_caffe_yolov2,
                        ReorgYoloLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(TestsReorgYolo_stride_2_smallest, ReorgYoloLayerTest, testCase_smallest,
                        ReorgYoloLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(TestsReorgYolo_stride_2, ReorgYoloLayerTest, testCase_stride_2,
                        ReorgYoloLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(TestsReorgYolo_stride_3, ReorgYoloLayerTest, testCase_stride_3,
                        ReorgYoloLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(TestsReorgYolo_smaller_h, ReorgYoloLayerTest, testCase_smaller_h,
                        ReorgYoloLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(TestsReorgYolo_batch_2, ReorgYoloLayerTest, testCase_batch_2,
                        ReorgYoloLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, ReorgYoloLayerTest,
                        ::testing::Combine(::testing::Values(inShapes[0]),                       //
                                           ::testing::Values(strides[0]),                        //
                                           ::testing::ValuesIn(netPrecisions),                   //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        ReorgYoloLayerTest::getTestCaseName);

}  // namespace
