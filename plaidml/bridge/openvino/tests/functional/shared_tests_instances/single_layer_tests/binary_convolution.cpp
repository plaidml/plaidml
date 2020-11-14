// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/binary_convolution.hpp"

using LayerTestsDefinitions::BinaryConvolutionLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::FP32,
};

/* ============= 2D Convolution ============= */
const std::vector<std::vector<size_t>> kernels = {{3, 3}, {3, 5}};
const std::vector<std::vector<size_t>> strides = {{1, 1}, {1, 3}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}, {0, 3}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0}, {0, 3}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}, {3, 1}};
const std::vector<size_t> numOutChannels = {1, 5};
const std::vector<ngraph::op::PadType> padTypes = {
    ngraph::op::PadType::EXPLICIT,  //
    ngraph::op::PadType::VALID      //
};

const auto conv2DParams_ExplicitPadding =
    ::testing::Combine(::testing::ValuesIn(kernels),                                                                //
                       ::testing::ValuesIn(strides),                                                                //
                       ::testing::ValuesIn(padBegins),                                                              //
                       ::testing::ValuesIn(padEnds),                                                                //
                       ::testing::ValuesIn(dilations),                                                              //
                       ::testing::ValuesIn(numOutChannels),                                                         //
                       ::testing::Values(ngraph::op::PadType::EXPLICIT),                                            //
                       ::testing::Values(ngraph::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT),  //
                       ::testing::Values(0.0));                                                                     //
const auto conv2DParams_AutoPadValid =
    ::testing::Combine(::testing::ValuesIn(kernels),                                                                //
                       ::testing::ValuesIn(strides),                                                                //
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),                                           //
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),                                           //
                       ::testing::ValuesIn(dilations),                                                              //
                       ::testing::ValuesIn(numOutChannels),                                                         //
                       ::testing::Values(ngraph::op::PadType::VALID),                                               //
                       ::testing::Values(ngraph::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT),  //
                       ::testing::Values(0.0));                                                                     //

INSTANTIATE_TEST_CASE_P(Convolution2D_ExplicitPadding, BinaryConvolutionLayerTest,
                        ::testing::Combine(conv2DParams_ExplicitPadding,                            //
                                           ::testing::ValuesIn(netPrecisions),                      //
                                           ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),  //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),     //
                        BinaryConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Convolution2D_AutoPadValid, BinaryConvolutionLayerTest,
                        ::testing::Combine(conv2DParams_AutoPadValid,  //
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),  //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),     //
                        BinaryConvolutionLayerTest::getTestCaseName);

/* ============= 3D Convolution ============= */
const std::vector<std::vector<size_t>> kernels3d = {{3, 3, 3}, {3, 5, 3}};
const std::vector<std::vector<ptrdiff_t>> paddings3d = {{0, 0, 0}, {0, 2, 0}};

const std::vector<std::vector<size_t>> strides3d = {{1, 1, 1}, {1, 2, 1}};
const std::vector<std::vector<size_t>> dilations3d = {{1, 1, 1}, {1, 2, 1}};

const auto conv3DParams_ExplicitPadding =
    ::testing::Combine(::testing::ValuesIn(kernels3d),                                                              //
                       ::testing::ValuesIn(strides3d),                                                              //
                       ::testing::ValuesIn(paddings3d),                                                             //
                       ::testing::ValuesIn(paddings3d),                                                             //
                       ::testing::ValuesIn(dilations3d),                                                            //
                       ::testing::Values(5),                                                                        //
                       ::testing::Values(ngraph::op::PadType::EXPLICIT),                                            //
                       ::testing::Values(ngraph::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT),  //
                       ::testing::Values(0.0));                                                                     //
const auto conv3DParams_AutoPadValid =
    ::testing::Combine(::testing::ValuesIn(kernels3d),                                                              //
                       ::testing::ValuesIn(strides3d),                                                              //
                       ::testing::Values(std::vector<ptrdiff_t>({0, 1, 0})),                                        //
                       ::testing::Values(std::vector<ptrdiff_t>({0, 1, 0})),                                        //
                       ::testing::ValuesIn(dilations3d),                                                            //
                       ::testing::Values(5),                                                                        //
                       ::testing::Values(ngraph::op::PadType::VALID),                                               //
                       ::testing::Values(ngraph::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT),  //
                       ::testing::Values(0.0));                                                                     //

INSTANTIATE_TEST_CASE_P(Convolution3D_ExplicitPadding, BinaryConvolutionLayerTest,
                        ::testing::Combine(conv3DParams_ExplicitPadding,                                //
                                           ::testing::ValuesIn(netPrecisions),                          //
                                           ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),  //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
                        BinaryConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Convolution3D_AutoPadValid, BinaryConvolutionLayerTest,
                        ::testing::Combine(conv3DParams_AutoPadValid,                                   //
                                           ::testing::ValuesIn(netPrecisions),                          //
                                           ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),  //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
                        BinaryConvolutionLayerTest::getTestCaseName);

/* ============= Smoke ============= */
const std::vector<std::vector<size_t>> kernel_smoke = {{3, 5, 3}};
const std::vector<std::vector<ptrdiff_t>> padding_smoke = {{0, 2, 0}};

const std::vector<std::vector<size_t>> strides_smoke = {{1, 2, 1}};
const std::vector<std::vector<size_t>> dilations_smoke = {{1, 2, 1}};

const auto convSmokeParams_ExplicitPadding =
    ::testing::Combine(::testing::ValuesIn(kernel_smoke),                                                           //
                       ::testing::ValuesIn(strides_smoke),                                                          //
                       ::testing::ValuesIn(padding_smoke),                                                          //
                       ::testing::ValuesIn(padding_smoke),                                                          //
                       ::testing::ValuesIn(dilations_smoke),                                                        //
                       ::testing::Values(5),                                                                        //
                       ::testing::Values(ngraph::op::PadType::EXPLICIT),                                            //
                       ::testing::Values(ngraph::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT),  //
                       ::testing::Values(0.0));                                                                     //
const auto convSmokeParams_AutoPadValid =
    ::testing::Combine(::testing::ValuesIn(kernel_smoke),                                                           //
                       ::testing::ValuesIn(strides_smoke),                                                          //
                       ::testing::Values(std::vector<ptrdiff_t>({0, 1, 0})),                                        //
                       ::testing::Values(std::vector<ptrdiff_t>({0, 1, 0})),                                        //
                       ::testing::ValuesIn(dilations_smoke),                                                        //
                       ::testing::Values(5),                                                                        //
                       ::testing::Values(ngraph::op::PadType::VALID),                                               //
                       ::testing::Values(ngraph::op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT),  //
                       ::testing::Values(0.0));                                                                     //

INSTANTIATE_TEST_CASE_P(smoke_ExplicitPadding, BinaryConvolutionLayerTest,
                        ::testing::Combine(convSmokeParams_ExplicitPadding,                             //
                                           ::testing::ValuesIn(netPrecisions),                          //
                                           ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),  //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
                        BinaryConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_AutoPadValid, BinaryConvolutionLayerTest,
                        ::testing::Combine(convSmokeParams_AutoPadValid,                                //
                                           ::testing::ValuesIn(netPrecisions),                          //
                                           ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),  //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
                        BinaryConvolutionLayerTest::getTestCaseName);

}  // namespace
