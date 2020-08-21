// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/convolution.hpp"
#include "single_layer_tests/convolution_backprop_data.hpp"
#include "single_layer_tests/group_convolution_backprop_data.hpp"

using LayerTestsDefinitions::ConvolutionBackpropDataLayerTest;
using LayerTestsDefinitions::ConvolutionLayerTest;
using LayerTestsDefinitions::GroupConvBackpropDataLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

/* ============= 2D Convolution ============= */
const std::vector<std::vector<size_t>> kernels = {{3, 5}};
const std::vector<std::vector<size_t>> strides = {{1, 3}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 3}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0}};
const std::vector<std::vector<size_t>> dilations = {{3, 1}};
const std::vector<size_t> numOutChannels = {5};

const auto conv2DParams_ExplicitPadding = ::testing::Combine(::testing::ValuesIn(kernels),                     //
                                                             ::testing::ValuesIn(strides),                     //
                                                             ::testing::ValuesIn(padBegins),                   //
                                                             ::testing::ValuesIn(padEnds),                     //
                                                             ::testing::ValuesIn(dilations),                   //
                                                             ::testing::ValuesIn(numOutChannels),              //
                                                             ::testing::Values(ngraph::op::PadType::EXPLICIT)  //
);
const auto conv2DParams_AutoPadValid = ::testing::Combine(::testing::ValuesIn(kernels),                       //
                                                          ::testing::ValuesIn(strides),                       //
                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  //
                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  //
                                                          ::testing::ValuesIn(dilations),                     //
                                                          ::testing::ValuesIn(numOutChannels),                //
                                                          ::testing::Values(ngraph::op::PadType::VALID)       //
);

INSTANTIATE_TEST_SUITE_P(Convolution_Smoke_2D_ExplicitPadding, ConvolutionLayerTest,
                         ::testing::Combine(conv2DParams_ExplicitPadding,                            //
                                            ::testing::ValuesIn(netPrecisions),                      //
                                            ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),  //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),     //
                         ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Convolution_Smoke_2D_AutoPadValid, ConvolutionLayerTest,
                         ::testing::Combine(conv2DParams_AutoPadValid,  //
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),  //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),     //
                         ConvolutionLayerTest::getTestCaseName);
/* ============= 3D Convolution ============= */
const std::vector<std::vector<size_t>> kernels3d = {{3, 5, 3}};
const std::vector<std::vector<ptrdiff_t>> paddings3d = {{0, 2, 0}};

const std::vector<std::vector<size_t>> strides3d = {{1, 2, 1}};
const std::vector<std::vector<size_t>> dilations3d = {{1, 2, 1}};

const auto conv3DParams_ExplicitPadding = ::testing::Combine(::testing::ValuesIn(kernels3d),                   //
                                                             ::testing::ValuesIn(strides3d),                   //
                                                             ::testing::ValuesIn(paddings3d),                  //
                                                             ::testing::ValuesIn(paddings3d),                  //
                                                             ::testing::ValuesIn(dilations3d),                 //
                                                             ::testing::Values(5),                             //
                                                             ::testing::Values(ngraph::op::PadType::EXPLICIT)  //
);
const auto conv3DParams_AutoPadValid = ::testing::Combine(::testing::ValuesIn(kernels3d),                        //
                                                          ::testing::ValuesIn(strides3d),                        //
                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),  //
                                                          ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),  //
                                                          ::testing::ValuesIn(dilations3d),                      //
                                                          ::testing::Values(5),                                  //
                                                          ::testing::Values(ngraph::op::PadType::VALID)          //
);

INSTANTIATE_TEST_SUITE_P(Convolution_Smoke_3D_ExplicitPadding, ConvolutionLayerTest,
                         ::testing::Combine(conv3DParams_ExplicitPadding,                                //
                                            ::testing::ValuesIn(netPrecisions),                          //
                                            ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),  //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
                         ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Convolution_Smoke_3D_AutoPadValid, ConvolutionLayerTest,
                         ::testing::Combine(conv3DParams_AutoPadValid,                                   //
                                            ::testing::ValuesIn(netPrecisions),                          //
                                            ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),  //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
                         ConvolutionLayerTest::getTestCaseName);

/* ============= 2D ConvolutionBackpropData ============= */
const std::vector<std::vector<size_t>> inputShapes2D = {{1, 3, 30, 30}};
const std::vector<std::vector<size_t>> kernels2D = {{3, 5}};
const std::vector<std::vector<size_t>> strides2D = {{1, 3}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {{1, 1}};
const std::vector<std::vector<size_t>> dilations2D = {{1, 2}};

const auto convbprop2DParams_ExplicitPadding = ::testing::Combine(::testing::ValuesIn(kernels2D),       //
                                                                  ::testing::ValuesIn(strides2D),       //
                                                                  ::testing::ValuesIn(padBegins2D),     //
                                                                  ::testing::ValuesIn(padEnds2D),       //
                                                                  ::testing::ValuesIn(dilations2D),     //
                                                                  ::testing::ValuesIn(numOutChannels),  //
                                                                  ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto convbprop2DParams_AutoPadValid = ::testing::Combine(::testing::ValuesIn(kernels2D),                     //
                                                               ::testing::ValuesIn(strides2D),                     //
                                                               ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  //
                                                               ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  //
                                                               ::testing::ValuesIn(dilations2D),                   //
                                                               ::testing::ValuesIn(numOutChannels),                //
                                                               ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_SUITE_P(ConvolutionBackpropData_Smoke_2D_ExplicitPadding, ConvolutionBackpropDataLayerTest,
                         ::testing::Combine(convbprop2DParams_ExplicitPadding,   //
                                            ::testing::ValuesIn(netPrecisions),  //
                                            ::testing::ValuesIn(inputShapes2D),  //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                         ConvolutionBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ConvolutionBackpropData_Smoke_2D_AutoPadValid, ConvolutionBackpropDataLayerTest,
                         ::testing::Combine(convbprop2DParams_AutoPadValid,      //
                                            ::testing::ValuesIn(netPrecisions),  //
                                            ::testing::ValuesIn(inputShapes2D),  //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                         ConvolutionBackpropDataLayerTest::getTestCaseName);

/* ============= 3D ConvolutionBackpropData ============= */
const std::vector<std::vector<size_t>> inputShapes3D = {{1, 16, 5, 5, 5}};
const std::vector<std::vector<size_t>> kernels3D = {{3, 3, 3}};
const std::vector<std::vector<size_t>> strides3D = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins3D = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds3D = {{1, 1, 1}};
const std::vector<std::vector<size_t>> dilations3D = {{2, 2, 2}};

const auto convbprop3DParams_ExplicitPadding = ::testing::Combine(::testing::ValuesIn(kernels3D),       //
                                                                  ::testing::ValuesIn(strides3D),       //
                                                                  ::testing::ValuesIn(padBegins3D),     //
                                                                  ::testing::ValuesIn(padEnds3D),       //
                                                                  ::testing::ValuesIn(dilations3D),     //
                                                                  ::testing::ValuesIn(numOutChannels),  //
                                                                  ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto convbrop3DParams_AutoPadValid = ::testing::Combine(::testing::ValuesIn(kernels3D),                        //
                                                              ::testing::ValuesIn(strides3D),                        //
                                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),  //
                                                              ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),  //
                                                              ::testing::ValuesIn(dilations3D),                      //
                                                              ::testing::ValuesIn(numOutChannels),                   //
                                                              ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_SUITE_P(ConvolutionBackpropData_Smoke_3D_ExplicitPadding, ConvolutionBackpropDataLayerTest,
                         ::testing::Combine(convbprop3DParams_ExplicitPadding,   //
                                            ::testing::ValuesIn(netPrecisions),  //
                                            ::testing::ValuesIn(inputShapes3D),  //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                         ConvolutionBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ConvolutionBackpropData_Smoke_3D_AutoPadValid, ConvolutionBackpropDataLayerTest,
                         ::testing::Combine(convbrop3DParams_AutoPadValid,       //
                                            ::testing::ValuesIn(netPrecisions),  //
                                            ::testing::ValuesIn(inputShapes3D),  //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                         ConvolutionBackpropDataLayerTest::getTestCaseName);

const std::vector<size_t> groupNumOutChannels = {16};
const std::vector<size_t> numGroups = {8};

/* ============= 2D GroupConvolution ============= */
const std::vector<std::vector<size_t>> groupInputShapes2D = {{1, 32, 10, 10}};

const auto groupConvBackpropData2DParams_ExplicitPadding =
    ::testing::Combine(::testing::ValuesIn(kernels2D),                     //
                       ::testing::ValuesIn(strides2D),                     //
                       ::testing::ValuesIn(padBegins2D),                   //
                       ::testing::ValuesIn(padEnds2D),                     //
                       ::testing::ValuesIn(dilations2D),                   //
                       ::testing::ValuesIn(groupNumOutChannels),           //
                       ::testing::ValuesIn(numGroups),                     //
                       ::testing::Values(ngraph::op::PadType::EXPLICIT));  //
const auto groupConvBackpropData2DParams_AutoPadValid =
    ::testing::Combine(::testing::ValuesIn(kernels2D),                     //
                       ::testing::ValuesIn(strides2D),                     //
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  //
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  //
                       ::testing::ValuesIn(dilations2D),                   //
                       ::testing::ValuesIn(groupNumOutChannels),           //
                       ::testing::ValuesIn(numGroups),                     //
                       ::testing::Values(ngraph::op::PadType::VALID));     //

INSTANTIATE_TEST_SUITE_P(GroupConvBackpropData_Smoke_2D_ExplicitPadding, GroupConvBackpropDataLayerTest,
                         ::testing::Combine(groupConvBackpropData2DParams_ExplicitPadding,        //
                                            ::testing::ValuesIn(netPrecisions),                   //
                                            ::testing::ValuesIn(groupInputShapes2D),              //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                         GroupConvBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupConvBackpropData_Smoke_2D_AutoPadValid, GroupConvBackpropDataLayerTest,
                         ::testing::Combine(groupConvBackpropData2DParams_AutoPadValid,           //
                                            ::testing::ValuesIn(netPrecisions),                   //
                                            ::testing::ValuesIn(groupInputShapes2D),              //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                         GroupConvBackpropDataLayerTest::getTestCaseName);

/* ============= 3D GroupConvolution ============= */

const auto groupConvBackpropData3DParams_ExplicitPadding =
    ::testing::Combine(::testing::ValuesIn(kernels3D),                     //
                       ::testing::ValuesIn(strides3D),                     //
                       ::testing::ValuesIn(padBegins3D),                   //
                       ::testing::ValuesIn(padEnds3D),                     //
                       ::testing::ValuesIn(dilations3D),                   //
                       ::testing::ValuesIn(groupNumOutChannels),           //
                       ::testing::ValuesIn(numGroups),                     //
                       ::testing::Values(ngraph::op::PadType::EXPLICIT));  //
const auto groupConvBackpropData3DParams_AutoPadValid =
    ::testing::Combine(::testing::ValuesIn(kernels3D),                        //
                       ::testing::ValuesIn(strides3D),                        //
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),  //
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),  //
                       ::testing::ValuesIn(dilations3D),                      //
                       ::testing::ValuesIn(groupNumOutChannels),              //
                       ::testing::ValuesIn(numGroups),                        //
                       ::testing::Values(ngraph::op::PadType::VALID));        //

INSTANTIATE_TEST_SUITE_P(GroupConvBackpropData_Smoke_3D_ExplicitPadding, GroupConvBackpropDataLayerTest,
                         ::testing::Combine(groupConvBackpropData3DParams_ExplicitPadding,        //
                                            ::testing::ValuesIn(netPrecisions),                   //
                                            ::testing::ValuesIn(inputShapes3D),                   //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                         GroupConvBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(GroupConvBackpropData_Smoke_3D_AutoPadValid, GroupConvBackpropDataLayerTest,
                         ::testing::Combine(groupConvBackpropData3DParams_AutoPadValid,           //
                                            ::testing::ValuesIn(netPrecisions),                   //
                                            ::testing::ValuesIn(inputShapes3D),                   //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                         GroupConvBackpropDataLayerTest::getTestCaseName);

}  // namespace
