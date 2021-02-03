// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/group_convolution_backprop_data.hpp"

using LayerTestsDefinitions::GroupConvBackpropDataLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<size_t> numOutChannels = {16, 32};
const std::vector<size_t> numGroups = {2, 8, 16};

/* ============= 2D GroupConvolution ============= */
const std::vector<std::vector<size_t>> inputShapes2D = {
    {1, 16, 10, 10},
    {1, 32, 10, 10},
};
const std::vector<std::vector<size_t>> kernels2D = {
    {1, 1},
    {3, 3},
};
const std::vector<std::vector<size_t>> strides2D = {
    {1, 1},
};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {
    {0, 0},
};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {
    {0, 0},
};
const std::vector<std::vector<size_t>> dilations2D = {
    {1, 1},
};

const auto groupConvBackpropData2DParams_ExplicitPadding =
    ::testing::Combine(::testing::ValuesIn(kernels2D),                     //
                       ::testing::ValuesIn(strides2D),                     //
                       ::testing::ValuesIn(padBegins2D),                   //
                       ::testing::ValuesIn(padEnds2D),                     //
                       ::testing::ValuesIn(dilations2D),                   //
                       ::testing::ValuesIn(numOutChannels),                //
                       ::testing::ValuesIn(numGroups),                     //
                       ::testing::Values(ngraph::op::PadType::EXPLICIT));  //
const auto groupConvBackpropData2DParams_AutoPadValid =
    ::testing::Combine(::testing::ValuesIn(kernels2D),                     //
                       ::testing::ValuesIn(strides2D),                     //
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  //
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),  //
                       ::testing::ValuesIn(dilations2D),                   //
                       ::testing::ValuesIn(numOutChannels),                //
                       ::testing::ValuesIn(numGroups),                     //
                       ::testing::Values(ngraph::op::PadType::VALID));     //

INSTANTIATE_TEST_CASE_P(GroupConvBackpropData2D_ExplicitPadding, GroupConvBackpropDataLayerTest,
                        ::testing::Combine(groupConvBackpropData2DParams_ExplicitPadding,               //
                                           ::testing::ValuesIn(netPrecisions),                          //
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                                           ::testing::Values(InferenceEngine::Layout::ANY),             //
                                           ::testing::Values(InferenceEngine::Layout::ANY),             //
                                           ::testing::ValuesIn(inputShapes2D),                          //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
                        GroupConvBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(GroupConvBackpropData2D_AutoPadValid, GroupConvBackpropDataLayerTest,
                        ::testing::Combine(groupConvBackpropData2DParams_AutoPadValid,                  //
                                           ::testing::ValuesIn(netPrecisions),                          //
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                                           ::testing::Values(InferenceEngine::Layout::ANY),             //
                                           ::testing::Values(InferenceEngine::Layout::ANY),             //
                                           ::testing::ValuesIn(inputShapes2D),                          //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
                        GroupConvBackpropDataLayerTest::getTestCaseName);

/* ============= 3D GroupConvolution ============= */
const std::vector<std::vector<size_t>> inputShapes3D = {
    {1, 16, 5, 5, 5},
    {1, 32, 5, 5, 5},
};
const std::vector<std::vector<size_t>> kernels3D = {
    {1, 1, 1},
    {3, 3, 3},
};
const std::vector<std::vector<size_t>> strides3D = {
    {1, 1, 1},
};
const std::vector<std::vector<ptrdiff_t>> padBegins3D = {
    {0, 0, 0},
};
const std::vector<std::vector<ptrdiff_t>> padEnds3D = {
    {0, 0, 0},
};
const std::vector<std::vector<size_t>> dilations3D = {
    {1, 1, 1},
};

const auto groupConvBackpropData3DParams_ExplicitPadding =
    ::testing::Combine(::testing::ValuesIn(kernels3D),                     //
                       ::testing::ValuesIn(strides3D),                     //
                       ::testing::ValuesIn(padBegins3D),                   //
                       ::testing::ValuesIn(padEnds3D),                     //
                       ::testing::ValuesIn(dilations3D),                   //
                       ::testing::ValuesIn(numOutChannels),                //
                       ::testing::ValuesIn(numGroups),                     //
                       ::testing::Values(ngraph::op::PadType::EXPLICIT));  //
const auto groupConvBackpropData3DParams_AutoPadValid =
    ::testing::Combine(::testing::ValuesIn(kernels3D),                        //
                       ::testing::ValuesIn(strides3D),                        //
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),  //
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),  //
                       ::testing::ValuesIn(dilations3D),                      //
                       ::testing::ValuesIn(numOutChannels),                   //
                       ::testing::ValuesIn(numGroups),                        //
                       ::testing::Values(ngraph::op::PadType::VALID));        //

INSTANTIATE_TEST_CASE_P(GroupConvBackpropData3D_ExplicitPadding, GroupConvBackpropDataLayerTest,
                        ::testing::Combine(groupConvBackpropData3DParams_ExplicitPadding,               //
                                           ::testing::ValuesIn(netPrecisions),                          //
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                                           ::testing::Values(InferenceEngine::Layout::ANY),             //
                                           ::testing::Values(InferenceEngine::Layout::ANY),             //
                                           ::testing::ValuesIn(inputShapes3D),                          //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
                        GroupConvBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(GroupConvBackpropData3D_AutoPadValid, GroupConvBackpropDataLayerTest,
                        ::testing::Combine(groupConvBackpropData3DParams_AutoPadValid,                  //
                                           ::testing::ValuesIn(netPrecisions),                          //
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                                           ::testing::Values(InferenceEngine::Layout::ANY),             //
                                           ::testing::Values(InferenceEngine::Layout::ANY),             //
                                           ::testing::ValuesIn(inputShapes3D),                          //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
                        GroupConvBackpropDataLayerTest::getTestCaseName);

/* ============= 2D GroupConvolution Smoke Tests============= */
const std::vector<size_t> smoke_groupNumOutChannels = {16};
const std::vector<size_t> smoke_numGroups = {8};

const std::vector<std::vector<size_t>> smoke_groupInputShapes2D = {{1, 32, 10, 10}};
const std::vector<std::vector<size_t>> smoke_kernels2D = {{3, 5}};
const std::vector<std::vector<size_t>> smoke_strides2D = {{1, 3}};
const std::vector<std::vector<ptrdiff_t>> smoke_padBegins2D = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> smoke_padEnds2D = {{1, 1}};
const std::vector<std::vector<size_t>> smoke_dilations2D = {{1, 2}};
const std::vector<size_t> smoke_numOutChannels = {5};

const auto smoke_groupConvBackpropData2DParams_ExplicitPadding = ::testing::Combine(  //
    ::testing::ValuesIn(smoke_kernels2D),                                             //
    ::testing::ValuesIn(smoke_strides2D),                                             //
    ::testing::ValuesIn(smoke_padBegins2D),                                           //
    ::testing::ValuesIn(smoke_padEnds2D),                                             //
    ::testing::ValuesIn(smoke_dilations2D),                                           //
    ::testing::ValuesIn(smoke_groupNumOutChannels),                                   //
    ::testing::ValuesIn(smoke_numGroups),                                             //
    ::testing::Values(ngraph::op::PadType::EXPLICIT)                                  //
);
const auto smoke_groupConvBackpropData2DParams_AutoPadValid = ::testing::Combine(  //
    ::testing::ValuesIn(smoke_kernels2D),                                          //
    ::testing::ValuesIn(smoke_strides2D),                                          //
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),                             //
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),                             //
    ::testing::ValuesIn(smoke_dilations2D),                                        //
    ::testing::ValuesIn(smoke_groupNumOutChannels),                                //
    ::testing::ValuesIn(smoke_numGroups),                                          //
    ::testing::Values(ngraph::op::PadType::VALID)                                  //
);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvBackpropData2D_ExplicitPadding, GroupConvBackpropDataLayerTest,
                        ::testing::Combine(                                              //
                            smoke_groupConvBackpropData2DParams_ExplicitPadding,         //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::ValuesIn(smoke_groupInputShapes2D),               //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        GroupConvBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvBackpropData2D_AutoPadValid, GroupConvBackpropDataLayerTest,
                        ::testing::Combine(                                              //
                            smoke_groupConvBackpropData2DParams_AutoPadValid,            //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::ValuesIn(smoke_groupInputShapes2D),               //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        GroupConvBackpropDataLayerTest::getTestCaseName);

/* ============= 3D GroupConvolution Smoke Tests============= */
const std::vector<std::vector<size_t>> smoke_inputShapes3D = {{1, 16, 5, 5, 5}};
const std::vector<std::vector<size_t>> smoke_kernels3D = {{3, 3, 3}};
const std::vector<std::vector<size_t>> smoke_strides3D = {{1, 1, 1}};
const std::vector<std::vector<ptrdiff_t>> smoke_padBegins3D = {{0, 0, 0}};
const std::vector<std::vector<ptrdiff_t>> smoke_padEnds3D = {{1, 1, 1}};
const std::vector<std::vector<size_t>> smoke_dilations3D = {{2, 2, 2}};

const auto smoke_groupConvBackpropData3DParams_ExplicitPadding = ::testing::Combine(  //
    ::testing::ValuesIn(smoke_kernels3D),                                             //
    ::testing::ValuesIn(smoke_strides3D),                                             //
    ::testing::ValuesIn(smoke_padBegins3D),                                           //
    ::testing::ValuesIn(smoke_padEnds3D),                                             //
    ::testing::ValuesIn(smoke_dilations3D),                                           //
    ::testing::ValuesIn(smoke_groupNumOutChannels),                                   //
    ::testing::ValuesIn(smoke_numGroups),                                             //
    ::testing::Values(ngraph::op::PadType::EXPLICIT)                                  //
);
const auto smoke_groupConvBackpropData3DParams_AutoPadValid = ::testing::Combine(  //
    ::testing::ValuesIn(smoke_kernels3D),                                          //
    ::testing::ValuesIn(smoke_strides3D),                                          //
    ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),                          //
    ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),                          //
    ::testing::ValuesIn(smoke_dilations3D),                                        //
    ::testing::ValuesIn(smoke_groupNumOutChannels),                                //
    ::testing::ValuesIn(smoke_numGroups),                                          //
    ::testing::Values(ngraph::op::PadType::VALID)                                  //
);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvBackpropData3D_ExplicitPadding, GroupConvBackpropDataLayerTest,
                        ::testing::Combine(                                              //
                            smoke_groupConvBackpropData3DParams_ExplicitPadding,         //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::ValuesIn(smoke_inputShapes3D),                    //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        GroupConvBackpropDataLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_GroupConvBackpropData3D_AutoPadValid, GroupConvBackpropDataLayerTest,
                        ::testing::Combine(                                              //
                            smoke_groupConvBackpropData3DParams_AutoPadValid,            //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::ValuesIn(smoke_inputShapes3D),                    //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        GroupConvBackpropDataLayerTest::getTestCaseName);

}  // namespace
