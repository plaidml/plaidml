// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/interpolate.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> inShapes = {
    {1, 1, 15, 20},
};

const std::vector<std::vector<size_t>> targetShapes = {
    {1, 1, 20, 30},
};

const std::vector<ngraph::op::v4::Interpolate::InterpolateMode> modesWithoutNearest = {
    // ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx,
    ngraph::op::v4::Interpolate::InterpolateMode::linear,  //
    ngraph::op::v4::Interpolate::InterpolateMode::cubic,   //
};

const std::vector<ngraph::op::v4::Interpolate::InterpolateMode> nearestMode = {
    ngraph::op::v4::Interpolate::InterpolateMode::nearest,
};

const std::vector<ngraph::op::v4::Interpolate::CoordinateTransformMode> coordinateTransformModes = {
    ngraph::op::v4::Interpolate::CoordinateTransformMode::tf_half_pixel_for_nn,
    ngraph::op::v4::Interpolate::CoordinateTransformMode::pytorch_half_pixel,
    ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel,
    ngraph::op::v4::Interpolate::CoordinateTransformMode::asymmetric,
    ngraph::op::v4::Interpolate::CoordinateTransformMode::align_corners,
};

const std::vector<ngraph::op::v4::Interpolate::ShapeCalcMode> shapeCalculationMode = {
    ngraph::op::v4::Interpolate::ShapeCalcMode::sizes,
    ngraph::op::v4::Interpolate::ShapeCalcMode::scales,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> nearestModes = {
    ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor,
    ngraph::op::v4::Interpolate::NearestMode::floor,
    ngraph::op::v4::Interpolate::NearestMode::round_prefer_ceil,
    ngraph::op::v4::Interpolate::NearestMode::ceil,
    ngraph::op::v4::Interpolate::NearestMode::simple,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> defaultNearestMode = {
    ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor,
};

const std::vector<std::vector<size_t>> pads = {
    {0, 0, 1, 0},
    {0, 0, 0, 0},
};

const std::vector<bool> antialias = {
    // Not enabled in Inference Engine
    //        true,
    false,
};

const std::vector<double> cubeCoefs = {
    -0.75f,
};

const std::vector<std::vector<int64_t>> defaultAxes = {
    // nGraph reference implementation does not support partial axes
    {0, 1, 2, 3},
};

const std::vector<std::vector<float>> defaultScales = {
    {1.0f, 1.0f, 1.33333f, 1.33333f},
};

const std::map<std::string, std::string> additional_config = {};

const auto interpolateCasesWithoutNearest = ::testing::Combine(  //
    ::testing::ValuesIn(modesWithoutNearest),                    //
    ::testing::ValuesIn(shapeCalculationMode),                   //
    ::testing::ValuesIn(coordinateTransformModes),               //
    ::testing::ValuesIn(defaultNearestMode),                     //
    ::testing::ValuesIn(antialias),                              //
    ::testing::ValuesIn(pads),                                   //
    ::testing::ValuesIn(pads),                                   //
    ::testing::ValuesIn(cubeCoefs),                              //
    ::testing::ValuesIn(defaultAxes),                            //
    ::testing::ValuesIn(defaultScales)                           //
);

const auto interpolateCases = ::testing::Combine(   //
    ::testing::ValuesIn(nearestMode),               //
    ::testing::ValuesIn(shapeCalculationMode),      //
    ::testing::ValuesIn(coordinateTransformModes),  //
    ::testing::ValuesIn(nearestModes),              //
    ::testing::ValuesIn(antialias),                 //
    ::testing::ValuesIn(pads),                      //
    ::testing::ValuesIn(pads),                      //
    ::testing::ValuesIn(cubeCoefs),                 //
    ::testing::ValuesIn(defaultAxes),               //
    ::testing::ValuesIn(defaultScales)              //
);

INSTANTIATE_TEST_CASE_P(Interpolate_Basic, InterpolateLayerTest,
                        ::testing::Combine(                                              //
                            interpolateCasesWithoutNearest,                              //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::ValuesIn(inShapes),                               //
                            ::testing::ValuesIn(targetShapes),                           //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),          //
                            ::testing::Values(additional_config)),                       //
                        InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Interpolate_Nearest, InterpolateLayerTest,
                        ::testing::Combine(                                              //
                            interpolateCases,                                            //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::ValuesIn(inShapes),                               //
                            ::testing::ValuesIn(targetShapes),                           //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),          //
                            ::testing::Values(additional_config)),                       //
                        InterpolateLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> targetShapesTailTest = {
    {1, 4, 10, 41},  // 10 * 41 is not multipler of 4, cover tail process code path
};

const std::vector<std::vector<float>> defaultScalesTailTest = {
    {1.0f, 1.0f, 0.33333f, 1.36666f},
};

const auto interpolateCasesWithoutNearestTail = ::testing::Combine(  //
    ::testing::ValuesIn(modesWithoutNearest),                        //
    ::testing::ValuesIn(shapeCalculationMode),                       //
    ::testing::ValuesIn(coordinateTransformModes),                   //
    ::testing::ValuesIn(defaultNearestMode),                         //
    ::testing::ValuesIn(antialias),                                  //
    ::testing::ValuesIn(pads),                                       //
    ::testing::ValuesIn(pads),                                       //
    ::testing::ValuesIn(cubeCoefs),                                  //
    ::testing::ValuesIn(defaultAxes),                                //
    ::testing::ValuesIn(defaultScalesTailTest)                       //
);

const auto interpolateCasesTail = ::testing::Combine(  //
    ::testing::ValuesIn(nearestMode),                  //
    ::testing::ValuesIn(shapeCalculationMode),         //
    ::testing::ValuesIn(coordinateTransformModes),     //
    ::testing::ValuesIn(nearestModes),                 //
    ::testing::ValuesIn(antialias),                    //
    ::testing::ValuesIn(pads),                         //
    ::testing::ValuesIn(pads),                         //
    ::testing::ValuesIn(cubeCoefs),                    //
    ::testing::ValuesIn(defaultAxes),                  //
    ::testing::ValuesIn(defaultScalesTailTest)         //
);

INSTANTIATE_TEST_CASE_P(Interpolate_Basic_2, InterpolateLayerTest,
                        ::testing::Combine(                                              //
                            interpolateCasesWithoutNearestTail,                          //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::ValuesIn(inShapes),                               //
                            ::testing::ValuesIn(targetShapesTailTest),                   //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),          //
                            ::testing::Values(additional_config)),                       //
                        InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Interpolate_Nearest_2, InterpolateLayerTest,
                        ::testing::Combine(                                              //
                            interpolateCasesTail,                                        //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::ValuesIn(inShapes),                               //
                            ::testing::ValuesIn(targetShapesTailTest),                   //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),          //
                            ::testing::Values(additional_config)),                       //
                        InterpolateLayerTest::getTestCaseName);

const auto smokeArgs = ::testing::Combine(                                                //
    ::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::linear),              //
    ::testing::Values(ngraph::op::v4::Interpolate::ShapeCalcMode::sizes),                 //
    ::testing::Values(ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel),  //
    ::testing::Values(ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor),      //
    ::testing::Values(false),                                                             //
    ::testing::Values(std::vector<size_t>({0, 0, 0, 0})),                                 //
    ::testing::Values(std::vector<size_t>({0, 0, 0, 0})),                                 //
    ::testing::Values(-0.5f),                                                             //
    ::testing::ValuesIn(defaultAxes),                                                     //
    ::testing::ValuesIn(defaultScales)                                                    //
);

INSTANTIATE_TEST_CASE_P(smoke, InterpolateLayerTest,
                        ::testing::Combine(                                              //
                            smokeArgs,                                                   //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::ValuesIn(inShapes),                               //
                            ::testing::ValuesIn(targetShapes),                           //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),          //
                            ::testing::Values(additional_config)),                       //
                        InterpolateLayerTest::getTestCaseName);
}  // namespace
