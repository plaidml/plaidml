// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/interpolate.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> prc = {
    InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> inShapes = {
    {1, 1, 15, 20},
};

const std::vector<std::vector<size_t>> targetShapes = {
    {1, 1, 20, 30},
};

const std::vector<ngraph::op::v4::Interpolate::InterpolateMode> modesWithoutNearest = {
    ngraph::op::v4::Interpolate::InterpolateMode::linear, ngraph::op::v4::Interpolate::InterpolateMode::cubic,
    // Not enabled in PlaidML
    // ngraph::op::v4::Interpolate::InterpolateMode::linear_onnx,
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

const std::vector<ngraph::op::v4::Interpolate::NearestMode> nearestModes = {
    ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor,
    ngraph::op::v4::Interpolate::NearestMode::floor,
    ngraph::op::v4::Interpolate::NearestMode::round_prefer_ceil,
    ngraph::op::v4::Interpolate::NearestMode::ceil,
    // If it is downsample and using simple mode, scales in InterpolateLayerTest::SetUp() has to be set correctly.
    ngraph::op::v4::Interpolate::NearestMode::simple,
};

const std::vector<ngraph::op::v4::Interpolate::NearestMode> defaultNearestMode = {
    ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor,
};

const std::vector<std::vector<size_t>> pads = {
    {0, 0, 1, 1},
    {0, 0, 0, 0},
};

const std::vector<bool> antialias = {
    // Not enabled in Inference Engine
    // Not enabled in PlaidML
    //        true,
    false,
};

const std::vector<double> cubeCoefs = {
    -0.75f,
};

const auto interpolateCasesWithoutNearest =
    ::testing::Combine(::testing::ValuesIn(modesWithoutNearest), ::testing::ValuesIn(coordinateTransformModes),
                       ::testing::ValuesIn(defaultNearestMode), ::testing::ValuesIn(antialias),
                       ::testing::ValuesIn(pads), ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs));

const auto interpolateCases =
    ::testing::Combine(::testing::ValuesIn(nearestMode), ::testing::ValuesIn(coordinateTransformModes),
                       ::testing::ValuesIn(nearestModes), ::testing::ValuesIn(antialias), ::testing::ValuesIn(pads),
                       ::testing::ValuesIn(pads), ::testing::ValuesIn(cubeCoefs));

const auto smokeArgSet =
    ::testing::Combine(::testing::Values(ngraph::op::v4::Interpolate::InterpolateMode::cubic),
                       ::testing::Values(ngraph::op::v4::Interpolate::CoordinateTransformMode::half_pixel),
                       ::testing::Values(ngraph::op::v4::Interpolate::NearestMode::round_prefer_floor),
                       ::testing::Values(false), ::testing::Values(std::vector<size_t>({0, 0, 0, 0})),
                       ::testing::Values(std::vector<size_t>({0, 0, 0, 0})), ::testing::Values(-0.75f));

INSTANTIATE_TEST_CASE_P(Interpolate_Basic, InterpolateLayerTest,
                        ::testing::Combine(interpolateCasesWithoutNearest, ::testing::ValuesIn(prc),
                                           ::testing::ValuesIn(inShapes), ::testing::ValuesIn(targetShapes),
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Interpolate_Nearest, InterpolateLayerTest,
                        ::testing::Combine(interpolateCases, ::testing::ValuesIn(prc), ::testing::ValuesIn(inShapes),
                                           ::testing::ValuesIn(targetShapes),
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        InterpolateLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, InterpolateLayerTest,
                        ::testing::Combine(smokeArgSet, ::testing::ValuesIn(prc),
                                           ::testing::Values(std::vector<size_t>({1, 1, 15, 20})),
                                           ::testing::Values(std::vector<size_t>({1, 1, 20, 30})),
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        InterpolateLayerTest::getTestCaseName);
}  // namespace
