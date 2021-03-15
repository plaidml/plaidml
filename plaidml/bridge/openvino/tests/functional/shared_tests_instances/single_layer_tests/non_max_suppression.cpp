// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/non_max_suppression.hpp"

using LayerTestsDefinitions::InputShapeParams;
using LayerTestsDefinitions::NmsLayerTest;

namespace {
// Each InputShapeParams consists of Number of batches, Number of boxes, Number of classes
const std::vector<InputShapeParams> inShapeParams = {InputShapeParams{1, 5, 1}};

const std::vector<int32_t> maxOutBoxPerClass = {5};
const std::vector<float> threshold = {0.3f};
const std::vector<float> sigmaThreshold = {0.0f};
const std::vector<ngraph::op::v5::NonMaxSuppression::BoxEncodingType> encodType = {
    ngraph::op::v5::NonMaxSuppression::BoxEncodingType::CENTER,
    ngraph::op::v5::NonMaxSuppression::BoxEncodingType::CORNER};
const std::vector<bool> sortResDesc = {false};
const std::vector<ngraph::element::Type> outType = {ngraph::element::i32};

const auto nmsParams = ::testing::Combine(
    ::testing::ValuesIn(inShapeParams),
    ::testing::Combine(::testing::Values(InferenceEngine::Precision::FP32),
                       ::testing::Values(InferenceEngine::Precision::I32),
                       ::testing::Values(InferenceEngine::Precision::FP32)),
    ::testing::ValuesIn(maxOutBoxPerClass), ::testing::ValuesIn(threshold), ::testing::ValuesIn(threshold),
    ::testing::ValuesIn(sigmaThreshold), ::testing::ValuesIn(encodType), ::testing::ValuesIn(sortResDesc),
    ::testing::ValuesIn(outType), ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_CASE_P(NMS, NmsLayerTest, nmsParams, NmsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, NmsLayerTest, nmsParams, NmsLayerTest::getTestCaseName);

}  // namespace
