// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/non_max_suppression.hpp"

using LayerTestsDefinitions;

namespace {
// Each InputShapeParams consists of Number of batches, Number of boxes, Number of classes
const std::vector<InputShapeParams> inShapeParams = {InputShapeParams{3, 100, 5}, InputShapeParams{1, 10, 50},
                                                     InputShapeParams{2, 50, 50}};

const std::vector<int32_t> maxOutBoxPerClass = {5, 20};
const std::vector<float> threshold = {0.3f, 0.7f};
const std::vector<float> sigmaThreshold = {0.0f, 0.5f};
const std::vector<op::v5::NonMaxSuppression::BoxEncodingType> encodType = {
    op::v5::NonMaxSuppression::BoxEncodingType::CENTER, op::v5::NonMaxSuppression::BoxEncodingType::CORNER};
const std::vector<bool> sortResDesc = {true, false};
const std::vector<element::Type> outType = {element::i32, element::i64};

const auto nmsParams = ::testing::Combine(
    ::testing::ValuesIn(inShapeParams),
    ::testing::Combine(::testing::Values(Precision::FP32), ::testing::Values(Precision::I32),
                       ::testing::Values(Precision::FP32)),
    ::testing::ValuesIn(maxOutBoxPerClass), ::testing::ValuesIn(threshold), ::testing::ValuesIn(threshold),
    ::testing::ValuesIn(sigmaThreshold), ::testing::ValuesIn(encodType), ::testing::ValuesIn(sortResDesc),
    ::testing::ValuesIn(outType), ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_CASE_P(NMS, NmsLayerTest, nmsParams, NmsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, NmsLayerTest, nmsParams, NmsLayerTest::getTestCaseName);

}  // namespace
