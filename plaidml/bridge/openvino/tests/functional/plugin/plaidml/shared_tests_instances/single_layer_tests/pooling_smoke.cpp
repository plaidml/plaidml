// Copyright (C) 2019 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

// Simplified pooling tests, suitable for CI. The main pooling tests are long and include errors in the reference
// implementation

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/pooling.hpp"

using LayerTestsDefinitions::PoolingLayerTest;
using LayerTestsDefinitions::poolSpecificParams;

namespace {
// Common params

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32};

const std::vector<std::vector<size_t>> kernels = {{3, 5}};
const std::vector<std::vector<size_t>> strides = {{1, 1}};
const std::vector<std::vector<size_t>> padBegins = {{0, 2}};
const std::vector<std::vector<size_t>> padEnds = {{0, 2}};
const std::vector<ngraph::op::RoundingType> roundingTypes = {ngraph::op::RoundingType::CEIL,
                                                             ngraph::op::RoundingType::FLOOR};
////* ========== Max Polling ========== */
const auto MaxPoolSmokeParams =
    ::testing::Combine(::testing::Values(ngraph::helpers::PoolingTypes::MAX),  //
                       ::testing::ValuesIn(kernels),                           //
                       ::testing::ValuesIn(strides),                           //
                       ::testing::ValuesIn(padBegins),                         //
                       ::testing::ValuesIn(padEnds),                           //
                       ::testing::ValuesIn(roundingTypes),                     //
                       ::testing::Values(ngraph::op::PadType::EXPLICIT),       //
                       ::testing::Values(false));  // placeholder value - exclude pad not applicable for max pooling

INSTANTIATE_TEST_SUITE_P(MaxPool_Smoke, PoolingLayerTest,
                         ::testing::Combine(MaxPoolSmokeParams,                                      //
                                            ::testing::ValuesIn(netPrecisions),                      //
                                            ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),  //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),     //
                         PoolingLayerTest::getTestCaseName);

////* ========== Avg Pooling ========== */

/* +========== Explicit Pad Floor Rounding ========== */
const auto avgPoolSmokeParams =
    ::testing::Combine(::testing::Values(ngraph::helpers::PoolingTypes::AVG),                    //
                       ::testing::ValuesIn(kernels),                                             //
                       ::testing::ValuesIn(strides),                                             //
                       ::testing::ValuesIn(padBegins),                                           //
                       ::testing::ValuesIn(std::vector<std::vector<size_t>>({{0, 0}, {1, 1}})),  //
                       ::testing::ValuesIn(roundingTypes),                                       //
                       ::testing::Values(ngraph::op::PadType::EXPLICIT),                         //
                       ::testing::Values(true, false));                                          //

INSTANTIATE_TEST_SUITE_P(AvgPool_Smoke, PoolingLayerTest,
                         ::testing::Combine(avgPoolSmokeParams,                                      //
                                            ::testing::ValuesIn(netPrecisions),                      //
                                            ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),  //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),     //
                         PoolingLayerTest::getTestCaseName);

////* ========== Avg and Max Polling Cases ========== */
/*    ========== Valid Pad Rounding Not Applicable ========== */
const auto allPools_ValidPad_Params = ::testing::Combine(
    ::testing::Values(ngraph::helpers::PoolingTypes::MAX, ngraph::helpers::PoolingTypes::AVG),  //
    ::testing::ValuesIn(kernels),                                                               //
    ::testing::ValuesIn(strides),                                                               //
    ::testing::Values(std::vector<size_t>({0, 0})),                                             //
    ::testing::Values(std::vector<size_t>({0, 0})),                                             //
    ::testing::Values(
        ngraph::op::RoundingType::FLOOR),  // placeholder value - Rounding Type not applicable for Valid pad type
    ::testing::Values(ngraph::op::PadType::VALID),  //
    ::testing::Values(false));                      // placeholder value - exclude pad not applicable for max pooling

INSTANTIATE_TEST_SUITE_P(BothPool_Smoke, PoolingLayerTest,
                         ::testing::Combine(allPools_ValidPad_Params,                                //
                                            ::testing::ValuesIn(netPrecisions),                      //
                                            ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),  //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),     //
                         PoolingLayerTest::getTestCaseName);

}  // namespace
