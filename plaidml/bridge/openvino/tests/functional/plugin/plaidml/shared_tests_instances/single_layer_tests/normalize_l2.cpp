// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/normalize_l2.hpp"

using LayerTestsDefinitions::NormalizeL2LayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,  //
};

const float eps = 1e-8;
const std::vector<int64_t> axes_empty;
const std::vector<int64_t> axes = {0};

INSTANTIATE_TEST_CASE_P(NormalizeL2CheckADD, NormalizeL2LayerTest,
                        ::testing::Combine(::testing::Values(eps),                                //
                                           ::testing::Values(ngraph::op::EpsMode::ADD),           //
                                           ::testing::Values(axes),                               //
                                           ::testing::ValuesIn(netPrecisions),                    //
                                           ::testing::Values(std::vector<size_t>({1, 9, 2, 4})),  //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),   //
                        NormalizeL2LayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(NormalizeL2CheckMax, NormalizeL2LayerTest,
                        ::testing::Combine(::testing::Values(eps),                                //
                                           ::testing::Values(ngraph::op::EpsMode::MAX),           //
                                           ::testing::Values(std::vector<int64_t>({0})),          //
                                           ::testing::ValuesIn(netPrecisions),                    //
                                           ::testing::Values(std::vector<size_t>({1, 9, 2, 4})),  //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),   //
                        NormalizeL2LayerTest::getTestCaseName);

//  INSTANTIATE_TEST_CASE_P(NormalizeL2CheckEmptyAxis, NormalizeL2LayerTest,
//                          ::testing::Combine(::testing::Values(eps),                                //
//                                             ::testing::Values(ngraph::op::EpsMode::MAX),           //
//                                             ::testing::Values(axes_empty),          //
//                                             ::testing::ValuesIn(netPrecisions),                    //
//                                             ::testing::Values(std::vector<size_t>({1, 9, 2, 4})),  //
//                                             ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),   //
//                          NormalizeL2LayerTest::getTestCaseName);
}  // namespace
