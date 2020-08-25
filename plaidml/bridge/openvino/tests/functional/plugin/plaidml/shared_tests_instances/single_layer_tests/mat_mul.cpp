// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <map>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/mat_mul.hpp"

using LayerTestsDefinitions::MatMulLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const auto params_NoTranspose = ::testing::Combine(::testing::Values(false),  //
                                                   ::testing::Values(false)   //
);

const auto params_ATranspose = ::testing::Combine(::testing::Values(true),  //
                                                  ::testing::Values(false)  //
);

const auto params_BTranspose = ::testing::Combine(::testing::Values(false),  //
                                                  ::testing::Values(true)    //
);

const std::vector<std::vector<std::vector<std::size_t>>> inputShapes = {
    {{1}, {1}},        //
    {{3, 3}, {3, 1}},  //
    {{5, 5}, {5, 5}},  //
};

const std::vector<std::vector<std::vector<std::size_t>>> inputShapesATrans = {
    {{1}, {1}},        //
    {{3, 1}, {3, 1}},  //
    {{5, 2}, {5, 5}},  //
};

const std::vector<std::vector<std::vector<std::size_t>>> inputShapesBTrans = {
    {{1}, {1}},        //
    {{3, 1}, {3, 1}},  //
    {{5, 2}, {5, 2}},  //
};

INSTANTIATE_TEST_SUITE_P(MatMul_NoTranspose, MatMulLayerTest,
                         ::testing::Combine(params_NoTranspose,                                   //
                                            ::testing::ValuesIn(netPrecisions),                   //
                                            ::testing::ValuesIn(inputShapes),                     //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                         MatMulLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MatMul_ATranspose, MatMulLayerTest,
                         ::testing::Combine(params_ATranspose,                                    //
                                            ::testing::ValuesIn(netPrecisions),                   //
                                            ::testing::ValuesIn(inputShapesATrans),               //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                         MatMulLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MatMul_BTranspose, MatMulLayerTest,
                         ::testing::Combine(params_BTranspose,                                    //
                                            ::testing::ValuesIn(netPrecisions),                   //
                                            ::testing::ValuesIn(inputShapesBTrans),               //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                         MatMulLayerTest::getTestCaseName);

}  // namespace
