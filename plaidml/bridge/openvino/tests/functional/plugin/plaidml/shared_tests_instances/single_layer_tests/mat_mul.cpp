// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <map>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/mat_mul.hpp"

using LayerTestsDefinitions::MatMulTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

std::vector<std::vector<size_t>> AShapes = {
    {1, 1},  //
};

std::vector<std::vector<size_t>> BShapes = {
    {1, 1},  //
};

INSTANTIATE_TEST_CASE_P(CompareWithRefs, MatMulTest,
                        ::testing::Combine(                                       //
                            ::testing::ValuesIn(netPrecisions),                   //
                            ::testing::ValuesIn(AShapes),                         //
                            ::testing::ValuesIn(BShapes),                         //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                        MatMulTest::getTestCaseName);

}  // namespace
