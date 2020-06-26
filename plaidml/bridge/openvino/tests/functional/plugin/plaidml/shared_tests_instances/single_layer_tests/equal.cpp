// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <map>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/equal.hpp"

using LayerTestsDefinitions::EqualLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const auto cases =
    ::testing::Combine(::testing::Values(std::vector<size_t>({1, 50}), std::vector<size_t>({1, 128})),  //
                       ::testing::ValuesIn(netPrecisions),                                              //
                       ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_CASE_P(CompareWithRefs, EqualLayerTest, cases, EqualLayerTest::getTestCaseName);
}  // namespace
