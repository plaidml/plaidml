// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/fake_quantize.hpp"

using LayerTestsDefinitions::FakeQuantizeLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32};

const std::vector<std::vector<size_t>> inputShapes = {{1, 1, 1, 1}, {3, 10, 5, 6}};
const std::vector<std::vector<size_t>> constShapes = {{1}};
const std::vector<size_t> levels = {16, 255, 256};

const auto fqParams = ::testing::Combine(::testing::ValuesIn(levels),        //
                                         ::testing::ValuesIn(constShapes));  //

INSTANTIATE_TEST_SUITE_P(FakeQuantize, FakeQuantizeLayerTest,
                         ::testing::Combine(fqParams,                                             //
                                            ::testing::ValuesIn(netPrecisions),                   //
                                            ::testing::ValuesIn(inputShapes),                     //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                         FakeQuantizeLayerTest::getTestCaseName);

}  // namespace
