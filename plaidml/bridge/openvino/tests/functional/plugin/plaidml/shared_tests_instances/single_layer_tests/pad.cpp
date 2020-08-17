// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <map>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/pad.hpp"

using LayerTestsDefinitions::PadLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<std::size_t>> inputShapes = {
    {std::vector<std::size_t>({3, 3}), std::vector<std::size_t>({300, 300})}};

const std::vector<std::vector<std::size_t>> loPads = {
    {std::vector<std::size_t>({0, 1}), std::vector<std::size_t>({5, 10})}};

const std::vector<std::vector<std::size_t>> hiPads = {
    {std::vector<std::size_t>({0, 1}), std::vector<std::size_t>({10, 10})}};

const std::vector<ngraph::op::PadMode> padModes = {{ngraph::op::PadMode::CONSTANT}};

INSTANTIATE_TEST_CASE_P(CompareWithRefs, PadLayerTest,
                        ::testing::Combine(::testing::ValuesIn(loPads),         //
                                           ::testing::ValuesIn(hiPads),         //
                                           ::testing::ValuesIn(padModes),       //
                                           ::testing::ValuesIn(netPrecisions),  //
                                           ::testing::ValuesIn(inputShapes),    //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        PadLayerTest::getTestCaseName);

}  // namespace
