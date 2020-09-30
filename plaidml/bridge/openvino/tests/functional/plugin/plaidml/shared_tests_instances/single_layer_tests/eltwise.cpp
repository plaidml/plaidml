// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/eltwise.hpp"
#include "common_test_utils/test_constants.hpp"
#include <vector>

using CommonTestUtils::OpType;
using LayerTestsDefinitions::EltwiseLayerTest;

namespace {
std::vector<std::vector<std::vector<size_t>>> inShapes = {
    {{2}},
    {{2, 200}},
    {{10, 200}},
    {{1, 10, 100}},
    {{4, 4, 16}},
    {{1, 1, 1, 3}},
    {{1, 2, 4}},
    {{1, 4, 4}},
    {{1, 4, 4, 1}},
    {{1, 4, 3, 2, 1, 3}},
    {{1, 3, 1, 1, 1, 3}, {1, 3, 1, 1, 1, 1}},
};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
    ngraph::helpers::InputLayerType::CONSTANT,
    ngraph::helpers::InputLayerType::PARAMETER,
};

std::vector<OpType> opTypes = {
    OpType::SCALAR,
    OpType::VECTOR,
};

std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypes = {ngraph::helpers::EltwiseTypes::MULTIPLY,
                                                             ngraph::helpers::EltwiseTypes::SUBTRACT,
                                                             ngraph::helpers::EltwiseTypes::ADD};

std::map<std::string, std::string> additional_config = {};

const auto multiply_params = ::testing::Combine(::testing::ValuesIn(inShapes),                       //
                                                ::testing::ValuesIn(eltwiseOpTypes),                 //
                                                ::testing::ValuesIn(secondaryInputTypes),            //
                                                ::testing::ValuesIn(opTypes),                        //
                                                ::testing::ValuesIn(netPrecisions),                  //
                                                ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),  //
                                                ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(CompareWithRefs, EltwiseLayerTest, multiply_params, EltwiseLayerTest::getTestCaseName);
}  // namespace
