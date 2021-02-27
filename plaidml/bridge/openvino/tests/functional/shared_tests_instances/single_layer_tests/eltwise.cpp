// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/eltwise.hpp"

using namespace LayerTestsDefinitions;

namespace {
std::vector<std::vector<std::vector<size_t>>> inShapes = {
    {{2}},
    {{2, 200}},
    {{10, 200}},
    {{1, 10, 100}},
    {{4, 4, 16}},
    {{1, 1, 1, 3}},
    {{2, 17, 5, 4}, {1, 17, 1, 1}},
    {{2, 17, 5, 1}, {1, 17, 1, 4}},
    {{1, 2, 4}},
    {{1, 4, 4}},
    {{1, 4, 4, 1}},
    {{1, 1, 1, 1, 1, 1, 3}},
    {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
};

std::vector<std::vector<std::vector<size_t>>> smokeShapes = {
    {{1, 2, 3, 4}},
    {{2, 2, 2, 2}},
    {{2, 1, 2, 1, 2, 2}},
};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I32,
};

std::vector<InferenceEngine::Precision> f32Precision = {
    InferenceEngine::Precision::FP32,
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
    ngraph::helpers::InputLayerType::CONSTANT,
    ngraph::helpers::InputLayerType::PARAMETER,
};

std::vector<CommonTestUtils::OpType> opTypes = {
    CommonTestUtils::OpType::SCALAR,
    CommonTestUtils::OpType::VECTOR,
};

std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypes = {
    ngraph::helpers::EltwiseTypes::ADD,           //
    ngraph::helpers::EltwiseTypes::MULTIPLY,      //
    ngraph::helpers::EltwiseTypes::SUBTRACT,      //
    ngraph::helpers::EltwiseTypes::DIVIDE,        //
    ngraph::helpers::EltwiseTypes::SQUARED_DIFF,  //
    ngraph::helpers::EltwiseTypes::MOD,           //
};

std::vector<ngraph::helpers::EltwiseTypes> f32OpTypes = {
    ngraph::helpers::EltwiseTypes::FLOOR_MOD,  //
    ngraph::helpers::EltwiseTypes::POWER,      //
};

std::map<std::string, std::string> additional_config = {};

const auto params = ::testing::Combine(                          //
    ::testing::ValuesIn(inShapes),                               //
    ::testing::ValuesIn(eltwiseOpTypes),                         //
    ::testing::ValuesIn(secondaryInputTypes),                    //
    ::testing::ValuesIn(opTypes),                                //
    ::testing::ValuesIn(netPrecisions),                          //
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
    ::testing::Values(InferenceEngine::Layout::ANY),             //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),          //
    ::testing::Values(additional_config)                         //
);

INSTANTIATE_TEST_CASE_P(Eltwise, EltwiseLayerTest, params, EltwiseLayerTest::getTestCaseName);

const auto smokeParams = ::testing::Combine(                     //
    ::testing::ValuesIn(smokeShapes),                            //
    ::testing::ValuesIn(eltwiseOpTypes),                         //
    ::testing::ValuesIn(secondaryInputTypes),                    //
    ::testing::ValuesIn(opTypes),                                //
    ::testing::ValuesIn(netPrecisions),                          //
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
    ::testing::Values(InferenceEngine::Layout::ANY),             //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),          //
    ::testing::Values(additional_config)                         //
);

INSTANTIATE_TEST_CASE_P(smokeEltwise, EltwiseLayerTest, smokeParams, EltwiseLayerTest::getTestCaseName);

const auto f32Params = ::testing::Combine(                       //
    ::testing::ValuesIn(inShapes),                               //
    ::testing::ValuesIn(f32OpTypes),                             //
    ::testing::ValuesIn(secondaryInputTypes),                    //
    ::testing::ValuesIn(opTypes),                                //
    ::testing::ValuesIn(f32Precision),                           //
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
    ::testing::Values(InferenceEngine::Layout::ANY),             //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),          //
    ::testing::Values(additional_config)                         //
);

INSTANTIATE_TEST_CASE_P(f32OnlyEltwise, EltwiseLayerTest, f32Params, EltwiseLayerTest::getTestCaseName);

const auto smokeF32Params = ::testing::Combine(                  //
    ::testing::ValuesIn(smokeShapes),                            //
    ::testing::ValuesIn(f32OpTypes),                             //
    ::testing::ValuesIn(secondaryInputTypes),                    //
    ::testing::ValuesIn(opTypes),                                //
    ::testing::ValuesIn(f32Precision),                           //
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
    ::testing::Values(InferenceEngine::Layout::ANY),             //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),          //
    ::testing::Values(additional_config)                         //
);

INSTANTIATE_TEST_CASE_P(smokeF32OnlyEltwise, EltwiseLayerTest, smokeF32Params, EltwiseLayerTest::getTestCaseName);

}  // namespace
