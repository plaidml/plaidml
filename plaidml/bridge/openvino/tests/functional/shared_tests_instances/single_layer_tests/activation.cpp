// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/activation.hpp"

using LayerTestsDefinitions::ActivationLayerTest;
using namespace ngraph::helpers;  // NOLINT[build/namespaces]
namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
    {Sigmoid, {{}}},
    {Tanh, {{}}},
    // {Relu,        {}},  // TODO
    {Exp, {{}}},
    {Log, {{}}},
    {Sign, {{}}},
    {Abs, {{}}},
    {Clamp, {{-2.0f, 2.0f}}},
    {Negative, {{}}},
    {Acos, {{}}},
    {Asin, {{}}},
    {Atan, {{}}},
    {Cos, {{}}},
    {Cosh, {{}}},
    {Floor, {{}}},
    {Sin, {{}}},
    {Sinh, {{}}},
    {Sqrt, {{}}},
    {Tan, {{}}},
    {Elu, {{0.1f}}},
    {Erf, {{}}},
    {HardSigmoid, {{0.2f, 0.5f}}},
    {Selu, {{1.6732f, 1.0507f}}},
    {Ceiling, {{}}},
    {Swish, {{1.0f}}},
    {Mish, {{}}},
    {HSwish, {{}}}
    // {SoftPlus,    {{}}}
};

// TODO
// const std::map<ActivationTypes, std::vector<std::vector<float>>> activationParamTypes = {
//     {PReLu, {{-0.01f}}},
//     {LeakyRelu, {{0.01f}}}
// };

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
    {{1, 50}, {{}}},
    {{1, 128}, {{}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> preluBasic = {
    {{1, 50}, {{1}, {50}}},
    {{1, 128}, {{1}, {128}}},
};

const auto basicCases = ::testing::Combine(::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),  //
                                           ::testing::ValuesIn(netPrecisions),                                    //
                                           ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),            //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)                     //
);

// const auto basicPreluCases = ::testing::Combine(
//         ::testing::ValuesIn(CommonTestUtils::combineParams(activationParamTypes)),
//         ::testing::ValuesIn(netPrecisions),
//         ::testing::ValuesIn(CommonTestUtils::combineParams(preluBasic)),
//         ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)
// );

INSTANTIATE_TEST_CASE_P(smoke, ActivationLayerTest, basicCases, ActivationLayerTest::getTestCaseName);
// INSTANTIATE_TEST_CASE_P(smoke_Prelu, ActivationLayerTest, basicPreluCases,
// ActivationLayerTest::getTestCaseName);

// INSTANTIATE_TEST_CASE_P(smoke_Prelu_Param, ActivationParamLayerTest, basicPreluCases,
// ActivationLayerTest::getTestCaseName);

}  // namespace
