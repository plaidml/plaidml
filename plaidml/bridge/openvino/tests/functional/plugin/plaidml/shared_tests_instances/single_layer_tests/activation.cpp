// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/activation.hpp"
#include "common_test_utils/test_constants.hpp"
#include <vector>

using LayerTestsDefinitions::ActivationLayerTest;
using ngraph::helpers::Abs;
using ngraph::helpers::Acos;
using ngraph::helpers::ActivationTypes;
using ngraph::helpers::Asin;
using ngraph::helpers::Atan;
using ngraph::helpers::Ceiling;
using ngraph::helpers::Clamp;
using ngraph::helpers::Cos;
using ngraph::helpers::Cosh;
using ngraph::helpers::Elu;
using ngraph::helpers::Erf;
using ngraph::helpers::Exp;
using ngraph::helpers::Floor;
// using ngraph::helpers::Gelu;  // not in opset1
using ngraph::helpers::HardSigmoid;
using ngraph::helpers::LeakyRelu;
using ngraph::helpers::Log;
// using ngraph::helpers::Mish;  // not in opset1
using ngraph::helpers::Negative;
using ngraph::helpers::PReLu;
using ngraph::helpers::Relu;
using ngraph::helpers::Selu;
using ngraph::helpers::Sigmoid;
using ngraph::helpers::Sign;
using ngraph::helpers::Sin;
using ngraph::helpers::Sinh;
using ngraph::helpers::Sqrt;
using ngraph::helpers::Tan;
using ngraph::helpers::Tanh;

namespace {
// Common params
const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
    {Sigmoid, {}},
    {Tanh, {}},
    {Relu, {}},
    {Exp, {}},
    {Log, {}},
    {Sign, {}},
    {Abs, {}},
    {Clamp, {{-2.0f, 2.0f}}},
    {Negative, {}},
    {Acos, {}},
    {Asin, {}},
    {Atan, {}},
    {Cos, {}},
    {Cosh, {}},
    {Floor, {}},
    {Sin, {}},
    {Sinh, {}},
    {Sqrt, {}},
    {Tan, {}},
    {Elu, {{0.1f}}},
    {Erf, {}},
    {HardSigmoid, {{0.2f, 0.5f}}},
    {Selu, {{1.6732f, 1.0507f}}},
    {Ceiling, {}},
    // {Mish,        {}},
    // Gelu
    // {HSwish,      {}},
    // {SoftPlus,    {}}
};

// TODO: Missing PReLU types, see
// inference-engine/tests/functional/plugin/cpu/shared_tests_instances/single_layer_tests/activation.cpp

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
    {{1, 50}, {{}}},
    {{1, 128}, {{}}},
};

const auto basicCases = ::testing::Combine(::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),  //
                                           ::testing::ValuesIn(netPrecisions),                                    //
                                           ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),            //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_SUITE_P(Activation_Basic, ActivationLayerTest, basicCases, ActivationLayerTest::getTestCaseName);

}  // namespace
