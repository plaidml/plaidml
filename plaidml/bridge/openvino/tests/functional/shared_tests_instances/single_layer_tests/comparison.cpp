// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/comparison.hpp"

using namespace LayerTestsDefinitions;
using namespace LayerTestsDefinitions::ComparisonParams;

namespace {

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> inputShapes = {
    {{1}, {{1}, {17}, {1, 1}, {2, 18}, {1, 1, 2}, {2, 2, 3}, {1, 1, 2, 3}}},
    {{2, 200}, {{1}, {200}, {1, 200}, {2, 200}, {2, 2, 200}}},
    {{1, 3, 20}, {{20}, {2, 1, 1}}},
    {{2, 17, 3, 4}, {{4}, {1, 3, 4}, {2, 1, 3, 4}}},
    {{2, 1, 1, 3, 1}, {{1}, {1, 3, 4}, {2, 1, 3, 4}, {1, 1, 1, 1, 1}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> smokeInputShapes = {
    {{5}, {{1}, {1, 1}, {2, 5}, {1, 1, 1}, {2, 2, 5}}},
};

std::vector<InferenceEngine::Precision> inputsPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

std::vector<ngraph::helpers::ComparisonTypes> comparisonOpTypes = {
    ngraph::helpers::ComparisonTypes::EQUAL,          //
    ngraph::helpers::ComparisonTypes::NOT_EQUAL,      //
    ngraph::helpers::ComparisonTypes::GREATER,        //
    ngraph::helpers::ComparisonTypes::GREATER_EQUAL,  //
    ngraph::helpers::ComparisonTypes::LESS,           //
    ngraph::helpers::ComparisonTypes::LESS_EQUAL,
};

std::vector<ngraph::helpers::InputLayerType> secondInputTypes = {
    ngraph::helpers::InputLayerType::CONSTANT,
    ngraph::helpers::InputLayerType::PARAMETER,
};

std::map<std::string, std::string> additional_config = {};

const auto ComparisonTestParams = ::testing::Combine(                  //
    ::testing::ValuesIn(CommonTestUtils::combineParams(inputShapes)),  //
    ::testing::ValuesIn(inputsPrecisions),                             //
    ::testing::ValuesIn(comparisonOpTypes),                            //
    ::testing::ValuesIn(secondInputTypes),                             //
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),        //
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),        //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),                //
    ::testing::Values(additional_config));

const auto SmokeComparisonTestParams = ::testing::Combine(                  //
    ::testing::ValuesIn(CommonTestUtils::combineParams(smokeInputShapes)),  //
    ::testing::ValuesIn(inputsPrecisions),                                  //
    ::testing::ValuesIn(comparisonOpTypes),                                 //
    ::testing::ValuesIn(secondInputTypes),                                  //
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),             //
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),             //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),                     //
    ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(CompareWithRefs, ComparisonLayerTest, ComparisonTestParams,
                        ComparisonLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_CompareWithRefs, ComparisonLayerTest, SmokeComparisonTestParams,
                        ComparisonLayerTest::getTestCaseName);

}  // namespace
