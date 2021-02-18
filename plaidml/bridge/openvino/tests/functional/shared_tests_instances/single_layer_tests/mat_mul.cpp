// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/mat_mul.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const std::vector<ShapeRelatedParams> shapeRelatedParams = {
    {{{1, 4, 5, 6}, false}, {{1, 4, 6, 4}, false}},
    {{{1, 3, 4, 8}, false}, {{5, 1, 8, 2}, false}},
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
    ngraph::helpers::InputLayerType::CONSTANT,
    ngraph::helpers::InputLayerType::PARAMETER,
};

std::map<std::string, std::string> additional_config = {};

INSTANTIATE_TEST_CASE_P(smoke, MatMulTest,
                        ::testing::Combine(                                              //
                            ::testing::ValuesIn(shapeRelatedParams),                     //
                            ::testing::ValuesIn(inputPrecisions),                        //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::ValuesIn(secondaryInputTypes),                    //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),          //
                            ::testing::Values(additional_config)),                       //
                        MatMulTest::getTestCaseName);
}  // namespace
