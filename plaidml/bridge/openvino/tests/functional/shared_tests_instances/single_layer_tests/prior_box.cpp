// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/prior_box.hpp"

using LayerTestsDefinitions::PriorBoxLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const std::vector<std::vector<float>> minSizes = {
    {256.0f},
    {256.0f, 128.0f},
};
const std::vector<std::vector<float>> maxSizes = {
    {315.0f},
};
const std::vector<std::vector<float>> aspectRatios = {
    {1.0f},
    {2.0f, 3.0f},
};
const std::vector<std::vector<float>> density = {
    {2.0f, 4.0f},
};
const std::vector<std::vector<float>> fixedRatios = {
    {2.0f},
};
const std::vector<std::vector<float>> fixedSizes = {
    {2.0f, 4.0f},
};
const std::vector<bool> clip = {
    false,
    true,
};
const std::vector<bool> flip = {
    false,
    true,
};
const std::vector<float> steps = {
    1.0f,
    0.0f,
    -1.0f,
};
const std::vector<float> offsets = {
    0.0f,
    0.5f,
};
const std::vector<std::vector<float>> variances = {
    {0.1f},
    {0.1f, 0.2f, 0.3f, 0.4f},
};
const std::vector<bool> scaleAllSizes = {
    false,
    true,
};
// The following added bool vectors are used to mask corresponding attributes.
const std::vector<bool> useFixedSizes = {
    false,
    true,
};
const std::vector<bool> useFixedRatios = {
    false,
    true,
};
// Combine supports up to 10 arguments
const auto layerSpecificParamsForMinSizeTest = ::testing::Combine(  //
    ::testing::ValuesIn(minSizes),                                  //
    ::testing::ValuesIn(maxSizes),                                  //
    ::testing::ValuesIn(aspectRatios),                              //
    ::testing::ValuesIn(density),                                   //
    ::testing::ValuesIn(fixedRatios),                               //
    ::testing::ValuesIn(fixedSizes),                                //
    ::testing::Values(false),                                       //
    ::testing::Values(false),                                       //
    ::testing::ValuesIn(steps),                                     //
    ::testing::ValuesIn(offsets)                                    //
);

const auto layerSpecificParamsForFixedSizeTest = ::testing::Combine(  //
    ::testing::Values(std::vector<float>({256.0f})),                  //
    ::testing::Values(std::vector<float>({315.0f})),                  //
    ::testing::ValuesIn(aspectRatios),                                //
    ::testing::ValuesIn(density),                                     //
    ::testing::ValuesIn(fixedRatios),                                 //
    ::testing::ValuesIn(fixedSizes),                                  //
    ::testing::Values(false),                                         //
    ::testing::Values(false),                                         //
    ::testing::ValuesIn(steps),                                       //
    ::testing::ValuesIn(offsets)                                      //
);

const auto layerSpecificParamsForSmokeTest = ::testing::Combine(  //
    ::testing::Values(std::vector<float>({256.0f})),              //
    ::testing::Values(std::vector<float>({315.0f})),              //
    ::testing::Values(std::vector<float>({2.0f, 3.0f})),          //
    ::testing::Values(std::vector<float>({2.0f, 4.0f})),          //
    ::testing::Values(std::vector<float>({2.0f})),                //
    ::testing::Values(std::vector<float>({2.0f, 4.0f})),          //
    ::testing::Values(true),                                      //
    ::testing::Values(true),                                      //
    ::testing::Values(1.0f),                                      //
    ::testing::Values(0.5f)                                       //
);

std::vector<std::vector<size_t>> layerShapes{
    {2, 3},
};
std::vector<std::vector<size_t>> imageShapes{
    {2},
};

INSTANTIATE_TEST_CASE_P(PriorBoxLayerMinSizeCheck, PriorBoxLayerTest,
                        ::testing::Combine(                                              //
                            layerSpecificParamsForMinSizeTest,                           //
                            ::testing::Values(std::vector<float>({0.1f})),               //
                            ::testing::ValuesIn(scaleAllSizes),                          //
                            ::testing::Values(false),                                    //
                            ::testing::Values(false),                                    //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::ValuesIn(layerShapes),                            //
                            ::testing::ValuesIn(imageShapes),                            //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),          //
                            ::testing::Values(std::map<std::string, std::string>({}))),  //
                        PriorBoxLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(PriorBoxLayerFixedSizeCheck, PriorBoxLayerTest,
                        ::testing::Combine(                                              //
                            layerSpecificParamsForFixedSizeTest,                         //
                            ::testing::Values(std::vector<float>({0.1f})),               //
                            ::testing::Values(false),                                    //
                            ::testing::Values(true),                                     //
                            ::testing::Values(true),                                     //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::ValuesIn(layerShapes),                            //
                            ::testing::ValuesIn(imageShapes),                            //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),          //
                            ::testing::Values(std::map<std::string, std::string>({}))),  //
                        PriorBoxLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, PriorBoxLayerTest,
                        ::testing::Combine(                                              //
                            layerSpecificParamsForSmokeTest,                             //
                            ::testing::Values(std::vector<float>({0.1f})),               //
                            ::testing::Values(true),                                     //
                            ::testing::Values(true),                                     //
                            ::testing::Values(true),                                     //
                            ::testing::Values(InferenceEngine::Precision::FP32),         //
                            ::testing::Values(std::vector<size_t>({2, 3})),              //
                            ::testing::Values(std::vector<size_t>({2})),                 //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),          //
                            ::testing::Values(std::map<std::string, std::string>({}))),  //
                        PriorBoxLayerTest::getTestCaseName);

}  // namespace
