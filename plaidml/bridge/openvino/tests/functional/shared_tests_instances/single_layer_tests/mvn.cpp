// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/mvn.hpp"

using LayerTestsDefinitions::MvnLayerTest;

const std::vector<std::vector<size_t>> inputShapes = {
    {1, 32, 17},        //
    {1, 37, 9},         //
    {1, 16, 5, 8},      //
    {2, 19, 5, 10},     //
    {7, 32, 2, 8},      //
    {5, 8, 3, 5},       //
    {4, 41, 6, 9},      //
    {1, 32, 8, 1, 6},   //
    {1, 9, 1, 15, 9},   //
    {6, 64, 6, 1, 18},  //
    {2, 31, 2, 9, 1},   //
    {10, 16, 5, 10, 6}  //
};

const std::vector<bool> acrossChannels = {
    true,  //
    false  //
};

const std::vector<bool> normalizeVariance = {
    true,  //
    false  //
};

const std::vector<double> epsilon = {0.000000001};

const auto MvnCases = ::testing::Combine(::testing::ValuesIn(inputShapes),                     //
                                         ::testing::Values(InferenceEngine::Precision::FP32),  //
                                         ::testing::ValuesIn(acrossChannels),                  //
                                         ::testing::ValuesIn(normalizeVariance),               //
                                         ::testing::ValuesIn(epsilon),                         //
                                         ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

const auto SmokeCases = ::testing::Combine(::testing::Values(std::vector<size_t>({3, 7, 16, 6})),  //
                                           ::testing::Values(InferenceEngine::Precision::FP32),    //
                                           ::testing::ValuesIn(acrossChannels),                    //
                                           ::testing::ValuesIn(normalizeVariance),                 //
                                           ::testing::ValuesIn(epsilon),                           //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_CASE_P(MVN, MvnLayerTest, MvnCases, MvnLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, MvnLayerTest, SmokeCases, MvnLayerTest::getTestCaseName);
