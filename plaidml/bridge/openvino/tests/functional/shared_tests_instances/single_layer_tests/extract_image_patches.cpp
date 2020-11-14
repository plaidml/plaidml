// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/extract_image_patches.hpp"

using LayerTestsDefinitions::ExtractImagePatchesTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::FP32,
};

const std::vector<std::vector<size_t>> kernels = {{3, 3}, {5, 5}};
const std::vector<std::vector<size_t>> strides = {{5, 5}, {1, 1}};
const std::vector<std::vector<size_t>> rates = {{1, 1}, {2, 2}};
const std::vector<ngraph::op::PadType> padTypes = {
    ngraph::op::PadType::VALID,       //
    ngraph::op::PadType::SAME_UPPER,  //
    ngraph::op::PadType::SAME_LOWER,  //
};

INSTANTIATE_TEST_CASE_P(ExtractImagePatches2D_AutoTestCheck, ExtractImagePatchesTest,
                        ::testing::Combine(::testing::Values(std::vector<size_t>({1, 3, 10, 10})),  //
                                           ::testing::ValuesIn(kernels),                            //
                                           ::testing::ValuesIn(strides),                            //
                                           ::testing::ValuesIn(rates),                              //
                                           ::testing::ValuesIn(padTypes),                           //
                                           ::testing::ValuesIn(netPrecisions),                      //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),     //
                        ExtractImagePatchesTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, ExtractImagePatchesTest,
                        ::testing::Combine(::testing::Values(std::vector<size_t>({1, 3, 10, 10})),  //
                                           ::testing::Values(std::vector<size_t>({3, 5})),          //
                                           ::testing::Values(std::vector<size_t>({5, 5})),          //
                                           ::testing::Values(std::vector<size_t>({1, 2})),          //
                                           ::testing::ValuesIn(padTypes),                           //
                                           ::testing::ValuesIn(netPrecisions),                      //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),     //
                        ExtractImagePatchesTest::getTestCaseName);

}  // namespace
