// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/deformable_psroi_pooling.hpp"
#include <vector>

using LayerTestsDefinitions::DeformablePSROIPoolingLayerTest;

namespace {

/* =============== 2 inputs without offsets =============== */
const auto deformablePSROIParams = ::testing::Combine(     //
    ::testing::Values(std::vector<size_t>{3, 8, 16, 16}),  // data input shape
    ::testing::Values(std::vector<size_t>{10, 5}),         // rois input shape
    // Empty offsets shape means test without optional third input
    ::testing::Values(std::vector<size_t>{}),                                                // offsets input shape
    ::testing::Values(2),                                                                    // output_dim
    ::testing::Values(2),                                                                    // group_size
    ::testing::ValuesIn(std::vector<float>{1.0, 0.5, 0.0625}),                               // spatial scale
    ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{1, 1}, {2, 2}, {3, 3}, {2, 3}}),  // spatial_bins_x_y
    ::testing::ValuesIn(std::vector<float>{0.0, 0.01, 0.5}),                                 // trans_std
    ::testing::Values(2));                                                                   // part_size

const auto deformablePSROICases_test_params = ::testing::Combine(  //
    deformablePSROIParams,                                         //
    ::testing::Values(InferenceEngine::Precision::FP32),           // Net precision
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));           // Device name

INSTANTIATE_TEST_CASE_P(smoke_WithoutOffsets, DeformablePSROIPoolingLayerTest, deformablePSROICases_test_params,
                        DeformablePSROIPoolingLayerTest::getTestCaseName);

/* =============== 3 inputs with offsets =============== */
const auto deformablePSROIParams_advanced = ::testing::Combine(      //
    ::testing::Values(std::vector<size_t>{2, 441, 63, 38}),          // data input shape
    ::testing::Values(std::vector<size_t>{30, 5}),                   // rois input shape
    ::testing::Values(std::vector<size_t>{30, 2, 3, 3}),             // offsets input shape
    ::testing::Values(49),                                           // output_dim
    ::testing::Values(3),                                            // group_size
    ::testing::ValuesIn(std::vector<float>{0.0625}),                 // spatial scale
    ::testing::ValuesIn(std::vector<std::vector<int64_t>>{{4, 4}}),  // spatial_bins_x_y
    ::testing::ValuesIn(std::vector<float>{0.1}),                    // trans_std
    ::testing::Values(3));                                           // part_size

const auto deformablePSROICases_test_params_advanced = ::testing::Combine(  //
    deformablePSROIParams_advanced,                                         //
    ::testing::Values(InferenceEngine::Precision::FP32),                    // Net precision
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));                    // Device name

INSTANTIATE_TEST_CASE_P(smoke_WithOffsets, DeformablePSROIPoolingLayerTest, deformablePSROICases_test_params_advanced,
                        DeformablePSROIPoolingLayerTest::getTestCaseName);
}  // namespace
