// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/space_to_batch.hpp"

using LayerTestsDefinitions::SpaceToBatchLayerTest;

namespace {

// spaceToBatchParamsTuple stb_only_test_cases[] = {
//     spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {1, 1, 2, 2}, InferenceEngine::Precision::FP32,
//                             CommonTestUtils::DEVICE_PLAIDML),
//     spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {1, 3, 2, 2}, InferenceEngine::Precision::FP32,
//                             CommonTestUtils::DEVICE_PLAIDML),
//     spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}, {1, 1, 4, 4}, InferenceEngine::Precision::FP32,
//                             CommonTestUtils::DEVICE_PLAIDML),
//     spaceToBatchParamsTuple({1, 1, 2, 2}, {0, 0, 0, 2}, {0, 0, 0, 0}, {2, 1, 2, 4}, InferenceEngine::Precision::FP32,
//                             CommonTestUtils::DEVICE_PLAIDML),
//     spaceToBatchParamsTuple({1, 1, 3, 2, 2}, {0, 0, 1, 0, 3}, {0, 0, 2, 0, 0}, {1, 1, 3, 2, 1},
//                             InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
// };

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32  //
};

INSTANTIATE_TEST_SUITE_P(smoke, SpaceToBatchLayerTest,
                         ::testing::Combine(::testing::Values(1, 1, 2, 2),                        //
                                            ::testing::Values(0, 0, 0, 0),                        //
                                            ::testing::Values(0, 0, 0, 0),                        //
                                            ::testing::Values(1, 1, 2, 2),                        //
                                            ::testing::ValuesIn(netPrecisions),                   //
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),  //
                         SpaceToBatchLayerTest::getTestCaseName);

}  // namespace
