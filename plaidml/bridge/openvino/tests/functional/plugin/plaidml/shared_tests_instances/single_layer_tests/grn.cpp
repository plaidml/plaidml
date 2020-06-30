// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/grn.hpp"

using LayerTestsDefinitions::GrnLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,  //
};

const float bias = 0.4;

//  INSTANTIATE_TEST_CASE_P(GrnCheck, GrnLayerTest,
//                          ::testing::Combine(::testing::ValuesIn(netPrecisions),                    //
//                                             ::testing::Values(std::vector<size_t>({1, 9, 2, 4})),  //
//                                             ::testing::Values(bias),                                //
//                                             ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),   //
//                          GrnLayerTest::getTestCaseName);

}  // namespace
