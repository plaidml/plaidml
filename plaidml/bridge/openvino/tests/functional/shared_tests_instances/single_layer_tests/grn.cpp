// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <map>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/grn.hpp"

using LayerTestsDefinitions::GrnLayerTest;

namespace {

INSTANTIATE_TEST_CASE_P(smoke, GrnLayerTest,
                        ::testing::Combine(::testing::Values(InferenceEngine::Precision::FP32),        //
                                           ::testing::Values(std::vector<std::size_t>({4, 3, 3, 6})),  //
                                           ::testing::Values(0.01),                                    //
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),        //
                        GrnLayerTest::getTestCaseName);

}  // namespace
