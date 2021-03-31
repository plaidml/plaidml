// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/shape_of.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::I32,
};

INSTANTIATE_TEST_CASE_P(smoke, ShapeOfLayerTest,
                        ::testing::Combine(                                        //
                            ::testing::ValuesIn(netPrecisions),                    //
                            ::testing::Values(std::vector<size_t>({10, 10, 10})),  //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),   //
                        ShapeOfLayerTest::getTestCaseName);
}  // namespace
