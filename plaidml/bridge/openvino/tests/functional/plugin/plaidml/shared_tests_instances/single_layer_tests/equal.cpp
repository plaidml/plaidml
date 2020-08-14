// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <map>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/equal.hpp"

using LayerTestsDefinitions::EqualLayerTest;

namespace {

const std::vector<InferenceEngine::Precision> inPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Precision> outPrecisions = {
    InferenceEngine::Precision::BOOL,
};

std::vector<std::vector<std::vector<size_t>>> inputShapes = {
    {{1}, {1}},                         //
    {{8}, {8}},                         //
    {{4, 5}, {4, 5}},                   //
    {{3, 4, 5}, {3, 4, 5}},             //
    {{2, 3, 4, 5}, {2, 3, 4, 5}},       //
    {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}  //
};

const auto cases = ::testing::Combine(::testing::ValuesIn(inputShapes),    //
                                      ::testing::ValuesIn(inPrecisions),   //
                                      ::testing::ValuesIn(outPrecisions),  //
                                      ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_CASE_P(CompareWithRefs, EqualLayerTest, cases, EqualLayerTest::getTestCaseName);
}  // namespace

// // Copyright (C) 2020 Intel Corporation
// // SPDX-License-Identifier: Apache-2.0
// //

// #pragma once

// #include "functional_test_utils/layer_test_utils.hpp"

// #include "ngraph_functions/builders.hpp"
// #include "ngraph_functions/utils/ngraph_helpers.hpp"

// #include <tuple>
// #include <string>
// #include <vector>
// #include <map>
// #include <memory>

// namespace LayerTestsDefinitions {

// using EqualTestParam = typename std::tuple<
//         std::vector<InferenceEngine::SizeVector>,  // Input shapes
//         InferenceEngine::Precision,                // Input precision
//         InferenceEngine::Precision,                // Output precision
//         LayerTestsUtils::TargetDevice>;            // Config

// class EqualLayerTest : public testing::WithParamInterface<EqualTestParam>,
//                        public LayerTestsUtils::LayerTestsCommon {
// public:
//     static std::string getTestCaseName(const testing::TestParamInfo<EqualTestParam>& obj);

// protected:
//     void SetUp() override;
// };

// }  // namespace LayerTestsDefinitions
