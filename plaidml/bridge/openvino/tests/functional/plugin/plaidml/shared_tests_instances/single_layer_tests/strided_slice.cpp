// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/strided_slice.hpp"

using LayerTestsDefinitions::StridedSliceLayerTest;
using LayerTestsDefinitions::stridedSliceParamsTuple;

namespace {

// n.b. adjusted slightly from upstream version as this version of the interpreter (or the op?) seems to be unhappy with
// empty vectors. Also, the -1 wrapping as if unsigned appears to be a printing error, not an underlying data error.
stridedSliceParamsTuple ss_only_test_cases[] = {
    stridedSliceParamsTuple({128, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1},              //
                            {0, 1, 1}, {0, 1, 1}, {1, 0, 0}, {0, 0, 0}, {0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({128, 1}, {0, 0, 0}, {0, 0, 0}, {1, 1, 1},              //
                            {1, 0, 1}, {1, 0, 1}, {0, 1, 0}, {0, 0, 0}, {0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({1, 12, 100}, {0, -1, 0}, {0, 0, 0}, {1, 1, 1},         //
                            {1, 0, 1}, {1, 0, 1}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({1, 12, 100}, {0, 9, 0}, {0, 11, 0}, {1, 1, 1},         //
                            {1, 0, 1}, {1, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({1, 12, 100}, {0, 1, 0}, {0, -1, 0}, {1, 1, 1},         //
                            {1, 0, 1}, {1, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({1, 12, 100}, {0, 9, 0}, {0, 8, 0}, {1, -1, 1},         //
                            {1, 0, 1}, {1, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({1, 12, 100}, {0, 9, 0}, {0, 7, 0}, {-1, -1, -1},       //
                            {1, 0, 1}, {1, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({1, 12, 100}, {0, 7, 0}, {0, 9, 0}, {-1, 1, -1},        //
                            {1, 0, 1}, {1, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({1, 12, 100}, {0, 4, 0}, {0, 9, 0}, {-1, 2, -1},        //
                            {1, 0, 1}, {1, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({1, 12, 100}, {0, 4, 0}, {0, 10, 0}, {-1, 2, -1},       //
                            {1, 0, 1}, {1, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({1, 12, 100}, {0, 9, 0}, {0, 4, 0}, {-1, -2, -1},       //
                            {1, 0, 1}, {1, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({1, 12, 100}, {0, 10, 0}, {0, 4, 0}, {-1, -2, -1},      //
                            {1, 0, 1}, {1, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({1, 12, 100}, {0, 11, 0}, {0, 0, 0}, {-1, -2, -1},      //
                            {1, 0, 1}, {1, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({1, 12, 100}, {0, -6, 0}, {0, -8, 0}, {-1, -2, -1},     //
                            {1, 0, 1}, {1, 0, 1}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({1, 12, 100, 1, 1}, {0, -1, 0, 0}, {0, 0, 0, 0}, {1, 1, 1, 1},         //
                            {1, 0, 1, 0}, {1, 0, 1, 0}, {0, 0, 0, 0}, {0, 1, 0, 1}, {0, 0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({2, 2, 2, 2}, {0, 0, 0, 0}, {2, 2, 2, 2}, {1, 1, 1, 1},                //
                            {1, 1, 1, 1}, {1, 1, 1, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({2, 2, 2, 2}, {1, 1, 1, 1}, {2, 2, 2, 2}, {1, 1, 1, 1},                //
                            {0, 0, 0, 0}, {1, 1, 1, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({2, 2, 2, 2}, {1, 1, 1, 1}, {2, 2, 2, 2}, {1, 1, 1, 1},                //
                            {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({2, 2, 4, 3}, {0, 0, 0, 0}, {2, 2, 4, 3}, {1, 1, 2, 1},                //
                            {1, 1, 1, 1}, {1, 1, 1, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({2, 2, 4, 2}, {1, 0, 0, 1}, {2, 2, 4, 2}, {1, 1, 2, 1},                //
                            {0, 1, 1, 0}, {1, 1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({1, 2, 4, 2}, {1, 0, 0, 0}, {1, 2, 4, 2}, {1, 1, -2, -1},              //
                            {1, 1, 1, 1}, {1, 1, 1, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
    stridedSliceParamsTuple({2, 2, 4, 2}, {1, 0, 0, 0}, {1, 2, 4, 2}, {1, 1, -2, -1},              //
                            {0, 1, 1, 1}, {1, 1, 1, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0},  //
                            InferenceEngine::Precision::FP32, CommonTestUtils::DEVICE_PLAIDML),
};

INSTANTIATE_TEST_CASE_P(StridedSlice, StridedSliceLayerTest, ::testing::ValuesIn(ss_only_test_cases),
                        StridedSliceLayerTest::getTestCaseName);

}  // namespace
