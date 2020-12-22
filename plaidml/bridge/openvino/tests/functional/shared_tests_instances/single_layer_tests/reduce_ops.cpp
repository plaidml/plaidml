// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/reduce_ops.hpp"

using LayerTestsDefinitions::ReduceOpsLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<bool> keepDims = {
    true,
    false,
};

const std::vector<std::vector<size_t>> inputShapes = {
    std::vector<size_t>{1, 2, 4, 4},
    std::vector<size_t>{3, 2, 5, 6},
};

const std::vector<std::vector<int>> axes = {{0}, {1}, {2}, {3}};

const std::vector<std::vector<int>> axesND = {
    {0, 1},       //
    {0, 2},       //
    {0, 3},       //
    {1, 2},       //
    {1, 3},       //
    {2, 3},       //
    {0, 1, 2},    //
    {0, 1, 3},    //
    {0, 2, 3},    //
    {1, 2, 3},    //
    {0, 1, 2, 3}  //
};

std::vector<CommonTestUtils::OpType> opTypes = {
    CommonTestUtils::OpType::SCALAR,  //
    CommonTestUtils::OpType::VECTOR,  //
};

const std::vector<ngraph::helpers::ReductionType> reductionTypes = {
    ngraph::helpers::ReductionType::Mean,  //
    ngraph::helpers::ReductionType::Min,   //
    ngraph::helpers::ReductionType::Max,   //
    ngraph::helpers::ReductionType::Sum,   //
    ngraph::helpers::ReductionType::Prod,  //
    ngraph::helpers::ReductionType::L1,    //
    ngraph::helpers::ReductionType::L2,    //
};

// ReduceLogicalAnd and ReduceLogicalOr require Boolean datatype which seems not well supported.
const std::vector<ngraph::helpers::ReductionType> reductionLogicalTypes = {
    ngraph::helpers::ReductionType::LogicalOr,   //
    ngraph::helpers::ReductionType::LogicalAnd,  //
};

INSTANTIATE_TEST_CASE_P(ReduceOneAxis, ReduceOpsLayerTest,
                        ::testing::Combine(testing::ValuesIn(axes),            //
                                           testing::ValuesIn(opTypes),         //
                                           testing::ValuesIn(keepDims),        //
                                           testing::ValuesIn(reductionTypes),  //
                                           testing::ValuesIn(netPrecisions),   //
                                           testing::ValuesIn(inputShapes),     //
                                           testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(ReduceMultiAxis, ReduceOpsLayerTest,
                        ::testing::Combine(testing::ValuesIn(axesND),                         //
                                           testing::Values(CommonTestUtils::OpType::VECTOR),  //
                                           testing::ValuesIn(keepDims),                       //
                                           testing::ValuesIn(reductionTypes),                 //
                                           testing::ValuesIn(netPrecisions),                  //
                                           testing::ValuesIn(inputShapes),                    //
                                           testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        ReduceOpsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, ReduceOpsLayerTest,
                        ::testing::Combine(testing::Values(axes[0]),                           //
                                           testing::Values(CommonTestUtils::OpType::VECTOR),   //
                                           testing::Values(false),                             //
                                           testing::ValuesIn(reductionTypes),                  //
                                           testing::Values(InferenceEngine::Precision::FP32),  //
                                           testing::Values(std::vector<size_t>{1, 2, 4, 4}),   //
                                           testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                        ReduceOpsLayerTest::getTestCaseName);
}  // namespace
