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

const std::vector<std::vector<size_t>> inputShapes = {
    std::vector<size_t>{1, 2, 4, 4},
    std::vector<size_t>{3, 2, 5, 6},
};

const std::vector<std::vector<int>> axes = {{0, 2}, {1, 3}};

std::vector<CommonTestUtils::OpType> opTypes = {
    CommonTestUtils::OpType::SCALAR,  //
    CommonTestUtils::OpType::VECTOR,  //
};

const std::vector<ngraph::helpers::ReductionType> reductionTypes = {
    // TODO(Liyang): LogicalAnd, LogicalOr, L1 fail tests, need fix
    //    ngraph::helpers::ReductionType::LogicalAnd,  //
    //    ngraph::helpers::ReductionType::LogicalOr,   //
    ngraph::helpers::ReductionType::Mean,  //
    ngraph::helpers::ReductionType::Min,   //
    ngraph::helpers::ReductionType::Max,   //
    ngraph::helpers::ReductionType::Sum,   //
    ngraph::helpers::ReductionType::Prod,  //
    //    ngraph::helpers::ReductionType::L1,          //
    ngraph::helpers::ReductionType::L2,  //
};

const auto paramsOneAxis = testing::Combine(testing::Values(std::vector<int>{0}),  //
                                            testing::ValuesIn(opTypes),            //
                                            testing::Values(true, false),          //
                                            testing::ValuesIn(reductionTypes),     //
                                            testing::ValuesIn(netPrecisions),      //
                                            testing::ValuesIn(inputShapes),        //
                                            testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_CASE_P(ReduceOneAxis, ReduceOpsLayerTest, paramsOneAxis, ReduceOpsLayerTest::getTestCaseName);

const auto params = testing::Combine(testing::ValuesIn(axes),            //
                                     testing::Values(opTypes[1]),        //
                                     testing::Values(true, false),       //
                                     testing::ValuesIn(reductionTypes),  //
                                     testing::ValuesIn(netPrecisions),   //
                                     testing::ValuesIn(inputShapes),     //
                                     testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_CASE_P(Reduce, ReduceOpsLayerTest, params, ReduceOpsLayerTest::getTestCaseName);
}  // namespace
