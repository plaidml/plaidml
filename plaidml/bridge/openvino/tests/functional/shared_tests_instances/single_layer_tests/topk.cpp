// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/topk.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const std::vector<int64_t> axes = {
    0,
    1,
    2,
};

const std::vector<int64_t> k = {
    1,
    5,
    10,
};

const std::vector<ngraph::opset4::TopK::Mode> modes = {
    ngraph::opset4::TopK::Mode::MIN,
    ngraph::opset4::TopK::Mode::MAX,
};

const std::vector<ngraph::opset4::TopK::SortType> sortTypes = {
    // TODO: According to OV specs, the behavior of Sort::NONE is undefined.
    // ngraph::opset4::TopK::SortType::NONE,
    ngraph::opset4::TopK::SortType::SORT_INDICES,
    ngraph::opset4::TopK::SortType::SORT_VALUES,
};

INSTANTIATE_TEST_CASE_P(TopK, TopKLayerTest,
                        ::testing::Combine(                                              //
                            ::testing::ValuesIn(k),                                      //
                            ::testing::ValuesIn(axes),                                   //
                            ::testing::ValuesIn(modes),                                  //
                            ::testing::ValuesIn(sortTypes),                              //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(std::vector<size_t>({10, 10, 10})),        //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
                        TopKLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke, TopKLayerTest,
                        ::testing::Combine(                                //
                            ::testing::ValuesIn(std::vector<int64_t>{5}),  //
                            ::testing::ValuesIn(std::vector<int64_t>{2}),  //
                            ::testing::ValuesIn(std::vector<ngraph::opset4::TopK::Mode>{
                                ngraph::opset4::TopK::Mode::MIN}),  //
                            ::testing::ValuesIn(std::vector<ngraph::opset4::TopK::SortType>{
                                ngraph::opset4::TopK::SortType::SORT_VALUES}),           //
                            ::testing::ValuesIn(netPrecisions),                          //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),  //
                            ::testing::Values(InferenceEngine::Layout::ANY),             //
                            ::testing::Values(std::vector<size_t>({10, 10, 10})),        //
                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),         //
                        TopKLayerTest::getTestCaseName);

}  // namespace
