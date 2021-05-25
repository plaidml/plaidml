// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/detection_output.hpp"

using namespace LayerTestsDefinitions;

namespace {

const int numClasses = 11;
const int backgroundLabelId = 0;
const std::vector<int> topK = {75};
const std::vector<std::vector<int>> keepTopK = {
    {50},
    {100},
};
const std::vector<std::string> codeType = {"caffe.PriorBoxParameter.CORNER", "caffe.PriorBoxParameter.CENTER_SIZE"};
const float nmsThreshold = 0.5f;
const float confidenceThreshold = 0.3f;
const std::vector<bool> clipAfterNms = {true, false};
const std::vector<bool> clipBeforeNms = {true, false};
const std::vector<bool> decreaseLabelId = {true, false};
const float objectnessScore = 0.4f;
const std::vector<size_t> numberBatch = {1};

const auto commonAttributes = ::testing::Combine(  //
    ::testing::Values(numClasses),                 //
    ::testing::Values(backgroundLabelId),          //
    ::testing::ValuesIn(topK),                     //
    ::testing::ValuesIn(keepTopK),                 //
    ::testing::ValuesIn(codeType),                 //
    ::testing::Values(nmsThreshold),               //
    ::testing::Values(confidenceThreshold),        //
    ::testing::ValuesIn(clipAfterNms),             //
    ::testing::ValuesIn(clipBeforeNms),            //
    ::testing::ValuesIn(decreaseLabelId));

const auto smokeAttributes = ::testing::Combine(                                      //
    ::testing::Values(numClasses),                                                    //
    ::testing::Values(backgroundLabelId),                                             //
    ::testing::ValuesIn(std::vector<int>{75}),                                        //
    ::testing::ValuesIn(std::vector<std::vector<int>>{{50}}),                         //
    ::testing::ValuesIn(std::vector<std::string>{"caffe.PriorBoxParameter.CORNER"}),  //
    ::testing::Values(nmsThreshold),                                                  //
    ::testing::Values(confidenceThreshold),                                           //
    ::testing::ValuesIn(std::vector<bool>{false}),                                    //
    ::testing::ValuesIn(std::vector<bool>{false}),                                    //
    ::testing::ValuesIn(std::vector<bool>{false}));

/* =============== 3 inputs cases =============== */

const std::vector<ParamsWhichSizeDepends> specificParams3In = {
    // variance_encoded_in_target, share_location, normalized, input_height, input_weight,
    // Location, Confidence, Priors, ArmConfidence, ArmLocation.
    ParamsWhichSizeDepends{true, true, true, 1, 1, {1, 60}, {1, 165}, {1, 1, 60}, {}, {}},
    ParamsWhichSizeDepends{false, true, true, 1, 1, {1, 60}, {1, 165}, {1, 2, 60}, {}, {}},
    ParamsWhichSizeDepends{true, true, false, 10, 10, {1, 60}, {1, 165}, {1, 1, 75}, {}, {}},
    ParamsWhichSizeDepends{false, true, false, 10, 10, {1, 60}, {1, 165}, {1, 2, 75}, {}, {}},
};

const auto params3Inputs = ::testing::Combine(  //
    commonAttributes,                           //
    ::testing::ValuesIn(specificParams3In),     //
    ::testing::ValuesIn(numberBatch),           //
    ::testing::Values(0.0f),                    //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_CASE_P(DetectionOutput3In, DetectionOutputLayerTest, params3Inputs,
                        DetectionOutputLayerTest::getTestCaseName);

const std::vector<ParamsWhichSizeDepends> smokeParams3In = {
    ParamsWhichSizeDepends{true, true, true, 1, 1, {1, 60}, {1, 165}, {1, 1, 60}, {}, {}},
};

const auto smoke3Inputs = ::testing::Combine(     //
    smokeAttributes,                              //
    ::testing::ValuesIn(smokeParams3In),          //
    ::testing::ValuesIn(std::vector<size_t>{1}),  //
    ::testing::Values(0.0f),                      //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_CASE_P(smoke3In, DetectionOutputLayerTest, smoke3Inputs, DetectionOutputLayerTest::getTestCaseName);

/* =============== 5 inputs cases =============== */

const std::vector<ParamsWhichSizeDepends> specificParams5In = {
    ParamsWhichSizeDepends{true, true, true, 1, 1, {1, 60}, {1, 165}, {1, 1, 60}, {1, 30}, {1, 60}},
    ParamsWhichSizeDepends{false, true, true, 1, 1, {1, 60}, {1, 165}, {1, 2, 60}, {1, 30}, {1, 60}},
    ParamsWhichSizeDepends{true, true, false, 10, 10, {1, 60}, {1, 165}, {1, 1, 75}, {1, 30}, {1, 60}},
    ParamsWhichSizeDepends{false, true, false, 10, 10, {1, 60}, {1, 165}, {1, 2, 75}, {1, 30}, {1, 60}},
};

const auto params5Inputs = ::testing::Combine(  //
    commonAttributes,                           //
    ::testing::ValuesIn(specificParams5In),     //
    ::testing::ValuesIn(numberBatch),           //
    ::testing::Values(objectnessScore),         //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_CASE_P(DetectionOutput5In, DetectionOutputLayerTest, params5Inputs,
                        DetectionOutputLayerTest::getTestCaseName);

const std::vector<ParamsWhichSizeDepends> smokeParams5In = {
    ParamsWhichSizeDepends{true, true, true, 1, 1, {1, 60}, {1, 165}, {1, 1, 60}, {1, 30}, {1, 60}},
};

const auto smoke5Inputs = ::testing::Combine(     //
    smokeAttributes,                              //
    ::testing::ValuesIn(smokeParams5In),          //
    ::testing::ValuesIn(std::vector<size_t>{1}),  //
    ::testing::Values(0.4f),                      //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_CASE_P(smoke5In, DetectionOutputLayerTest, smoke5Inputs, DetectionOutputLayerTest::getTestCaseName);

}  // namespace
