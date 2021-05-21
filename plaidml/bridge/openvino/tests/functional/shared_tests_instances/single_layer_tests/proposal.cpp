// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/proposal.hpp"
#include <vector>

using namespace LayerTestsDefinitions;

namespace {

const std::vector<size_t> baseSizes = {16, 8};
const std::vector<size_t> preNmsTopns = {80};
const std::vector<size_t> postNmsTopns = {10};
const std::vector<float> nmsThresholds = {1.0f, 0.5f};
const std::vector<size_t> minSizes = {0, 8};
const std::vector<std::vector<float>> ratios = {{1.0f}};
const std::vector<std::vector<float>> scales = {{1.0f, 2.0f}};
// feat_stride, normalize, box_size_scale, box_coordinate_scale, infer_probs
// are hard coded in openvino op setup

const auto clipNMSAttributes = ::testing::Combine(  //
    ::testing::ValuesIn(baseSizes),                 //
    ::testing::ValuesIn(preNmsTopns),               //
    ::testing::ValuesIn(postNmsTopns),              //
    ::testing::ValuesIn(nmsThresholds),             //
    ::testing::ValuesIn(minSizes),                  //
    ::testing::ValuesIn(ratios),                    //
    ::testing::ValuesIn(scales),                    //
    ::testing::Values(true),                        //
    ::testing::Values(true),                        //
    ::testing::Values(""));
const auto clipNMSParams = ::testing::Combine(  //
    clipNMSAttributes,                          //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));
INSTANTIATE_TEST_CASE_P(clipNMS, ProposalLayerTest, clipNMSParams, ProposalLayerTest::getTestCaseName);

const auto noClipAttributes = ::testing::Combine(  //
    ::testing::ValuesIn(baseSizes),                //
    ::testing::ValuesIn(preNmsTopns),              //
    ::testing::ValuesIn(postNmsTopns),             //
    ::testing::ValuesIn(nmsThresholds),            //
    ::testing::ValuesIn(minSizes),                 //
    ::testing::ValuesIn(ratios),                   //
    ::testing::ValuesIn(scales),                   //
    ::testing::Values(false),                      //
    ::testing::Values(false),                      //
    ::testing::Values(""));
const auto noClipParams = ::testing::Combine(  //
    noClipAttributes,                          //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));
INSTANTIATE_TEST_CASE_P(noClipNMS, ProposalLayerTest, noClipParams, ProposalLayerTest::getTestCaseName);

const auto tfDecodeAttributes = ::testing::Combine(  //
    ::testing::ValuesIn(baseSizes),                  //
    ::testing::ValuesIn(preNmsTopns),                //
    ::testing::ValuesIn(postNmsTopns),               //
    ::testing::ValuesIn(nmsThresholds),              //
    ::testing::ValuesIn(minSizes),                   //
    ::testing::ValuesIn(ratios),                     //
    ::testing::ValuesIn(scales),                     //
    ::testing::Values(true),                         //
    ::testing::Values(false),                        //
    ::testing::Values("tensorflow"));
const auto tfDecodeParams = ::testing::Combine(  //
    tfDecodeAttributes,                          //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));
INSTANTIATE_TEST_CASE_P(tfDecode, ProposalLayerTest, tfDecodeParams, ProposalLayerTest::getTestCaseName);

const auto smokeAttributes = ::testing::Combine(        //
    ::testing::Values(8),                               //
    ::testing::Values(50),                              //
    ::testing::Values(5),                               //
    ::testing::Values(0.5f),                            //
    ::testing::Values(8),                               //
    ::testing::Values(std::vector<float>{1.0f}),        //
    ::testing::Values(std::vector<float>{1.0f, 2.0f}),  //
    ::testing::Values(true),                            //
    ::testing::Values(true),                            //
    ::testing::Values(""));
const auto smokeParams = ::testing::Combine(  //
    smokeAttributes,                          //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));
INSTANTIATE_TEST_CASE_P(smoke, ProposalLayerTest, smokeParams, ProposalLayerTest::getTestCaseName);

}  // namespace
