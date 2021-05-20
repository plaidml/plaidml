// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/proposal.hpp"
#include <vector>

using namespace LayerTestsDefinitions;

namespace {

const std::vector<size_t> base_sizes = {16, 8};
const std::vector<size_t> pre_nms_topns = {40, 80};
const std::vector<size_t> post_nms_topns = {20, 40};
const std::vector<float> nms_threshes = {1.0f, 0.5f};
const std::vector<size_t> min_sizes = {0, 8};
const std::vector<std::vector<float>> ratios = {{1.0f}};
const std::vector<std::vector<float>> scales = {{1.0f, 2.0f}};
const std::vector<bool> clip_before_nms = {true, false};
const std::vector<bool> clip_after_nms = {true, false};
const std::vector<std::string> frameworks = {"", "tensorflow"};
// feat_stride, normalize, box_size_scale, box_coordinate_scale, infer_probs
// are hard coded in openvino op setup

const auto proposalAttributes = ::testing::Combine(  //
    ::testing::ValuesIn(base_sizes),                 //
    ::testing::ValuesIn(pre_nms_topns),              //
    ::testing::ValuesIn(post_nms_topns),             //
    ::testing::ValuesIn(nms_threshes),               //
    ::testing::ValuesIn(min_sizes),                  //
    ::testing::ValuesIn(ratios),                     //
    ::testing::ValuesIn(scales),                     //
    ::testing::ValuesIn(clip_before_nms),            //
    ::testing::ValuesIn(clip_after_nms),             //
    ::testing::ValuesIn(frameworks));

const auto proposalParams = ::testing::Combine(  //
    proposalAttributes,                          //
    ::testing::Values(CommonTestUtils::DEVICE_PLAIDML));

INSTANTIATE_TEST_CASE_P(Proposal, ProposalLayerTest, proposalParams, ProposalLayerTest::getTestCaseName);

const auto smokeAttributes = ::testing::Combine(        //
    ::testing::Values(8),                               //
    ::testing::Values(80),                              //
    ::testing::Values(40),                              //
    ::testing::Values(1.0f),                            //
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
