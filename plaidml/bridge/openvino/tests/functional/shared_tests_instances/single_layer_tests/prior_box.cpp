// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/prior_box.hpp"

using LayerTestsDefinitions::PriorBoxLayerTest;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    // InferenceEngine::Precision::FP16,
};

// The count of tests will be too large for CI, close most checks here
const std::vector<std::vector<float>> minSizes = {{256.0f}};
const std::vector<std::vector<float>> maxSizes = {{315.0f}};
const std::vector<std::vector<float>> aspectRatios = {
    {1.0f},
    // {2.0f, 3.0f}
};
const std::vector<std::vector<float>> density = {{2.0f, 4.0f}};
const std::vector<std::vector<float>> fixedRatios = {{2.0f, 4.0f}};
const std::vector<std::vector<float>> fixedSizes = {{2.0f, 4.0f}};
const std::vector<bool> clip = {
    false,
    // true
};
const std::vector<bool> flip = {
    false,
    // true
};
const std::vector<float> steps = {
    1.0f,
    // 0.0f,
    // -1.0f
};
const std::vector<float> offsets = {
    0.0f,
    // 0.5f
};
const std::vector<std::vector<float>> variances = {
    {0.1f},
    // {0.1f, 0.2f, 0.3f, 0.4f}
};
const std::vector<bool> scaleAllSizes = {false, true};
// The following added bool vectors are used to mask corresponding attributes.
const std::vector<bool> useFixedSizes = {false, true};
const std::vector<bool> useFixedRatios = {false, true};
// Combine supports up to 10 arguments
const auto layerSpecificParams = ::testing::Combine(
    ::testing::ValuesIn(minSizes), ::testing::ValuesIn(maxSizes), ::testing::ValuesIn(aspectRatios),
    ::testing::ValuesIn(density), ::testing::ValuesIn(fixedRatios), ::testing::ValuesIn(fixedSizes),
    ::testing::ValuesIn(clip), ::testing::ValuesIn(flip), ::testing::ValuesIn(steps), ::testing::ValuesIn(offsets));

std::vector<std::vector<size_t>> layerShapes{{2, 3}};
std::vector<std::vector<size_t>> imageShapes{{8, 16}};

INSTANTIATE_TEST_CASE_P(smoke, PriorBoxLayerTest,
                        ::testing::Combine(layerSpecificParams, ::testing::ValuesIn(variances),
                                           ::testing::ValuesIn(scaleAllSizes), ::testing::ValuesIn(useFixedSizes),
                                           ::testing::ValuesIn(useFixedRatios), ::testing::ValuesIn(netPrecisions),
                                           ::testing::ValuesIn(layerShapes), ::testing::ValuesIn(imageShapes),
                                           ::testing::Values(CommonTestUtils::DEVICE_PLAIDML),
                                           ::testing::Values(std::map<std::string, std::string>({}))),
                        PriorBoxLayerTest::getTestCaseName);
}  // namespace
