#include <cstddef>
#include <map>
#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/grn.hpp"

using LayerTestsDefinitions::GrnLayerTest;

namespace {

INSTANTIATE_TEST_SUITE_P(GrnCheck1, GrnLayerTest,
                         ::testing::Combine(::testing::Values(InferenceEngine::Precision::FP32),
                                            ::testing::Values(std::vector<std::size_t>({4, 3, 3, 6})),
                                            ::testing::Values(0.01),
                                            ::testing::Values(CommonTestUtils::DEVICE_PLAIDML)),
                         GrnLayerTest::getTestCaseName);

}  // namespace
