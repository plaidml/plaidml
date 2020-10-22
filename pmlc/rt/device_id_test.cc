// Copyright 2020 Intel Corporation

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "pmlc/rt/internal.h"

using ::testing::Not;

namespace pmlc::rt {
namespace {

MATCHER(IsDeviceID, negation ? "is not a valid device identifier"
                             : "is a valid device identifier") {
  return std::regex_match(arg, getDeviceIDRegex());
}

class ValidTest : public ::testing::TestWithParam<const char *> {};

TEST_P(ValidTest, Matches) { EXPECT_THAT(GetParam(), IsDeviceID()); }

INSTANTIATE_TEST_SUITE_P(ValidDeviceIDs, ValidTest,
                         ::testing::Values("llvm_cpu.0", "intel_gen.0",
                                           "vulkan.0", "gen9-intel-vulkan.0"));

class InvalidTest : public ::testing::TestWithParam<const char *> {};

TEST_P(InvalidTest, Rejects) { EXPECT_THAT(GetParam(), Not(IsDeviceID())); }

INSTANTIATE_TEST_SUITE_P(
    InvalidDeviceIDs, InvalidTest,
    ::testing::Values("no_device_index", "no_device_dot0", "9leading_digit.0",
                      "-leading-dash.0", "trailing-dash-.0", "double--dash.0"));

} // namespace
} // namespace pmlc::rt
