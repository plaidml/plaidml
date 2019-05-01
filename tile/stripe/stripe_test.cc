// Copyright 2019, Intel Corporation

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "tile/stripe/stripe.h"

using ::testing::Combine;
using ::testing::Eq;
using ::testing::Ne;
using ::testing::Values;

namespace vertexai {
namespace tile {
namespace stripe {
namespace {

using LocParam = std::tuple<Location, std::string>;

class StripeLocMatchTest : public ::testing::TestWithParam<LocParam> {};

TEST_P(StripeLocMatchTest, PatternMatches) {
  LocParam param = GetParam();
  EXPECT_THAT(std::get<0>(param), Eq(std::get<1>(param)));
}

class StripeLocFailTest : public ::testing::TestWithParam<LocParam> {};

TEST_P(StripeLocFailTest, PatternFails) {
  LocParam param = GetParam();
  EXPECT_THAT(std::get<0>(param), Ne(std::get<1>(param)));
}

INSTANTIATE_TEST_CASE_P(EmptyLocMatch, StripeLocMatchTest, Combine(Values(Location{}), Values("")));

INSTANTIATE_TEST_CASE_P(EmptyLocFail, StripeLocFailTest,
                        Combine(Values(Location{}), Values("*", "*[]", "*[*]", "*/*", "Dev", "Dev/*")));

INSTANTIATE_TEST_CASE_P(DevNoUnitLocMatch, StripeLocMatchTest,
                        Combine(Values(Location{{{"Dev"}}}), Values("*", "*[]", "Dev", "Dev[]")));

INSTANTIATE_TEST_CASE_P(DevNoUnitLocFail, StripeLocFailTest,
                        Combine(Values(Location{{{"Dev"}}}),
                                Values("", "*[*]", "*/*", "Dev/*", "XYZ", "Dev[4]", "Dev[*]")));

INSTANTIATE_TEST_CASE_P(DevZeroUnitLocMatch, StripeLocMatchTest,
                        Combine(Values(Location{{{"Dev", {}}}}), Values("*", "*[]", "Dev", "Dev[]")));

INSTANTIATE_TEST_CASE_P(DevZeroUnitLocFail, StripeLocFailTest,
                        Combine(Values(Location{{{"Dev", {}}}}),
                                Values("", "*[*]", "*/*", "Dev/*", "XYZ", "Dev[4]", "Dev[*]")));

INSTANTIATE_TEST_CASE_P(DevOneUnitLocMatch, StripeLocMatchTest,
                        Combine(Values(Location{{{"Dev", {4}}}}), Values("*", "*[*]", "Dev", "Dev[4]", "Dev[*]")));

INSTANTIATE_TEST_CASE_P(DevOneUnitLocFail, StripeLocFailTest,
                        Combine(Values(Location{{{"Dev", {4}}}}),
                                Values("", "*[]", "*/*", "Dev[]", "Dev/*", "XYZ", "Dev[6]")));

INSTANTIATE_TEST_CASE_P(NestedDevLocMatch, StripeLocMatchTest,
                        Combine(Values(Location{{{"Dev", {4}}, {"NDev", {6}}}}),
                                Values("*/*", "Dev/*", "Dev/*[6]", "*/NDev", "*[4]/*", "Dev/NDev")));

INSTANTIATE_TEST_CASE_P(NestedDevLocFail, StripeLocFailTest,
                        Combine(Values(Location{{{"Dev", {4}}, {"NDev", {6}}}}),
                                Values("", "*", "*[]", "*[*]", "Dev", "Dev[4]", "Dev[*]", "Dev[]", "XYZ", "Dev[6]")));

class StripeLocNoThrowTest : public ::testing::TestWithParam<const char*> {};

TEST_P(StripeLocNoThrowTest, ParserDoesNotThrow) {
  EXPECT_NO_THROW({
    if (Location() == GetParam()) {
      SUCCEED();
    }
  });
}

INSTANTIATE_TEST_CASE_P(ValidPatterns, StripeLocNoThrowTest,
                        Values("foo", "foo[]", "foo[1, *  ]/bar", "foo[1, *  ]/*[*]"));

class StripeLocThrowTest : public ::testing::TestWithParam<const char*> {};

TEST_P(StripeLocThrowTest, ParserThrows) {
  EXPECT_THROW(
      {
        if (Location() == GetParam()) {
          SUCCEED();
        }
      },
      std::runtime_error);
}

INSTANTIATE_TEST_CASE_P(InvalidPatterns, StripeLocThrowTest,
                        Values("foo[1, *  ]qux/bar", "foo[1, florp ]/bar", "foo[1, 2* ]/bar"));

}  // namespace
}  // namespace stripe
}  // namespace tile
}  // namespace vertexai
