#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "base/util/error.h"
#include "plaidml/base/base.h"
#include "plaidml/base/status.h"
#include "plaidml/base/status_strings.h"

using ::testing::Eq;
using ::testing::StrEq;
using ::testing::StrNe;

namespace vertexai {
namespace {

constexpr char kSomethingBadStr[] = "Something bad happened";

TEST(StatusTest, SetStatus) {
  SetLastStatus(VAI_STATUS_UNKNOWN, kSomethingBadStr);
  EXPECT_THAT(vai_last_status(), Eq(VAI_STATUS_UNKNOWN));
  EXPECT_THAT(vai_last_status_str(), StrEq(kSomethingBadStr));
}

TEST(StatusTest, BadSetStatus) {
  SetLastStatus(VAI_STATUS_UNKNOWN, nullptr);
  EXPECT_THAT(vai_last_status(), Eq(VAI_STATUS_INTERNAL));
  EXPECT_THAT(vai_last_status_str(), StrEq(status_strings::kInternal));
}

TEST(StatusTest, ClearStatus) {
  SetLastStatus(VAI_STATUS_UNKNOWN, kSomethingBadStr);
  EXPECT_THAT(vai_last_status(), Eq(VAI_STATUS_UNKNOWN));
  EXPECT_THAT(vai_last_status_str(), StrEq(kSomethingBadStr));
  vai_clear_status();
  EXPECT_THAT(vai_last_status(), Eq(VAI_STATUS_OK));
  EXPECT_THAT(vai_last_status_str(), StrEq(status_strings::kOk));
}

TEST(StatusTest, SetLastException) {
  SetLastException(std::make_exception_ptr(error::Unknown{kSomethingBadStr}));
  EXPECT_THAT(vai_last_status(), Eq(VAI_STATUS_UNKNOWN));
  EXPECT_THAT(vai_last_status_str(), StrEq(kSomethingBadStr));
}

TEST(StatusTest, OOM) {
  SetLastException(std::make_exception_ptr(std::bad_alloc{}));
  EXPECT_THAT(vai_last_status(), Eq(VAI_STATUS_RESOURCE_EXHAUSTED));
  EXPECT_THAT(vai_last_status_str(), StrEq(status_strings::kOom));
}

}  // namespace
}  // namespace vertexai
