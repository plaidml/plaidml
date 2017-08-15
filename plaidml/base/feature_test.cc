#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "plaidml/base/base.h"
#include "plaidml/base/status_strings.h"

using ::testing::Eq;
using ::testing::IsNull;
using ::testing::StrEq;

namespace vertexai {
namespace {

TEST(FeatureTest, NoFeatures) {
  EXPECT_THAT(vai_query_feature(VAI_FEATURE_ID_RESERVED), IsNull());
  EXPECT_THAT(vai_last_status(), Eq(VAI_STATUS_NOT_FOUND));
  EXPECT_THAT(vai_last_status_str(), StrEq(status_strings::kNoSuchFeature));
}
}  // namespace
}  // namespace vertexai
