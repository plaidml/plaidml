#include <gmock/gmock.h>
#include <google/protobuf/util/time_util.h>
#include <gtest/gtest.h>

#include <thread>

#include "base/context/context.h"
#include "base/util/error.h"
#include "base/util/logging.h"

using ::testing::Eq;

namespace vertexai {
namespace context {
namespace {

TEST(ContextTest, DeadlinePropagation) {
  Context context;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
  context.set_deadline(deadline);

  Activity activity{context, "context::TestActivity"};
  context = activity.ctx();

  EXPECT_THAT(context.deadline(), Eq(deadline));
}

TEST(ContextTest, Cancellation) {
  auto gate = std::make_shared<Gate>();
  Context context;

  EXPECT_THAT(context.cancelled(), Eq(false));

  context.set_gate(gate);
  EXPECT_THAT(context.cancelled(), Eq(false));

  gate->Close().wait();
  EXPECT_THAT(context.cancelled(), Eq(true));

  ASSERT_THROW(context.CheckCancelled(), error::Cancelled);
}

}  // namespace
}  // namespace context
}  // namespace vertexai
