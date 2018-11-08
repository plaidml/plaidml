#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fstream>
#include <utility>

#include "base/eventing/file/eventlog.h"
#include "base/eventing/file/eventlog.pb.h"
#include "base/util/compat.h"
#include "base/util/logging.h"
#include "testing/matchers.h"

using ::testing::Eq;
using ::testing::EqualsProtoText;

namespace vertexai {
namespace eventing {
namespace file {
namespace {

constexpr static char kTestFilename[] = "eventlog.gz";

class EventLogTest : public ::testing::Test {
 protected:
  EventLogTest() { config_.set_filename(kTestFilename); }

  void SetUp() override { eventlog_ = compat::make_unique<EventLog>(config_); }

  proto::EventLog config_;
  std::unique_ptr<EventLog> eventlog_;
};

TEST_F(EventLogTest, CanReport) {
  {
    context::proto::Event event;
    event.set_verb("Hello, World!");
    eventlog_->LogEvent(std::move(event));
    eventlog_.reset();
  }

  {
    Reader reader{kTestFilename};
    context::proto::Event event;
    EXPECT_THAT(reader.Read(&event), Eq(true));
    EXPECT_THAT(event.activity_id().stream_uuid().length(), Eq(16));
    event.clear_activity_id();
    EXPECT_THAT(event, EqualsProtoText(R"(verb: "Hello, World!")"));
    EXPECT_THAT(reader.Read(&event), Eq(false));
  }
}

}  // namespace
}  // namespace file
}  // namespace eventing
}  // namespace vertexai
