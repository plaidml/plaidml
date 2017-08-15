#include "base/context/eventlog.h"

#include <string>
#include <utility>

#include "base/context/context.h"

namespace gp = google::protobuf;

namespace vertexai {
namespace context {

Clock::Clock() : clock_uuid_(GetRandomUUID()) {}

void Clock::LogActivity(const Context& ctx, const char* verb, gp::Duration start_time, gp::Duration end_time) const {
  if (ctx.is_logging_events()) {
    proto::Event event;
    *event.mutable_parent_instance_uuid() = ToByteString(ctx.activity_uuid());
    event.set_verb(verb);
    *event.mutable_instance_uuid() = ToByteString(GetRandomUUID());
    *event.mutable_clock_uuid() = ToByteString(clock_uuid_);
    *event.mutable_start_time() = start_time;
    *event.mutable_end_time() = end_time;
    *event.mutable_domain_uuid() = ToByteString(ctx.domain_uuid());

    ctx.eventlog()->LogEvent(std::move(event));
  }
}

const Clock& HighResolutionClock() {
  static const Clock hrt_clock;
  return hrt_clock;
}

}  // namespace context
}  // namespace vertexai
