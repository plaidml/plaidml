#include "base/context/eventlog.h"

#include <string>
#include <utility>

#include "base/context/context.h"

namespace gp = google::protobuf;

namespace vertexai {
namespace context {

EventLog::EventLog() : stream_uuid_(GetRandomUUID()) {}

EventLog::EventLog(boost::uuids::uuid stream_uuid) : stream_uuid_(stream_uuid) {}

std::size_t EventLog::GetClockIndex(const Clock* clock) {
  std::lock_guard<std::mutex> lock{mu_};
  auto res = clock_indicies_.insert(std::make_pair(clock, 0));
  if (res.second) {
    res.first->second = clock_indicies_.size();
  }
  return res.first->second;
}

void Clock::LogActivity(const Context& ctx, const char* verb, gp::Duration start_time, gp::Duration end_time) const {
  if (ctx.is_logging_events()) {
    proto::Event event;
    *event.mutable_parent_id() = ctx.activity_id();
    event.set_verb(verb);
    event.mutable_activity_id()->set_index(ctx.eventlog()->AllocActivityIndex());
    event.mutable_clock_id()->set_index(ctx.eventlog()->GetClockIndex(this));
    *event.mutable_start_time() = start_time;
    *event.mutable_end_time() = end_time;
    *event.mutable_domain_id() = ctx.domain_id();

    ctx.eventlog()->LogEvent(std::move(event));
  }
}

}  // namespace context
}  // namespace vertexai
