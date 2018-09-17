#pragma once

#include <atomic>
#include <map>
#include <mutex>

#include <boost/uuid/uuid.hpp>

#include "base/context/context.pb.h"

namespace vertexai {
namespace context {

class Context;

// Clock simplifies logging events relative to some arbitrary clock.
// To use it, instantiate one Clock per clock source, and use it for recording activities relative to that source.
class Clock {
 public:
  // Logs an activity that occurred relative to this clock.  The activity verb should be in the same
  // space as the verbs used for the Activity object: strings localized to the namespace of the
  // creating component, e.g. "context::Test".  Note that if there's any cost at all to computing
  // the start and end times, the caller should check to see whether event logging's enabled.
  void LogActivity(const Context& ctx, const char* verb, google::protobuf::Duration start_time,
                   google::protobuf::Duration end_time) const;
};

// An EventLog is the abstract interface for a thing that accepts event data.
// Each log represents a logical stream of events; activities have a unique
// index within their stream.
class EventLog {
 public:
  EventLog();
  explicit EventLog(boost::uuids::uuid stream_uuid);
  virtual ~EventLog() {}

  virtual void LogEvent(proto::Event event) = 0;

  virtual void FlushAndClose() = 0;

  boost::uuids::uuid stream_uuid() const { return stream_uuid_; }

  std::size_t AllocActivityIndex() { return ++prev_activity_index_; }

  std::size_t GetClockIndex(const Clock* clock);

 private:
  std::mutex mu_;
  boost::uuids::uuid stream_uuid_;
  std::atomic_size_t prev_activity_index_;
  std::map<const Clock*, std::size_t> clock_indicies_;
};

// A simple converter from a std::chrono::duration to a Duration proto.
template <typename D>
void StdDurationToProto(google::protobuf::Duration* proto, const D& duration) {
  auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
  auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(duration - seconds);
  proto->set_seconds(seconds.count());
  proto->set_nanos(nanos.count());
}

}  // namespace context
}  // namespace vertexai
