#pragma once

#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid.hpp>

#include <mutex>

#include "base/context/context.pb.h"

namespace vertexai {
namespace context {

class Context;

// An EventLog is the abstract interface for a thing that accepts event data.
class EventLog {
 public:
  virtual ~EventLog() {}

  virtual void LogEvent(proto::Event event) = 0;

  virtual void FlushAndClose() = 0;
};

// Clock simplifies logging events relative to some arbitrary clock.
// To use it, instantiate one Clock per clock source, and use it for recording activities relative to that source.
class Clock {
 public:
  Clock();

  // Logs an activity that occurred relative to this clock.  The activity verb should be in the same
  // space as the verbs used for the Activity object: strings localized to the namespace of the
  // creating component, e.g. "context::Test".  Note that if there's any cost at all to computing
  // the start and end times, the caller should check to see whether event logging's enabled.
  void LogActivity(const Context& ctx, const char* verb, google::protobuf::Duration start_time,
                   google::protobuf::Duration end_time) const;

  const boost::uuids::uuid& uuid() const { return clock_uuid_; }

 private:
  boost::uuids::uuid clock_uuid_;
};

// The clock instance corresponding to the local high-resolution clock.
const Clock& HighResolutionClock();

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
