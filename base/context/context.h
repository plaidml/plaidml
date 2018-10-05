// Copyright 2018 Intel Corporation.

#pragma once

#include <atomic>
#include <chrono>
#include <exception>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include <boost/uuid/nil_generator.hpp>
#include <boost/uuid/uuid.hpp>

#include "base/context/eventlog.h"
#include "base/context/gate.h"
#include "base/util/uuid.h"

namespace vertexai {
namespace context {

// Context tracks various bits of information related to the current activity being performed, potentially
// asynchronously and potentially across RPCs.
//
// Specifically, Context tracks:
//   The current activity (i.e. what's going on)
//   The current deadline
//   The current eventlog
//   The current cancellation state
//
// Context is a relatively lightweight value type (copies are for all intents and purposes identical to originals), but
// for performance, a Context is typically passed from one function to another by passing a const Context reference.
class Context {
 public:
  // Setters for building up Context objects.
  // TODO: Consider replacing these with a Builder pattern.
  Context& set_deadline(const std::chrono::steady_clock::time_point& deadline) {
    deadline_ = deadline;
    return *this;
  }
  Context& set_gate(const std::shared_ptr<Gate>& gate) {
    gate_ = gate;
    return *this;
  }
  Context& set_eventlog(std::shared_ptr<EventLog> eventlog) {
    eventlog_ = eventlog;
    return *this;
  }
  Context& set_is_logging_events(bool is_logging_events) {
    is_logging_events_ = is_logging_events;
    return *this;
  }
  Context& set_activity_id(proto::ActivityID activity_id) {
    activity_id_ = activity_id;
    return *this;
  }
  Context& set_domain_id(proto::ActivityID domain_id) {
    domain_id_ = domain_id;
    return *this;
  }

  // Deadline: the time point by which the context's activity should be complete.
  const std::chrono::steady_clock::time_point& deadline() const { return deadline_; }

  // Returns true if the context's associated gate (if any) has been closed.
  bool cancelled() const;

  // Throws an exception (of type Cancelled) if the context's associated gate (if any) has been closed.
  void CheckCancelled() const;

  // Gets the current gate (if any).
  const std::shared_ptr<Gate>& gate() const { return gate_; }

  // Gets the current event log (if any).
  const std::shared_ptr<EventLog>& eventlog() const { return eventlog_; }

  // Gets whether event logging is enabled or not.
  bool is_logging_events() const { return is_logging_events_ && eventlog_; }

  // Gets the current activity's instance id.
  proto::ActivityID activity_id() const { return activity_id_; }

  // Gets the current activity's domain id.
  proto::ActivityID domain_id() const { return domain_id_; }

  // Gets the full activity id, including the stream uuid.  This should be used for cross-stream references.
  proto::ActivityID full_activity_id() const {
    context::proto::ActivityID aid = activity_id();
    aid.set_stream_uuid(ToByteString(eventlog()->stream_uuid()));
    return aid;
  }

 private:
  static boost::uuids::nil_generator nil_uuid_gen;

  std::chrono::steady_clock::time_point deadline_{std::chrono::steady_clock::time_point::max()};
  std::shared_ptr<EventLog> eventlog_;
  bool is_logging_events_ = false;
  std::shared_ptr<Gate> gate_;
  proto::ActivityID activity_id_;
  proto::ActivityID domain_id_;
};

// Activity works with the current context's eventlog to automatically track the beginning and end of an event, using
// the system's high-resolution clock.  It's moveable, so that it can outlive the current thread.  Activities can be
// nested; to do this, pass the activity's context to sub-activities.
class Activity {
 public:
  // Construct an activity.  The verb can be an arbitrary string, but it should be localized to the
  // namespace of the creating component.  For example, "context::Test" makes a fine verb.
  Activity() {}
  Activity(const Context& parent, const std::string& verb, bool set_domain_id = false);
  ~Activity();

  Activity(const Activity& activity) = delete;
  Activity(Activity&& activity) noexcept = default;
  Activity& operator=(const Activity& activity) = delete;
  Activity& operator=(Activity&& activity) noexcept = default;

  const Context& ctx() const { return ctx_; }
  Context* mutable_ctx() { return &ctx_; }

  // Adds metadata to the activity's final report.
  // This is *not* synchronized.
  // If metadata cannot be added (out-of-memory, typically), the metadata is dropped.
  void AddMetadata(const google::protobuf::Message& metadata) noexcept;

 private:
  static google::protobuf::Duration Now() noexcept;

  Context ctx_;
  proto::Event final_event_;
};

}  // namespace context
}  // namespace vertexai
