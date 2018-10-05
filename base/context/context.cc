// Copyright 2018 Intel Corporation.

#include "base/context/context.h"

#include <random>
#include <string>
#include <utility>

#include "base/context/eventlog.h"
#include "base/util/logging.h"
#include "base/util/type_url.h"

namespace bu = boost::uuids;
namespace pb = google::protobuf;

namespace vertexai {
namespace context {

bu::nil_generator Context::nil_uuid_gen;

bool Context::cancelled() const { return gate_ && !gate_->is_open(); }

void Context::CheckCancelled() const {
  if (gate_) {
    gate_->CheckIsOpen();
  }
}

pb::Duration Activity::Now() noexcept {
  pb::Duration result;
  StdDurationToProto(&result, std::chrono::high_resolution_clock::now().time_since_epoch());
  return result;
}

Activity::Activity(const Context& parent, const std::string& verb, bool set_domain_id) : ctx_{parent} {
  if (ctx_.is_logging_events()) {
    auto idx = ctx_.eventlog()->AllocActivityIndex();
    if (set_domain_id) {
      proto::ActivityID did;
      did.set_index(idx);
      ctx_.set_domain_id(did);
    }

    proto::Event event;
    *event.mutable_parent_id() = ctx_.activity_id();
    event.set_verb(verb);

    proto::ActivityID aid;
    aid.set_index(idx);
    ctx_.set_activity_id(aid);

    *event.mutable_activity_id() = aid;
    *event.mutable_start_time() = Now();
    *event.mutable_domain_id() = ctx_.domain_id();
    ctx_.eventlog()->LogEvent(std::move(event));

    *final_event_.mutable_activity_id() = aid;
  }
}

Activity::~Activity() {
  // N.B. If this Activity has been moved-from, the shared pointers in ctx_ will be empty.
  if (ctx_.is_logging_events()) {
    *final_event_.mutable_end_time() = Now();
    ctx_.eventlog()->LogEvent(std::move(final_event_));
  }
}

void Activity::AddMetadata(const google::protobuf::Message& metadata) noexcept {
  if (ctx_.is_logging_events()) {
    try {
      auto md_event = final_event_;
      md_event.add_metadata()->PackFrom(metadata, kTypeVertexAI);
      ctx_.eventlog()->LogEvent(std::move(md_event));
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to record eventlog metadata: " << e.what();
    } catch (...) {
      LOG(WARNING) << "Failed to record eventlog metadata";
    }
  }
}

}  // namespace context
}  // namespace vertexai
