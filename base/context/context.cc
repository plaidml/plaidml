// Copyright Vertex.AI.

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

Activity::Activity(const Context& parent, const std::string& verb, bool set_domain_uuid) : ctx_{parent} {
  auto instance_uuid = GetRandomUUID();
  if (set_domain_uuid) {
    ctx_.set_domain_uuid(instance_uuid);
  }
  if (ctx_.is_logging_events()) {
    auto instance_uuid_str = ToByteString(instance_uuid);

    proto::Event event;
    *event.mutable_parent_instance_uuid() = ToByteString(ctx_.activity_uuid());
    event.set_verb(verb);
    *event.mutable_instance_uuid() = instance_uuid_str;
    *event.mutable_clock_uuid() = ToByteString(HighResolutionClock().uuid());
    *event.mutable_start_time() = Now();
    *event.mutable_domain_uuid() = ToByteString(ctx_.domain_uuid());
    ctx_.eventlog()->LogEvent(std::move(event));

    ctx_.set_activity_uuid(instance_uuid);

    *final_event_.mutable_instance_uuid() = std::move(instance_uuid_str);
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
