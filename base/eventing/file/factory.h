#pragma once

#include <memory>

#include "base/eventing/file/eventlog.pb.h"
#include "base/util/any_factory.h"

namespace vertexai {
namespace eventing {
namespace file {

class EventLogFactory final : public TypedAnyFactory<context::EventLog, proto::EventLog> {
 public:
  std::unique_ptr<context::EventLog> MakeTypedInstance(const context::Context& ctx,
                                                       const proto::EventLog& config) override;
};

}  // namespace file
}  // namespace eventing
}  // namespace vertexai
