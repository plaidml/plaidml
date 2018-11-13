#include "base/eventing/file/factory.h"

#include "base/eventing/file/eventlog.h"
#include "base/util/any_factory_map.h"
#include "base/util/compat.h"

namespace vertexai {
namespace eventing {
namespace file {

std::unique_ptr<context::EventLog> EventLogFactory::MakeTypedInstance(const context::Context& ctx,
                                                                      const proto::EventLog& config) {
  return std::make_unique<EventLog>(config);
}

[[gnu::unused]] char reg = []() -> char {
  AnyFactoryMap<context::EventLog>::Instance()->Register(std::make_unique<EventLogFactory>());
  return 0;
}();

}  // namespace file
}  // namespace eventing
}  // namespace vertexai
