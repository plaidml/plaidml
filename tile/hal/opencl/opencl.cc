// Copyright 2017, Vertex.AI. CONFIDENTIAL

#include "tile/hal/opencl/opencl.h"

#include "base/util/any_factory_map.h"
#include "base/util/compat.h"
#include "tile/hal/opencl/driver.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

std::unique_ptr<hal::Driver> DriverFactory::MakeTypedInstance(const context::Context& ctx,
                                                              const proto::Driver& config) {
  return compat::make_unique<Driver>(ctx, config);
}

[[gnu::unused]] char reg = []() -> char {
  AnyFactoryMap<hal::Driver>::Instance()->Register(compat::make_unique<DriverFactory>());
  return 0;
}();

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
