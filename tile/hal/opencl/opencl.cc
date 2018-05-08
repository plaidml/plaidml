// Copyright 2017, Vertex.AI.

#include "base/util/compat.h"
#include "base/util/factory.h"
#include "tile/hal/opencl/driver.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

[[gnu::unused]] char reg = []() -> char {
  FactoryRegistrar<hal::Driver>::Instance()->Register(
      [](const context::Context& ctx) -> std::unique_ptr<hal::Driver> { return compat::make_unique<Driver>(ctx); });
  return 0;
}();

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
