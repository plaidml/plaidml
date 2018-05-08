// Copyright 2017, Vertex.AI. CONFIDENTIAL

#include "base/util/compat.h"
#include "base/util/factory.h"
#include "tile/hal/cpu/driver.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

[[gnu::unused]] char reg = []() -> char {
  FactoryRegistrar<hal::Driver>::Instance()->Register(
      [](const context::Context& ctx) -> std::unique_ptr<hal::Driver> {
        //
        return compat::make_unique<Driver>(ctx);
      },
      -9);
  return 0;
}();

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
