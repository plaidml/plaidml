// Copyright 2017-2018 Intel Corporation.

#include "base/util/compat.h"
#include "base/util/factory.h"
#include "tile/hal/cpu/driver.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

[[gnu::unused]] char reg = []() -> char {
  FactoryRegistrar<hal::Driver>::Instance()->Register(
      "llvm",                                                                        //
      [](const context::Context& ctx) { return compat::make_unique<Driver>(ctx); },  //
      FactoryPriority::LOW);
  return 0;
}();

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
