// Copyright 2017-2018 Intel Corporation.

#include "base/util/compat.h"
#include "base/util/factory.h"
#include "tile/hal/opencl/driver.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

[[gnu::unused]] char reg = []() -> char {
  FactoryRegistrar<hal::Driver>::Instance()->Register(
      "opencl",                                                                      //
      [](const context::Context& ctx) { return compat::make_unique<Driver>(ctx); },  //
#ifdef __APPLE__
      FactoryPriority::DEFAULT);
#else
      FactoryPriority::HIGH);
#endif
  return 0;
}();

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
