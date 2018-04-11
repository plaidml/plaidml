// Copyright 2017, Vertex.AI.

#include <gtest/gtest.h>

#include "base/util/type_url.h"
#include "tile/base/platform_test.h"
#include "tile/hal/opencl/opencl.h"
#include "tile/platform/local_machine/platform.h"

using ::testing::ValuesIn;

namespace vertexai {
namespace tile {
namespace testing {
namespace {

std::vector<std::function<std::unique_ptr<Platform>()>> MakeFactories() {
  std::vector<std::function<std::unique_ptr<Platform>()>> factories;
  factories.emplace_back(std::function<std::unique_ptr<Platform>()>([] {
    context::Context ctx;
    hal::opencl::proto::Driver driver;
    local_machine::proto::Platform config;
    config.add_hals()->PackFrom(driver, kTypeVertexAI);
    config.add_hardware_configs()->mutable_sel()->set_value(true);
    return compat::make_unique<local_machine::Platform>(context::Context(), config);
  }));
  return factories;
}

INSTANTIATE_TEST_CASE_P(OclHal, PlatformTest, ValuesIn(MakeFactories()));

}  // namespace
}  // namespace testing
}  // namespace tile
}  // namespace vertexai
