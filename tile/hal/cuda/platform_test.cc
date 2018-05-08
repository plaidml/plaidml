// Copyright 2018, Vertex.AI.

#include <gtest/gtest.h>

#include "base/util/type_url.h"
#include "tile/base/platform_test.h"
#include "tile/hal/cuda/hal.h"
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
    local_machine::proto::Platform config;
    config.add_hardware_configs()->mutable_sel()->set_value(true);
    return compat::make_unique<local_machine::Platform>(context::Context(), config);
  }));
  return factories;
}

INSTANTIATE_TEST_CASE_P(CudaHal, PlatformTest, ValuesIn(MakeFactories()));

}  // namespace
}  // namespace testing
}  // namespace tile
}  // namespace vertexai
