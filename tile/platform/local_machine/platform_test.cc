// Copyright 2017, Vertex.AI.

#include "tile/base/platform_test.h"

#include <gtest/gtest.h>

#include "base/config/config.h"
#include "base/context/context.h"
#include "base/util/any_factory_map.h"
#include "base/util/compat.h"
#include "base/util/type_url.h"
#include "plaidml/plaidml.pb.h"
#ifdef HAVE_CPU_HAL
#include "tile/hal/cpu/cpu.pb.h"
#endif  // defined(HAVE_CPU_HAL)
#include "testing/plaidml_config.h"
#include "tile/hal/opencl/opencl.pb.h"
#include "tile/platform/local_machine/local_machine.pb.h"
#include "tile/platform/local_machine/platform.h"

using ::testing::ValuesIn;

namespace vertexai {
namespace tile {
namespace testing {
namespace {

#ifdef HAVE_CPU_HAL
std::string ReadFile(const std::string& name) {
  std::ifstream file(name);
  return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}
#endif

void LoadConfig(const std::string& config_str, std::vector<std::function<std::unique_ptr<Platform>()>>* factories) {
  auto config = vertexai::ParseConfig<plaidml::proto::Config>(config_str);
  factories->emplace_back(std::function<std::unique_ptr<Platform>()>([config] {
    context::Context ctx;
    return vertexai::AnyFactoryMap<tile::Platform>::Instance()->MakeInstance(ctx, config.platform());
  }));
}

std::vector<std::function<std::unique_ptr<Platform>()>> MakeFactories() {
  std::vector<std::function<std::unique_ptr<Platform>()>> factories;
  LoadConfig(vertexai::testing::PlaidMLConfig(), &factories);
#ifdef HAVE_CPU_HAL
  LoadConfig(ReadFile("testing/tile_llvm.json"), &factories);
#endif  // defined(HAVE_CPU_HAL)
  return factories;
}

INSTANTIATE_TEST_CASE_P(LocalMachine, PlatformTest, ValuesIn(MakeFactories()));

}  // namespace
}  // namespace testing
}  // namespace tile
}  // namespace vertexai
