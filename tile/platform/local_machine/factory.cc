// Copyright 2017-2018 Intel Corporation.

#include "base/util/any_factory_map.h"
#include "base/util/compat.h"
#include "tile/platform/local_machine/local_machine.h"
#include "tile/platform/local_machine/platform.h"

namespace vertexai {
namespace tile {
namespace local_machine {

std::unique_ptr<tile::Platform> PlatformFactory::MakeTypedInstance(const context::Context& ctx,
                                                                   const proto::Platform& config) {
  return compat::make_unique<Platform>(ctx, config);
}

[[gnu::unused]] char reg = []() -> char {
  AnyFactoryMap<tile::Platform>::Instance()->Register(compat::make_unique<PlatformFactory>());
  return 0;
}();

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
