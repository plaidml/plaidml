// Copyright 2017-2018 Intel Corporation.

#include "base/util/any_factory_map.h"
#include "base/util/compat.h"
#include "base/util/env.h"
#include "tile/platform/local_machine/local_machine.h"
#include "tile/platform/local_machine/platform.h"
#include "tile/platform/stripejit/platform.h"

namespace vertexai {
namespace tile {
namespace local_machine {

class JitPlatformFactory final : public TypedAnyFactory<tile::Platform, proto::Platform> {
 public:
  std::unique_ptr<tile::Platform> MakeTypedInstance(const context::Context& ctx,
                                                    const proto::Platform& config) override;
};

std::unique_ptr<tile::Platform> JitPlatformFactory::MakeTypedInstance(const context::Context& ctx,
                                                                      const proto::Platform& config) {
  return std::make_unique<stripejit::Platform>();
}

[[gnu::unused]] char reg_jit = []() -> char {
  if (env::Get("STRIPE_JIT") == "1") {
    AnyFactoryMap<tile::Platform>::Instance()->Register(std::make_unique<JitPlatformFactory>());
  }
  return 0;
}();

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
