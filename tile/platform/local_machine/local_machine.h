// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>

#include "base/util/any_factory.h"
#include "tile/base/platform.h"
#include "tile/platform/local_machine/local_machine.pb.h"

namespace vertexai {
namespace tile {
namespace local_machine {

class PlatformFactory final : public TypedAnyFactory<tile::Platform, proto::Platform> {
 public:
  std::unique_ptr<tile::Platform> MakeTypedInstance(const context::Context& ctx,
                                                    const proto::Platform& config) override;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
