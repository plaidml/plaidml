// Copyright 2017, Vertex.AI. CONFIDENTIAL

#pragma once

#include <memory>

#include "base/util/any_factory.h"
#include "tile/base/hal.h"
#include "tile/hal/cpu/cpu.pb.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

class DriverFactory final : public TypedAnyFactory<Driver, proto::Driver> {
 public:
  std::unique_ptr<hal::Driver> MakeTypedInstance(const context::Context& ctx, const proto::Driver& config) final;
};

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
