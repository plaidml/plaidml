// Copyright 2017, Vertex.AI.

#pragma once

#include <memory>

#include "base/util/any_factory.h"
#include "tile/base/hal.h"
#include "tile/hal/opencl/opencl.pb.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

class DriverFactory final : public TypedAnyFactory<Driver, proto::Driver> {
 public:
  std::unique_ptr<hal::Driver> MakeTypedInstance(const context::Context& ctx, const proto::Driver& config) final;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
