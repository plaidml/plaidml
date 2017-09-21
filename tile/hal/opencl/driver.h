// Copyright 2017, Vertex.AI. CONFIDENTIAL

#pragma once

#include <memory>
#include <vector>

#include "base/context/context.h"
#include "tile/base/hal.h"
#include "tile/hal/opencl/device_set.h"
#include "tile/hal/opencl/opencl.pb.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

class Driver final : public hal::Driver {
 public:
  Driver(const context::Context& ctx, const proto::Driver& config);

  const std::vector<std::shared_ptr<hal::DeviceSet>>& device_sets() final;

 private:
  const std::shared_ptr<proto::Driver> config_;
  std::vector<std::shared_ptr<hal::DeviceSet>> device_sets_;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
