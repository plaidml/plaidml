// Copyright 2017, Vertex.AI. CONFIDENTIAL

#pragma once

#include <memory>
#include <vector>

#include "base/context/context.h"
#include "tile/base/hal.h"
#include "tile/hal/cpu/cpu.pb.h"
#include "tile/hal/cpu/device_set.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

class Driver final : public hal::Driver {
 public:
  Driver(const context::Context& ctx, const proto::Driver& config);

  const std::vector<std::shared_ptr<hal::DeviceSet>>& device_sets() final;

 private:
  const std::shared_ptr<proto::Driver> config_;
  std::vector<std::shared_ptr<hal::DeviceSet>> device_sets_;
};

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
