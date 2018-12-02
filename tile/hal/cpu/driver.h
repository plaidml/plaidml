// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <vector>

#include "base/context/context.h"
#include "tile/base/hal.h"
#include "tile/hal/cpu/device_set.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

class Driver final : public hal::Driver {
 public:
  explicit Driver(const context::Context& ctx);

  const std::vector<std::shared_ptr<hal::DeviceSet>>& device_sets() final;

 private:
  std::vector<std::shared_ptr<hal::DeviceSet>> device_sets_;
};

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
