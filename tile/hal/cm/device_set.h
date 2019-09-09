// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "base/context/context.h"
#include "tile/base/hal.h"
#include "tile/hal/cm/cm.pb.h"
#include "tile/hal/cm/device.h"
#include "tile/hal/cm/host_memory.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

class DeviceSet final : public hal::DeviceSet {
 public:
  explicit DeviceSet(const context::Context& ctx);

  const std::vector<std::shared_ptr<hal::Device>>& devices() final;

  Memory* host_memory() final;

 private:
  std::vector<std::shared_ptr<hal::Device>> devices_;
  std::unique_ptr<Memory> host_memory_;
};

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
