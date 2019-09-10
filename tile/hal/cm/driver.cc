// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/driver.h"

#include <utility>

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

Driver::Driver(const context::Context& ctx) {
  context::Activity enumerating{ctx, "tile::hal::cm::Enumerating"};
  auto device_set = std::make_shared<DeviceSet>(enumerating.ctx());
  if (device_set->devices().size()) {
    device_sets_.emplace_back(std::move(device_set));
  }
}

const std::vector<std::shared_ptr<hal::DeviceSet>>& Driver::device_sets() { return device_sets_; }

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
