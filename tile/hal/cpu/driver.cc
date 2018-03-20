// Copyright 2017, Vertex.AI. CONFIDENTIAL

#include "tile/hal/cpu/driver.h"

#include <memory>
#include <utility>

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

Driver::Driver(const context::Context& ctx, const proto::Driver& config)
    : config_{std::make_shared<proto::Driver>(config)} {
  device_sets_.emplace_back(std::make_shared<DeviceSet>());
}

const std::vector<std::shared_ptr<hal::DeviceSet>>& Driver::device_sets() { return device_sets_; }

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
