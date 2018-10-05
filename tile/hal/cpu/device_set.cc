// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cpu/device_set.h"

#include <utility>

#include "base/util/error.h"
#include "tile/hal/cpu/device.h"
#include "tile/hal/cpu/memory.h"
#include "tile/hal/util/selector.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

DeviceSet::DeviceSet() : host_memory_{new Memory} { devices_.emplace_back(new Device); }

const std::vector<std::shared_ptr<hal::Device>>& DeviceSet::devices() { return devices_; }

hal::Memory* DeviceSet::host_memory() { return host_memory_.get(); }

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
