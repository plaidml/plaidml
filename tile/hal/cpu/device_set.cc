// Copyright 2017, Vertex.AI. CONFIDENTIAL

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

DeviceSet::DeviceSet() : host_memory_{new Memory} {
#if ENABLE_CPU_DEVICE
    devices_.emplace_back(new Device);
#endif // ENABLE_CPU_DEVICE
}

const std::vector<std::shared_ptr<hal::Device>>& DeviceSet::devices() { return devices_; }

hal::Memory* DeviceSet::host_memory() { return host_memory_.get(); }

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
