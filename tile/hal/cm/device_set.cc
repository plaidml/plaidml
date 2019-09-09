// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/device_set.h"

#include <string>
#include <utility>

#include <boost/regex.hpp>

#include "base/util/compat.h"
#include "base/util/env.h"
#include "base/util/error.h"
#include "tile/hal/cm/err.h"
#include "tile/hal/cm/runtime.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

DeviceSet::DeviceSet(const context::Context& ctx) {
  context::Activity platform_activity{ctx, "tile::hal::cm::Platform"};
  proto::PlatformInfo pinfo;
  pinfo.set_name("CM");

  context::Activity device_activity{platform_activity.ctx(), "tile::hal::cm::Device"};
  proto::DeviceInfo info;
  info.set_platform_name(pinfo.name());
  info.set_vendor("Intel(R) Corporation");
  info.set_name("Intel(R) Gen9 HD Graphics");
  info.set_mem_base_addr_align(0x1000);

  device_activity.AddMetadata(info);
  *info.mutable_platform_id() = platform_activity.ctx().activity_id();

  auto dev = std::make_shared<Device>(device_activity.ctx(), std::move(info));

  std::shared_ptr<Device> first_dev;
  first_dev = dev;

  devices_.emplace_back(std::move(dev));

  host_memory_ = std::make_unique<HostMemory>(first_dev->device_state());
}

const std::vector<std::shared_ptr<hal::Device>>& DeviceSet::devices() { return devices_; }

Memory* DeviceSet::host_memory() { return host_memory_.get(); }

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
