// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/opencl/driver.h"

#include <utility>

#include "tile/hal/opencl/info.h"
#include "tile/hal/opencl/ocl.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

Driver::Driver(const context::Context& ctx) {
  context::Activity enumerating{ctx, "tile::hal::opencl::Enumerating"};

  cl_uint platform_count;
  clGetPlatformIDs(0, nullptr, &platform_count);
  std::vector<cl_platform_id> platforms(platform_count);
  clGetPlatformIDs(platforms.size(), platforms.data(), nullptr);

  for (std::uint32_t pidx = 0; pidx < platforms.size(); ++pidx) {
    auto device_set = std::make_shared<DeviceSet>(enumerating.ctx(), pidx, platforms[pidx]);
    if (device_set->devices().size()) {
      device_sets_.emplace_back(std::move(device_set));
    }
  }
}

const std::vector<std::shared_ptr<hal::DeviceSet>>& Driver::device_sets() { return device_sets_; }

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
