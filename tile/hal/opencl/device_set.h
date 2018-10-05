// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "base/context/context.h"
#include "tile/base/hal.h"
#include "tile/hal/opencl/device.h"
#include "tile/hal/opencl/host_memory.h"
#include "tile/hal/opencl/ocl.h"
#include "tile/hal/opencl/opencl.pb.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

// DeviceSet implements the hal::DeviceSet model as a single OpenCL platform.
class DeviceSet final : public hal::DeviceSet {
 public:
  DeviceSet(const context::Context& ctx, std::uint32_t pidx, cl_platform_id pid);

  const std::vector<std::shared_ptr<hal::Device>>& devices() final;

  Memory* host_memory() final;

 private:
  std::vector<std::shared_ptr<hal::Device>> devices_;
  std::unique_ptr<Memory> host_memory_;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
