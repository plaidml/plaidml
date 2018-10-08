// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/opencl/device.h"

#include <utility>

#include "base/util/compat.h"
#include "tile/hal/opencl/compiler.h"
#include "tile/hal/opencl/executor.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

Device::Device(const context::Context& ctx, const CLObj<cl_context>& cl_ctx, cl_device_id did, proto::DeviceInfo info)
    : device_state_{std::make_shared<DeviceState>(ctx, cl_ctx, did, std::move(info))},
      compiler_{compat::make_unique<Compiler>(device_state_)},
      executor_{compat::make_unique<Executor>(device_state_)} {}

void Device::Initialize(const hal::proto::HardwareSettings& settings) { device_state_->Initialize(settings); }

std::string Device::description() {  //
  return device_state()->info().vendor() + " " + device_state()->info().name() + " (OpenCL)";
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
