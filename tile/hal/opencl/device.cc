// Copyright 2017, Vertex.AI.

#include "tile/hal/opencl/device.h"

#include <utility>

#include "base/util/compat.h"
#include "tile/hal/opencl/compiler.h"
#include "tile/hal/opencl/executor.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

Device::Device(const context::Context& ctx, const CLObj<cl_context>& cl_ctx, cl_device_id did,
               const std::shared_ptr<proto::Driver>& config, proto::DeviceInfo info)
    : device_state_{std::make_shared<DeviceState>(ctx, cl_ctx, did, config, std::move(info))},
      compiler_{compat::make_unique<Compiler>(device_state_)},
      executor_{compat::make_unique<Executor>(device_state_)} {}

void Device::Initialize(const context::Context& ctx, const hal::proto::HardwareSettings& settings) {
  device_state_->Initialize(ctx, settings);
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
