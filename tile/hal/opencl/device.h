// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/context/context.h"
#include "tile/base/hal.h"
#include "tile/hal/opencl/device_state.h"
#include "tile/hal/opencl/ocl.h"
#include "tile/hal/opencl/opencl.pb.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

// Device implements the hal::Device model as a single OpenCL device.
class Device final : public hal::Device {
 public:
  Device(const context::Context& ctx, const CLObj<cl_context>& cl_ctx, cl_device_id did, proto::DeviceInfo info);

  void Initialize(const hal::proto::HardwareSettings& settings) final;

  std::string description() final;

  hal::Compiler* compiler() final { return compiler_.get(); }

  hal::Loader* loader() final { return nullptr; /* TODO: Support offline compilation */ }

  const std::unordered_map<std::string, std::unique_ptr<hal::Loader>>& il_loader_map() final { return il_loader_map_; }

  hal::Executor* executor() final { return executor_.get(); }

  const std::shared_ptr<DeviceState>& device_state() const { return device_state_; }

 private:
  const std::shared_ptr<DeviceState> device_state_;
  const std::unique_ptr<hal::Compiler> compiler_;
  const std::unordered_map<std::string, std::unique_ptr<hal::Loader>> il_loader_map_;
  const std::unique_ptr<hal::Executor> executor_;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
