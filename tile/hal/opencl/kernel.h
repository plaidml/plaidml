// Copyright 2017, Vertex.AI. CONFIDENTIAL

#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "tile/base/hal.h"
#include "tile/hal/opencl/device_state.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

class Kernel final : public hal::Kernel {
 public:
  Kernel(const std::shared_ptr<DeviceState>& device_state, CLObj<cl_kernel> kernel, const lang::KernelInfo& info,
         boost::uuids::uuid kernel_uuid);

  std::shared_ptr<hal::Event> Run(const context::Context& ctx, const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                  const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                  bool enable_profiling) final;

 private:
  std::mutex mu_;
  std::shared_ptr<DeviceState> device_state_;
  CLObj<cl_kernel> kernel_;
  lang::KernelInfo ki_;
  boost::uuids::uuid kernel_uuid_;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
