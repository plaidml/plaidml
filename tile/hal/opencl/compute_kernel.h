// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "tile/base/hal.h"
#include "tile/hal/opencl/device_state.h"
#include "tile/hal/opencl/kernel.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

class ComputeKernel final : public Kernel {
 public:
  ComputeKernel(const std::shared_ptr<DeviceState>& device_state, CLObj<cl_kernel> kernel, const lang::KernelInfo& info,
                context::proto::ActivityID kernel_id);

  std::shared_ptr<hal::Event> Run(const context::Context& ctx, const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                  const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                  bool enable_profiling) final;

 private:
  std::mutex mu_;
  std::shared_ptr<DeviceState> device_state_;
  CLObj<cl_kernel> kernel_;
  lang::KernelInfo ki_;
  context::proto::ActivityID kernel_id_;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
