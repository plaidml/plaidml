// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "tile/base/hal.h"
#include "tile/hal/opencl/buffer.h"
#include "tile/hal/opencl/device_state.h"
#include "tile/hal/opencl/kernel.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

class ZeroKernel final : public Kernel {
 public:
  ZeroKernel(const std::shared_ptr<DeviceState>& device_state, const lang::KernelInfo& kinfo,
             context::proto::ActivityID kid);

  std::shared_ptr<hal::Event> Run(const context::Context& ctx, const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                  const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                  bool enable_profiling) final;

 private:
  std::shared_ptr<DeviceState> device_state_;
  lang::KernelInfo kinfo_;
  context::proto::ActivityID kid_;

  CLObj<cl_event> FillBufferImpl(const DeviceState::Queue& queue, Buffer* buf, void* pattern, size_t pattern_size,
                                 const std::vector<cl_event>& deps);
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
