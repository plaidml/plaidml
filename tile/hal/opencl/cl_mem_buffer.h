// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <vector>

#include "tile/hal/opencl/buffer.h"
#include "tile/hal/opencl/device_state.h"
#include "tile/hal/opencl/ocl.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

// A Buffer implemented using a cl_mem object.
class CLMemBuffer final : public Buffer, public std::enable_shared_from_this<CLMemBuffer> {
 public:
  CLMemBuffer(const std::shared_ptr<DeviceState>& device_state, std::uint64_t size, CLObj<cl_mem> mem);

  void SetKernelArg(const CLObj<cl_kernel>& kernel, std::size_t index) final;

  boost::future<void*> MapCurrent(const std::vector<std::shared_ptr<hal::Event>>& deps) final;
  boost::future<void*> MapDiscard(const std::vector<std::shared_ptr<hal::Event>>& deps) final;
  std::shared_ptr<hal::Event> Unmap(const context::Context& ctx) final;

  cl_mem mem() const final { return mem_.get(); }

 private:
  static CLObj<cl_mem> MakeMem(const std::shared_ptr<DeviceState>& device_state, std::uint64_t size);

  const std::shared_ptr<DeviceState> device_state_;
  const CLObj<cl_mem> mem_;
  void* base_ = nullptr;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
