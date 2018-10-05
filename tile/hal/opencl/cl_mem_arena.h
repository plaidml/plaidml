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

// An Arena implemented using a cl_mem object.
class CLMemArena final : public hal::Arena {
 public:
  CLMemArena(const std::shared_ptr<DeviceState>& device_state, std::uint64_t size, CLObj<cl_mem> mem);

  std::shared_ptr<hal::Buffer> MakeBuffer(std::uint64_t offset, std::uint64_t size) final;

 private:
  const std::shared_ptr<DeviceState> device_state_;
  std::uint64_t size_;
  const CLObj<cl_mem> mem_;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
