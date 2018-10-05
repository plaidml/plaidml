// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/opencl/cl_mem_arena.h"

#include <utility>

#include "base/util/error.h"
#include "tile/hal/opencl/cl_mem_buffer.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

CLMemArena::CLMemArena(const std::shared_ptr<DeviceState>& device_state, std::uint64_t size, CLObj<cl_mem> mem)
    : device_state_{device_state}, size_{size}, mem_{std::move(mem)} {}

std::shared_ptr<hal::Buffer> CLMemArena::MakeBuffer(std::uint64_t offset, std::uint64_t size) {
  if (size_ < offset || size_ < size || size_ < (offset + size)) {
    throw error::OutOfRange{"Requesting memory outside arena bounds"};
  }

  Err err;
  cl_buffer_region region;
  region.origin = offset;
  region.size = size;
  CLObj<cl_mem> mem = clCreateSubBuffer(mem_.get(), 0, CL_BUFFER_CREATE_TYPE_REGION, &region, err.ptr());
  Err::Check(err, "Unable to allocate memory");
  return std::make_shared<CLMemBuffer>(device_state_, size, std::move(mem));
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
