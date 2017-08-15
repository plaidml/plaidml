// Copyright 2017, Vertex.AI.

#include "tile/hal/opencl/local_memory.h"

#include <utility>

#include "tile/hal/opencl/cl_mem_arena.h"
#include "tile/hal/opencl/cl_mem_buffer.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

LocalMemory::LocalMemory(const std::shared_ptr<DeviceState>& device_state) : device_state_{device_state} {}

std::shared_ptr<hal::Buffer> LocalMemory::MakeBuffer(std::uint64_t size, BufferAccessMask /* access */) {
  Err err;
  CLObj<cl_mem> mem = clCreateBuffer(device_state_->cl_ctx().get(), CL_MEM_READ_WRITE, size, nullptr, err.ptr());
  Err::Check(err, "Unable to allocate device-local memory");
  return std::make_shared<CLMemBuffer>(device_state_, size, std::move(mem));
}

std::shared_ptr<hal::Arena> LocalMemory::MakeArena(std::uint64_t size, BufferAccessMask /* access */) {
  Err err;
  CLObj<cl_mem> mem = clCreateBuffer(device_state_->cl_ctx().get(), CL_MEM_READ_WRITE, size, nullptr, err.ptr());
  Err::Check(err, "Unable to allocate device-local memory");
  return std::make_shared<CLMemArena>(device_state_, size, std::move(mem));
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
