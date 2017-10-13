// Copyright 2017, Vertex.AI.

#include "tile/hal/opencl/global_memory.h"

#include <utility>

#include "tile/hal/opencl/cl_mem_arena.h"
#include "tile/hal/opencl/cl_mem_buffer.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

GlobalMemory::GlobalMemory(const std::shared_ptr<DeviceState>& device_state) : device_state_{device_state} {}

std::shared_ptr<hal::Buffer> GlobalMemory::MakeBuffer(std::uint64_t size, BufferAccessMask /* access */) {
  Err err;
  CLObj<cl_mem> mem = clCreateBuffer(device_state_->cl_ctx().get(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size,
                                     nullptr, err.ptr());
  Err::Check(err, "Unable to allocate host-local memory");
  return std::make_shared<CLMemBuffer>(device_state_, size, std::move(mem));
}

std::shared_ptr<hal::Arena> GlobalMemory::MakeArena(std::uint64_t size, BufferAccessMask /* access */) {
  Err err;
  CLObj<cl_mem> mem = clCreateBuffer(device_state_->cl_ctx().get(), CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size,
                                     nullptr, err.ptr());
  Err::Check(err, "Unable to allocate host-local memory");
  return std::make_shared<CLMemArena>(device_state_, size, std::move(mem));
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
