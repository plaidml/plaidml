// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/host_memory.h"

#include <utility>

#include "tile/hal/cm/mem_arena.h"
#include "tile/hal/cm/mem_buffer.h"
#include "tile/hal/cm/runtime.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

HostMemory::HostMemory(std::shared_ptr<DeviceState> device_state) : device_state_{std::move(device_state)} {}

std::shared_ptr<hal::Buffer> HostMemory::MakeBuffer(std::uint64_t size, BufferAccessMask /* access */) {
  if (size < 16) {
    size = 16;
  }
  uint64_t buf_alignment_overflow_size = 3 * sizeof(float);
  size += buf_alignment_overflow_size;

  auto align = device_state_->info().mem_base_addr_align();

  void* void_buf_ = CM_ALIGNED_MALLOC(size, align);
  memset(void_buf_, 0, size);

  CmBufferUP* pCmBuffer = nullptr;
  return std::make_shared<CMMemBuffer>(device_state_, size, pCmBuffer, void_buf_);
}

std::shared_ptr<hal::Arena> HostMemory::MakeArena(std::uint64_t size, BufferAccessMask /* access */) {
  return std::make_shared<CMMemArena>(device_state_, size);
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
