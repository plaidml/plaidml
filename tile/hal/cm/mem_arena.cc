// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/mem_arena.h"

#include <utility>

#include "base/util/error.h"
#include "tile/hal/cm/mem_buffer.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

CMMemArena::CMMemArena(std::shared_ptr<DeviceState> device_state, std::uint64_t size)
    : device_state_{device_state}, size_{size} {}

std::shared_ptr<hal::Buffer> CMMemArena::MakeBuffer(std::uint64_t offset, std::uint64_t size) {
  if (size_ < offset || size_ < size || size_ < (offset + size)) {
    throw error::OutOfRange{"Requesting memory outside arena bounds"};
  }

  auto align = device_state_->info().mem_base_addr_align();

  void* base = CM_ALIGNED_MALLOC(size, align);
  memset(base, 0, size);

  CmBufferUP* pCmBuffer = nullptr;
  return std::make_shared<CMMemBuffer>(device_state_, size, pCmBuffer, base);
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
