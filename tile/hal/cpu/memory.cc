// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cpu/memory.h"

#include <utility>

#include "tile/hal/cpu/arena.h"
#include "tile/hal/cpu/buffer.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

std::shared_ptr<hal::Buffer> Memory::MakeBuffer(std::uint64_t size, BufferAccessMask /* access */) {
  return std::make_shared<Arena>(size)->MakeBuffer(0, size);
}

std::shared_ptr<hal::Arena> Memory::MakeArena(std::uint64_t size, BufferAccessMask /* access */) {
  return std::make_shared<Arena>(size);
}

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
