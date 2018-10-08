// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ratio>

#include "tile/base/hal.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

class Memory final : public hal::Memory {
 public:
  Memory() {}

  std::uint64_t size_goal() const final {
    // TODO: Actually query the system physical memory size.
    return 16 * std::giga::num;
  }
  BufferAccessMask AllowedAccesses() const final { return BufferAccessMask::ALL; }
  std::size_t ArenaBufferAlignment() const final { return alignof(long double); /* TODO: Use std::max_align_t */ }

  std::shared_ptr<hal::Buffer> MakeBuffer(std::uint64_t size, BufferAccessMask access) final;
  std::shared_ptr<hal::Arena> MakeArena(std::uint64_t size, BufferAccessMask access) final;
};

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
