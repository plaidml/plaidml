// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "tile/base/hal.h"
#include "tile/hal/cm/device_state.h"
#include "tile/hal/cm/err.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

class HostMemory final : public Memory {
 public:
  explicit HostMemory(std::shared_ptr<DeviceState> device_state);

  std::uint64_t size_goal() const final {
    // TODO: Actually query the system physical memory size.
    return 16 * std::giga::num;
  }

  BufferAccessMask AllowedAccesses() const final { return BufferAccessMask::ALL; }

  std::size_t ArenaBufferAlignment() const final { return device_state_->info().mem_base_addr_align(); }

  std::shared_ptr<hal::Buffer> MakeBuffer(std::uint64_t size, BufferAccessMask access) final;

  std::shared_ptr<hal::Arena> MakeArena(std::uint64_t size, BufferAccessMask access) final;

 private:
  std::shared_ptr<DeviceState> device_state_;
};

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
