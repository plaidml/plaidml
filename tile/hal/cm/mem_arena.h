// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <vector>

#include "tile/hal/cm/buffer.h"
#include "tile/hal/cm/device_state.h"
#include "tile/hal/cm/err.h"
#include "tile/hal/cm/mem_buffer.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

class CMMemArena final : public hal::Arena {
 public:
  CMMemArena(std::shared_ptr<DeviceState> device_state, std::uint64_t size);

  std::shared_ptr<hal::Buffer> MakeBuffer(std::uint64_t offset, std::uint64_t size) final;

 private:
  std::shared_ptr<DeviceState> device_state_;
  std::uint64_t size_;
};

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
