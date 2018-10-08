// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>

#include "tile/platform/local_machine/devinfo.h"
#include "tile/platform/local_machine/mem_strategy.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// DirectMemStrategy manages memory by copying buffers to and from devices.
class DirectMemStrategy final : public MemStrategy {
 public:
  DirectMemStrategy(const std::shared_ptr<DevInfo>& devinfo, hal::Memory* source);

  std::shared_ptr<MemChunk> MakeChunk(const context::Context& ctx, std::uint64_t size) const final;

 private:
  std::shared_ptr<DevInfo> devinfo_;
  hal::Memory* source_ = nullptr;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
