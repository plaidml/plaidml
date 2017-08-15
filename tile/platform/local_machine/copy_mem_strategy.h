// Copyright 2017, Vertex.AI.

#pragma once

#include <memory>

#include "tile/platform/local_machine/devinfo.h"
#include "tile/platform/local_machine/mem_strategy.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// CopyMemStrategy manages memory by copying buffers to and from devices.
class CopyMemStrategy final : public MemStrategy {
 public:
  explicit CopyMemStrategy(const std::shared_ptr<DevInfo>& devinfo);

  std::shared_ptr<MemChunk> MakeChunk(const context::Context& ctx, std::uint64_t size) const final;

 private:
  std::shared_ptr<DevInfo> devinfo_;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
