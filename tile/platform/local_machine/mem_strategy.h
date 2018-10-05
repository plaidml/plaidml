// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>

#include "base/context/context.h"
#include "tile/platform/local_machine/mem_chunk.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// A MemStrategy encapsulates some mechanism for managing Tile memory buffers.
class MemStrategy {
 public:
  virtual ~MemStrategy() {}

  // Allocates a memory object for kernels to use.
  virtual std::shared_ptr<MemChunk> MakeChunk(const context::Context& ctx, std::uint64_t size) const = 0;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
