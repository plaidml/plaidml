// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>

#include "tile/base/hal.h"
#include "tile/platform/local_machine/devinfo.h"
#include "tile/platform/local_machine/mem_cache.h"
#include "tile/platform/local_machine/mem_strategy.h"

namespace vertexai {
namespace tile {
namespace local_machine {

// TmpMemStrategy provides a source of temporary memory chunks.
//
// Chunks allocated by TmpMemStrategy may not be directly accessible to the host; map and unmap calls may fail.
//
// Memory described by chunks may be reused when the chunk is deleted; callers must make sure to maintain chunk
// references as long as the underlying memory is in use.
class TmpMemStrategy final : public MemStrategy {
 public:
  TmpMemStrategy(const std::shared_ptr<DevInfo>& devinfo, hal::Memory* source);

  std::shared_ptr<MemChunk> MakeChunk(const context::Context& ctx, std::uint64_t size) const final;

 private:
  std::shared_ptr<DevInfo> devinfo_;
  hal::Memory* source_ = nullptr;
  std::shared_ptr<MemCache> cache_;
};

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
