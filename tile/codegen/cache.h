// Copyright 2018, Intel Corp.

#pragma once

#include <string>

#include "tile/codegen/access.h"

namespace vertexai {
namespace tile {
namespace codegen {

struct CacheInfo {
  Indexes indexes;
  BufferAccess far;
  BufferAccess near;
  Indexes xfer_indexes;
  BufferAccess xfer_far;
  BufferAccess xfew_near;
};

CacheInfo ComputeCacheInfo(const Indexes& indexes, const BufferAccess& access);

void ApplyCache(stripe::proto::Block* block, const std::string& buffer);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
