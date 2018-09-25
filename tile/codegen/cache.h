// Copyright 2018, Intel Corp.

#pragma once

#include <string>
#include <vector>

#include "tile/codegen/access.h"

namespace vertexai {
namespace tile {
namespace codegen {

struct CacheInfo {
  std::vector<stripe::Index> idxs;
  stripe::BufferAccess far;
  stripe::BufferAccess near;
  std::vector<stripe::Index> xfer_idxs;
  stripe::BufferAccess xfer_far;
  stripe::BufferAccess xfer_near;
};

inline bool operator==(const CacheInfo& lhs, const CacheInfo& rhs) {
  return std::tie(lhs.idxs, lhs.far, lhs.near, lhs.xfer_idxs, lhs.xfer_far, lhs.xfer_near) ==  //
         std::tie(rhs.idxs, rhs.far, rhs.near, rhs.xfer_idxs, rhs.xfer_far, rhs.xfer_near);
}

std::ostream& operator<<(std::ostream& os, const CacheInfo& x);

CacheInfo ComputeCacheInfo(const std::vector<stripe::Index>& idxs, const stripe::BufferAccess& access);

void ApplyCache(std::shared_ptr<stripe::Block> outer, size_t inner_stmt_id, const std::string& buffer);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
