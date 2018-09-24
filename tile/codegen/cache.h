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

  inline bool operator==(const CacheInfo& o) const {
    return std::tie(idxs, far, near, xfer_idxs, xfer_far, xfer_near) ==  //
           std::tie(o.idxs, o.far, o.near, o.xfer_idxs, o.xfer_far, o.xfer_near);
  }
};

std::ostream& operator<<(std::ostream& os, const CacheInfo& x);

CacheInfo ComputeCacheInfo(const std::vector<stripe::Index>& idxs, const stripe::BufferAccess& access);

void ApplyCache(stripe::Block* block, const std::string& buffer);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
