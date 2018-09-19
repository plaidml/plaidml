// Copyright 2018, Intel Corp.

#include "tile/codegen/cache.h"

namespace vertexai {
namespace tile {
namespace codegen {

CacheInfo ComputeCacheInfo(const Indexes& indexes, const BufferAccess& access) {
  CacheInfo r;

  // Get index count and verify consistency
  size_t idx_count = indexes.size();
  if (access.strides.size() != idx_count) {
    throw std::runtime_error("Invalid strides in ComputeCacheInfo");
  }

  // Copy across initial state
  r.indexes = indexes;
  r.far = access;

  // Make a list of index ids which are actually relevant
  std::vector<size_t> iids;
  for (size_t i = 0; i < idx_count; i++) {
    if (r.indexes[i].range > 1 && r.far.strides[i] != 0) {
      iids.push_back(i);
    }
  }

  // Sort ranges by absolute far stride
  std::sort(iids.begin(), iids.end(),
            [&](size_t a, size_t b) { return std::abs(r.far.strides[a]) < std::abs(r.far.strides[b]); });

  /*
  // Merge indexes.  Basically, we copy indexes from indexes to xfer_indexes
  // and merge them as we go.  We merge whenever a new index is an even multiple
  // of another and its ranges overlap.
  for (size_t i : iids) {  // For each meaningful index
    for (size_t j = 0; j < xfer_indexes.size(); j++) {

    for (auto& mi : merged_) {  // Look for an index in merged to merge it with
      // Compute the divisor + remainder
      auto div = std::div(std::abs(oi.stride), std::abs(mi.stride));
      // If it's mergeable, adjust the index in question and break
      if (div.rem == 0 && mi.range >= div.quot) {
        mi.range += div.quot * (oi.range - 1);
        mi.name += string("_") + oi.name;
        oi.merge_scale = div.quot * Sign(oi.stride);
        break;
      }
    }
    // Create a new index if this index was not merged.
    if (oi.merge_scale == 0) {
      merged_.emplace_back(oi);
      oi.merge_scale = Sign(oi.stride);
    }
  }
  */

  return r;
}

void ApplyCache(stripe::proto::Block* block, const std::string& buffer) {
  auto accesses = ComputeAccess(*block, buffer);
  if (accesses.size() != 1) {
    throw std::runtime_error("Currently we don't support multi-access caching");
  }
  auto access = accesses[0];
  if (access.is_write) {
    throw std::runtime_error("Currently we only support caching of reads");
  }
  ComputeCacheInfo(access.indexes, access.access);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
