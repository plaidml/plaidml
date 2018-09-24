// Copyright 2018, Intel Corp.

#include "tile/codegen/cache.h"

#include <algorithm>

#include "base/util/stream_container.h"

namespace vertexai {
namespace tile {
namespace codegen {

std::ostream& operator<<(std::ostream& os, const CacheInfo& x) {
  os << "Base: idxs=" << StreamContainer(x.idxs) << ", far=" << x.far << ", near=" << x.near << "\n";
  os << "Xfer: idxs=" << StreamContainer(x.xfer_idxs) << ", far=" << x.xfer_far << ", near=" << x.xfer_near << "\n";
  return os;
}

static int64_t Sign(int64_t x) { return (x < 0 ? -1 : 1); }

CacheInfo ComputeCacheInfo(const std::vector<stripe::Index>& idxs, const stripe::BufferAccess& access) {
  CacheInfo r;

  // Get index count and verify consistency
  size_t idx_count = idxs.size();
  if (access.strides.size() != idx_count) {
    throw std::runtime_error("Invalid strides in ComputeCacheInfo");
  }

  // Copy across initial state
  r.idxs = idxs;
  r.far = access;

  // Make a list of index ids which are actually relevant
  std::vector<size_t> iids;
  for (size_t i = 0; i < idx_count; i++) {
    if (r.idxs[i].range > 1 && r.far.strides[i] != 0) {
      iids.push_back(i);
    }
  }

  // Sort ranges by absolute far stride
  std::sort(iids.begin(), iids.end(),
            [&](size_t a, size_t b) { return std::abs(r.far.strides[a]) < std::abs(r.far.strides[b]); });

  // Merge indexes.  Basically, we copy indexes from indexes to xfer_indexes
  // and merge them as we go.  We merge whenever a new index is an even multiple
  // of another and its ranges overlap.  We also track the multiplier and new ID
  std::vector<size_t> merge_into;
  std::vector<int64_t> merge_scale;
  for (size_t i : iids) {  // For each meaningful index
    bool merged = false;
    // Extract orignal stride info
    int64_t ostride = r.far.strides[i];
    uint64_t orange = r.idxs[i].range;
    const std::string& oname = r.idxs[i].name;
    for (size_t j = 0; j < r.xfer_idxs.size(); j++) {  // Look for an index in merged to merge it with
      // Get refs for merged stride info
      int64_t& mstride = r.xfer_far.strides[j];
      uint64_t& mrange = r.xfer_idxs[j].range;
      std::string& mname = r.xfer_idxs[j].name;
      // Compute the divisor + remainder
      auto div = std::div(std::abs(ostride), std::abs(mstride));
      // If it's mergeable, adjust the index in question and break
      if (div.rem == 0 && mrange >= div.quot) {
        mrange += div.quot * (orange - 1);
        mname += std::string("_") + oname;
        merge_into.push_back(j);
        merge_scale.push_back(div.quot * Sign(ostride));
        merged = true;
        break;
      }
    }
    // Create a new index if this index was not merged.
    if (!merged) {
      merge_into.push_back(r.xfer_idxs.size());
      merge_scale.push_back(Sign(ostride));
      r.xfer_idxs.push_back(r.idxs[i]);
      r.xfer_far.strides.push_back(std::abs(ostride));
    }
  }

  // Compute near strides
  int64_t cur_stride = 1;
  for (const auto& idx : r.xfer_idxs) {
    r.xfer_near.strides.push_back(cur_stride);
    cur_stride *= idx.range;
  }
  r.near.strides.resize(r.far.strides.size());
  for (size_t ri = 0; ri < iids.size(); ri++) {
    size_t i = iids[ri];
    r.near.strides[i] = r.xfer_near.strides[merge_into[ri]] * merge_scale[ri];
  }
  return r;
}

void ApplyCache(stripe::Block* block, const std::string& buffer) {
  auto accesses = ComputeAccess(*block, buffer);
  if (accesses.size() != 1) {
    throw std::runtime_error("Currently we don't support multi-access caching");
  }
  auto access = accesses[0];
  if (access.is_write) {
    throw std::runtime_error("Currently we only support caching of reads");
  }
  ComputeCacheInfo(access.idxs, access.access);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
