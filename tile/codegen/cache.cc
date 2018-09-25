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
  CacheInfo info;

  // Get index count and verify consistency
  size_t idx_count = idxs.size();
  if (access.strides.size() != idx_count) {
    throw std::runtime_error("Invalid strides in ComputeCacheInfo");
  }

  // Copy across initial state
  info.idxs = idxs;
  info.far = access;

  // Make a list of index ids which are actually relevant
  std::vector<size_t> iids;
  for (size_t i = 0; i < idx_count; i++) {
    if (info.idxs[i].range > 1 && info.far.strides[i] != 0) {
      iids.push_back(i);
    }
  }

  // Sort ranges by absolute far stride
  std::sort(iids.begin(), iids.end(),
            [&](size_t a, size_t b) { return std::abs(info.far.strides[a]) < std::abs(info.far.strides[b]); });

  // Merge indexes.  Basically, we copy indexes from indexes to xfer_indexes
  // and merge them as we go.  We merge whenever a new index is an even multiple
  // of another and its ranges overlap.  We also track the multiplier and new ID
  std::vector<size_t> merge_into;
  std::vector<int64_t> merge_scale;
  for (size_t i : iids) {  // For each meaningful index
    bool merged = false;
    // Extract orignal stride info
    int64_t ostride = info.far.strides[i];
    uint64_t orange = info.idxs[i].range;
    const std::string& oname = info.idxs[i].name;
    for (size_t j = 0; j < info.xfer_idxs.size(); j++) {  // Look for an index in merged to merge it with
      // Get refs for merged stride info
      int64_t& mstride = info.xfer_far.strides[j];
      uint64_t& mrange = info.xfer_idxs[j].range;
      std::string& mname = info.xfer_idxs[j].name;
      // Compute the divisor + remainder
      auto div = std::div(std::abs(ostride), std::abs(mstride));
      // If it's mergeable, adjust the index in question and break
      if (div.rem == 0 && mrange >= static_cast<uint64_t>(div.quot)) {
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
      merge_into.push_back(info.xfer_idxs.size());
      merge_scale.push_back(Sign(ostride));
      info.xfer_idxs.push_back(info.idxs[i]);
      info.xfer_far.strides.push_back(std::abs(ostride));
    }
  }

  // Compute near strides
  int64_t cur_stride = 1;
  for (const auto& idx : info.xfer_idxs) {
    info.xfer_near.strides.push_back(cur_stride);
    cur_stride *= idx.range;
  }
  info.near.strides.resize(info.far.strides.size());
  for (size_t ri = 0; ri < iids.size(); ri++) {
    size_t i = iids[ri];
    info.near.strides[i] = info.xfer_near.strides[merge_into[ri]] * merge_scale[ri];
  }
  return info;
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
