// Copyright 2018, Intel Corporation

#include "tile/ocl_exec/kern_info.h"
#include "tile/math/util.h"

namespace vertexai {
namespace tile {
namespace codegen {

using math::IsPo2;

void KernelInfo::compute_gidx_packing() {
  const auto& idxs = block->idxs;

  // First, put index number + size into an array
  std::vector<std::pair<size_t, size_t>> to_place;
  for (size_t i = 0; i < idxs.size(); i++) {
    to_place.emplace_back(i, idxs[i].range);
  }

  // Then sort by size (biggest to smallest) and non-po2 first
  std::sort(to_place.begin(), to_place.end(),
            [](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) -> bool {
              return (std::make_tuple(IsPo2(a.second), a.second) > std::make_tuple(IsPo2(b.second), b.second));
            });

  // Now, place indexes into gid buckets, always use the smallest 'bucket'
  std::array<std::vector<size_t>, 3> buckets;
  for (const auto& p : to_place) {
    size_t which = std::min_element(group_dims.begin(), group_dims.end()) - group_dims.begin();
    buckets[which].push_back(p.first);
    group_dims[which] *= p.second;
  }

  // Now generate extraction data
  for (size_t i = 0; i < 3; i++) {
    size_t cur_below = 1;
    for (const size_t idx_id : buckets[i]) {
      auto& info = gidx_extract[idxs[idx_id].name];
      size_t cur = cur_below * idxs[idx_id].range;
      info.gid_base = i;
      if (cur != group_dims[i]) {
        info.pre_mod = cur;
      }
      info.pre_div = cur_below;
      cur_below = cur;
    }
  }
}

KernelInfo::KernelInfo(const std::shared_ptr<stripe::Block>& block_ptr)
    : block(block_ptr), local_dims(3, 1), group_dims(3, 1) {
  compute_gidx_packing();

  for (const auto& kvp : gidx_extract) {
    std::cout << kvp.first << " " << kvp.second.gid_base << " " << kvp.second.pre_mod << " " << kvp.second.pre_div
              << "\n";
  }
  std::cout << *block;
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
