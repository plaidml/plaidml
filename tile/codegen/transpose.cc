// Copyright 2018, Intel Corporation

#include "tile/codegen/transpose.h"

#include "tile/codegen/localize.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

namespace {

struct BufferUsage {
  TileShape shape;
  Block* base_block;
  Refinement* base_ref;
};

using BufferUsageMap = std::map<std::string, BufferUsage>;

void CollectUsage(BufferUsageMap* usages, const Block& block, const AliasMap& map, const Tags& alloc_reqs) {
  auto acc_idxs = block.accumulation_idxs();
  for (const auto& ref : block.ref_ins()) {
    auto alias = map.at(ref->into);
    if (HasTags(*alias.base_block, alloc_reqs)) {
      auto it = usages->find(alias.base_name);
      if (it == usages->end()) {
        BufferUsage usage{TileShape(ref->access.size()), alias.base_block, alias.base_ref};
        std::tie(it, std::ignore) = usages->emplace(alias.base_name, usage);
      }
      for (size_t i = 0; i < ref->access.size(); i++) {
        const auto& access = ref->access[i];
        for (const auto& idx : acc_idxs) {
          if (access.getMap().count(idx->name)) {
            it->second.shape[i] += idx->range;
          }
        }
      }
    }
  }
}

using SizeStridePair = std::pair<uint64_t, int64_t>;

struct CompareSizeStridePair {
  bool operator()(const SizeStridePair& lhs, const SizeStridePair& rhs) const {
    if (lhs.first != rhs.first) {
      return lhs.first > rhs.first;
    }
    return lhs.second < rhs.second;
  }
};

}  // namespace

void TransposePass(Block* root, const proto::TransposePass& options) {
  auto reqs = FromProto(options.reqs());
  auto alloc_reqs = FromProto(options.alloc_reqs());
  BufferUsageMap usages;
  RunOnBlocks(root, reqs, [&](auto map, auto block) { CollectUsage(&usages, *block, map, alloc_reqs); });
  for (auto& item : usages) {
    size_t stride_one_idx = 0;
    uint64_t stride_one_range = 0;
    for (size_t i = 0; i < item.second.shape.size(); i++) {
      auto dim = item.second.shape[i];
      if (dim > stride_one_range) {
        stride_one_idx = i;
        stride_one_range = dim;
      }
    }
    auto base_ref = item.second.base_ref;
    std::multimap<SizeStridePair, size_t, CompareSizeStridePair> idxs_by_size;
    for (size_t i = 0; i < base_ref->shape.dims.size(); i++) {
      if (i != stride_one_idx) {
        auto dim = base_ref->shape.dims[i];
        idxs_by_size.emplace(std::make_pair(dim.size, dim.stride), i);
      }
    }
    IVLOG(3, "base: " << item.first                                                              //
                      << ", shape: " << item.second.shape                                        //
                      << ", stride_one: (" << stride_one_range << ", " << stride_one_idx << ")"  //
                      << ", idxs_by_size: " << idxs_by_size);
    IVLOG(3, "    old_ref: " << *base_ref);
    // Adjust strides
    int64_t stride = 1;
    {
      auto& dim = base_ref->shape.dims[stride_one_idx];
      dim.stride = stride;
      stride *= dim.size;
    }
    for (const auto& idx : idxs_by_size) {
      auto& dim = base_ref->shape.dims[idx.second];
      dim.stride = stride;
      stride *= dim.size;
    }
    IVLOG(3, "    new_ref: " << *base_ref);
    // Propagate the changes
    FixupRefs(item.second.base_block, base_ref->into);
  }
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
