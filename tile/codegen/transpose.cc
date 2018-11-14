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
      IVLOG(3, "block: " << block.name                                  //
                         << ", ref: " << *ref                           //
                         << ", base_block: " << alias.base_block->name  //
                         << ", base: " << alias.base_name);
    }
  }
}

}  // namespace

void TransposePass(Block* root, const proto::TransposePass& options) {
  auto reqs = FromProto(options.reqs());
  auto alloc_reqs = FromProto(options.alloc_reqs());
  BufferUsageMap usages;
  RunOnBlocks(root, reqs, [&](auto map, auto block) { CollectUsage(&usages, *block, map, alloc_reqs); });
  for (auto& item : usages) {
    std::multimap<uint64_t, size_t, std::greater<uint64_t>> sorted_idxs;
    auto base_ref = item.second.base_ref;
    for (size_t i = 0; i < item.second.shape.size(); i++) {
      auto dim = item.second.shape[i];
      sorted_idxs.emplace(dim, i);
    }
    IVLOG(3, "base: " << item.first                        //
                      << ", shape: " << item.second.shape  //
                      << ", sorted_idxs: " << sorted_idxs);
    IVLOG(3, "    old_ref: " << *base_ref);
    // Adjust strides
    int64_t stride = 1;
    for (const auto& idx : sorted_idxs) {
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
