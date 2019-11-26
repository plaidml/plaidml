// Copyright 2018, Intel Corporation

#include "tile/codegen/idx_order.h"
#include "base/util/any_factory_map.h"
#include "tile/codegen/tile.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT
using namespace math;    // NOLINT

struct RefInfo {
  std::string name;  // Refinement name
  Index* low_idx;    // The lowest index (stride 1)
  size_t size;       // Refinement size
};

bool IsGlobalRef(const Refinement& ref) {
  return ref.location.devs.size() == 0 || ref.location.devs[0].name == "GLOBAL";
}

bool IsRegisterRef(const Refinement& ref) {
  return ref.location.devs.size() > 0 && ref.location.devs[0].name == "REGISTER";
}

Index* StrideOneIndex(Block* block, const Refinement& ref) {
  const auto& dims = ref.interior_shape.dims;
  int n_dim = dims.size();
  for (int i = n_dim - 1; i >= 0; --i) {
    if (dims[i].stride > 1) {
      continue;
    }
    if ((i == 0) || (i > 0 && dims[i - 1].stride > 1)) {
      auto& acc_map = ref.access[i].getMap();
      for (auto& kvp : acc_map) {
        if (kvp.first.size() > 0 && kvp.second == 1) {
          return block->idx_by_name(kvp.first);
        }
      }
    }
  }
  return nullptr;
}

void ReorderIndex(Block* block, bool global_only, bool apply_inner) {
  // Find out the largest refinement and use it as the reference
  std::vector<RefInfo> ref_info;
  Block *target = apply_inner ? (block->SubBlock(0).get()) : block;
  for (const auto& ref : block->refs) {
    if (IsRegisterRef(ref) || (global_only && !IsGlobalRef(ref))) {
      continue;
    }
    size_t size = block->exterior_shape(ref.into()).sizes_product();
    RefInfo ri = {ref.into(), StrideOneIndex(target, ref), size};
    if (ri.low_idx) {
      ref_info.push_back(ri);
    }
  }

  // Sort ref_info by size
  std::sort(ref_info.begin(), ref_info.end(),
    [](const RefInfo& r0, const RefInfo& r1) { return r0.size > r1.size; });
  std::vector<Index> new_idxs;
  std::set<Index*> used_idx;

  // Push the lowest index
  for (auto& ri : ref_info) {
    if (used_idx.find(ri.low_idx) == used_idx.end()) {
      new_idxs.push_back(*ri.low_idx);
      used_idx.insert(ri.low_idx);
    }
  }

  // Push the remaining index
  for (auto iter = target->idxs.rbegin(); iter != target->idxs.rend(); ++iter) {
    if (used_idx.find(&(*iter)) == used_idx.end()) {
      new_idxs.push_back(*iter);
      used_idx.insert(&(*iter));
    }
  }

  std::reverse(new_idxs.begin(), new_idxs.end());
  target->idxs = new_idxs;
}

void IdxOrderPass::Apply(CompilerState* state) const {
  auto reqs = stripe::FromProto(options_.reqs());
  RunOnBlocks(state->entry(), reqs,
              [](const AliasMap& alias_map, stripe::Block* block) {  //
                ReorderIndex(block, true, false);
              },
              true);
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<IdxOrderPass, proto::IdxOrderPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
