// Copyright 2018, Intel Corporation

#include <set>

#include "tile/codegen/kernel_tag.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT
using namespace math;    // NOLINT

static bool IsEltwiseRef(const Refinement& ref, Block* block) {
  std::map<std::string, size_t> idx_set;
  for (const auto& idx : block->idxs) {
    if (idx.affine == Affine()) {
      idx_set.emplace(idx.name, idx.range);
    }
  }
  auto shape = block->exterior_shape(ref.into());
  for (size_t i = 0; i < ref.access.size(); ++i) {
    const auto& acc = ref.access[i];
    if (acc == Affine()) {
      continue;
    }
    const auto& acc_map = acc.getMap(); 
    if (acc_map.size() != 1 || acc_map.begin()->second != 1) {
      return false;
    }
    const std::string& acc_idx = acc_map.begin()->first;
    if (idx_set.find(acc_idx) == idx_set.end()) {
      return false;
    }
    if (idx_set[acc_idx] != shape.dims[i].size) {
      return false;
    }
    idx_set.erase(acc_idx);
  }
  return idx_set.empty(); 
}

void EltwiseTag(const AliasMap& alias_map, Block* block) {
  if (block->has_tag("eltwise") || block->has_tag("zero")) {
    return;
  }
  // Make sure no real agg ops
  for (const auto& ref : block->refs) {
    if (ref.agg_op != "" && ref.agg_op != "assign") {
      return;
    }
  }
  for (const auto& ref : block->refs) {
    if (IsEltwiseRef(ref, block)) {
      block->set_tag("eltwise");
      if (block->has_tag("contraction")) {
        block->remove_tag("contraction");
      }
      break;
    }
  }
}

void KernelTag(const AliasMap& alias_map, Block* block) {
  // If the block is not tagged as element-wise, and its input or output is
  // actually element-wise, tag it as element-wise.
  EltwiseTag(alias_map, block);
  // We can change more tags...
}

void KernelTagPass::Apply(stripe::Block* root) const {
  auto reqs = stripe::FromProto(options_.reqs());
  RunOnBlocks(root, reqs,
              [](const AliasMap& alias_map, stripe::Block* block) {  //
                 KernelTag(alias_map, block);
              },
              false);
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<KernelTagPass, proto::KernelTagPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
