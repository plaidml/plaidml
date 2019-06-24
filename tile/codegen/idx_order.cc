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

// Reorder the index in cache block. Let the low bits of thread ID
// be the low stride of ref
void ReorderIndex(Block* block, const Refinement& ref) {
  std::map<std::string, size_t> idx_stride;
  TensorShape shape = block->exterior_shape(ref.into());
  for (size_t i = 0; i < ref.access.size(); ++i) {
    auto& acc_map = ref.access[i].getMap();
    if (acc_map.size() == 1 && acc_map.begin()->first != "") {
      idx_stride.emplace(acc_map.begin()->first, shape.dims[i].stride);
    }
  }
  std::sort(block->idxs.begin(), block->idxs.end(),
    [idx_stride](const Index& idx0, const Index& idx1) {
      auto it0 = idx_stride.find(idx0.name);
      if (it0 != idx_stride.end()) {
        auto it1 = idx_stride.find(idx1.name);
        if (it1 != idx_stride.end()) {
          return it0->second >  it1->second;
        }
      }
      return false;
    }
  );
}

void IdxOrder(const AliasMap& alias_map, Block* block, const proto::IdxOrderPass& options) {
  // Find out the largest refinement and use it as the reference
  size_t max_size = 0;
  std::string max_ref;
  for (const auto& ref : block->refs) {
    size_t size = block->exterior_shape(ref.into()).sizes_product();
    if (size > max_size) {
      max_size = size;
      max_ref = ref.into();
    }
  }
  if (max_size > 0) {
    auto ref_it = block->ref_by_into(max_ref);
    ReorderIndex(block, *ref_it);
  }
}

void IdxOrderPass::Apply(CompilerState* state) const {
  auto reqs = stripe::FromProto(options_.reqs());
  RunOnBlocks(state->entry(), reqs,
              [this](const AliasMap& alias_map, stripe::Block* block) {  //
                IdxOrder(alias_map, block, options_);
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
