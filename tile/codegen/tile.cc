// Copyright 2018, Intel Corp.

#include "tile/codegen/tile.h"

#include "base/util/printstring.h"

namespace vertexai {
namespace tile {
namespace codegen {

using stripe::proto::Intrinsic;

void ApplyTile(stripe::proto::Block* outer, const lang::TileShape& tile) {
  // Create a new inner block
  stripe::proto::Block inner;
  // Move all statements from the outer block into the inner block
  inner.mutable_stmts()->Swap(outer->mutable_stmts());
  // Move all constraints on the outer block into the inner block
  inner.mutable_constraints()->Swap(outer->mutable_constraints());
  // Add indicies to the inner block
  inner.mutable_idxs()->CopyFrom(outer->idxs());
  for (int i = 0; i < outer->idxs_size(); i++) {
    auto range = outer->idxs(i).range();
    auto outer_idx = outer->mutable_idxs(i);
    auto inner_idx = inner.mutable_idxs(i);
    // Replace the indices on the outer block with 'outer indicies'
    // Make ranges of the outer blocks: [ceil(ri / ti), ceil(rj / tj), ceil(rk / tk), ...]
    outer_idx->set_range((outer_idx->range() + tile[i] - 1) / tile[i]);
    // Make ranges of the inner blocks: [ti, tk, tk]
    inner_idx->set_range(tile[i]);
    inner_idx->set_factor(tile[i]);
    // For each index i, if (r_i/t_i) is not integral, add a constraint to the inner block of `o_i * t_i + i_i < r_i`.
    if (range % tile[i]) {
      auto constraint = inner.add_constraints();
      for (int j = 0; j < outer->idxs_size(); j++) {
        constraint->add_lhs(i == j ? 1 : 0);
      }
      constraint->set_rhs(range);
    }
  }
  // Copy all refinements from outer to inner block
  inner.mutable_ref_ins()->CopyFrom(outer->ref_ins());
  inner.mutable_ref_outs()->CopyFrom(outer->ref_outs());
  // Multiply each stride in the outer block refinements by the appropriate tile size
  for (auto& ref : *outer->mutable_ref_ins()) {
    auto access = ref.mutable_access();
    for (int i = 0; i < access->strides_size(); i++) {
      access->set_strides(i, access->strides(i) * tile[i]);
    }
  }
  for (auto& ref : *outer->mutable_ref_outs()) {
    auto access = ref.mutable_access();
    for (int i = 0; i < access->strides_size(); i++) {
      access->set_strides(i, access->strides(i) * tile[i]);
    }
  }
  // Make the inner block the sole stmt of the outer block
  outer->add_stmts()->mutable_block()->Swap(&inner);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
