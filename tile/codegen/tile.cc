// Copyright 2018, Intel Corp.

#include "tile/codegen/tile.h"

#include "base/util/printstring.h"
#include "tile/lang/intrinsics.h"

namespace vertexai {
namespace tile {
namespace codegen {

void ApplyTile(stripe::proto::Block* outer, const lang::TileShape& tile) {
  // Create a new inner block
  stripe::proto::Block inner;
  // Move all statements from the outer block into the inner block
  inner.mutable_stmts()->Swap(outer->mutable_stmts());
  // Move all constraints on the outer block into the inner block
  // Replace ci * i in each constraint with ci * (oi * ti + ii), for each index i
  inner.mutable_constraints()->Swap(outer->mutable_constraints());
  // Add indicies to the inner block
  for (int i = 0; i < outer->index_names_size(); i++) {
    auto name = outer->index_names(i);
    auto range = outer->index_ranges(i);
    // Replace the indices on the outer block with 'outer indicies'
    outer->set_index_names(i, printstring("%s.o", name.c_str()));
    inner.add_index_names(printstring("%s.i", name.c_str()));
    // Make ranges of the outer blocks: [ceil(ri / ti), ceil(rj / tj), ceil(rk / tk), ...]
    outer->set_index_ranges(i, std::ceil(range / tile[i]));
    // Make ranges of the inner blocks: [ti, tk, tk]
    inner.add_index_ranges(tile[i]);
    // For each index i, if (r_i/t_i) is not integral, add a constraint to the inner block of `o_i * t_i + i_i < r_i`.
    if (range % tile[i]) {
      auto constraint = inner.add_constraints();
      for (int j = 0; j < outer->index_names_size(); j++) {
        if (i == j) {
          constraint->add_lhs(tile[i]);
        } else {
          constraint->add_lhs(0);
        }
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
    ref.set_agg_op(lang::intrinsic::ASSIGN);
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
