// Copyright 2018, Intel Corp.

#include "tile/codegen/tile.h"

#include "base/util/printstring.h"
#include "tile/lang/stripe.h"

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

inline bool IsLegal(int64_t rule, int64_t candidate) {  //
  return rule == -1 || candidate == rule;
}

void FindStencilMatches(std::set<StencilMatch>* into,                  //
                        const std::vector<StencilCriteria>& criteria,  //
                        const stripe::proto::Block& block,             //
                        const std::vector<size_t>& cur) {
  if (cur.size() == criteria.size()) {
    // base case
    StencilMatch match{
        1,                                                 // total
        std::vector<std::string>(block.idxs_size(), "*"),  // names
        lang::TileShape(block.idxs_size(), 1)              // tile
    };
    for (size_t i = 0; i < cur.size(); i++) {
      size_t j = cur[i];
      match.names[j] = criteria[i].name;
      if (criteria[i].size == -1) {
        match.tile[j] = block.idxs(j).range();
      } else {
        match.tile[j] = criteria[i].size;
      }
      match.total *= match.tile[j];
    }
    into->emplace(match);
  } else {
    size_t i = cur.size();
    const auto& rule = criteria[i];
    if (rule.out_strides.size() == static_cast<size_t>(block.ref_outs_size()) &&  //
        rule.in_strides.size() == static_cast<size_t>(block.ref_ins_size())) {
      for (int j = 0; j < block.idxs_size(); j++) {
        if (std::find(cur.cbegin(), cur.cend(), j) == cur.cend()) {
          bool is_legal = true;
          for (size_t k = 0; k < rule.out_strides.size(); k++) {
            is_legal &= IsLegal(rule.out_strides[k], block.ref_outs(k).access().strides(j));
          }
          for (size_t k = 0; k < rule.in_strides.size(); k++) {
            is_legal &= IsLegal(rule.in_strides[k], block.ref_ins(k).access().strides(j));
          }
          if (is_legal) {
            // found a match on this index, keep going
            std::vector<size_t> next = cur;
            next.push_back(j);
            FindStencilMatches(into, criteria, block, next);
          }
        }
      }
    }
  }
}

StencilMatch FindBestStencil(const std::vector<std::vector<StencilCriteria>>& criteria,  //
                             const stripe::proto::Block& block) {
  std::set<StencilMatch> matches;
  for (const auto& rules : criteria) {
    FindStencilMatches(&matches, rules, block, {});
  }
  if (matches.empty()) {
    StencilMatch fallback{
        1,                                                 // total
        std::vector<std::string>(block.idxs_size(), "*"),  // names
        lang::TileShape(block.idxs_size(), 1)              // tile
    };
    for (int i = 0; i < block.idxs_size(); i++) {
      auto range = block.idxs(i).range();
      fallback.total *= range;
      fallback.tile[i] = range;
    }
    LOG(WARNING) << "Fallback: " << fallback;
    return fallback;
    // throw std::runtime_error("Could not find suitable tile");
  }
  return *matches.rbegin();
}

void TilePass(stripe::proto::Block* block, const TileGenerator& generator) {
  bool is_leaf = true;
  for (auto& stmt : *block->mutable_stmts()) {
    if (stmt.has_block()) {
      TilePass(stmt.mutable_block(), generator);
      is_leaf = false;
    }
  }
  if (is_leaf) {
    ApplyTile(block, generator(*block));
  }
}

void TilePass(stripe::proto::Block* block, const std::vector<std::vector<StencilCriteria>>& criteria) {
  TilePass(block, [&criteria](const stripe::proto::Block& block) {  //
    return FindBestStencil(criteria, block).tile;
  });
}

MAKE_LOGGABLE(StencilMatch, match, os) {
  os << match.total << ":" << to_string(match.names) << ":" << to_string(match.tile);
  return os;
}

bool operator==(const StencilMatch& lhs, const StencilMatch& rhs) {  //
  return std::tie(lhs.total, lhs.names, lhs.tile) == std::tie(rhs.total, rhs.names, rhs.tile);
}

bool operator<(const StencilMatch& lhs, const StencilMatch& rhs) {  //
  return std::tie(lhs.total, lhs.names, lhs.tile) < std::tie(rhs.total, rhs.names, rhs.tile);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
