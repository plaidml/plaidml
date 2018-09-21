// Copyright 2018, Intel Corp.

#include "tile/codegen/tile.h"

#include "base/util/logging.h"
#include "base/util/printstring.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

void ApplyTile(Block* outer, const TileShape& tile) {
  // Create a new inner block
  auto inner = std::make_shared<Block>();
  // Block inner;
  // Move all statements from the outer block into the inner block
  std::swap(inner->stmts, outer->stmts);
  // Move all constraints on the outer block into the inner block
  std::swap(inner->constraints, outer->constraints);
  // Add indicies to the inner block
  inner->idxs = outer->idxs;
  for (size_t i = 0; i < outer->idxs.size(); i++) {
    int64_t range = outer->idxs[i].range;
    auto& outer_idx = outer->idxs[i];
    auto& inner_idx = inner->idxs[i];
    // Replace the indices on the outer block with 'outer indicies'
    // Make ranges of the outer blocks: [ceil(ri / ti), ceil(rj / tj), ceil(rk / tk), ...]
    outer_idx.range = (outer_idx.range + tile[i] - 1) / tile[i];
    // Make ranges of the inner blocks: [ti, tk, tk]
    inner_idx.range = tile[i];
    inner_idx.factor = tile[i];
    // For each index i, if (r_i/t_i) is not integral, add a constraint to the inner block of `o_i * t_i + i_i < r_i`.
    if (range % tile[i]) {
      std::vector<int64_t> lhs;
      for (size_t j = 0; j < outer->idxs.size(); j++) {
        lhs.push_back(i == j ? 1 : 0);
      }
      inner->constraints.emplace_back(Constraint{lhs, range});
    }
  }
  // Copy all refinements from outer to inner block
  inner->refs = outer->refs;
  // Multiply each stride in the outer block refinements by the appropriate tile size
  for (auto& ref : outer->refs) {
    for (size_t i = 0; i < ref.access.strides.size(); i++) {
      ref.access.strides[i] *= tile[i];
    }
  }
  // Make the inner block the sole stmt of the outer block
  outer->stmts = {inner};
}

inline bool IsLegal(int64_t rule, int64_t candidate) {  //
  return rule == -1 || candidate == rule;
}

void FindStencilMatches(std::set<StencilMatch>* into,                  //
                        const std::vector<StencilCriteria>& criteria,  //
                        const Block& block,                            //
                        const std::vector<size_t>& cur) {
  if (cur.size() == criteria.size()) {
    // base case
    StencilMatch match{
        1,                                                 // total
        std::vector<std::string>(block.idxs.size(), "*"),  // names
        TileShape(block.idxs.size(), 1),                   // tile
        false                                              // is_fallback
    };
    for (size_t i = 0; i < cur.size(); i++) {
      size_t j = cur[i];
      match.names[j] = criteria[i].name;
      if (criteria[i].size == -1) {
        match.tile[j] = block.idxs[j].range;
      } else {
        match.tile[j] = criteria[i].size;
      }
      match.total *= match.tile[j];
    }
    into->emplace(match);
  } else {
    size_t i = cur.size();
    const auto& rule = criteria[i];
    auto ref_outs = block.ref_outs();
    auto ref_ins = block.ref_ins();
    if (rule.out_strides.size() == ref_outs.size() &&  //
        rule.in_strides.size() == ref_ins.size()) {
      for (size_t j = 0; j < block.idxs.size(); j++) {
        if (std::find(cur.cbegin(), cur.cend(), j) == cur.cend()) {
          bool is_legal = true;
          for (size_t k = 0; k < rule.out_strides.size(); k++) {
            is_legal &= IsLegal(rule.out_strides[k], ref_outs[k].access.strides[j]);
          }
          for (size_t k = 0; k < rule.in_strides.size(); k++) {
            is_legal &= IsLegal(rule.in_strides[k], ref_ins[k].access.strides[j]);
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
                             Block* block) {
  std::set<StencilMatch> matches;
  for (const auto& rules : criteria) {
    FindStencilMatches(&matches, rules, *block, {});
  }
  if (matches.empty()) {
    StencilMatch fallback{
        1,                                                  // total
        std::vector<std::string>(block->idxs.size(), "*"),  // names
        TileShape(block->idxs.size(), 1),                   // tile
        true                                                // is_fallback
    };
    for (size_t i = 0; i < block->idxs.size(); i++) {
      auto range = block->idxs[i].range;
      fallback.total *= range;
      fallback.tile[i] = range;
    }
    LOG(WARNING) << "Fallback: " << fallback;
    block->annotations.emplace("is_fallback", std::make_shared<BoolAnnotation>(true));
    return fallback;
  }
  block->annotations.emplace("is_fallback", std::make_shared<BoolAnnotation>(false));
  return *matches.rbegin();
}

void TilePass(Block* block, const TileGenerator& generator) {
  bool is_leaf = true;
  for (auto stmt : block->stmts) {
    if (stmt->kind() == StmtKind::Block) {
      TilePass(std::dynamic_pointer_cast<Block>(stmt).get(), generator);
      is_leaf = false;
    }
  }
  if (is_leaf) {
    ApplyTile(block, generator(block));
  }
}

void TilePass(Block* block, const std::vector<std::vector<StencilCriteria>>& criteria) {
  TilePass(block, [&criteria](Block* block) {  //
    return FindBestStencil(criteria, block).tile;
  });
}

std::ostream& operator<<(std::ostream& os, const StencilMatch& match) {
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
