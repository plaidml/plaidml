// Copyright 2018, Intel Corp.

#include "tile/codegen/tile.h"

#include "base/util/logging.h"
#include "base/util/printstring.h"
#include "base/util/stream_container.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

void ApplyTile(Block* outer, const TileShape& tile) {
  // Verify tile shape is correct
  if (outer->idxs.size() != tile.size()) {
    throw std::runtime_error("Invalid tile specified");
  }
  // Make a 'by-name' version of tile
  std::map<std::string, size_t> tile_by_name;
  for (size_t i = 0; i < outer->idxs.size(); i++) {
    tile_by_name[outer->idxs[i].name] = tile[i];
  }
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
    if (range % tile[i]) {
      inner->constraints.emplace_back(Affine(outer_idx.name, -1) + int64_t(outer_idx.range - 1));
    }
    // Replace the indices on the outer block with 'outer indicies'
    // Make ranges of the outer blocks: [ceil(ri / ti), ceil(rj / tj), ceil(rk / tk), ...]
    outer_idx.range = (outer_idx.range + tile[i] - 1) / tile[i];
    // Make ranges of the inner blocks: [ti, tk, tk]
    inner_idx.range = tile[i];
    inner_idx.factor = tile[i];
    if (outer_idx.factor > 0 && outer_idx.factor % tile[i] != 0) {
      throw std::runtime_error("ApplyTile: unhandled uneven subtiling");
    }
    outer_idx.factor /= tile[i];
    // For each index i, if (r_i/t_i) is not integral, add a constraint to the inner block of `o_i * t_i + i_i < r_i`.
  }
  // Copy all refinements from outer to inner block
  inner->refs = outer->refs;
  // Fix the sizes on the outer blocks
  // TODO: How to handle 'skips'
  for (auto& ref : outer->refs) {
    for (size_t i = 0; i < ref.access.size(); i++) {
      auto& aff = ref.access[i];
      int64_t low = 0;
      int64_t high = 0;
      for (const auto& kvp : aff.getMap()) {
        if (kvp.first.empty()) {
          continue;
        }
        if (kvp.second > 0) {
          high += kvp.second * (tile_by_name[kvp.first] - 1);
        } else {
          low += kvp.second * (tile_by_name[kvp.first] - 1);
        }
      }
      ref.shape.dims[i].size = high - low + 1;
    }
  }
  // Multiply each stride in the outer block refinements by the appropriate tile size
  for (auto& ref : outer->refs) {
    for (auto& aff : ref.access) {
      for (auto& kvp : aff.mutateMap()) {
        if (!kvp.first.empty()) {
          kvp.second *= tile_by_name[kvp.first];
        }
      }
    }
  }

  // Make the inner block the sole stmt of the outer block
  outer->stmts = {inner};
}

inline bool IsLegal(int64_t rule, int64_t candidate) {  //
  return rule == -1 || candidate == rule;
}

using StencilMapEntry = std::pair<std::string, std::string>;
using StencilMap = std::vector<StencilMapEntry>;

void FindStencilMatches(std::set<StencilMatch>* into,  //
                        const StencilSpec& spec,       //
                        const Block& block,            //
                        const StencilMap& cur) {
  if (cur.size() == spec.idxs.size()) {
    // base case
    StencilMatch match{1, {}, false};
    std::map<std::string, StencilIndexMatch> idx_matches;
    for (const auto& idx : block.idxs) {
      idx_matches[idx.name] = StencilIndexMatch{idx.name, "*", 1};
    }
    for (size_t i = 0; i < cur.size(); i++) {
      const auto& item = cur[i];
      auto it = std::find_if(spec.idxs.cbegin(), spec.idxs.cend(),
                             [&item](const StencilIndex& idx) { return idx.name == item.second; });
      if (it == spec.idxs.end()) {
        throw std::runtime_error("Invalid idx name");
      }
      StencilIndexMatch idx_match;
      if (it->size != -1) {
        idx_match = StencilIndexMatch{item.first, item.second, static_cast<uint64_t>(it->size)};
      } else {
        auto block_idx = block.idx_by_name(item.first);
        idx_match = StencilIndexMatch{item.first, item.second, block_idx->range};
      }
      idx_matches[idx_match.block_idx_name] = idx_match;
    }
    size_t total_tiles = 1;
    for (const auto& idx : block.idxs) {
      auto tile = idx_matches.at(idx.name);
      size_t num_tiles = (idx.range + tile.value - 1) / tile.value;
      total_tiles *= num_tiles;
      match.cost *= num_tiles * tile.value;
      match.idxs.push_back(tile);
    }
    match.cost += spec.alpha * total_tiles;
    IVLOG(4, "Candidate: " << match);
    into->emplace(match);
  } else {
    size_t i = cur.size();
    const auto& rule = spec.idxs[i];
    auto ref_outs = block.ref_outs();
    auto ref_ins = block.ref_ins();
    if (rule.out_strides.size() == ref_outs.size() &&  //
        rule.in_strides.size() == ref_ins.size()) {
      for (const auto& idx : block.idxs) {
        auto it = std::find_if(cur.cbegin(), cur.cend(),
                               [&idx](const StencilMapEntry& item) { return item.first == idx.name; });
        if (it == cur.end()) {
          bool is_legal = true;
          for (size_t k = 0; k < rule.out_strides.size(); k++) {
            is_legal &= IsLegal(rule.out_strides[k], ref_outs[k]->FlatAccess().get(idx.name));
          }
          for (size_t k = 0; k < rule.in_strides.size(); k++) {
            is_legal &= IsLegal(rule.in_strides[k], ref_ins[k]->FlatAccess().get(idx.name));
          }
          if (is_legal) {
            // found a match on this index, keep going
            auto next = cur;
            next.push_back(std::make_pair(idx.name, rule.name));
            FindStencilMatches(into, spec, block, next);
          }
        }
      }
    }
  }
}

StencilMatch FindBestStencil(const std::vector<StencilSpec>& specs,  //
                             Block* block) {
  std::set<StencilMatch> matches;
  for (const auto& spec : specs) {
    FindStencilMatches(&matches, spec, *block, {});
  }
  if (matches.empty()) {
    StencilMatch fallback{1, {}, true};
    for (const auto& idx : block->idxs) {
      fallback.cost *= idx.range;
      fallback.idxs.emplace_back(StencilIndexMatch{idx.name, "*", idx.range});
    }
    IVLOG(3, "Fallback: " << fallback);
    block->annotations.emplace("is_fallback", std::make_shared<BoolAnnotation>(true));
    return fallback;
  }
  block->annotations.emplace("is_fallback", std::make_shared<BoolAnnotation>(false));
  return *matches.begin();
}

void TilePass(Block* block, const TileGenerator& generator) {
  bool is_leaf = true;
  for (auto stmt : block->stmts) {
    if (stmt->kind() == StmtKind::Block) {
      TilePass(Block::Downcast(stmt).get(), generator);
      is_leaf = false;
    }
  }
  if (is_leaf) {
    ApplyTile(block, generator(block));
  }
}

void TilePass(Block* block, const std::vector<StencilSpec>& specs) {
  TilePass(block, [&specs](Block* block) {
    auto stencil = FindBestStencil(specs, block);
    TileShape tile;
    for (const auto& idx : stencil.idxs) {
      tile.push_back(idx.value);
    }
    return tile;
  });
}

std::ostream& operator<<(std::ostream& os, const StencilIndexMatch& idx) {
  os << idx.block_idx_name << "->" << idx.stencil_idx_name << ":" << idx.value;
  return os;
}

bool operator==(const StencilIndexMatch& lhs, const StencilIndexMatch& rhs) {
  return std::tie(lhs.block_idx_name, lhs.stencil_idx_name, lhs.value) ==  //
         std::tie(rhs.block_idx_name, rhs.stencil_idx_name, rhs.value);
}

bool operator<(const StencilIndexMatch& lhs, const StencilIndexMatch& rhs) {
  return std::tie(lhs.block_idx_name, lhs.stencil_idx_name, lhs.value) <  //
         std::tie(rhs.block_idx_name, rhs.stencil_idx_name, rhs.value);
}

std::ostream& operator<<(std::ostream& os, const StencilMatch& match) {
  os << match.cost << ":" << StreamContainer(match.idxs);
  return os;
}

bool operator==(const StencilMatch& lhs, const StencilMatch& rhs) {
  return std::tie(lhs.cost, lhs.idxs) ==  //
         std::tie(rhs.cost, rhs.idxs);
}

bool operator<(const StencilMatch& lhs, const StencilMatch& rhs) {
  return std::tie(lhs.cost, lhs.idxs) <  //
         std::tie(rhs.cost, rhs.idxs);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
