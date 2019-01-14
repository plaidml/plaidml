// Copyright 2018, Intel Corporation

#include "tile/codegen/tile.h"

#include "base/util/logging.h"
#include "base/util/lookup.h"
#include "base/util/stream_container.h"
#include "base/util/throw.h"
#include "tile/codegen/math.h"
#include "tile/codegen/tags.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

namespace {

bool HasConstraints(const Block& block, const Index& idx) {  //
  for (const auto& constraint : block.constraints) {
    if (constraint.get(idx.name)) {
      return true;
    }
  }
  return false;
}

bool HasAffines(const Block& block, const Index& idx) {
  for (auto stmt : block.stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      for (auto& inner_idx : inner->idxs) {
        if (inner_idx.affine.get(idx.name)) {
          return true;
        }
      }
    }
  }
  return false;
}

}  // namespace

bool ApplyTile(Block* outer, const TileShape& shape, bool elide_trivial, bool copy_tags) {
  // Verify tile shape is correct
  if (outer->idxs.size() != shape.size()) {
    throw_with_trace(std::runtime_error("Invalid tile specified"));
  }
  // IVLOG(3, "Doing tiling " << shape << ":\n" << *outer);
  // Make a 'by-name' version of tile and check for trivality
  bool all_min = true;
  bool all_max = true;
  std::map<std::string, size_t> tile_by_name;
  for (size_t i = 0; i < outer->idxs.size(); i++) {
    tile_by_name[outer->idxs[i].name] = shape[i];
    if (shape[i] != 1) {
      all_min = false;
    }
    if (shape[i] != outer->idxs[i].range) {
      all_max = false;
    }
  }
  if (elide_trivial && (all_min || all_max)) {
    return false;
  }
  // Create a new inner block
  auto inner = std::make_shared<Block>();
  if (copy_tags) {
    inner->tags = outer->tags;
  }
  inner->name = outer->name;
  inner->location = outer->location;
  // Move all statements from the outer block into the inner block
  std::swap(inner->stmts, outer->stmts);
  // Move all constraints on the outer block into the inner block
  std::swap(inner->constraints, outer->constraints);
  // Add indicies to the inner block
  inner->idxs = outer->idxs;
  std::vector<Index> passthru_idxs;
  for (size_t i = 0; i < outer->idxs.size(); i++) {
    auto& outer_idx = outer->idxs[i];
    auto& inner_idx = inner->idxs[i];
    if (outer_idx.range == 1) {
      // For indexes without a range, just make a passthru for the inner block
      inner_idx.affine = Affine(outer_idx.name);
      continue;
    }
    // Needs passthru:
    // 1. tiling is uneven on this index
    // 2. constraints on outer that use this index
    // 3. affines on interior blocks that refer to this index
    if ((outer_idx.range % shape[i]) || HasConstraints(*inner, outer_idx) || HasAffines(*inner, outer_idx)) {
      Index passthru_idx{
          inner->unique_idx_name(outer_idx.name),           // name
          1,                                                // range
          {outer_idx.name, static_cast<int64_t>(shape[i])}  // affine
      };
      // Update any inner constraints that refer to this index
      for (auto& constraint : inner->constraints) {
        constraint.substitute(outer_idx.name, Affine(outer_idx.name) + Affine(passthru_idx.name));
      }
      // Update any interior idxs that refer to this index
      for (auto stmt : inner->stmts) {
        auto interior = Block::Downcast(stmt);
        if (interior) {
          for (auto& idx : interior->idxs) {
            idx.affine.substitute(outer_idx.name, Affine(outer_idx.name) + Affine(passthru_idx.name));
          }
        }
      }
      if (outer_idx.range % shape[i]) {
        // For each index i, if (r_i/t_i) is not integral, add a constraint to the inner block of `o_i * t_i + i_i <
        // r_i`. Or solving to make it in the form poly >= 0, ((r_i - 1) - p_i - i_i >= 0).
        inner->constraints.emplace_back(Affine(passthru_idx.name, -1) +  //
                                        Affine(inner_idx.name, -1) +     //
                                        int64_t(outer_idx.range - 1));
      }
      // Finally add the passthru_idx
      passthru_idxs.emplace_back(passthru_idx);
    }
    // Replace the indices on the outer block with 'outer indicies'
    // Make ranges of the outer blocks: [ceil(ri / ti), ceil(rj / tj), ceil(rk / tk), ...]
    outer_idx.range = IntDivCeil(outer_idx.range, shape[i]);
    // Make ranges of the inner blocks: [ti, tk, tk]
    inner_idx.range = shape[i];
  }
  // Append all passthru_idxs, we defer this since this may cause inner->idxs to realloc
  std::copy(passthru_idxs.begin(), passthru_idxs.end(), std::back_inserter(inner->idxs));
  // Copy all in/out refinements from outer to inner block
  inner->refs = outer->refs;
  // Remove allocs on the outer refs
  outer->refs.erase(
      std::remove_if(outer->refs.begin(), outer->refs.end(), [&](const auto& ref) { return ref.dir == RefDir::None; }),
      outer->refs.end());
  for (auto& ref : outer->refs) {
    // Fix the sizes on the outer blocks
    ref.interior_shape = inner->exterior_shape(ref.into);
    // Save any renames till the inner block (ie, only one, inner or outer needs to rename)
    ref.into = ref.from;
    for (auto& aff : ref.access) {
      // Since we're taking a single block and turning it into two (e.g. outer and inner),
      // arrange for only one of the blocks to do the constant pointer bumping.
      // It probably doesn't matter whether the outer or the inner does the bumping.
      aff.setConstant(0);
      // Multiply each stride in the outer block refinements by the appropriate tile size
      for (auto& kvp : aff.mutateMap()) {
        if (!kvp.first.empty()) {
          kvp.second *= tile_by_name[kvp.first];
        }
      }
    }
  }
  // Make the inner block the sole stmt of the outer block
  outer->stmts = {inner};
  return true;
}

inline bool IsLegal(int64_t rule, int64_t candidate) {  //
  return rule == -1 || candidate == rule;
}

using StencilMapEntry = std::pair<std::string, std::string>;
using StencilMap = std::vector<StencilMapEntry>;

void FindStencilMatches(std::set<StencilMatch>* into,  //
                        const proto::Stencil& spec,    //
                        const Block& block,            //
                        const StencilMap& cur) {
  if (cur.size() == static_cast<size_t>(spec.idxs_size())) {
    // base case
    StencilMatch match{1, {}};
    std::map<std::string, StencilIndexMatch> idx_matches;
    for (const auto& idx : block.idxs) {
      idx_matches[idx.name] = StencilIndexMatch{idx.name, "*", 1};
    }
    for (size_t i = 0; i < cur.size(); i++) {
      const auto& item = cur[i];
      auto it = std::find_if(                        //
          spec.idxs().cbegin(),                      //
          spec.idxs().cend(),                        //
          [&item](const proto::StencilIndex& idx) {  //
            return idx.name() == item.second;
          });
      if (it == spec.idxs().end()) {
        throw_with_trace(std::runtime_error("Invalid idx name"));
      }
      StencilIndexMatch idx_match;
      if (it->size() != -1) {
        idx_match = StencilIndexMatch{item.first, item.second, static_cast<uint64_t>(it->size())};
      } else {
        auto block_idx = block.idx_by_name(item.first);
        idx_match = StencilIndexMatch{item.first, item.second, block_idx->range};
      }
      idx_matches[idx_match.block_idx_name] = idx_match;
    }
    size_t total_tiles = 1;
    for (const auto& idx : block.idxs) {
      auto tile = safe_at(idx_matches, idx.name);
      size_t num_tiles = IntDivCeil(idx.range, tile.value);
      total_tiles *= num_tiles;
      match.cost *= num_tiles * tile.value;
      match.idxs.push_back(tile);
    }
    match.cost += spec.startup_cost() * total_tiles;
    IVLOG(4, "Candidate: " << match);
    into->emplace(match);
  } else {
    size_t i = cur.size();
    const auto& rule = spec.idxs(i);
    auto ref_outs = block.ref_outs();
    auto ref_ins = block.ref_ins();
    if (ref_outs.size() == static_cast<size_t>(rule.outs_size()) &&  //
        ref_ins.size() == static_cast<size_t>(rule.ins_size())) {
      for (const auto& idx : block.idxs) {
        auto it = std::find_if(                    //
            cur.cbegin(),                          //
            cur.cend(),                            //
            [&idx](const StencilMapEntry& item) {  //
              return item.first == idx.name;
            });
        if (it == cur.end()) {
          bool is_legal = true;
          for (size_t k = 0; k < ref_outs.size(); k++) {
            is_legal &= IsLegal(rule.outs(k), ref_outs[k]->FlatAccess().get(idx.name));
          }
          for (size_t k = 0; k < ref_ins.size(); k++) {
            is_legal &= IsLegal(rule.ins(k), ref_ins[k]->FlatAccess().get(idx.name));
          }
          if (is_legal) {
            // found a match on this index, keep going
            auto next = cur;
            next.push_back(std::make_pair(idx.name, rule.name()));
            FindStencilMatches(into, spec, block, next);
          }
        }
      }
    }
  }
}

boost::optional<StencilMatch> FindBestStencil(const std::vector<proto::Stencil>& specs,  //
                                              const Block& block) {
  std::set<StencilMatch> matches;
  for (const auto& spec : specs) {
    FindStencilMatches(&matches, spec, block, {});
  }
  if (!matches.empty()) {
    return *matches.begin();
  }
  return boost::none;
}

struct StencilPassOptions {
  Tags reqs;
  std::vector<proto::Stencil> specs;
  Tags set_outer;
  Tags set_inner;
};

void ApplyIndexTags(Block* block, const StencilMatch& match) {
  for (const auto& idx_match : match.idxs) {
    if (idx_match.stencil_idx_name == "*") {
      continue;
    }
    auto idx = block->idx_by_name(idx_match.block_idx_name);
    if (idx) {
      idx->set_tag(str(boost::format("stencil_%1%") % idx_match.stencil_idx_name));
    }
  }
}

void StencilPassRecurse(Block* block, const StencilPassOptions& options) {
  for (auto stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      StencilPassRecurse(inner.get(), options);
    }
  }
  if (block->has_tags(options.reqs)) {
    auto match = FindBestStencil(options.specs, *block);
    if (!match) {
      return;
    }
    TileShape tile;
    for (const auto& idx : match->idxs) {
      tile.push_back(idx.value);
    }
    ApplyTile(block, tile, false);
    ApplyIndexTags(block, *match);
    block->add_tags(options.set_outer);
    auto inner = block->SubBlock(0);
    ApplyIndexTags(inner.get(), *match);
    inner->add_tags(options.set_inner);
  }
}

void StencilPass(Block* block, const proto::StencilPass& options) {
  StencilPassOptions sopts = {
      FromProto(options.reqs()),       // reqs
      {},                              // specs
      FromProto(options.outer_set()),  // set_outer
      FromProto(options.inner_set())   // set_inner
  };
  for (const auto& stencil : options.stencils()) {
    sopts.specs.push_back(stencil);
  }
  StencilPassRecurse(block, sopts);
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
