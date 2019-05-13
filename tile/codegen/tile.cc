// Copyright 2018, Intel Corporation

#include <set>

#include "tile/codegen/tile.h"

#include "base/util/logging.h"
#include "base/util/lookup.h"
#include "base/util/stream_container.h"
#include "base/util/throw.h"
#include "tile/math/util.h"
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
  for (const auto& dev : block.location.devs) {
    for (const auto& unit : dev.units) {
      if (unit.get(idx.name)) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

void PromoteCondition(Block* outer, std::shared_ptr<Block> inner,  //
                      const std::set<std::string>& passthru, bool insert_back) {
  // Wrap a block containing the conditions out of inner
  auto wrapper = std::make_shared<Block>();
  wrapper->refs = outer->refs;

  // Copy all affine index from inner to wrapper
  for (auto& idx : inner->idxs) {
    if (idx.affine != Affine()) {
      wrapper->idxs.push_back(idx);
      idx.affine = Affine(idx.name);
    }
  }

  for (auto& cons : inner->constraints) {
    // If all variables are passthru affines, cons can be promoted
    bool selected = true;
    for (const auto& mvp : cons.getMap()) {
      if (mvp.first != "" && passthru.find(mvp.first) == passthru.end()) {
        selected = false;
        break;
      }
    }
    if (selected) {
      wrapper->constraints.push_back(cons);
      cons.mutateMap().clear();
    }
  }

  // accesses in passthru ref should be zero
  for (auto& ref : wrapper->refs) {
    for (auto& aff : ref.mut().access) {
      aff.mutateMap().clear();
    }
  }

  inner->constraints.erase(std::remove(inner->constraints.begin(),  //
                                       inner->constraints.end(), Affine()),
                           inner->constraints.end());

  // Collect all used passthru index for inner, and remove the ununsed passthru index
  std::set<std::string> used_idx;
  for (const auto& cons : inner->constraints) {
    auto& cons_map = cons.getMap();
    for (const auto& mvp : cons_map) {
      if (mvp.first != "") {
        used_idx.insert(mvp.first);
      }
    }
  }
  for (const auto& ref : inner->refs) {
    for (const auto& aff : ref.access) {
      auto& aff_map = aff.getMap();
      for (const auto& mvp : aff_map) {
        if (mvp.first != "") {
          used_idx.insert(mvp.first);
        }
      }
    }
  }
  inner->idxs.erase(
      std::remove_if(inner->idxs.begin(), inner->idxs.end(),                                                       //
                     [used_idx](const Index& idx) -> bool { return used_idx.find(idx.name) == used_idx.end(); }),  //
      inner->idxs.end());

  wrapper->stmts = {inner};
  if (insert_back) {
    outer->stmts.push_back(wrapper);
  } else {
    outer->stmts.insert(outer->stmts.begin(), wrapper);
  }
}

bool ApplyTile(Block* outer, const TileShape& shape, bool elide_trivial, bool copy_tags, bool interleave,
               bool split_unaligned, const std::string& location_idx_tag) {
  // A block is split by value on index
  struct DimSplitPoint {
    std::string index;
    size_t value;
    size_t step;
  };
  std::vector<DimSplitPoint> split_points;

  // Verify tile shape is correct
  if (outer->idxs.size() != shape.size()) {
    throw_with_trace(std::runtime_error("Invalid tile specified"));
  }
  // IVLOG(3, "Doing tiling " << shape << ":\n" << *outer);
  // Make a 'by-name' version of tile and check for trivality
  bool all_min = true;
  bool all_max = true;
  std::map<std::string, size_t> tile_by_name;
  std::map<std::string, size_t> range_by_name;
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
    inner->set_attrs(*outer);
  }
  inner->name = outer->name;
  // Move all statements from the outer block into the inner block
  std::swap(inner->stmts, outer->stmts);
  // Move all constraints on the outer block into the inner block
  std::swap(inner->constraints, outer->constraints);
  // Add indicies to the inner block
  inner->idxs = outer->idxs;

  // Find all index used by load_index
  // used_idx maps the original index to the passthru index
  std::set<std::string> used_idx;
  for (const auto& stmt : inner->stmts) {
    if (stmt->kind() == StmtKind::LoadIndex) {
      auto load_index = LoadIndex::Downcast(stmt);
      for (const auto& kvp : load_index->from.getMap()) {
        if (kvp.first != "") {
          used_idx.emplace(kvp.first);
        }
      }
    }
  }

  std::vector<Index> passthru_idxs;
  for (size_t i = 0; i < outer->idxs.size(); i++) {
    auto& outer_idx = outer->idxs[i];
    auto& inner_idx = inner->idxs[i];
    if (outer_idx.affine != Affine()) {
      // For indexes which are calculated, just make a passthru for the inner block
      inner_idx.affine = Affine(outer_idx.name);
      continue;
    }
    // Needs passthru:
    // 1. tiling is uneven on this index
    // 2. constraints on outer that use this index
    // 3. affines on interior blocks that refer to this index
    // 4. the index is used by load_index
    // TODO: We can optimize the passthru index if outer index is always 0
    if ((outer_idx.range % shape[i]) || HasConstraints(*inner, outer_idx) || HasAffines(*inner, outer_idx) ||
        used_idx.count(outer_idx.name)) {
      Index passthru_idx{
          inner->unique_idx_name(outer_idx.name),                            // name
          1,                                                                 // range
          {outer_idx.name, interleave ? 1 : static_cast<int64_t>(shape[i])}  // affine
      };
      int64_t local_mul = interleave ? math::RoundUp(outer_idx.range, shape[i]) : 1;
      Affine replacement = Affine(outer_idx.name) * local_mul + Affine(passthru_idx.name);
      // Update any inner constraints that refer to this index
      for (auto& constraint : inner->constraints) {
        constraint.substitute(outer_idx.name, replacement);
      }
      // Update any LoadIndexes that use this index
      for (const auto& stmt : inner->stmts) {
        if (stmt->kind() == StmtKind::LoadIndex) {
          LoadIndex::Downcast(stmt)->from.substitute(outer_idx.name, replacement);
        }
      }
      // Update any interior idxs that refer to this index
      for (auto stmt : inner->stmts) {
        auto interior = Block::Downcast(stmt);
        if (interior) {
          for (auto& idx : interior->idxs) {
            idx.affine.substitute(outer_idx.name, replacement);
          }
        }
      }
      if (outer_idx.range % shape[i]) {
        // For each index i, if (r_i/t_i) is not integral, add a constraint to the inner block of `o_i * t_i + i_i <
        // r_i`. Or solving to make it in the form poly >= 0, ((r_i - 1) - p_i - i_i >= 0).
        inner->constraints.emplace_back(Affine(passthru_idx.name, -1) +       //
                                        Affine(inner_idx.name, -local_mul) +  //
                                        int64_t(outer_idx.range - 1));
      }

      if (split_unaligned && (outer_idx.range % shape[i] > 0)) {
        split_points.push_back({passthru_idx.name, outer_idx.range / shape[i], shape[i]});
      }

      // Finally add the passthru_idx
      passthru_idxs.emplace_back(passthru_idx);
    }

    // Replace the indices on the outer block with 'outer indicies'
    // Make ranges of the outer blocks: [ceil(ri / ti), ceil(rj / tj), ceil(rk / tk), ...]
    outer_idx.range = math::RoundUp(outer_idx.range, shape[i]);
    range_by_name[outer_idx.name] = outer_idx.range;
    // Make ranges of the inner blocks: [ti, tk, tk]
    inner_idx.range = shape[i];
  }
  // Append all passthru_idxs, we defer this since this may cause inner->idxs to realloc
  std::copy(passthru_idxs.begin(), passthru_idxs.end(), std::back_inserter(inner->idxs));

  // Copy the location to the inner block.
  inner->location = outer->location;

  if (location_idx_tag.size()) {
    // Rewrite the outer block's location, replacing instances of the
    // tag with an affine of the loop indices.
    std::string tag = "#" + location_idx_tag;
    Affine tile_unit;
    std::int64_t scale = 1;
    // N.B. We walk the indices in reverse order just to match the
    // unrolling order; it doesn't actually matter, but it makes the
    // generated code look a little nicer.
    for (auto idx_it = outer->idxs.rbegin(); idx_it != outer->idxs.rend(); ++idx_it) {
      if (idx_it->range > 1) {
        tile_unit.mutateMap()[idx_it->name] = scale;
        scale *= idx_it->range;
      }
    }
    for (auto& dev : outer->location.devs) {
      for (auto& unit : dev.units) {
        unit.substitute(tag, tile_unit);
      }
    }

    if (tile_unit.isConstant()) {
      // Replace the tag on the inner location with the constant.
      for (auto& dev : inner->location.devs) {
        for (auto& unit : dev.units) {
          unit.substitute(tag, tile_unit);
        }
      }
    } else {
      // Create an index for the inner block, and update the inner
      // location to use it.
      std::string inner_idx_name = inner->unique_idx_name(location_idx_tag);
      inner->idxs.emplace_back(Index{inner_idx_name, 1, tile_unit});
      for (auto& dev : inner->location.devs) {
        for (auto& unit : dev.units) {
          auto& umap = unit.mutateMap();
          auto it = umap.find(tag);
          if (it != umap.end()) {
            umap[inner_idx_name] = it->second;
            umap.erase(it);
          }
        }
      }
    }
  }

  // Copy all in/out refinements from outer to inner block
  inner->refs = outer->refs;

  // Remove allocs on the outer refs
  for (auto it = outer->refs.begin(), limit = outer->refs.end(); it != limit;) {
    if (it->dir == RefDir::None) {
      it = outer->refs.erase(it);
    } else {
      ++it;
    }
  }

  // Compute for each reference / for each dimension / the minimal offset
  std::map<std::string, std::vector<int64_t>> zero_points;
  for (auto& ref : inner->refs) {
    auto& vec = zero_points[ref.into()];
    for (auto& aff : ref.access) {
      int64_t min = 0;
      for (auto& kvp : aff.getMap()) {
        if (kvp.first != "" && kvp.second < 0) {
          min += kvp.second * (tile_by_name[kvp.first] - 1);
        }
      }
      vec.push_back(min);
    }
  }
  // Adjust inner references
  for (auto& ref : inner->refs) {
    const auto& zeros = zero_points[ref.into()];
    for (size_t i = 0; i < ref.access.size(); i++) {
      auto& aff = ref.mut().access[i];
      // We pick the outer block to do the bumping so it only happens once
      // But we still need to adjust the zero point of the inner block
      aff.setConstant(-zeros[i]);
      // Since we're taking a single block and turning it into two (e.g. outer and inner),
      // arrange for only one of the blocks to do the constant pointer bumping.
      if (interleave) {
        // Multiply each stride in the inner block refinements by the appropriate range
        for (auto& kvp : aff.mutateMap()) {
          if (!kvp.first.empty()) {
            kvp.second *= range_by_name[kvp.first];
          }
        }
      }
    }
  }
  for (auto& ref : outer->refs) {
    // Fix the sizes on the outer block
    ref.mut().interior_shape = inner->exterior_shape(ref.into());
    const auto& zeros = zero_points[ref.into()];
    for (size_t i = 0; i < ref.access.size(); i++) {
      auto& aff = ref.mut().access[i];
      aff += stripe::Affine(zeros[i]);
      if (!interleave) {
        // Multiply each stride in the outer block refinements by the appropriate tile size
        for (auto& kvp : aff.mutateMap()) {
          if (!kvp.first.empty()) {
            kvp.second *= tile_by_name[kvp.first];
          }
        }
      }
    }
    // Let inner's from be the outer's into
    inner->ref_by_into(ref.into())->mut().from = ref.into();
  }

  if (split_unaligned && split_points.size() > 0) {
    outer->stmts = {};
    std::set<std::string> passthru;
    for (const auto& sp : split_points) {
      passthru.insert(sp.index);
    }

    size_t n_points = split_points.size();
    for (size_t i = 0; i < n_points; ++i) {
      auto unaligned = stripe::CloneBlock(*inner);
      for (size_t j = 0; j < i; ++j) {
        const auto& sp = split_points[j];
        // Find the corresponding constraints and replace it
        for (auto& cons : unaligned->constraints) {
          auto& cons_map = cons.mutateMap();
          if (cons_map.size() == 3 && cons_map.find(sp.index) != cons_map.end()) {
            cons_map.clear();
            cons_map.emplace(sp.index, -1);
            cons_map.emplace("", (sp.value - 1) * sp.step);
          }
        }
      }
      const auto& sp = split_points[i];
      Affine unaligned_cons(-sp.value * sp.step);
      auto& aff_map = unaligned_cons.mutateMap();
      aff_map.emplace(sp.index, 1);
      unaligned->constraints.push_back(unaligned_cons);
      if (i > 0) {
        PromoteCondition(outer, unaligned, passthru, true);
      } else {
        outer->stmts.push_back(unaligned);
      }
    }

    // Remove all original constraints in inner
    for (const auto& sp : split_points) {
      for (auto& cons : inner->constraints) {
        auto& cons_map = cons.mutateMap();
        if (cons_map.find(sp.index) != cons_map.end()) {
          cons_map.clear();
        }
      }
    }
    inner->constraints.erase(std::remove(inner->constraints.begin(),  //
                                         inner->constraints.end(), Affine()),
                             inner->constraints.end());
    for (const auto& sp : split_points) {
      Affine aff(sp.index, -1);
      aff.setConstant((sp.value - 1) * sp.step);
      inner->constraints.push_back(aff);
    }
    PromoteCondition(outer, inner, passthru, false);
  } else {
    outer->stmts = {inner};
  }
  return true;
}

// A simple wrapper to provide an ordering to object vectors that
// we're going to be processing with std::next_permutation() --
// e.g. if we used pointers as comparison values, our order of
// iteration could vary run-to-run, creating non-determinism.
template <typename V>
class Orderer {
 public:
  Orderer(std::size_t ord, V value) : ord_{ord}, value_{std::forward<V>(value)} {}

  void set_ord(std::size_t ord) { ord_ = ord; }
  std::size_t ord() const { return ord_; }

  V& operator*() { return value_; }
  const V& operator*() const { return value_; }

  V& operator->() { return value_; }
  const V& operator->() const { return value_; }

  bool operator<(const Orderer<V>& other) const { return ord() < other.ord(); }

 private:
  std::size_t ord_;
  V value_;
};

template <typename V>
void swap(Orderer<V>& v1, Orderer<V>& v2) {
  std::size_t v1o = v1.ord();
  v1.set_ord(v2.ord());
  v2.set_ord(v1o);
  std::swap(*v1, *v2);
}

boost::optional<StencilMatch> FindBestStencil(const std::vector<proto::Stencil>& specs, stripe::Block* block) {
  boost::optional<StencilMatch> lowest_cost_match;

  std::vector<Orderer<Refinement*>> ref_ins;
  std::size_t ord = 0;
  for (auto* ref : block->ref_ins()) {
    ref_ins.emplace_back(ord++, ref);
  }

  std::vector<Orderer<Refinement*>> ref_outs;
  ord = 0;
  for (auto* ref : block->ref_outs()) {
    ref_outs.emplace_back(ord++, ref);
  }

  auto check_legal = [](std::int64_t rule, std::int64_t candidate) { return rule == -1 || candidate == rule; };

  for (const auto& spec : specs) {
    IVLOG(4, "Attempting to match spec: " << spec.DebugString());

    // Verify spec rule compatibility upfront.
    bool sizes_match = true;
    for (const auto& rule : spec.idxs()) {
      if (ref_outs.size() != static_cast<size_t>(rule.outs_size()) ||
          ref_ins.size() != static_cast<size_t>(rule.ins_size())) {
        sizes_match = false;
        break;
      }
    }
    if (!sizes_match) {
      continue;
    }

    std::vector<Orderer<const Index*>> idxs;
    ord = 0;
    for (auto& idx : block->idxs) {
      idxs.emplace_back(ord++, &idx);
    }

    // Add virtual indexes for this spec so that we consider inefficent but valid matches.
    std::vector<Index> virtual_idxs;
    for (int k = idxs.size(); k < spec.idxs_size(); k++) {
      auto idx_name = block->unique_idx_name("$" + std::to_string(k));
      virtual_idxs.emplace_back(Index{idx_name, 1});
    }
    for (auto& idx : virtual_idxs) {
      idxs.emplace_back(ord++, &idx);
    }

    // The match-building logic, used when we've found a permutation
    // of idxs, ref_ins, and ref_outs that matches the stencil's
    // rules.  We've pulled out of the permutation-walking loops in
    // order to make the loops a bit more straightforward to
    // understand.
    auto build_stencil_match = [&]() {
      StencilMatch match{1, {}};
      std::map<std::string, StencilIndexMatch> idx_matches;
      for (const auto& idx : block->idxs) {
        idx_matches[idx.name] = StencilIndexMatch{idx.name, "*", 1};
      }
      auto idx_it = idxs.begin();
      auto rule_it = spec.idxs().begin();
      while (rule_it != spec.idxs().end()) {
        StencilIndexMatch idx_match;
        if (rule_it->size() != -1) {
          idx_match = StencilIndexMatch{(*idx_it)->name, rule_it->name(), static_cast<uint64_t>(rule_it->size())};
        } else {
          auto block_idx = block->idx_by_name((*idx_it)->name);
          idx_match = StencilIndexMatch{(*idx_it)->name, rule_it->name(), block_idx->range};
        }
        idx_matches[idx_match.block_idx_name] = idx_match;

        ++rule_it;
        ++idx_it;
      }
      size_t total_tiles = 1;
      for (const auto& idx : block->idxs) {
        auto tile = safe_at(idx_matches, idx.name);
        size_t num_tiles = math::RoundUp(idx.range, tile.value);
        total_tiles *= num_tiles;
        match.cost *= num_tiles * tile.value;
        match.idxs.push_back(tile);
      }
      for (const auto& ref_in : ref_ins) {
        match.ref_ins.emplace_back(*ref_in);
      }
      for (const auto& ref_out : ref_outs) {
        match.ref_outs.emplace_back(*ref_out);
      }
      match.cost += spec.startup_cost() * total_tiles;
      IVLOG(4, "Candidate: " << match);
      return match;
    };

    IVLOG(4, "Beginning iteration");

    do {      // Iterate through idxs permutations
      do {    // Iterate through ref_ins permutations
        do {  // Iterate through ref_outs permutations
          bool is_legal = true;

          // Loop over spec rules and block idxs.
          auto idx_it = idxs.begin();
          auto rule_it = spec.idxs().begin();
          for (; is_legal && rule_it != spec.idxs().end(); ++idx_it, ++rule_it) {
            // Walk through the (spec rule in, ref_in) pairs, and see
            // if the access for the current block idx is compatible
            // with the spec rule.
            auto rule_in_it = rule_it->ins().begin();
            auto ref_in_it = ref_ins.begin();
            for (; is_legal && rule_in_it != rule_it->ins().end(); ++rule_in_it, ++ref_in_it) {
              is_legal &= check_legal(*rule_in_it, (*ref_in_it)->FlatAccess().get((*idx_it)->name));
            }

            // Walk through the (spec rule out, ref_out) pairs, and
            // see if the access for the current block idx is
            // compatible with the spec rule.
            auto rule_out_it = rule_it->outs().begin();
            auto ref_out_it = ref_outs.begin();
            for (; is_legal && rule_out_it != rule_it->outs().end(); ++rule_out_it, ++ref_out_it) {
              is_legal &= check_legal(*rule_out_it, (*ref_out_it)->FlatAccess().get((*idx_it)->name));
            }
          }

          if (is_legal) {
            // This permutation of idxs, ref_ins, and ref_outs matches
            // the stencil.  Generate the corresponding StencilMatch,
            // and accumulate it into lowest_cost_match.
            auto match = build_stencil_match();
            if (!lowest_cost_match || match < *lowest_cost_match) {
              IVLOG(4, "  Accepting match: " << match);
              lowest_cost_match = std::move(match);
            }
          }
        } while (std::next_permutation(ref_outs.begin(), ref_outs.end()));
      } while (std::next_permutation(ref_ins.begin(), ref_ins.end()));
    } while (std::next_permutation(idxs.begin(), idxs.end()));
  }  // Loop over specs

  IVLOG(4, "Returning lowest-cost match");

  return lowest_cost_match;
}

struct StencilPassOptions {
  Tags reqs;
  std::vector<proto::Stencil> specs;
  Tags set_outer;
  Tags set_inner;
  std::vector<Tags> set_inputs;
  std::vector<Tags> set_outputs;
};

void ApplyIndexTags(Block* block, const StencilMatch& match) {
  for (const auto& idx_match : match.idxs) {
    if (idx_match.stencil_idx_name == "*") {
      continue;
    }
    auto idx = block->idx_by_name(idx_match.block_idx_name);
    if (idx) {
      idx->set_tag("stencil");
      idx->set_tag(str(boost::format("stencil_%1%") % idx_match.stencil_idx_name));
    }
  }
}

void ApplyRefTags(Block* block, const StencilMatch& match, const StencilPassOptions& options) {
  auto ref_it = match.ref_ins.begin();
  for (const auto& tags : options.set_inputs) {
    if (ref_it == match.ref_ins.end()) {
      break;
    }
    (*ref_it++)->add_tags(tags);
  }

  ref_it = match.ref_outs.begin();
  for (const auto& tags : options.set_outputs) {
    if (ref_it == match.ref_outs.end()) {
      break;
    }
    (*ref_it++)->add_tags(tags);
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
    auto match = FindBestStencil(options.specs, block);
    if (!match) {
      return;
    }
    ApplyRefTags(block, *match, options);
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

void StencilPass::Apply(Block* block) const {
  StencilPassOptions sopts = {
      FromProto(options_.reqs()),       // reqs
      {},                               // specs
      FromProto(options_.outer_set()),  // set_outer
      FromProto(options_.inner_set())   // set_inner
  };
  for (const auto& stencil : options_.stencils()) {
    sopts.specs.push_back(stencil);
  }
  for (const auto& input_set : options_.inputs_set()) {
    sopts.set_inputs.emplace_back(FromProto(input_set.tags()));
  }
  for (const auto& output_set : options_.outputs_set()) {
    sopts.set_outputs.emplace_back(FromProto(output_set.tags()));
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

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<StencilPass, proto::StencilPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
