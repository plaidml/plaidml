// Copyright 2018, Intel Corporation

#include "tile/codegen/stencil.h"

#include <map>
#include <utility>

#include "base/util/logging.h"
#include "base/util/lookup.h"
#include "base/util/stream_container.h"
#include "base/util/throw.h"
#include "tile/codegen/tile.h"
#include "tile/math/util.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

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
std::ostream& operator<<(std::ostream& os, const Orderer<V>& v) {
  os << *v << ":" << v.ord();
  return os;
}

template <typename V>
void swap(Orderer<V>& v1, Orderer<V>& v2) {
  std::size_t v1o = v1.ord();
  v1.set_ord(v2.ord());
  v2.set_ord(v1o);
  std::swap(*v1, *v2);
}

boost::optional<StencilMatch> FindBestStencil(const std::vector<proto::Stencil>& specs, const bool is_strict_dims,
                                              stripe::Block* block) {
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
    if (block->idxs.size() > 8) {
      // Skip attempting to stencil when permutation space is too large
      // TODO: There are much better algorithms to use for stenciling
      continue;
    }
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
      StencilMatch match{1, false, {}};
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
        // If we should skip non strict tiles - with a dimension
        // that is not no-remainder dividing the block shape's
        // dimension - mark the match appropriately.
        match.skip_non_strict = is_strict_dims && (match.skip_non_strict || (idx.range % tile.value));
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
          // IVLOG(5, "Evaluating, ref_outs = " << ref_outs << ", ref_ins = " << ref_ins << ", idxs = " << idxs);
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
            if (!match.skip_non_strict && (!lowest_cost_match || match < *lowest_cost_match)) {
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
  bool is_strict_dims;
  bool copy_tags;
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
    auto match = FindBestStencil(options.specs, options.is_strict_dims, block);
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
    if (options.copy_tags) {
      inner->set_attrs(*block);
    }
    inner->add_tags(options.set_inner);
  }
}

void StencilPass::Apply(CompilerState* state) const {
  StencilPassOptions sopts = {
      FromProto(options_.reqs()),       // reqs
      {},                               // specs
      FromProto(options_.outer_set()),  // set_outer
      FromProto(options_.inner_set()),  // set_inner
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

  sopts.is_strict_dims = options_.is_strict_dims();
  sopts.copy_tags = options_.copy_tags();

  StencilPassRecurse(state->entry(), sopts);
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
