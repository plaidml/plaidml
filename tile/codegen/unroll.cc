// Copyright 2018, Intel Corporation

#include "tile/codegen/unroll.h"

#include <list>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/util/logging.h"
#include "base/util/throw.h"
#include "tile/codegen/alias.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

namespace {

struct IndexValue {
  const Index* idx;
  size_t value;
};

using EnumerateFunc = std::function<void(const std::vector<IndexValue>& idxs)>;

void EnumerateIndexes(const std::vector<IndexValue>& idxs, size_t idx_num, const EnumerateFunc& func) {
  if (idx_num == idxs.size()) {
    func(idxs);
  } else {
    for (size_t i = 0; i < idxs[idx_num].idx->range; i++) {
      auto next = idxs;
      next[idx_num].value = i;
      EnumerateIndexes(next, idx_num + 1, func);
    }
  }
}

void EvalWithValues(Block* block, const std::map<std::string, int64_t>& fixed) {
  IVLOG(3, "EvalWithValues> block: " << block->name << ", fixed: " << fixed);
  block->location = PartialEval(block->location, fixed);
  for (auto& ref : block->refs) {
    ref.mut().location = PartialEval(ref.location, fixed);
    for (auto& aff : ref.mut().access) {
      aff = aff.partial_eval(fixed);
    }
    if (ref.cache_unit) {
      ref.mut().cache_unit = ref.cache_unit->partial_eval(fixed);
    }
  }
  for (auto& constraint : block->constraints) {
    constraint = constraint.partial_eval(fixed);
  }
  for (const auto& stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      for (auto& idx : inner->idxs) {
        idx.affine = idx.affine.partial_eval(fixed);
      }
      std::map<std::string, int64_t> next;
      for (const auto& item : fixed) {
        if (item.first[0] == '#') {
          next.emplace(item.first, item.second);
        }
      }
      std::vector<std::string> prune_idxs;
      for (const auto& idx : inner->idxs) {
        if (idx.range == 1 && idx.affine.isConstant()) {
          prune_idxs.push_back(idx.name);
          next.emplace(idx.name, idx.affine.constant());
        }
      }
      EvalWithValues(inner.get(), next);
      for (const auto& idx_name : prune_idxs) {
        auto it = std::find_if(inner->idxs.begin(), inner->idxs.end(),
                               [&idx_name](const Index& idx) { return idx.name == idx_name; });
        inner->idxs.erase(it);
      }
    }
  }
}

struct ExpansionRule {
  std::string rule;
  std::string idx_name() const { return rule; }
  std::string tag_name() const { return rule.substr(1); }
  bool is_valid() const { return !rule.empty(); }
  bool is_tag() const { return is_valid() && rule.substr(0, 1) == "#"; }
};

struct ExpansionValue {
  size_t last_stride = 1;
  size_t value = 0;

  void operator+=(const IndexValue& idx_val) {
    value += last_stride * idx_val.value;
    last_stride *= idx_val.idx->range;
  }
};

using RefMap = std::map<std::tuple<std::string, std::vector<stripe::Affine>>, std::string>;

void EvalInner(Block* outer,                                                                //
               Block* block,                                                                //
               RefMap* ref_map,                                                             //
               const std::vector<IndexValue>& idxs,                                         //
               const AliasMap& outer_alias_map,                                             //
               std::map<std::string, std::vector<std::vector<Extent>>>* ref_write_extents,  //
               const std::map<std::string, Affine>& aff_idxs,                               //
               const proto::UnrollPass& options) {
  IVLOG(3, "EvalInner> " << outer->name << " -> " << block->name);
  std::map<std::string, int64_t> fixed;
  ExpansionRule rule{options.expand_idx()};
  ExpansionValue expand_val;
  std::stringstream ss;
  ss << "%";
  if (!options.part_name().empty()) {
    ss << options.part_name();
  }
  for (const auto& idx_val : idxs) {
    fixed.emplace(idx_val.idx->name, idx_val.value);
    ss << "_" << idx_val.idx->name << "_" << idx_val.value;
    if (rule.is_tag()) {
      // The rule is tagged, like: '#foo'. Accumulate tagged idxs into primary.
      if (idx_val.idx->has_tag(rule.tag_name())) {
        expand_val += idx_val;
      }
    } else if (rule.is_valid()) {
      // The rule is not tagged, like: 'foo'. Accumulate specifc idx into primary.
      if (idx_val.idx->name == rule.idx_name()) {
        expand_val += idx_val;
      }
    }
  }
  if (rule.is_valid()) {
    fixed.emplace(rule.idx_name(), expand_val.value);
  }
  EvalWithValues(block, fixed);

  AliasMap block_alias_map{outer_alias_map, block};
  std::map<std::string, std::vector<std::vector<Extent>>> current_ref_write_extents;
  for (const auto& block_ref : block->refs) {
    if (!IsWriteDir(block_ref.dir)) {
      continue;
    }

    const auto& ai = block_alias_map.at(block_ref.into());

    if (block_ref.agg_op.size() && !IsReadDir(block_ref.dir)) {
      // We're aggregating into this refinement, so as we unroll, we
      // need to add in the read direction if the underlying memory
      // overlaps with any refinement previously written by an
      // unrolled step.
      auto name_extents = ref_write_extents->find(ai.base_name);
      if (name_extents != ref_write_extents->end()) {
        bool found_overlap = false;
        for (const auto& extents : name_extents->second) {
          if (CheckOverlap(ai.extents, extents)) {
            found_overlap = true;
            break;
          }
        }
        if (found_overlap) {
          block_ref.mut().dir = RefDir::InOut;
        }
      }
    }

    // Keep track of the fact that we're writing to this ref, so
    // that we can add the read direction to affected subsequent
    // blocks as needed.
    current_ref_write_extents[ai.base_name].emplace_back(ai.extents);
  }

  // Accumulate the extents written by the current block into the extents tracked over the unrolling.
  for (auto& ref_extents : current_ref_write_extents) {
    auto& extents = (*ref_write_extents)[ref_extents.first];
    extents.insert(extents.end(), ref_extents.second.begin(), ref_extents.second.end());
  }

  std::unordered_map<std::string, std::string> block_alloc_map;  // block.into -> outer.into
  for (const auto& ref : block->refs) {
    if (ref.dir != RefDir::None) {
      continue;
    }
    // This refinement is an alloc; in order for the inner blocks to
    // access it, we need to push it to the outer block.
    auto alloc = ref.WithInto(outer->unique_ref_name(ref.into()));
    block_alloc_map[ref.into()] = alloc.into();
    outer->refs.emplace(std::move(alloc));
  }

  for (const auto& stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (!inner) {
      continue;
    }
    inner->name += ss.str();
    for (auto& inner_ref : inner->refs) {
      if (inner_ref.from.empty()) {
        continue;
      }
      auto block_ref = block->ref_by_into(inner_ref.from);
      if (block_ref->dir == RefDir::None) {
        inner_ref.mut().from = block_alloc_map[block_ref->into()];
        continue;
      }
      for (size_t i = 0; i < inner_ref.access.size(); i++) {
        inner_ref.mut().access[i] += block_ref->access[i];
      }
      inner_ref.mut().dir = UnionDir(inner_ref.dir, block_ref->dir);
      auto outer_ref = outer->ref_by_into(block_ref->from);
      IVLOG(3, "  outer_ref: " << *outer_ref);
      IVLOG(3, "  block_ref: " << *block_ref);
      IVLOG(3, "  inner_ref: " << inner_ref);
      inner_ref.mut().from = block_ref->from;
      IVLOG(3, "    from = " << inner_ref.from);
      if (!options.make_views()) {
        continue;
      }
      if (block_ref->interior_shape == outer_ref->interior_shape) {
        IVLOG(3, "    no view, same shape");
        continue;
      }
      auto key = std::make_tuple(outer_ref->from, inner_ref.access);
      auto it_inserted = ref_map->emplace(key, outer_ref->into());
      if (it_inserted.second) {
        it_inserted.first->second = outer->unique_ref_name(outer_ref->into());
      }
      auto view = block_ref->WithInto(it_inserted.first->second);
      view.from = outer_ref->from;
      IVLOG(3, "    make view: " << view.into() << " from: " << view.from << " via: " << block_ref->from);
      if (inner_ref.cache_unit || outer_ref->cache_unit) {
        IVLOG(3, "    with cache: " << *outer_ref->cache_unit);
        view.cache_unit = outer_ref->cache_unit;
      }
      for (size_t i = 0; i < inner_ref.access.size(); i++) {
        auto const_access = block_ref->access[i].constant();
        view.access[i] = outer_ref->access[i] + const_access;
        inner_ref.mut().access[i] -= const_access;
      }
      inner_ref.mut().from = view.into();
      IVLOG(2, "view: " << view);
      if (outer->ref_by_into(view.into(), false) == outer->refs.end()) {
        outer->refs.emplace(std::move(view));
      }
    }
    // Rewrite the inner affine indices in terms of the parent block
    // (where they're going to be inserted).
    for (auto& idx : inner->idxs) {
      if (idx.affine != Affine{}) {
        idx.affine.substitute(aff_idxs);
      }
    }
  }
}

void UnrollBlock(Block* outer,                     //
                 Block* block,                     //
                 const AliasMap& outer_alias_map,  //
                 const StatementIt& it_stmt,       //
                 const Tags& reqs,                 //
                 const proto::UnrollPass& options) {
  if (block->has_tags(reqs)) {
    RefMap ref_map;
    std::vector<IndexValue> idxs;
    idxs.reserve(block->idxs.size());
    std::map<std::string, Affine> aff_idxs;
    for (const auto& idx : block->idxs) {
      if (idx.affine == Affine{}) {
        idxs.emplace_back(IndexValue{&idx, 0});
      } else {
        aff_idxs[idx.name] = idx.affine;
      }
    }
    std::map<std::string, std::vector<std::vector<Extent>>> ref_write_extents;
    std::list<StatementList> cloned_statements;
    EnumerateIndexes(idxs, 0, [&](const std::vector<IndexValue>& idxs) {
      auto clone = CloneBlock(*block);
      EvalInner(outer, clone.get(), &ref_map, idxs, outer_alias_map, &ref_write_extents, aff_idxs, options);
      cloned_statements.emplace_back(std::move(clone->stmts));
    });
    while (cloned_statements.size()) {
      for (auto cs_it = cloned_statements.begin(); cs_it != cloned_statements.end();) {
        if (cs_it->size()) {
          outer->stmts.emplace(it_stmt, std::move(cs_it->front()));
          cs_it->pop_front();
          ++cs_it;
        } else {
          cs_it = cloned_statements.erase(cs_it);
        }
      }
    }
    outer->erase_stmt(it_stmt);
  } else {
    AliasMap alias_map{outer_alias_map, block};
    PreIterate(block, [&](const StatementIt& it) {
      auto inner = Block::Downcast(*it);
      if (inner) {
        UnrollBlock(block, inner.get(), alias_map, it, reqs, options);
      }
    });
  }
}

}  // namespace

void UnrollPass::Apply(CompilerState* state) const {
  auto reqs = FromProto(options_.reqs());
  Block* root = state->entry();
  AliasMap base_alias_map;
  AliasMap alias_map{base_alias_map, root};
  PreIterate(root, [&](const StatementIt& it) {
    auto inner = Block::Downcast(*it);
    if (inner) {
      UnrollBlock(root, inner.get(), alias_map, it, reqs, options_);
    }
  });
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<UnrollPass, proto::UnrollPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
