// Copyright 2018, Intel Corporation

#include "tile/codegen/tidy.h"

#include "base/util/logging.h"
#include "tile/codegen/alias.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

void PruneIndexes(Block* block, const Tags& exclude_tags) {
  // Find all the indexes to remove
  std::set<const Index*> to_remove;
  std::map<std::string, int64_t> idx_values;
  for (const auto& idx : block->idxs) {
    if (!idx.has_any_tags(exclude_tags) && idx.range == 1 && idx.affine == 0) {
      to_remove.emplace(&idx);
      idx_values.emplace(idx.name, 0);
    }
  }
  // Remove from refinements
  for (auto& refs : block->refs) {
    for (auto& aff : refs.mut().access) {
      aff = aff.partial_eval(idx_values);
    }
  }
  // Remove from constraints
  for (auto& con : block->constraints) {
    con = con.partial_eval(idx_values);
  }
  // Remove from index list
  block->idxs.erase(std::remove_if(block->idxs.begin(), block->idxs.end(),
                                   [&to_remove](const Index& idx) { return to_remove.count(&idx); }),
                    block->idxs.end());
  // Remove from load index statements
  for (auto& stmt : block->stmts) {
    auto inner = LoadIndex::Downcast(stmt);
    if (inner) {
      inner->from = inner->from.partial_eval(idx_values);
    }
  }
  // Remove from inner blocks
  for (auto& stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      for (auto& inner_idx : inner->idxs) {
        inner_idx.affine = inner_idx.affine.partial_eval(idx_values);
      }
    }
  }
  // Recurse
  for (auto& stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      PruneIndexes(inner.get(), exclude_tags);
    }
  }
}

void PruneRefinements(const AliasMap& alias_map, Block* block) {
  for (const auto& stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      AliasMap inner_map(alias_map, inner.get());
      PruneRefinements(inner_map, inner.get());
    }
  }
  auto use_count = alias_map.RefUseCounts(*block);
  IVLOG(2, "PruneRefinements> " << block->name);
  IVLOG(3, "    use_count: " << use_count);
  std::set<std::string> to_remove;
  for (const auto& ref : block->refs) {
    if (!use_count.count(ref.into())) {
      to_remove.emplace(ref.into());
    }
  }
  if (!to_remove.empty()) {
    IVLOG(2, "    to_remove: " << to_remove);
  }
  for (const auto& name : to_remove) {
    block->refs.erase(block->ref_by_into(name));
  }
  // Recurse
  for (auto& stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      AliasMap inner_map(alias_map, inner.get());
      PruneRefinements(inner_map, inner.get());
    }
  }
}

void PruneIndexesPass::Apply(CompilerState* state) const {
  auto reqs = FromProto(options_.reqs());
  RunOnBlocks(state->entry(), reqs, [](const AliasMap& map, Block* block) {  //
    PruneIndexes(block, {});
  });
}

void PruneRefinementsPass::Apply(CompilerState* state) const {
  auto reqs = FromProto(options_.reqs());
  RunOnBlocks(state->entry(), reqs, [](const AliasMap& alias_map, Block* block) {  //
    PruneRefinements(alias_map, block);
  });
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<PruneIndexesPass, proto::PruneIndexesPass>::Register();
  CompilePassFactory<PruneRefinementsPass, proto::PruneRefinementsPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
