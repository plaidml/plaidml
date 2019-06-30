// Copyright 2018, Intel Corporation

#include "tile/codegen/tidy.h"

#include "base/util/logging.h"
#include "tile/codegen/alias.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

bool NotUsedInRefs(Block* block, const std::string& idx_name) {
  for (const auto& ref : block->refs) {
    for (const auto& acc : ref.access) {
      if (acc.getMap().find(idx_name) != acc.getMap().end()) {
        return false;
      }
    } 
  }
  return true;
}

bool NotUsedInConstraints(Block* block, const std::string& idx_name) {
  for (const auto& cons : block->constraints) {
    if (cons.getMap().find(idx_name) != cons.getMap().end()) {
      return false;
    }
  }
  return true;
}

bool UsedInIdxs(Block* block, const std::string& idx_name) {
  for (const auto& idx : block->idxs) {
    if (idx.affine != Affine()) {
      if (idx.affine.getMap().find(idx_name) != idx.affine.getMap().end()) {
        return true;
      }
    }
  }
  return false;
}

bool NotUsedInStmts(Block* block, const std::string& idx_name) {
  for (const auto& stmt : block->stmts) {
    auto load_index = LoadIndex::Downcast(stmt);
    if (load_index) {
      if (load_index->from.getMap().find(idx_name) != load_index->from.getMap().end()) {
        return false;
      }
      continue;
    }
    auto sub_block = Block::Downcast(stmt);
    if (sub_block) {
      if (UsedInIdxs(sub_block.get(), idx_name)) {
        return false;
      }
      continue;
    } 
  }
  return true;
}

bool NotUsedInBlock(Block* block, const std::string& idx_name) {
  return NotUsedInRefs(block, idx_name) &&
         NotUsedInConstraints(block, idx_name) &&
         NotUsedInStmts(block, idx_name);
}

void PruneIndexes(Block* block, const Tags& exclude_tags) {
  // Find all the indexes to remove
  std::set<const Index*> to_remove;
  std::map<std::string, int64_t> idx_values;
  for (const auto& idx : block->idxs) {
    if (!idx.has_any_tags(exclude_tags)) {
      if (idx.range == 1 && idx.affine == 0) {
        to_remove.emplace(&idx);
        idx_values.emplace(idx.name, 0);
      }
      else if (idx.affine != Affine()) {
        if (NotUsedInBlock(block, idx.name)) {
          to_remove.emplace(&idx);
          idx_values.emplace(idx.name, 0);
        }
      }
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
