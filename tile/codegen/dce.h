// Copyright 2018, Intel Corporation

#pragma once

#include <map>
#include <memory>
#include <set>
#include <vector>

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/deps.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

// Traverse backward and the innermost first
template <typename F>
void RunOnBlocksRecurseBackward(const AliasMap& map, stripe::Block* block, const stripe::Tags& reqs, const F& func,
                                bool rec_func) {
  bool run_func = block->has_tags(reqs) || reqs.count("all") > 0;
  if (!run_func || rec_func) {
    for (auto stmt_it = block->stmts.rbegin(); stmt_it != block->stmts.rend(); ++stmt_it) {
      auto inner = stripe::Block::Downcast(*stmt_it);
      if (inner) {
        AliasMap inner_map(map, inner.get());
        RunOnBlocksRecurse(inner_map, inner.get(), reqs, func, rec_func);
      }
    }
    // Remove all statements tagged "removed"
    if (block->stmts.size() > 0) {
      block->stmts.erase(
          std::remove_if(block->stmts.begin(), block->stmts.end(),  //
                         [](const std::shared_ptr<stripe::Statement>& stmt) { return stmt.get()->has_tag("removed"); }),
          block->stmts.end());
    }
  }
  if (run_func) {
    func(map, block);
  }
}

template <typename F>
void RunOnBlocksBackward(stripe::Block* root, const stripe::Tags& reqs, const F& func, bool rec_func = false) {
  AliasMap base;
  AliasMap root_map(base, root);
  RunOnBlocksRecurseBackward(root_map, root, reqs, func, rec_func);
}

void DeadCodeElimination(const AliasMap& alias_map, stripe::Block* block);

inline void DeadCodeEliminationPass(stripe::Block* root, const proto::GenericPass& options, bool fix_deps) {
  auto reqs = stripe::FromProto(options.reqs());
  RunOnBlocksBackward(root, reqs,
                      [](const AliasMap& alias_map, stripe::Block* block) {  //
                        DeadCodeElimination(alias_map, block);
                      },
                      true);

  RunOnBlocks(root, reqs,
              [&](const AliasMap& map, stripe::Block* block) {  //
                if (fix_deps) {
                  // Rebuild deps
                  ComputeDepsForBlock(block, map);
                } else {
                  // Clean up deps after use
                  for (auto& stmt : block->stmts) {
                    stmt.get()->deps.clear();
                  }
                }
              },
              true);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
