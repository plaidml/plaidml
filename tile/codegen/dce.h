// Copyright 2018, Intel Corporation

#pragma once

#include <memory>
#include <set>

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
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

class DeadCodeEliminationPass final : public CompilePass {
 public:
  explicit DeadCodeEliminationPass(const proto::DeadCodeEliminationPass& options) : options_{options} {}
  void Apply(stripe::Block* root) const final;

 private:
  proto::DeadCodeEliminationPass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
