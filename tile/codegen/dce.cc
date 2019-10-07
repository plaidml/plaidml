// Copyright 2018, Intel Corporation

#include "tile/codegen/dce.h"

#include <map>
#include <set>
#include <string>
#include <vector>

#include "tile/codegen/deps.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT
using namespace math;    // NOLINT

// Remove the inputs that are not used and the outputs that are not defined.
// Note that the buffer use information in AliasMap may not be correct.
static void PruneRefinements(Block* block) {
  std::set<std::string> uses;
  std::set<std::string> defs;
  for (const auto& stmt : block->stmts) {
    switch (stmt->kind()) {
      case StmtKind::Load: {
        auto load = Load::Downcast(stmt);
        uses.insert(load->from);
        defs.insert(load->into);
      } break;
      case StmtKind::Store: {
        auto store = Store::Downcast(stmt);
        uses.insert(store->from);
        defs.insert(store->into);
      } break;
      case StmtKind::Constant: {
        auto constant = Constant::Downcast(stmt);
        defs.insert(constant->name);
      } break;
      case StmtKind::LoadIndex: {
        auto load_index = LoadIndex::Downcast(stmt);
        defs.insert(load_index->into);
      } break;
      case StmtKind::Special: {
        auto special = Special::Downcast(stmt);
        uses.insert(special->inputs.begin(), special->inputs.end());
        defs.insert(special->outputs.begin(), special->outputs.end());
      } break;
      case StmtKind::Intrinsic: {
        auto intrinsic = Intrinsic::Downcast(stmt);
        uses.insert(intrinsic->inputs.begin(), intrinsic->inputs.end());
        defs.insert(intrinsic->outputs.begin(), intrinsic->outputs.end());
      } break;
      case StmtKind::Block: {
        auto sub_block = Block::Downcast(stmt);
        for (const auto& ref : sub_block->refs) {
          if (ref.dir == RefDir::In) {
            uses.insert(ref.from);
          } else if (ref.dir == RefDir::Out) {
            defs.insert(ref.from);
          } else if (ref.dir == RefDir::InOut) {
            uses.insert(ref.from);
            defs.insert(ref.from);
          }
        }
      } break;
    }
  }

  for (auto& ref : block->refs) {
    switch (ref.dir) {
      case RefDir::In: {
        // If the In refinement is not used, remove it.
        if (uses.find(ref.into()) == uses.end()) {
          ref.mut().set_tag("removed");
        }
      } break;
      case RefDir::Out: {
        // If the Out refinement is not defined, remove it.
        if (defs.find(ref.into()) == defs.end()) {
          ref.mut().set_tag("removed");
        }
      } break;
      case RefDir::InOut: {
        // If the InOut refinement is not either used or defined, remove it.
        if (uses.find(ref.into()) == uses.end() && defs.find(ref.into()) == defs.end()) {
          ref.mut().set_tag("removed");
        }
        if (uses.find(ref.into()) == uses.end() && !ref.has_tag("initialized")) {
          // never used and initialized before, set as Out
          ref.mut().dir = RefDir::Out;
        } else if (defs.find(ref.into()) == defs.end()) {
          // never defined, set as In
          ref.mut().dir = RefDir::In;
        }
      } break;
      case RefDir::None: {
        // If the None refinement is not either used or defined, remove it.
        if (uses.find(ref.into()) == uses.end() && defs.find(ref.into()) == defs.end()) {
          ref.mut().set_tag("removed");
        }
      } break;
    }
  }
  for (auto it = block->refs.begin(), limit = block->refs.end(); it != limit;) {
    if (it->has_tag("removed")) {
      it = block->refs.erase(it);
    } else {
      ++it;
    }
  }
}

// To determine if the output of the stmt is one of the block outputs.
bool IsResultBlockOutput(const Statement* stmt, const std::set<std::string>& outputs) {
  switch (stmt->kind()) {
    case StmtKind::Load: {
      auto load = dynamic_cast<const Load*>(stmt);
      if (outputs.find(load->into) != outputs.end()) {
        return true;
      }
    } break;
    case StmtKind::Store: {
      auto store = dynamic_cast<const Store*>(stmt);
      if (outputs.find(store->into) != outputs.end()) {
        return true;
      }
    } break;
    case StmtKind::Constant: {
      auto constant = dynamic_cast<const Constant*>(stmt);
      if (outputs.find(constant->name) != outputs.end()) {
        return true;
      }
    } break;
    case StmtKind::LoadIndex: {
      auto load_index = dynamic_cast<const LoadIndex*>(stmt);
      if (outputs.find(load_index->into) != outputs.end()) {
        return true;
      }
    } break;
    case StmtKind::Special: {
      auto special = dynamic_cast<const Special*>(stmt);
      for (const auto& output : special->outputs) {
        if (outputs.find(output) != outputs.end()) {
          return true;
        }
      }
    } break;
    case StmtKind::Intrinsic: {
      auto intrinsic = dynamic_cast<const Intrinsic*>(stmt);
      for (const auto& output : intrinsic->outputs) {
        if (outputs.find(output) != outputs.end()) {
          return true;
        }
      }
    } break;
    case StmtKind::Block: {
      auto sub_block = dynamic_cast<const Block*>(stmt);
      for (const auto& ref : sub_block->refs) {
        if ((ref.dir == RefDir::Out || ref.dir == RefDir::InOut) && outputs.find(ref.from) != outputs.end()) {
          return true;
        }
      }
    } break;
  }  // switch
  return false;
}

void DeadCodeElimination(const AliasMap& alias_map, Block* block) {
  // If this is root, just remove useless refinements
  if (alias_map.parent_block() == nullptr) {
    PruneRefinements(block);
    return;
  }

  // If the output of a statement is in the block output (non-temporary) list,
  // the statement is used
  std::set<std::string> outputs;
  for (const auto& ref : block->refs) {
    if ((ref.dir == RefDir::Out || ref.dir == RefDir::InOut) && !ref.has_tag("tmp")) {
      outputs.insert(ref.into());
    }
  }

  ComputeDepsForBlock(block, alias_map);
  // Map a statement to its uses
  std::map<Statement*, std::vector<Statement*>> stmt_uses;
  for (const auto& stmt : block->stmts) {
    for (const auto& dep_stmt_it : stmt->deps) {
      Statement* dep_stmt = dep_stmt_it->get();
      stmt_uses[dep_stmt].push_back(stmt.get());
    }
  }

  // Traverse backward
  for (auto stmt_it = block->stmts.rbegin(); stmt_it != block->stmts.rend(); ++stmt_it) {
    Statement* stmt = stmt_it->get();
    bool stmt_is_used = IsResultBlockOutput(stmt, outputs);
    if (!stmt_is_used) {
      std::vector<Statement*>& stmt_list = stmt_uses[stmt];
      stmt_is_used = std::any_of(stmt_list.begin(), stmt_list.end(),  //
                                 [](Statement* stmt) { return !stmt->has_tag("removed"); });
    }
    if (!stmt_is_used) {
      // If the stmt's output is not either the block's output or
      // used by a live stmt, remove it.
      stmt->set_tag("removed");
    }
  }

  // Remove all marked "removed" statements
  block->erase_stmts(            //
      std::remove_if(            //
          block->stmts.begin(),  //
          block->stmts.end(),    //
          [](const auto& stmt) { return stmt->has_tag("removed"); }),
      block->stmts.end());

  // Clean up refinements
  // Do not use AliasMap here, which may not be correct after code elimination
  PruneRefinements(block);

  // Check if this block can be removed after cleaning
  if (std::none_of(block->refs.begin(), block->refs.end(), [](const Refinement& ref) {
        return (ref.dir == RefDir::InOut || ref.dir == RefDir::Out) && !ref.has_tag("tmp");
      })) {
    block->set_tag("removed");
  }
}

void DeadCodeEliminationPass::Apply(CompilerState* state) const {
  auto reqs = stripe::FromProto(options_.reqs());
  RunOnBlocksBackward(
      state->entry(), reqs,
      [](const AliasMap& alias_map, stripe::Block* block) {  //
        DeadCodeElimination(alias_map, block);
      },
      true);

  RunOnBlocks(
      state->entry(), reqs,
      [&](const AliasMap& map, stripe::Block* block) {  //
        if (options_.fix_deps()) {
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

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<DeadCodeEliminationPass, proto::DeadCodeEliminationPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
