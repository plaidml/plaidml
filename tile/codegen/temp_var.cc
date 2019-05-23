// Copyright 2018, Intel Corporation

#include "tile/codegen/temp_var.h"
#include "base/util/any_factory_map.h"
#include "tile/codegen/tile.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT
using namespace math;    // NOLINT

// Mark #temp_var for the multiple used variables to explicitly delcare it
// It avoids duplicated/redundant loads for the save variable
void TempVar(const AliasMap& alias_map, Block* block, const proto::TempVarPass& options) {
  std::map<std::string, Statement*> var_def_stmt;
  std::map<Statement*, size_t> stmt_uses;
  for (const auto& stmt : block->stmts) {
    stmt_uses.emplace(stmt.get(), 0);
  }
  for (const auto& stmt : block->stmts) {
    switch (stmt->kind()) {
      case StmtKind::Load: {
        auto load = Load::Downcast(stmt);
        var_def_stmt.emplace(load->into, stmt.get());
      } break;
      case StmtKind::Store: {
        auto store = Store::Downcast(stmt);
        auto def_stmt = var_def_stmt[store->from];
        ++stmt_uses[def_stmt];
      } break;
      case StmtKind::Special: {
        auto special = Special::Downcast(stmt);
        for (const auto& out : special->outputs) {
          var_def_stmt.emplace(out, stmt.get());
        }
        for (const auto& in : special->inputs) {
          auto it = var_def_stmt.find(in);
          if (it != var_def_stmt.end()) {
            ++stmt_uses[it->second];
          }
        }
      } break;
      case StmtKind::Intrinsic: {
        auto intrinsic = Intrinsic::Downcast(stmt);
        for (const auto& out : intrinsic->outputs) {
          var_def_stmt.emplace(out, stmt.get());
        }
        for (const auto& in : intrinsic->inputs) {
          auto it = var_def_stmt.find(in);
          if (it != var_def_stmt.end()) {
            ++stmt_uses[it->second];
          }
        }
      } break;
      case StmtKind::Constant:
      case StmtKind::LoadIndex:
      case StmtKind::Block:
        break;
    }
  }
  for (const auto& stmt : block->stmts) {
    if ((stmt->kind() == StmtKind::Load || stmt->kind() == StmtKind::Intrinsic || stmt->kind() == StmtKind::Special) && stmt_uses[stmt.get()] > 1) {
      stmt->set_tag("temp_var");
    }
  }
}

void TempVarPass::Apply(CompilerState* state) const {
  auto reqs = stripe::FromProto(options_.reqs());
  RunOnBlocks(state->entry(), reqs,
              [this](const AliasMap& alias_map, stripe::Block* block) {  //
                TempVar(alias_map, block, options_);
              },
              true);
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<TempVarPass, proto::TempVarPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
