// Copyright 2018, Intel Corporation

#include <map>
#include <set>
#include <vector>

#include "tile/codegen/deps.h"
#include "tile/codegen/reorder.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT
using namespace math;    // NOLINT

typedef std::shared_ptr<Statement> StmtPtr;

struct StmtList {
  bool eltwise;                   // If all stmts in the list are element-wise blocks
  std::vector<StmtPtr> stmts;     // Statements in order
  std::vector<size_t> shape;      // The block shape
  std::set<std::string> inputs;   // input refinements
  std::set<std::string> outputs;  // output refinements;
  std::set<StmtPtr> deps;         // The stmts that this list depends on
};

static bool RefMatchIdx(const Refinement& ref, Block* block) {
  std::set<std::string> idx_set;
  for (const auto& idx : block->idxs) {
    idx_set.insert(idx.name);
  }
  for (const auto& acc : ref.access) {
    if (acc == Affine()) {
      continue;
    }
    const auto& acc_map = acc.getMap();
    if (acc_map.size() > 1) {
      return false;
    }
    if (idx_set.find(acc_map.begin()->first) == idx_set.end()) {
      return false;
    }
    idx_set.erase(acc_map.begin()->first);
  }
  return true;
}

static bool Depends(std::shared_ptr<StmtList> sl0, std::shared_ptr<StmtList> sl1) {
  for (const auto& stmt : sl0->stmts) {
    if (sl1->deps.find(stmt) != sl1->deps.end()) {
      return true;
    }
  }
  return false;
}

// If s0 and s1 may be fused, they should:
// 1) be blocks with same shape
// 2) be element-wise
// 3) s0's output is s1's input
static bool MayFuse(std::shared_ptr<StmtList> s0, std::shared_ptr<StmtList> s1) {
  if (!s0->eltwise || !s1->eltwise || s0->shape != s1->shape) {
    return false;
  }
  for (const auto& out : s0->outputs) {
    if (s1->inputs.find(out) != s1->inputs.end()) {
      return true;
    }
  }
  return false;
}

void ReorderBlocksPass::Apply(CompilerState* state) const {
  stripe::Block* root = state->entry();
  auto main_block = root->SubBlock(0);
  if (!main_block->has_tag("main")) {
    throw std::runtime_error("Input non-root block for block reordering pass.");
  }

  // Compute deps
  AliasMap base;
  AliasMap root_map(base, root);
  AliasMap alias_map(root_map, main_block.get());
  ComputeDepsForBlock(main_block.get(), alias_map);

  std::vector<std::shared_ptr<StmtList>> stmt_list;
  StatementList done;

  for (const auto& stmt : main_block->stmts) {
    if (ZeroBlock(stmt)) {
      done.push_back(stmt);
      continue;
    }
    auto block = Block::Downcast(stmt);
    auto sl = std::make_shared<StmtList>();
    sl->stmts.push_back(stmt);
    for (const auto& dep_stmt_it : stmt->deps) {
      sl->deps.insert(*dep_stmt_it);
    }
    if (block && block->has_tag("eltwise")) {
      sl->eltwise = true;
      for (const auto& ref : block->refs) {
        if (!RefMatchIdx(ref, block.get())) {
          sl->eltwise = false;
        }
        if (IsReadDir(ref.dir)) {
          sl->inputs.insert(ref.from);
        }
        if (IsWriteDir(ref.dir)) {
          sl->outputs.insert(ref.from);
        }
      }
      if (sl->eltwise) {
        IVLOG(3, "Block: " << block->name);
      }
    } else {
      sl->eltwise = false;
    }
    stmt_list.push_back(sl);
  }

  while (stmt_list.size() > 0) {
    std::shared_ptr<StmtList> sl0 = stmt_list.front();
    stmt_list.erase(stmt_list.begin());
    bool fused = false;
    // Decide if sl0 can be fused by any following stmt
    for (auto it = stmt_list.begin(); it != stmt_list.end(); ++it) {
      auto& sl1 = *it;
      if (MayFuse(sl0, sl1)) {
        IVLOG(3, "Fuse: ");
        for (const auto& s : sl0->stmts) {
          auto block = Block::Downcast(s);
          IVLOG(3, "    " << block->name);
        }
        for (const auto& s : sl1->stmts) {
          auto block = Block::Downcast(s);
          IVLOG(3, "    " << block->name);
        }
        // merge stmts
        sl1->stmts.insert(sl1->stmts.begin(), sl0->stmts.begin(), sl0->stmts.end());
        // merge deps
        for (auto& s : sl0->stmts) {
          if (sl1->deps.find(s) != sl1->deps.end()) {
            sl1->deps.erase(s);
          }
        }
        // merge inputs and outputs
        sl1->deps.insert(sl0->deps.begin(), sl0->deps.end());
        for (auto& out : sl0->outputs) {
          if (sl1->inputs.find(out) != sl1->inputs.end()) {
            sl1->inputs.erase(out);
          }
        }
        sl1->inputs.insert(sl0->inputs.begin(), sl0->inputs.end());
        fused = true;
        break;
      }
      // If sl1 depends on sl0, sl0 can't be merged with the later blocks
      if (Depends(sl0, sl1)) {
        break;
      }
    }
    // If sl0 is not fused, put it into ready list
    if (!fused) {
      done.insert(done.end(), sl0->stmts.begin(), sl0->stmts.end());
    }
  }
  main_block->stmts = done;
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<ReorderBlocksPass, proto::ReorderBlocksPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
