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

struct StmtList;
typedef std::shared_ptr<Statement> StmtPtr;
typedef std::shared_ptr<StmtList> StmtListPtr;

struct StmtList {
  std::vector<StmtPtr> stmts;                                     // Statements in order
  std::unordered_map<std::string, std::vector<size_t>> inputs;    // input refinements
  std::unordered_map<std::string, std::vector<size_t>> outputs;   // output refinements;
  std::unordered_set<StmtListPtr> deps;                           // The stmts that this list directly depends on
  std::unordered_set<StmtListPtr> rdeps;                          // The reverse of deps
  bool in_eltwise;                                                // Whether the first stmt is element-wise
  bool out_eltwise;                                               // Whether the last stmt is element-wise
  bool skip;                                                      // Skip the removed stmts
  bool defer;                                                     // Whether defer processing the StmtList
};

// Whether sl1 directly depends on sl0
bool Depends(StmtListPtr sl0, StmtListPtr sl1) {
  return sl1->deps.find(sl0) != sl1->deps.end();
}

// Return true if sl1 does not have any dependency or depends on only sl0
bool DependsOnly(StmtListPtr sl0, StmtListPtr sl1) {
  if (sl1->deps.empty()) {
    return true;
  }
  if (sl1->deps.size() > 1) {
    return false;
  }
  return sl1->deps.find(sl0) != sl1->deps.end();
}

std::vector<size_t> MakeRefShape(Block* block, const Refinement& ref) {
  std::vector<size_t> shape;
  for (const auto& acc : ref.access) {
    const auto& acc_map = acc.getMap();
    if (acc_map.size() > 1) {
      return {};
    }
    if (acc_map.size() == 0) {
      shape.push_back(1);
      continue;
    }
    auto& idx_name = acc_map.begin()->first;
    if (idx_name == "") {
      shape.push_back(1);
    }
    else {
      auto idx = block->idx_by_name(idx_name);
      shape.push_back((idx->affine == Affine()) ? idx->range : 1);
    }
  }
  return shape;
}

bool MayFuse(StmtListPtr s0, StmtListPtr s1) {
  if (!s1->in_eltwise) {
    return false;
  }
  bool same_shape = false;
  // s1->in_eltwise must be true here
  if (s0->out_eltwise) {
    // Check if the index are same
    auto b0 = Block::Downcast(s0->stmts.back());
    auto b1 = Block::Downcast(s1->stmts.front());
    if (b0 && b1) {
      same_shape = b0->sorted_idx_ranges() == b1->sorted_idx_ranges();
      if (!same_shape) {
        // We can't fuse eltwise blocks with different shapes
        return false;
      }
    }
  }
  // Check if there is any common refinement
  for (const auto& out_it : s0->outputs) {
    const auto& in_it = s1->inputs.find(out_it.first);
    if (in_it != s1->inputs.end()) {
      // Their shape should be same
      if (out_it.second == in_it->second) {
        return true;
      }
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

  // The stmts in done list
  std::unordered_set<StmtPtr> done_stmts;
  // The processing stmt list
  std::vector<std::shared_ptr<StmtList>> processing;
  // final stmt list
  StatementList done;
  // Map the statement to the statement list
  std::unordered_map<StmtPtr, StmtListPtr> stmt2list;

  // Initialize the statement lists
  for (const auto& stmt : main_block->stmts) {
    if (ZeroBlock(stmt)) {
      done.push_back(stmt);
      done_stmts.insert(stmt);
      continue;
    }
    auto block = Block::Downcast(stmt);
    auto sl = std::make_shared<StmtList>();
    sl->stmts.push_back(stmt);
    stmt2list.emplace(stmt, sl);
    sl->skip = false;
    sl->defer = false;
    sl->in_eltwise = false;
    sl->out_eltwise = false;
    if (block) {
      if (block->has_tag("eltwise")) {
        sl->in_eltwise = true;
        sl->out_eltwise = true;
        for (const auto& ref : block->refs) {
          std::vector<size_t> rs = MakeRefShape(block.get(), ref);
          if (IsReadDir(ref.dir)) {
            sl->inputs.emplace(ref.from, rs);
          }
          if (IsWriteDir(ref.dir)) {
            sl->outputs.emplace(ref.from, rs);
          }
        }
      } else if (block->has_tag("contraction")) {
        for (const auto& ref : block->refs) {
          if (IsWriteDir(ref.dir)) {
            std::vector<size_t> rs = MakeRefShape(block.get(), ref);
            sl->outputs.emplace(ref.from, rs);
          }
        }
      }
    }
    processing.push_back(sl);
  }

  // Set the direct dependencies
  for (auto& sl0 : processing) {
    // Now there is only one statement in sl
    auto& stmt = sl0->stmts[0];
    for (const auto& dep_stmt_it : stmt->deps) {
      if (done_stmts.find(*dep_stmt_it) == done_stmts.end()) {
        // If the dep stmt is not in the done set, it is a real dependency
        auto sl1 = stmt2list[*dep_stmt_it];
        sl0->deps.insert(sl1);
        sl1->rdeps.insert(sl0);
      }
    }
  }

  // Topological sort for processing on deps field
  while (!processing.empty()) {
    auto it = processing.begin();
    auto first_defer = processing.end();
    // Find a valid (skip is false) StmtList without dependencies
    while (it != processing.end()) {
      if ((*it)->skip) {
        // Break the loop and erase the skip StmtList
        break;
      }
      if ((*it)->deps.empty()) {
        if (!(*it)->defer) {
          // If it is not deferred, process it at once
          break;
        }
        // Otherwise, save the first deferred StmtList
        if (first_defer == processing.end()) {
          first_defer = it;
        }
      }
      ++it;
    }
    if (it == processing.end()) {
      if (first_defer == processing.end()) {
        // If there is not any StmtList without dependency
        throw std::runtime_error("Statement lists with dependencies depend on each other.");
      }
      // There is only the deferred StmtList without dependency.
      // We have to process it then.
      it = first_defer;
    }
    auto sl0 = *it;
    processing.erase(it);
    if (sl0->skip) {
      continue;
    }

    // First StmtList that may be fused into sl0
    auto first_may_fuse = processing.end();
    // First dependency of sl0
    auto first_dep = processing.end();
    // Find the StmtLists that may be fused after sl0
    for (auto sl1_it = processing.begin(); sl1_it != processing.end(); ++sl1_it) {
      auto& sl1 = *sl1_it;
      if (sl1->skip) {
        continue;
      }
      if (MayFuse(sl0, sl1)) {
        if (DependsOnly(sl0, sl1)) {
          sl1->skip = true;
          // merge stmts
          sl0->stmts.insert(sl0->stmts.end(), sl1->stmts.begin(), sl1->stmts.end());
          // merge dependencies
          if (Depends(sl0, sl1)) {
            sl0->rdeps.erase(sl1);
          }
          for (auto& sl2 : sl1->rdeps) {
            sl2->deps.erase(sl1);
            sl2->deps.insert(sl0);
            sl0->rdeps.insert(sl2);
          }
          // merge inputs and outputs
          sl0->inputs.insert(sl1->inputs.begin(), sl1->inputs.end());
          sl0->outputs.insert(sl1->outputs.begin(), sl1->outputs.end());
          sl0->out_eltwise = sl1->out_eltwise;
          sl0->defer = false;
        }
        else {
          // If we can't fuse sl1 immediately, save "it could be fused" for later use
          if (first_may_fuse == processing.end()) {
            first_may_fuse = sl1_it;
          }
        }
      }
      else {
        if (first_dep == processing.end() && Depends(sl0, *sl1_it)) {
          first_dep = sl1_it;
        }
      }
    }

    // Whether sl0 can be done
    bool sl0_done = true;
    if (first_may_fuse != processing.end()) {
      // sl1 depends on something that has not been done.
      // We put sl0 before sl1 as close as possible now
      if (first_dep == processing.end() || first_may_fuse <= first_dep) {
        // Nothing depends on sl0 so far, fuse sl0 and sl1
        auto sl1 = *first_may_fuse;
        // merge statements
        sl0->stmts.insert(sl0->stmts.end(), sl1->stmts.begin(), sl1->stmts.end());
        // merge dependencies and reverse dependencies
        for (auto& sl2 : sl1->deps) {
          sl2->rdeps.erase(sl1);
          if (sl2 != sl0) {
            sl2->rdeps.insert(sl0);
            sl0->deps.insert(sl2);
          }
        }
        for (auto& sl2 : sl1->rdeps) {
          sl2->deps.erase(sl1);
          sl2->deps.insert(sl0);
          sl0->rdeps.insert(sl2);
        }
        // merge inputs and outputs
        sl0->inputs.insert(sl1->inputs.begin(), sl1->inputs.end());
        sl0->outputs.insert(sl1->outputs.begin(), sl1->outputs.end());
        sl0->out_eltwise = sl1->out_eltwise;
        // Reuse the slot in processing
        *first_may_fuse = sl0;
        // Keep it to see if it can fuse more in the future
        sl0_done = false;
      }
      else {
        if (!sl0->deps.empty() || !sl0->defer) {
          // We have to stop here and insert sl0 before the first dependency of sl0
          sl0->defer = true;
          sl0_done = false;
          processing.insert(first_dep, sl0);
        }
        // else: If sl0 is already deferred, make it done
      }
    }
    if (sl0_done && sl0->deps.empty()) {
      // Remove deps in the remaining stmt lists
      for (auto& sl1 : sl0->rdeps) {
        sl1->deps.erase(sl0);
      } 
      done.insert(done.end(), sl0->stmts.begin(), sl0->stmts.end());
    }
  }
  // Replace the stmts in the main block
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
