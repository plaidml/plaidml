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
  std::vector<StmtPtr> stmts;                          // Statements in order
  std::map<std::string, std::vector<size_t>> inputs;   // input refinements
  std::map<std::string, std::vector<size_t>> outputs;  // output refinements;
  std::set<StmtPtr> direct_deps;                       // The stmts that this list directly depends on
  std::set<StmtPtr> transitive_deps;                   // The stmts that this list transitively depends on
  bool in_eltwise;                                     // Whether the first stmt is element-wise
  bool out_eltwise;                                    // Whether the last stmt is element-wise
  bool skip;                                           // Skip the removed stmts
};

// Determine if any of deps depends on any of stmts
static bool Depends(const std::vector<StmtPtr>& stmts, const std::set<StmtPtr>& deps) {
  for (const auto& stmt : stmts) {
    if (deps.find(stmt) != deps.end()) {
      return true;
    }
  }
  return false;
}

static std::vector<size_t> MakeRefShape(Block* block, const Refinement& ref) {
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

static bool MayFuse(std::shared_ptr<StmtList> s0, std::shared_ptr<StmtList> s1) {
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
  std::set<StmtPtr> done_stmts;
  // Initial stmt list with direct dependencies
  std::vector<std::shared_ptr<StmtList>> direct_list;
  // stmt list with transitive dependencies
  std::vector<std::shared_ptr<StmtList>> transitive_list;
  // stmt list after merging
  std::vector<std::shared_ptr<StmtList>> ready_list;
  // final stmt list
  StatementList done;

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
    for (const auto& dep_stmt_it : stmt->deps) {
      if (done_stmts.find(*dep_stmt_it) == done_stmts.end()) {
        sl->direct_deps.insert(*dep_stmt_it);
      }
    }
    sl->skip = false;
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
    direct_list.push_back(sl);
  }

  // Make transitive dependencies for stmt_list
  while (!direct_list.empty()) {
    auto it = direct_list.begin();
    // Find a stmt list without dependency
    while (it != direct_list.end() && !(*it)->direct_deps.empty()) {
      ++it;
    }
    if (it == direct_list.end()) {
      throw std::runtime_error("Statement lists with direct dependencies depend on each other.");
    }
    auto& sl0 = *it;
    // Pass its transitive dependencies to its successors
    for (auto& sl1 : direct_list) {
      if (sl0 != sl1 && Depends(sl0->stmts, sl1->direct_deps)) {
        sl1->transitive_deps.insert(sl0->stmts.begin(), sl0->stmts.end());
        sl1->transitive_deps.insert(sl0->transitive_deps.begin(), sl0->transitive_deps.end());
        for (const auto& stmt : sl0->stmts) {
          if (sl1->direct_deps.find(stmt) != sl1->direct_deps.end()) {
            sl1->direct_deps.erase(stmt);
          }
        }
      }
    }
    transitive_list.push_back(*it);
    direct_list.erase(it);
  }
  // direct_deps are no longer valid after this point

  // Topological sort for transitive_list on transitive_deps field
  while (!transitive_list.empty()) {
    auto it = transitive_list.begin();
    // Find a valid (skip is false) StmtList without transitive dependencies
    while (it != transitive_list.end() && !(*it)->transitive_deps.empty() && !(*it)->skip) {
      ++it;
    }
    if (it == transitive_list.end()) {
      throw std::runtime_error("Statement lists with transitive dependencies depend on each other.");
    }
    auto sl0 = *it;
    transitive_list.erase(it);
    if (sl0->skip) {
      continue;
    }
    // Find the StmtLists that may be fused after sl0
    for (auto sl1_it = transitive_list.begin(); sl1_it != transitive_list.end(); ++sl1_it) {
      auto& sl1 = *sl1_it;
      if (sl1->skip) {
        continue;
      }
      if (MayFuse(sl0, sl1)) {
        // Make sure there is no any block depends on sl0, which sl1 depends on.
        bool can_merge = true;
        for (auto sl2_it = transitive_list.begin(); sl2_it != transitive_list.end(); ++sl2_it) {
          auto& sl2 = *sl2_it;
          if ((!sl2->skip) && Depends(sl0->stmts, sl2->transitive_deps)
              && Depends(sl2->stmts, sl1->transitive_deps)) {
            can_merge = false;
            break;
          }
        }
        if (can_merge) {
          sl1->skip = true;
          // merge stmts
          sl0->stmts.insert(sl0->stmts.end(), sl1->stmts.begin(), sl1->stmts.end());
          // merge deps
          for (auto& s : sl0->stmts) {
            if (sl1->transitive_deps.find(s) != sl1->transitive_deps.end()) {
              sl1->transitive_deps.erase(s);
            }
          }
          sl0->transitive_deps.insert(sl1->transitive_deps.begin(), sl1->transitive_deps.end());
          // merge inputs and outputs
          for (auto& out_it : sl0->outputs) {
            if (sl1->inputs.find(out_it.first) != sl1->inputs.end()) {
              sl1->inputs.erase(out_it.first);
            }
          }
          sl0->inputs.insert(sl1->inputs.begin(), sl1->inputs.end());
          sl0->outputs.insert(sl1->outputs.begin(), sl1->outputs.end());
          sl0->out_eltwise = sl1->out_eltwise;
        }
      }
    }
    if (sl0->transitive_deps.empty()) {
      // Remove deps in the remaining stmt lists
      for (auto& sl1 : transitive_list) {
        if (sl0 != sl1) {
          for (const auto& stmt : sl0->stmts) {
            if (sl1->transitive_deps.find(stmt) != sl1->transitive_deps.end()) {
              sl1->transitive_deps.erase(stmt);
            }
          }
        }
      }
      done.insert(done.end(), sl0->stmts.begin(), sl0->stmts.end());
    }
    else {
      transitive_list.insert(transitive_list.begin(), sl0);
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
