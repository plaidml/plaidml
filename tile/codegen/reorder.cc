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
  std::set<StmtPtr> deps;                              // The stmts that this list depends on
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
  for (const auto& out_it : s0->outputs) {
    const auto& in_it = s1->inputs.find(out_it.first);
    if (in_it != s1->inputs.end()) {
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
    sl->skip = false;
    if (block) {
      if (block->has_tag("eltwise")) {
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
    stmt_list.push_back(sl);
  }
  while (stmt_list.size() > 0) {
    std::shared_ptr<StmtList> sl0 = stmt_list.front();
    stmt_list.erase(stmt_list.begin());
    if (sl0->skip) {
      continue;
    }
    // Whether sl0 is merged forward into the following StmtList
    bool merged_forward = false;
    // Whether any stmt is merged backward to sl0
    bool merged_backward = false;
    // Whether it is currently forward merge
    bool merging_forward = true;
    // the stmts between sl0 and sl1 while backward merge
    std::vector<StmtPtr> between_stmts;
    // Insertion point if backward merge
    std::vector<std::shared_ptr<StmtList>>::iterator insert_pos;
    // Decide if sl0 can be fused by any following stmt
    for (auto it = stmt_list.begin(); it != stmt_list.end(); ++it) {
      auto& sl1 = *it;
      if (sl1->skip) {
        continue;
      }
      if (!merging_forward) {
        // If it is backward merge, sl1 must not depend on the above stmts
        if (Depends(between_stmts, sl1->deps)) {
          between_stmts.insert(between_stmts.end(), sl1->stmts.begin(), sl1->stmts.end());
          continue;
        }
      }
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
        if (merging_forward) {
          // Forward merge sl0 into sl1
          // merge stmts
          sl1->stmts.insert(sl1->stmts.begin(), sl0->stmts.begin(), sl0->stmts.end());
          // merge deps
          for (auto& s : sl0->stmts) {
            if (sl1->deps.find(s) != sl1->deps.end()) {
              sl1->deps.erase(s);
            }
          }
          sl1->deps.insert(sl0->deps.begin(), sl0->deps.end());
          // merge inputs and outputs
          for (auto& out_it : sl0->outputs) {
            if (sl1->inputs.find(out_it.first) != sl1->inputs.end()) {
              sl1->inputs.erase(out_it.first);
            }
          }
          sl1->inputs.insert(sl0->inputs.begin(), sl0->inputs.end());
          sl1->outputs.insert(sl0->outputs.begin(), sl0->outputs.end());
          merged_forward = true;
          break;
        }
        else {
          // Backward merge sl1 into sl0
          sl1->skip = true;
          // merge stmts
          sl0->stmts.insert(sl0->stmts.end(), sl1->stmts.begin(), sl1->stmts.end());
          // merge deps
          for (auto& s : sl0->stmts) {
            if (sl1->deps.find(s) != sl1->deps.end()) {
              sl1->deps.erase(s);
            }
          }
          sl1->deps.insert(sl0->deps.begin(), sl0->deps.end());
          // merge inputs and outputs
          for (auto& out_it : sl0->outputs) {
            if (sl1->inputs.find(out_it.first) != sl1->inputs.end()) {
              sl1->inputs.erase(out_it.first);
            }
          }
          sl0->inputs.insert(sl1->inputs.begin(), sl1->inputs.end());
          sl0->outputs.insert(sl1->outputs.begin(), sl1->outputs.end());
          merged_backward = true;
        }
      }
      // If sl1 depends on sl0, sl0 can't be merged with the later blocks
      if (merging_forward) {
        if (Depends(sl0->stmts, sl1->deps)) {
          // Start backward merge, insert sl0 before sl1 later
          merging_forward = false;
          insert_pos = it;
        }
      }
      // Do not use "else" here. It should be executed if the above block
      // sets merging_forward to false
      if (!merging_forward) {
        between_stmts.insert(between_stmts.end(), sl1->stmts.begin(), sl1->stmts.end());
      }
    }
    if (!merged_forward) {
      if (merged_backward) {
        // Something backward is merged into sl0. So insert sl0 into stmt_list.
        stmt_list.insert(insert_pos, sl0);
      }
      else {
        // Put sl0 into ready list
        done.insert(done.end(), sl0->stmts.begin(), sl0->stmts.end());
      }
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
