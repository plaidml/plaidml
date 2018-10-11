// Copyright 2018, Intel Corp.

#include "tile/codegen/fuse.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

boost::optional<FusionPlan> ComputeFusionPlan(const Block& a, const Block& b, const std::string& buf_name) {
  FusionPlan plan;
  // This is quite hueristic right now, but still beats our prior implementation
  auto it_a = a.ref_by_from(buf_name);
  if (it_a == a.refs.end()) {
    return boost::none;
  }
  auto it_b = b.ref_by_from(buf_name);
  if (it_b == b.refs.end()) {
    return boost::none;
  }
  assert(it_a->access.size() == it_b->access.size());
  for (size_t i = 0; i < it_a->access.size(); i++) {
    const Affine& poly_a = it_a->access[i];
    const Affine& poly_b = it_b->access[i];
    if (poly_a.getMap().size() != 1 || poly_a.getMap().begin()->first.empty()) {
      return boost::none;
    }
    if (poly_b.getMap().size() != 1 || poly_b.getMap().begin()->first.empty()) {
      return boost::none;
    }
    std::string idx_a = poly_a.getMap().begin()->first;
    std::string idx_b = poly_b.getMap().begin()->first;
    if (plan.remap_a.find(idx_a) != plan.remap_a.end()) {
      return boost::none;
    }
    if (plan.remap_a.find(idx_a) != plan.remap_a.end()) {
      return boost::none;
    }
    plan.remap_a.emplace(idx_a, idx_a);
    plan.remap_b.emplace(idx_b, idx_a);
  }
  return plan;
}

std::shared_ptr<Block> FusionRefactor(const stripe::Block& orig, const std::map<std::string, std::string>& mapping) {
  // Make empty inner and outer blocks, and put inner into outer
  auto outer = std::make_shared<Block>();
  auto inner = std::make_shared<Block>();
  outer->stmts.push_back(inner);
  // Move / rename each index to the appropriate block
  for (const auto& idx : orig.idxs) {
    auto it = mapping.find(idx.name);
    if (it == mapping.end()) {
      inner->idxs.push_back(idx);
    } else {
      inner->idxs.emplace_back(idx.name, it->second, 1, 1);
      outer->idxs.push_back(idx);
      outer->idxs.back().name = it->second;
    }
  }
  // Sort outer indexes by names
  std::sort(outer->idxs.begin(), outer->idxs.end(), [](const Index& a, const Index& b) { return a.name < b.name; });
  // Copy constraints to inner block
  inner->constraints = orig.constraints;
  // Copy statements to the inner block
  inner->stmts = orig.stmts;
  // Copy refinements to both blocks
  outer->refs = orig.refs;
  inner->refs = orig.refs;
  // Rename mapped, and remove unmapped access elements from outer refinements
  // Also expand sizes base on inner indexes that have been removed.
  for (auto& ref : outer->refs) {
    for (size_t i = 0; i < ref.access.size(); i++) {
      auto& acc = ref.access[i];
      int64_t max_val = ref.shape.dims[i].size - 1;
      Affine r = acc.constant();
      for (const auto& kvp : acc.getMap()) {
        auto it = mapping.find(kvp.first);
        if (it == mapping.end()) {
          if (kvp.first != "") {
            if (kvp.second < 0) {
              throw std::runtime_error("FusionRefactor: Unable to handle negative strides");
            }
            max_val += (orig.idx_by_name(kvp.first)->range - 1) * kvp.second;
          }
          continue;
        }
        r += Affine(it->second, kvp.second);
      }
      ref.shape.dims[i].size = max_val + 1;
      acc = r;
    }
  }
  // Remove mapped access elements from inner refinements
  for (auto& ref : inner->refs) {
    for (auto& acc : ref.access) {
      Affine r;
      for (const auto& kvp : acc.getMap()) {
        if (kvp.first != "" && !mapping.count(kvp.first)) {
          r += Affine(kvp.first, kvp.second);
        }
      }
      acc = r;
    }
  }
  // Update (and copy) any inner blocks that
  return outer;
}

bool FuseBlocks(const AliasMap& scope, Block* a, Block* b) {
  // If indexes don't match, fail
  if (!(a->idxs == b->idxs)) {
    IVLOG(2, "Fuse failed dues to mismatched indexes");
    return false;
  }
  // If constraints don't match, fail
  if (!(a->constraints == b->constraints)) {
    IVLOG(2, "Fuse failed dues to mismatched constraints");
    return true;
  }
  // Make AliasMaps for the two blocks
  AliasMap a_map(scope, *a);
  AliasMap b_map(scope, *b);
  // Start by copying A's reference across
  auto r = std::make_shared<Block>();
  r->refs = a->refs;
  // Walk over refinements in B and move them across
  // Rename duplicate refinements in B to their name in A
  // Otherwise make a new unique name (keeping original if possible)
  std::map<std::string, std::string> remap_b;
  for (const auto& new_ref : b->refs) {
    // If it's a local, always safe to copy if across
    // Check if b matches something in the existing block
    bool merged = false;
    for (auto& old_ref : a->refs) {
      auto atype = AliasInfo::Compare(a_map.at(old_ref.into), b_map.at(new_ref.into));
      if (atype == AliasType::Partial) {
        // Conflict, if either do any writing, we have a problem
        if (IsWriteDir(new_ref.dir) || IsWriteDir(old_ref.dir)) {
          IVLOG(2, "Fuse failed dues to mismatched aliases: " << old_ref.into << " vs " << new_ref.into);
          return false;  // Fuse will not work, bail
        }
      } else if (atype == AliasType::Exact) {
        remap_b[new_ref.into] = old_ref.into;
        old_ref.dir = UnionDir(old_ref.dir, new_ref.dir);
        merged = true;
        break;
      }
    }
    if (!merged) {
      // Copy across as a new ref
      std::string new_name = r->unique_ref_name(new_ref.into);
      remap_b[new_ref.into] = new_name;
      r->refs.push_back(new_ref);
      r->refs.back().into = new_name;
    }
  }
  // We are now safe (cannot fail), move new reference over A's
  std::swap(a->refs, r->refs);
  // Now move across statements, updating references as we do:
  for (const auto& stmt : b->stmts) {
    switch (stmt->kind()) {
      case StmtKind::Load: {
        auto lstmt = Load::Downcast(stmt);
        lstmt->from = remap_b.at(lstmt->from);
      } break;
      case StmtKind::Store: {
        auto sstmt = Store::Downcast(stmt);
        sstmt->into = remap_b.at(sstmt->into);
      } break;
      case StmtKind::Special: {
        auto sstmt = Special::Downcast(stmt);
        for (auto& s : sstmt->inputs) {
          s = remap_b.at(s);
        }
        for (auto& s : sstmt->outputs) {
          s = remap_b.at(s);
        }
      } break;
      case StmtKind::Block: {
        auto bstmt = Block::Downcast(stmt);
        for (auto& ref : bstmt->refs) {
          ref.from = remap_b.at(ref.from);
        }
      } break;
      case StmtKind::Constant:
      case StmtKind::Intrinsic:
        break;
    }
    a->stmts.push_back(stmt);
  }
  // All is well
  return true;
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
