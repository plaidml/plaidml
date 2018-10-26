// Copyright 2018, Intel Corp.

#include "tile/codegen/fuse.h"
#include "tile/codegen/localize.h"
#include "tile/codegen/tile.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

boost::optional<FusionPlan> ComputeFusionPlan(const Block& a, const Block& b, const std::string& buf_name) {
  FusionPlan plan;
  plan.tile_a = TileShape(a.idxs.size(), 1);
  plan.tile_b = TileShape(b.idxs.size(), 1);
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
    int64_t mul_a = poly_a[idx_a];
    int64_t mul_b = poly_b[idx_b];
    if (mul_a % mul_b != 0) {
      return boost::none;
    }
    for (size_t i = 0; i < b.idxs.size(); i++) {
      if (b.idxs[i].name == idx_b) {
        plan.tile_b[i] = mul_a / mul_b;
      }
    }
    plan.remap_a.emplace(idx_a, idx_a);
    plan.remap_b.emplace(idx_b, idx_a);
  }
  return plan;
}

void FlattenTrivial(stripe::Block* outer) {
  auto it = outer->stmts.begin();
  while (it != outer->stmts.end()) {
    // Skip non blocks
    if ((*it)->kind() != StmtKind::Block) {
      ++it;
      continue;
    }
    auto inner = Block::Downcast(*it);
    // Skip non-trival blocks
    if (inner->constraints.size()) {
      ++it;
      continue;
    }
    uint64_t range = 1;
    for (const auto& idx : inner->idxs) {
      range *= idx.range;
    }
    if (range != 1) {
      ++it;
      continue;
    }
    bool renames = false;
    for (const auto& ref : inner->refs) {
      if (ref.from != "" && ref.into != ref.from) {
        renames = true;
      }
    }
    // TODO: renames technically can be applied to inner statements,
    // but it's really annoying!
    if (renames) {
      ++it;
      continue;
    }
    // Move out inner statements
    for (auto& stmt : inner->stmts) {
      if (stmt->kind() == StmtKind::Block) {
        auto deep = Block::Downcast(stmt);
        // Rewrite any copied down indexes
        for (auto& idx : deep->idxs) {
          if (idx.from != "") {
            idx.factor *= inner->idx_by_name(idx.from)->factor;
            idx.from = inner->idx_by_name(idx.from)->from;
          }
        }
      }
      outer->stmts.insert(it, stmt);
    }
    auto it_old = it;
    ++it;
    outer->stmts.erase(it_old);
  }
}

std::shared_ptr<Block> FusionRefactor(const stripe::Block& orig,                          //
                                      const std::map<std::string, std::string>& mapping,  //
                                      const TileShape& tile,                              //
                                      const std::string& location) {
  // Possibly tile
  auto tiled = std::make_shared<Block>(orig);
  ApplyTile(tiled.get(), tile, "fusion_tile", location);
  // Make empty inner and outer blocks, and put inner into outer
  auto outer = std::make_shared<Block>();
  auto inner = std::make_shared<Block>();
  outer->stmts.push_back(inner);
  // Move / rename each index to the appropriate block
  for (const auto& idx : tiled->idxs) {
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
  inner->constraints = tiled->constraints;
  // Copy statements to the inner block
  inner->stmts = tiled->stmts;
  // Copy refinements to both blocks
  outer->refs = tiled->refs;
  inner->refs = tiled->refs;
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
            max_val += (tiled->idx_by_name(kvp.first)->range - 1) * kvp.second;
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
  // Remove any trivial loops remaining
  FlattenTrivial(outer.get());
  // Return final result
  return outer;
}

bool FuseBlocks(const AliasMap& scope, Block* a, Block* b) {
  // If indexes don't match, fail
  if (!(a->idxs == b->idxs)) {
    IVLOG(3, "Fuse failed dues to mismatched indexes");
    return false;
  }
  // If constraints don't match, fail
  if (!(a->constraints == b->constraints)) {
    IVLOG(3, "Fuse failed dues to mismatched constraints");
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
          IVLOG(3, "Fuse failed dues to mismatched aliases: " << old_ref.into << " vs " << new_ref.into);
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
  // Load all the scalars that exist as of block A
  std::set<std::string> all_scalars;
  std::map<std::string, std::string> smap_b;
  for (const auto& stmt : a->stmts) {
    for (const auto& name : stmt->scalar_defs()) {
      all_scalars.emplace(name);
    }
  }
  auto def_scalar = [&](const std::string& orig) -> std::string {
    if (all_scalars.count(orig) == 0) {
      all_scalars.emplace(orig);
      smap_b[orig] = orig;
      return orig;
    }
    for (size_t i = 0; true; i++) {
      std::string with_suffix = orig + "_" + std::to_string(i);
      if (all_scalars.count(with_suffix) == 0) {
        all_scalars.emplace(with_suffix);
        smap_b[orig] = with_suffix;
        return with_suffix;
      }
    }
    return "";
  };
  // Now move across statements, updating references/scalars as we do:
  for (const auto& stmt : b->stmts) {
    switch (stmt->kind()) {
      case StmtKind::Load: {
        auto op = Load::Downcast(stmt);
        op->into = def_scalar(op->into);
        op->from = remap_b.at(op->from);
      } break;
      case StmtKind::Store: {
        auto op = Store::Downcast(stmt);
        op->into = remap_b.at(op->into);
        op->from = smap_b.at(op->from);
      } break;
      case StmtKind::Special: {
        auto op = Special::Downcast(stmt);
        for (auto& s : op->inputs) {
          s = remap_b.at(s);
        }
        for (auto& s : op->outputs) {
          s = remap_b.at(s);
        }
      } break;
      case StmtKind::Block: {
        auto op = Block::Downcast(stmt);
        for (auto& ref : op->refs) {
          ref.from = remap_b.at(ref.from);
        }
      } break;
      case StmtKind::Constant: {
        auto op = Constant::Downcast(stmt);
        op->name = def_scalar(op->name);
      } break;
      case StmtKind::Intrinsic: {
        auto op = Intrinsic::Downcast(stmt);
        for (auto& in : op->inputs) {
          in = smap_b.at(in);
        }
        for (auto& out : op->outputs) {
          out = def_scalar(out);
        }
      } break;
    }
    a->stmts.push_back(stmt);
  }
  // All is well
  return true;
}

void FusionPass(const AliasMap& scope, Block* block, FusionStrategy* strategy) {
  // Start with the first statement, and keep tying to fuse until you can't anymore, then move to the next
  auto it = block->stmts.begin();
  while (it != block->stmts.end()) {
    // If it's not a block, forget it!
    if ((*it)->kind() != StmtKind::Block) {
      ++it;
      continue;
    }
    while (true) {
      // Get block everytime in case it's updated
      auto block1 = Block::Downcast(*it);
      IVLOG(3, "Attempting fusion on block:\n" << *block1);
      // Get the next statement
      auto it_next = it;
      it_next++;
      // If there is no next statement, I'm done with this block
      if (it_next == block->stmts.end()) {
        break;
      }
      // If it's not a block, forget it
      if ((*it_next)->kind() != StmtKind::Block) {
        break;
      }
      auto block2 = Block::Downcast(*it_next);
      // Get the list of outputs for this block
      std::set<std::string> outs_for_fuse;
      for (const auto& ro : block1->ref_outs()) {
        IVLOG(3, "Considering output: " << ro->from);
        outs_for_fuse.emplace(ro->from);
      }
      IVLOG(3, "Outs for fuse size: " << outs_for_fuse.size());
      std::string fuse_on = "";
      // Check if it's a match to any of the inputs on the next block
      for (const auto& ri : block2->ref_ins()) {
        IVLOG(3, "Considering input: " << ri->from);
        if (outs_for_fuse.count(ri->from)) {
          fuse_on = ri->from;
          break;
        }
      }
      // Nothing to fuse on, done with this block
      if (fuse_on == "") {
        IVLOG(3, "Nothing to fuse on");
        break;
      }
      IVLOG(3, "Fuse on = " << fuse_on);
      // Compute a fusion plan for the two blocks, if fails, give up
      auto plan = ComputeFusionPlan(*block1, *block2, fuse_on);
      if (!plan) {
        IVLOG(3, "Fusion plan failed");
        break;
      }
      // Now call the strategy to see if we should fuse
      if (!strategy->attempt_fuse(*block1, *block2)) {
        IVLOG(3, "Fusion denied by strategy");
        break;
      }
      // Do the appropriate refactors
      auto ref1 = FusionRefactor(*block1, plan->remap_a, plan->tile_a, "");
      auto ref2 = FusionRefactor(*block2, plan->remap_b, plan->tile_b, "");
      // IVLOG(3, "Fusion refactor 1:\n" << *ref1);
      // IVLOG(3, "Fusion refactor 2:\n" << *ref2);
      // Try the actual fusion
      if (!FuseBlocks(scope, ref1.get(), ref2.get())) {
        strategy->fusion_failed();
        IVLOG(3, "Actual fusion failed");
        break;
      }
      IVLOG(3, "Fused block:\n" << *ref1);
      // If it worked, update
      *it = ref1;
      block->stmts.erase(it_next);
      strategy->on_fused(scope, ref1.get());
    }
    it++;
  }
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
