// Copyright 2018, Intel Corporation

#include "tile/codegen/fuse.h"

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

#include <boost/format.hpp>

#include "base/util/stream_container.h"
#include "base/util/throw.h"
#include "tile/codegen/localize.h"
#include "tile/codegen/tile.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

bool NoInnerBlock(Block* block) {
  for (const auto& stmt : block->stmts) {
    if (stmt->kind() == StmtKind::Block) {
      return false;
    }
  }
  return true;
}

bool InnerConstraints(Block* block) {
  if (block->constraints.size() > 0) {
    return true;
  }
  for (const auto& stmt : block->stmts) {
    auto sub = Block::Downcast(stmt);
    if (sub && InnerConstraints(sub.get())) {
      return true;
    }
  }
  return false;
}

void AddInnerTags(Block* block, const Tags& tags) {
  for (auto stmt : block->stmts) {
    auto sub = Block::Downcast(stmt);
    if (sub) {
      sub->add_tags(tags);
    }
  }
}

static std::vector<Affine> TranslatedContraints(const AliasMap& map, std::map<std::string, std::string> remap,
                                                const Block& in) {
  std::vector<Affine> out;
  AliasMap inner = AliasMap(map, const_cast<Block*>(&in));
  std::map<std::string, stripe::Affine> remapped;
  for (const auto& kvp : inner.idx_sources()) {
    auto it = remap.find(kvp.first);
    if (it == remap.end()) {
      remapped.emplace(kvp);
    } else {
      remapped.emplace(kvp.first, Affine(it->second));
    }
  }
  for (const auto& aff : in.constraints) {
    IVLOG(4, "Remap = " << remap);
    IVLOG(4, "Translating " << aff << " to " << aff.sym_eval(remapped));
    out.push_back(aff.sym_eval(remapped));
  }
  std::sort(out.begin(), out.end());
  return out;
}

static Affine TranslateAffine(const Affine& src, const AliasMap& map, std::map<std::string, std::string> remap,
                              const Block* in) {
  Affine dest;
  AliasMap inner = AliasMap(map, const_cast<Block*>(in));
  std::map<std::string, stripe::Affine> remapped;
  for (const auto& kvp : inner.idx_sources()) {
    auto it = remap.find(kvp.first);
    if (it == remap.end()) {
      remapped.emplace(kvp);
    } else {
      remapped.emplace(kvp.first, Affine(it->second));
    }
  }
  return dest.sym_eval(remapped);
}

boost::optional<FusionPlan> ComputeFusionPlan(const AliasMap& scope, const Block& a, const Block& b,
                                              const std::string& buf_name) {
  IVLOG(3, "ComputeFusionPlan for " << buf_name << " between " << a.name << " and " << b.name);
  FusionPlan plan;
  plan.tile_a = TileShape(a.idxs.size(), 1);
  plan.a_interleave = false;
  plan.tile_b = TileShape(b.idxs.size(), 1);
  plan.b_interleave = false;
  // This is quite hueristic right now, but still beats our prior implementation
  auto it_a = a.ref_by_from(buf_name, false);
  if (it_a == a.refs.end()) {
    IVLOG(3, "ComputeFusionPlan: buffer name unknown in block a");
    return boost::none;
  }
  auto it_b = b.ref_by_from(buf_name, false);
  if (it_b == b.refs.end()) {
    IVLOG(3, "ComputeFusionPlan: buffer name unknown in block b");
    return boost::none;
  }
  assert(it_a->access.size() == it_b->access.size());
  for (size_t i = 0; i < it_a->access.size(); i++) {
    const Affine& poly_a = it_a->access[i];
    const Affine& poly_b = it_b->access[i];
    if (poly_a == 0 && poly_b == 0) {
      continue;
    }
    if (poly_a.getMap().size() != 1 || poly_a.getMap().begin()->first.empty()) {
      IVLOG(3, "ComputeFusionPlan: complex access in a: " << poly_a.toString());
      return boost::none;
    }
    if (poly_b.getMap().size() != 1 || poly_b.getMap().begin()->first.empty()) {
      IVLOG(3, "ComputeFusionPlan: complex access in b: " << poly_b.toString());
      return boost::none;
    }
    std::string idx_a = poly_a.getMap().begin()->first;
    std::string idx_b = poly_b.getMap().begin()->first;
    if (plan.remap_a.find(idx_a) != plan.remap_a.end()) {
      IVLOG(3, "ComputeFusionPlan: duplicate index");
      return boost::none;
    }
    int64_t mul_a = poly_a[idx_a];
    int64_t mul_b = poly_b[idx_b];
    if (mul_a % mul_b != 0) {
      IVLOG(3, "ComputeFusionPlan: uneven index division");
      return boost::none;
    }
    for (size_t i = 0; i < b.idxs.size(); i++) {
      if (b.idxs[i].name == idx_b) {
        int64_t mul = mul_a / mul_b;
        size_t a_range = a.idx_by_name(idx_a)->range;
        size_t b_range = b.idx_by_name(idx_b)->range;
        if (mul > 1) {
          plan.tile_b[i] = mul_a / mul_b;
        } else if (a_range != b_range) {
          if (b_range > a_range) {
            plan.tile_b[i] = b_range / a_range;
            plan.b_interleave = true;
          } else {
            IVLOG(3, "ComputeFusionPlan: b_range is less than a_range");
            return boost::none;
          }
        }
      }
    }
    plan.remap_a.emplace(idx_a, idx_a);
    plan.remap_b.emplace(idx_b, idx_a);
  }
  // Process the index that are not in the ref access
  for (size_t i = 0; i < b.idxs.size(); ++i) {
    const auto& idx = b.idxs[i];
    if (plan.remap_b.find(idx.name) == plan.remap_b.end()) {
      plan.tile_b[i] = idx.range;
    }
  }

  // Translate the constraints
  if (TranslatedContraints(scope, plan.remap_a, a) != TranslatedContraints(scope, plan.remap_b, b)) {
    IVLOG(3, "Remap a: " << plan.remap_a);
    IVLOG(3, "Remap b: " << plan.remap_b);
    IVLOG(3, "ComputeFusionPlan: incompatible constraints");
    IVLOG(4, "    a: " << StreamContainer(a.constraints));
    IVLOG(4, "    b: " << StreamContainer(b.constraints));
    return boost::none;
  }
  // Compute induced remappings
  for (const auto& idx_b : b.idxs) {
    if (idx_b.affine != Affine()) {
      for (const auto& idx_a : a.idxs) {
        if (idx_b.affine == idx_a.affine) {
          plan.remap_b.emplace(idx_b.name, idx_a.name);
        }
      }
    }
  }
  // Translate constraints
  for (const auto& constraint : b.constraints) {
    for (const auto& term : constraint.getMap()) {
      auto it = plan.remap_b.find(term.first);
      if (it == plan.remap_b.end()) {
        plan.remap_b.emplace(term.first, term.first);
      }
    }
  }
  return plan;
}

void FlattenTrivial(stripe::Block* outer) {
  IVLOG(4, "FlattenTrivial before:\n" << *outer);
  auto it = outer->stmts.begin();
  while (it != outer->stmts.end()) {
    auto inner = Block::Downcast(*it);
    // Skip non blocks
    if (!inner) {
      IVLOG(4, "FlattenTrivial: skip> non-block");
      ++it;
      continue;
    }
    uint64_t range = 1;
    for (const auto& idx : inner->idxs) {
      range *= idx.range;
    }
    if (range != 1) {
      IVLOG(4, "FlattenTrivial: skip> range != 1");
      ++it;
      continue;
    }
    bool renames = false;
    for (const auto& ref : inner->refs) {
      if (ref.from != "" && ref.into() != ref.from) {
        renames = true;
      }
    }
    // TODO: renames technically can be applied to inner statements,
    // but it's really annoying!
    if (renames) {
      IVLOG(4, "FlattenTrivial: skip> renames");
      ++it;
      continue;
    }
    // Move out inner statements
    for (auto& stmt : inner->stmts) {
      auto deep = Block::Downcast(stmt);
      if (deep) {
        // Rewrite any copied down indexes
        for (auto& idx : deep->idxs) {
          std::vector<std::string> names;
          for (const auto& item : idx.affine.getMap()) {
            if (item.first != "") {
              names.push_back(item.first);
            }
          }
          for (const auto& name : names) {
            idx.affine.substitute(name, inner->idx_by_name(name)->affine);
          }
        }
      }
      outer->stmts.insert(it, stmt);
    }
    auto it_old = it;
    ++it;
    outer->erase_stmt(it_old);
  }

  IVLOG(4, "FlattenTrivial after:\n" << *outer);
}

std::shared_ptr<Block> FusionRefactor(const stripe::Block& orig,                          //
                                      const std::map<std::string, std::string>& mapping,  //
                                      const TileShape& tile,                              //
                                      bool interleave,                                    //
                                      bool elide_trivial) {
  IVLOG(3, "FusionRefactor:\n" << orig);
  IVLOG(3, "mapping: " << StreamContainer(mapping) << ", tile: " << tile);
  // Possibly tile
  auto tiled = std::make_shared<Block>(orig);
  ApplyTile(tiled.get(), tile, elide_trivial, true, interleave);
  // IVLOG(3, "Tiled:\n" << *tiled);
  // Make empty inner and outer blocks, and put inner into outer
  auto outer = std::make_shared<Block>();
  outer->name = tiled->name;
  auto inner = std::make_shared<Block>();
  inner->name = tiled->name;
  outer->set_attrs(*tiled);
  outer->stmts.push_back(inner);
  // Put constraints on outer block and rewrite
  for (const auto& aff : tiled->constraints) {
    Affine out;
    for (const auto& kvp : aff.getMap()) {
      auto it = mapping.find(kvp.first);
      if (it != mapping.end()) {
        out += Affine(it->second, kvp.second);
      } else {
        out += Affine(kvp.first, kvp.second);
      }
    }
    outer->constraints.push_back(out);
  }
  // Move / rename each index to the appropriate block
  for (const auto& idx : tiled->idxs) {
    auto it = mapping.find(idx.name);
    if (it == mapping.end()) {
      if (idx.affine != Affine()) {
        IVLOG(3, "Affine idx: " << idx.name);
        outer->idxs.push_back(idx);
        inner->idxs.emplace_back(Index{idx.name, 1, idx.name});
      } else {
        IVLOG(3, "New idx: " << idx.name);
        inner->idxs.push_back(idx);
      }
    } else {
      IVLOG(3, "Existing idx: " << idx.name);
      inner->idxs.emplace_back(Index{idx.name, 1, it->second});
      outer->idxs.push_back(idx);
      outer->idxs.back().name = it->second;
    }
  }
  outer->location = tiled->location;
  // Sort outer indexes by names
  std::sort(outer->idxs.begin(), outer->idxs.end(), [](const Index& a, const Index& b) { return a.name < b.name; });
  // Copy statements to the inner block
  inner->stmts = tiled->stmts;
  // Copy refinements to both blocks
  outer->refs = tiled->refs;
  inner->refs = tiled->refs;
  // Rename mapped, and remove unmapped access elements from outer refinements
  // Also expand sizes base on inner indexes that have been removed.
  for (auto& ref : outer->refs) {
    for (size_t i = 0; i < ref.access.size(); i++) {
      auto& acc = ref.mut().access[i];
      int64_t min_val = 0;
      int64_t max_val = ref.interior_shape.dims[i].size - 1;
      Affine affine = acc.constant();
      for (const auto& kvp : acc.getMap()) {
        auto it = mapping.find(kvp.first);
        if (it == mapping.end()) {
          if (kvp.first != "") {
            if (kvp.second < 0) {
              min_val += (tiled->idx_by_name(kvp.first)->range - 1) * kvp.second;
            } else {
              max_val += (tiled->idx_by_name(kvp.first)->range - 1) * kvp.second;
            }
          }
          continue;
        }
        affine += Affine(it->second, kvp.second);
      }
      ref.mut().interior_shape.dims[i].size = max_val - min_val + 1;
      acc = affine;
    }
  }
  // Remove mapped access elements from inner refinements
  for (auto& ref : inner->refs) {
    // Rename from to match outer into
    ref.mut().from = ref.into();
    // If original was an allocation, make R/W.
    if (ref.dir == RefDir::None) {
      ref.mut().dir = RefDir::InOut;
    }
    // Update accesses
    for (auto& acc : ref.mut().access) {
      Affine affine;
      for (const auto& kvp : acc.getMap()) {
        if (kvp.first != "" && !mapping.count(kvp.first)) {
          affine += Affine(kvp.first, kvp.second);
        }
      }
      acc = affine;
    }
  }
  // Remove any trivial loops remaining
  FlattenTrivial(outer.get());
  // Return final result
  IVLOG(3, "Refactor output:\n" << *outer);
  return outer;
}

bool FuseBlocks(const AliasMap& scope, Block* block_a, Block* block_b) {
  // If indexes don't match, fail
  if (block_a->idxs != block_b->idxs) {
    IVLOG(3, "Fuse failed due to mismatched indexes");
    IVLOG(3, "A: " << block_a->idxs);
    IVLOG(3, "B: " << block_b->idxs);
    return false;
  }
  // If constraints don't match, fail
  if (block_a->constraints != block_b->constraints) {
    IVLOG(3, "Fuse failed due to mismatched constraints");
    return false;
  }
  // If locations don't match, fail
  if (block_a->location != block_b->location) {
    IVLOG(3, "Fuse failed due to mismatched locations");
    return false;
  }
  // Make AliasMaps for the two blocks
  AliasMap a_map(scope, block_a);
  AliasMap b_map(scope, block_b);
  // Start by copying A's reference across
  auto tmp = std::make_shared<Block>();
  tmp->refs = block_a->refs;
  // Walk over refinements in B and move them across
  // Rename duplicate refinements in B to their name in A
  // Otherwise make a new unique name (keeping original if possible)
  std::map<std::string, std::string> remap_b;
  for (const auto& new_ref : block_b->refs) {
    // If it's a local, always safe to copy if across
    // Check if b matches something in the existing block
    bool merged = false;
    for (auto& old_ref : block_a->refs) {
      auto atype = AliasInfo::Compare(a_map.at(old_ref.into()), b_map.at(new_ref.into()));
      if (atype == AliasType::Partial) {
        // Conflict, if either do any writing, we have a problem
        if (IsWriteDir(new_ref.dir) || IsWriteDir(old_ref.dir)) {
          IVLOG(3, "Fuse failed due to mismatched aliases: " << old_ref.into() << " vs " << new_ref.into());
          return false;  // Fuse will not work, bail
        }
      } else if (atype == AliasType::Exact) {
        remap_b[new_ref.into()] = old_ref.into();
        old_ref.mut().dir = UnionDir(old_ref.dir, new_ref.dir);
        merged = true;
        break;
      }
    }
    if (!merged) {
      // Copy across as a new ref
      std::string new_name = tmp->unique_ref_name(new_ref.into());
      remap_b[new_ref.into()] = new_name;
      tmp->refs.emplace(new_ref.WithInto(std::move(new_name)));
    }
  }
  // We are now safe (cannot fail), move new reference over A's
  std::swap(block_a->refs, tmp->refs);
  if (!block_a->name.empty()) {
    block_a->name = str(boost::format("%s+%s") % block_a->name % block_b->name);
  } else if (!block_b->name.empty()) {
    block_a->name = block_b->name;
  }
  // Load all the scalars that exist as of block A
  std::set<std::string> all_scalars;
  std::map<std::string, std::string> scalar_rename;
  for (const auto& stmt : block_a->stmts) {
    for (const auto& name : stmt->scalar_defs()) {
      all_scalars.emplace(name);
    }
  }
  auto def_scalar = [&](const std::string& orig) -> std::string {
    if (all_scalars.count(orig) == 0) {
      all_scalars.emplace(orig);
      scalar_rename[orig] = orig;
      return orig;
    }
    for (size_t i = 0; true; i++) {
      std::string with_suffix = orig + "_" + std::to_string(i);
      if (all_scalars.count(with_suffix) == 0) {
        all_scalars.emplace(with_suffix);
        scalar_rename[orig] = with_suffix;
        return with_suffix;
      }
    }
    return "";
  };
  // Now move across statements, updating references/scalars as we do:
  for (const auto& stmt : block_b->stmts) {
    switch (stmt->kind()) {
      case StmtKind::Load: {
        auto op = Load::Downcast(stmt);
        op->into = def_scalar(op->into);
        op->from = remap_b.at(op->from);
      } break;
      case StmtKind::Store: {
        auto op = Store::Downcast(stmt);
        op->into = remap_b.at(op->into);
        op->from = scalar_rename.at(op->from);
      } break;
      case StmtKind::LoadIndex: {
        // Currently we can't reach here actually because LoadIndex
        // takes only index which must not be the output of block_a.
        // If we get here in the future, the following code need
        // to be tested.
        auto op = LoadIndex::Downcast(stmt);
        op->into = def_scalar(op->into);
        op->from = TranslateAffine(op->from, scope, remap_b, block_b);
      }
      case StmtKind::Special: {
        auto op = Special::Downcast(stmt);
        for (auto& in : op->inputs) {
          in = remap_b.at(in);
        }
        for (auto& out : op->outputs) {
          out = remap_b.at(out);
        }
      } break;
      case StmtKind::Block: {
        auto op = Block::Downcast(stmt);
        for (auto& ref : op->refs) {
          ref.mut().from = remap_b.at(ref.from);
        }
      } break;
      case StmtKind::Constant: {
        auto op = Constant::Downcast(stmt);
        op->name = def_scalar(op->name);
      } break;
      case StmtKind::Intrinsic: {
        auto op = Intrinsic::Downcast(stmt);
        for (auto& in : op->inputs) {
          in = scalar_rename.at(in);
        }
        for (auto& out : op->outputs) {
          out = def_scalar(out);
        }
      } break;
    }
    block_a->stmts.push_back(stmt);
  }
  // All is well
  return true;
}

void FusionInner(const AliasMap& scope, Block* block, TagFusionStrategy* strategy, bool no_inner, bool no_constraints) {
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
      IVLOG(3, "Attempting fusion on block:\n" << block1->name);
      // Get the next statement
      auto it_next = it;
      it_next++;
      // If there is no next statement, I'm done with this block
      if (it_next == block->stmts.end()) {
        break;
      }
      // Convert to block
      auto block2 = Block::Downcast(*it_next);
      // If it's not a block, forget it
      if (!block2) {
        break;
      }
      if (block1->refs.size() + block2->refs.size() - 1 > strategy->Options().max_refs()) {
        // Too many refinements in a block for the particular platform
        break;
      }
      // Get the list of outputs for this block
      std::set<std::string> outs_for_fuse;
      // Do not use block1->ref_outs() because we need also InOut refs
      for (const auto& ro : block1->refs) {
        if (IsWriteDir(ro.dir)) {
          IVLOG(3, "Considering output: " << ro.from);
          outs_for_fuse.emplace(ro.from);
        }
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
      auto plan = ComputeFusionPlan(scope, *block1, *block2, fuse_on);
      if (!plan) {
        IVLOG(3, "Fusion plan failed");
        break;
      }
      // Now call the strategy to see if we should fuse
      if (!strategy->AttemptFuse(*block, *block1, *block2)) {
        IVLOG(3, "Fusion denied by strategy");
        break;
      }
      // Do the appropriate refactors
      auto refactor1 = FusionRefactor(*block1, plan->remap_a, plan->tile_a, plan->a_interleave);
      auto inner1 = refactor1->SubBlock(0, true);
      auto refactor2 = FusionRefactor(*block2, plan->remap_b, plan->tile_b, plan->b_interleave,
                                      (refactor1->idxs_product() > 1) || (inner1 == nullptr));
      if (no_inner) {
        // Check if there is any inner block in refactor1 and refactor2
        if (!NoInnerBlock(refactor1.get()) || !NoInnerBlock(refactor2.get())) {
          IVLOG(3, "Inner block exists");
          break;
        }
      }
      if (no_constraints) {
        if (InnerConstraints(refactor1.get()) || InnerConstraints(refactor2.get())) {
          IVLOG(3, "Generate constraints if fused.");
          break;
        }
      }
      if (refactor1->idxs_product() == 1 && refactor2->idxs_product() == 1) {
        auto inner2 = refactor2->SubBlock(0);
        if ((inner1 && inner2 == nullptr) || (inner1 == nullptr && inner2)) {
          IVLOG(3, "Cannot fuse inner blocks when the outer block is trivial.")
          break;
        }
        if (inner1 && inner2 && inner1->has_tag("eltwise") && inner2->has_tag("eltwise") &&
            inner1->idxs_product() != inner2->idxs_product()) {
          IVLOG(3, "Cannot fuse inner eltwise blocks when the outer block is trivial.")
          break;
        }
      }

      // Now copy computed indexes if it's safe
      for (const auto& idx : refactor1->idxs) {
        if (idx.affine != stripe::Affine() && !refactor2->idx_by_name(idx.name)) {
          refactor2->idxs.push_back(idx);
        }
      }
      for (const auto& idx : refactor2->idxs) {
        if (idx.affine != stripe::Affine() && !refactor1->idx_by_name(idx.name)) {
          refactor1->idxs.push_back(idx);
        }
      }
      // Sort one more time
      std::sort(refactor1->idxs.begin(), refactor1->idxs.end(),
                [](const Index& a, const Index& b) { return a.name < b.name; });
      std::sort(refactor2->idxs.begin(), refactor2->idxs.end(),
                [](const Index& a, const Index& b) { return a.name < b.name; });

      AddInnerTags(refactor1.get(), FromProto(strategy->Options().a_inner_set()));
      AddInnerTags(refactor2.get(), FromProto(strategy->Options().b_inner_set()));

      // IVLOG(3, "Fusion refactor 1:\n" << *refactor1);
      // IVLOG(3, "Fusion refactor 2:\n" << *refactor2);
      // Try the actual fusion
      if (!FuseBlocks(scope, refactor1.get(), refactor2.get())) {
        strategy->OnFailed();
        IVLOG(3, "Actual fusion failed");
        break;
      }
      IVLOG(3, "Fused block:\n" << *refactor1);
      // If it worked, update
      *it = refactor1;
      block->erase_stmt(it_next);
      strategy->OnFused(scope, refactor1.get(), *block1, *block2);
    }
    it++;
  }
}

static void FusionPassRecurse(const AliasMap& map, stripe::Block* block, TagFusionStrategy* strategy) {
  FusionInner(map, block, strategy, strategy->NoInner(), strategy->NoConstraints());
  for (const auto& stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      AliasMap inner_map(map, inner.get());
      FusionPassRecurse(inner_map, inner.get(), strategy);
    }
  }
}

void FusionPass::Apply(CompilerState* state) const {
  stripe::Block* root = state->entry();
  AliasMap base;
  AliasMap root_map(base, root);
  // Check if we should fuse this block
  TagFusionStrategy strategy(options_);
  FusionPassRecurse(root_map, root, &strategy);
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<FusionPass, proto::FusionPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
