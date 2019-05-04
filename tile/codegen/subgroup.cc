// Copyright 2019, Intel Corp.

#include "tile/codegen/subgroup.h"

#include "tile/codegen/tile.h"

namespace vertexai {
namespace tile {
namespace codegen {

struct SubgroupPlan {
  size_t subgroup_size;
  std::vector<std::string> idxs;
  std::map<std::string, size_t> subgroup_tile;
  std::map<std::string, size_t> extra_tile;
  std::map<std::string, std::string> ref_idx;
  std::string thread_idx;
};

class SubgroupCostModel {

public:
  SubgroupCostModel(stripe::Block *block, const SubgroupPlan &plan,
                    const proto::SubgroupPass &options)
      : block_(block), plan_(plan), options_(options) {}

  void ComputeCost() {
    // Compute total tile size for each index
    std::map<std::string, size_t> tot_tile;
    for (const auto& idx : plan_.idxs) {
      tot_tile[idx] = plan_.subgroup_tile[idx] * plan_.extra_tile[idx];
    }
    // Compute the amount of memory transfer for each refinement
    size_t tot_mem = 0;
    double tot_mem_io = 0;
    plan_.ref_idx.clear();
    plan_.thread_idx = "";
    for (const auto& ref : block_->refs) {
      TensorShape tile_shape = ref.ApplyTile(tot_tile);
      // Get overall memory cost
      int subgroup = 0;
      size_t mem = tile_shape.sizes_product_bytes();
      double cache_miss = tile_shape.memory_io(options_.cache_width());
      double cache_hit = mem - cache_miss;
      double mem_io = cache_miss * options_.mem_latency() + cache_hit * options_.cache_latency();
      // Figure out if we have a single unique stride 1 subgroup block
      for (size_t i = 0; i < ref.access.size(); i++) {
        if (ref.interior_shape.dims[i].stride != 1) {
          continue;
        }
        for (const auto& kvp : ref.access[i].getMap()) {
          if (kvp.first != "" && kvp.second == 1 &&
              plan_.subgroup_tile.at(kvp.first) == static_cast<size_t>(plan_.subgroup_size)) {
            subgroup++;
            plan_.ref_idx[ref.into()] = kvp.first;
          }
        }
      }
      if (subgroup > 1) {
        subgroup = 0;
        plan_.ref_idx.erase(ref.into());
      }
      // If it works, set thread_idx, otherwise increase memory
      if (subgroup == 1) {
        if (ref.dir == stripe::RefDir::Out) {
          plan_.thread_idx = plan_.ref_idx[ref.into()];
        }
      } else {
        mem *= plan_.subgroup_size;
        mem_io *= plan_.subgroup_size;
      }
      // More than one subgroup index per refinment = madness
      if (subgroup > 1) {
        return;
      }
      tot_mem += mem;
      tot_mem_io += mem_io;
    }
    // Fail if we don't actually do subgrouping on output
    if (plan_.thread_idx == "") {
      return;
    }
    // Fail if any input uses thead_idx and isn't subgrouped on it
    for (const auto& ref : block_->refs) {
      if (plan_.ref_idx.count(ref.into()) && plan_.ref_idx.at(ref.into()) == plan_.thread_idx) {
        continue;  // Access to therad_idx is safe when it's the subgroup index
      }
      if (ref.FlatAccess()[plan_.thread_idx] != 0) {
        // Otherwise, use of thread_idx is disallowed!
        return;
      }
    }
    // Fail if we go over memory
    if (tot_mem > static_cast<size_t>(options_.max_mem())) {
      IVLOG(2, "subgroup: " << plan_.subgroup_tile << ", extra: " << plan_.extra_tile << ", mem: " << tot_mem
                            << ", mem_io: " << tot_mem_io << ", cost: INF");
      return;
    }
    // Compute compute
    size_t tot_ops = 1;
    for (const auto& kvp : tot_tile) {
      tot_ops *= kvp.second;
    }
    // Cost = mem / compute
    IVLOG(2, "subgroup: " << plan_.subgroup_tile << ", extra: " << plan_.extra_tile << ", mem: " << tot_mem
                          << ", mem_io: " << tot_mem_io << ", cost: " << tot_mem_io / static_cast<double>(tot_ops));
    double cost = tot_mem_io / static_cast<double>(tot_ops);
    if (cost < best_cost_) {
      best_cost_ = cost;
      best_plan_ = plan_;
    }
  }

  void BestCostRecursive(size_t idx_num) {
    if (idx_num == plan_.idxs.size()) {
      ComputeCost();
      return;
    }
    std::string idx = plan_.idxs[idx_num];
    // Try using a subgroup or not for each index
    for (size_t do_subgroup = 0; do_subgroup < 2; do_subgroup++) {
      // Get the range of this index
      size_t range = block_->idx_by_name(idx)->range;
      if (range % plan_.subgroup_size != 0 && do_subgroup == 1) {
        continue; // Skip the subgroup if not even divison by subgroup size
      }
      if (do_subgroup) {
        // If we are doing subgrouping, set + reduce range
        plan_.subgroup_tile[idx] = plan_.subgroup_size;
        range /= plan_.subgroup_size;
      } else {
        plan_.subgroup_tile[idx] = 1;
      }
      // Now try all even divisors for remaining size
      for (size_t ts = 1; ts <= range; ts++) {
        if (range % ts != 0) {
          continue;
        }
        plan_.extra_tile[idx] = ts;
        BestCostRecursive(idx_num + 1);
      }
    }
    return;
  }

  double BestCost(SubgroupPlan *best_plan) {
    best_cost_ = std::numeric_limits<double>::infinity();
    int subgroup_size = options_.min_subgroup_size();
    while (subgroup_size <= options_.max_subgroup_size()) {
      plan_.subgroup_size = subgroup_size;
      BestCostRecursive(0);
      subgroup_size <<= 1;
    }
    *best_plan = best_plan_;
    return best_cost_;
  }

private:
  stripe::Block *block_;
  SubgroupPlan plan_;
  double best_cost_;
  SubgroupPlan best_plan_;
  proto::SubgroupPass options_;
};

void Subgroup(stripe::Block *block, const AliasMap &map,
              const proto::SubgroupPass &options) {
  if (block->constraints.size()) {
    IVLOG(1, "Failed due to constraints");
    return;
  }
  if (block->ref_outs().size() != 1) {
    IVLOG(1, "Giving up due to outputs == 1");
    return;
  }
  // Setup an empty plan
  SubgroupPlan plan;
  for (const auto &idx : block->idxs) {
    if (idx.affine == stripe::Affine()) {
      plan.idxs.push_back(idx.name);
    } else {
      IVLOG(1, "Failed due to passthrus");
      return; // Right now we don't handle this case
    }
  }
  // Compute optimal plan
  SubgroupCostModel model(block, plan, options);
  double cost = model.BestCost(&plan);
  IVLOG(2, "Cost = " << cost);
  // If it's not valid, forget it
  if (std::isinf(cost)) {
    return;
  }
  // Tell everyone about it
  IVLOG(2, *block);
  IVLOG(2, plan.subgroup_size);
  IVLOG(2, plan.subgroup_tile);
  IVLOG(2, plan.extra_tile);
  IVLOG(2, plan.ref_idx);
  IVLOG(2, plan.thread_idx);

  // Compute some per-index data
  std::map<std::string, size_t> inner_tile;
  std::map<std::string, stripe::Affine> replace;
  std::vector<stripe::Index> inner_idxs;
  for (const auto &idx : plan.idxs) {
    inner_tile[idx] = plan.subgroup_tile[idx] * plan.extra_tile[idx];
    inner_idxs.emplace_back(idx + "_e", plan.extra_tile[idx]);
    if (plan.subgroup_tile[idx] == size_t(plan.subgroup_size)) {
      inner_idxs.emplace_back(idx + "_i", plan.subgroup_tile[idx]);
      replace[idx] = stripe::Affine(idx + "_e") * plan.subgroup_tile[idx] +
                     stripe::Affine(idx + "_i");
    } else {
      replace[idx] = stripe::Affine(idx + "_e");
    }
  }
  // Compute all the refinements for the various blocks
  std::set<stripe::Refinement> reg_allocs;    // Allocations of register caches
  std::set<stripe::Refinement> reg_passthrus; // Passthru's for register caches
  std::set<stripe::Refinement> reg_inners;    // Passthru's for register caches
  std::map<std::string, stripe::Refinement>
      orig_by_name; // Passthru's for register caches
  std::map<std::string, stripe::Refinement>
      inner_by_name; // Passthru's for register caches
  for (const auto &oref : block->refs) {
    // Modify original reference to remove constants
    stripe::Refinement ref = oref;
    for (auto &aff : ref.access) {
      aff.setConstant(0);
    }
    // Start with the normal interior tiling
    auto ref_tile = inner_tile;
    auto ridx = plan.ref_idx[ref.into()];
    // If subgrouped, just use outer tiling
    if (ridx.size()) {
      ref_tile[ridx] = plan.extra_tile[ridx];
    }
    std::vector<size_t> sizes = ref.ApplyTile(ref_tile).sizes();
    auto reg_shape = SimpleShape(ref.interior_shape.type, sizes);
    // If subgrouped, add a banked dimension
    if (ridx.size()) {
      reg_shape.dims.emplace_back(0, plan.subgroup_tile[ridx]);
    }
    // Build the actual allocation
    std::vector<stripe::Affine> reg_access(reg_shape.dims.size());
    stripe::Refinement reg_ref =
        stripe::Refinement(stripe::RefDir::None, "", ref.into() + "_reg",
                           reg_access, reg_shape, ref.agg_op);
    if (ridx.size()) {
      reg_ref.bank_dim = stripe::BankDimension{sizes.size()};
    }
    reg_allocs.emplace(reg_ref);
    // Now modify to make the passthru
    reg_ref.from = reg_ref.into();
    reg_ref.dir = stripe::RefDir::InOut;
    reg_passthrus.emplace(reg_ref);
    // Now, make the innermost register refinements
    auto ref_repl = replace;
    if (ridx.size()) {
      ref_repl[ridx] = stripe::Affine(ridx + "_e");
    }
    reg_ref.access.clear();
    for (auto poly : ref.access) {
      reg_ref.access.push_back(poly.sym_eval(ref_repl));
    }
    if (ridx.size()) {
      reg_ref.access.push_back(stripe::Affine(ridx + "_i"));
    }
    orig_by_name[ref.into()] = ref;
    inner_by_name[ref.into()] = reg_ref;
    reg_ref = reg_ref.WithInto(ref.into());
    reg_ref.dir = ref.dir;
    reg_inners.emplace(reg_ref);
  }

  // Now, prepare to tile the block
  TileShape threaded_ts, accum_ts, inner_ts;
  stripe::Affine out_flat = block->ref_outs()[0]->FlatAccess();
  for (const auto &idx : plan.idxs) {
    size_t prod = 1;
    prod *= (idx == plan.thread_idx ? 1 : plan.subgroup_tile[idx]);
    prod *= plan.extra_tile[idx];
    inner_ts.push_back(prod);
    if (out_flat[idx] == 0) {
      prod = block->idx_by_name(idx)->range;
    }
    accum_ts.push_back(prod);
    if (idx == plan.thread_idx) {
      prod *= plan.subgroup_tile[idx];
    }
    threaded_ts.push_back(prod);
  }

  // Do the tiling and name each block
  stripe::Block *outer = block;
  ApplyTile(outer, threaded_ts, false);
  stripe::Block *thread = block->SubBlock(0).get();
  ApplyTile(thread, accum_ts, false, false, true);
  stripe::Block *accum = thread->SubBlock(0).get();
  ApplyTile(accum, inner_ts, false);
  stripe::Block *inner = accum->SubBlock(0).get();

  // Change up the inner indexes
  inner->idxs = inner_idxs;

  // Add in the register refinements
  outer->refs.insert(reg_allocs.begin(), reg_allocs.end());
  thread->refs.insert(reg_passthrus.begin(), reg_passthrus.end());
  accum->refs.insert(reg_passthrus.begin(), reg_passthrus.end());
  inner->refs = reg_inners;

  // Pass the thread_id through
  accum->idxs.emplace_back("thread_idx", 1, stripe::Affine(plan.thread_idx));
  inner->idx_by_name(plan.thread_idx + "_i")->range = 1;
  inner->idx_by_name(plan.thread_idx + "_i")->affine =
      stripe::Affine("thread_idx");

  // Adjust offset of other subgroup refinement
  for (auto &ref : thread->refs) {
    if (plan.ref_idx[ref.into()] != "" &&
        plan.ref_idx[ref.into()] != plan.thread_idx) {
      ref.mut().access[ref.access.size() - 1] = stripe::Affine(plan.thread_idx);
    }
  }

  // Make the base transfer blocks
  std::map<std::string, std::shared_ptr<stripe::Block>> xfer_blocks;
  for (const auto &kvp : orig_by_name) {
    std::string ri = kvp.first;
    stripe::Refinement orig = kvp.second;
    auto xfer = std::make_shared<stripe::Block>();

    for (const auto &idx : inner_idxs) {
      if (orig.FlatAccess()[idx.name.substr(0, idx.name.size() - 2)]) {
        xfer->idxs.push_back(idx);
      }
    }
    std::string ref_idx = plan.ref_idx[ri];
    auto repl = replace;
    if (ref_idx.size()) {
      repl[ref_idx] =
          stripe::Affine(ref_idx + "_e", plan.subgroup_tile[ref_idx]);
    }
    for (auto &poly : orig.access) {
      poly = poly.sym_eval(repl);
    }
    xfer->refs.emplace(orig);
    xfer->refs.emplace(inner_by_name[ri]);
    if (ref_idx.size()) {
      xfer->idx_by_name(ref_idx + "_i")->range = 1;
      xfer->idx_by_name(ref_idx + "_i")->affine = stripe::Affine("thread_idx");
    }
    for (auto &xref : xfer->refs) {
      for (auto &dim : xref.mut().interior_shape.dims) {
        dim.size = 1;
      }
    }
    xfer_blocks[ri] = xfer;
  }

  // Add them in the appropriate places and add the load/stores as required
  for (const stripe::Refinement *ref : block->ref_ins()) {
    auto load = xfer_blocks[ref->into()];
    load->stmts.push_back(std::make_shared<stripe::Load>(ref->into(), "$x"));
    load->stmts.push_back(
        std::make_shared<stripe::Store>("$x", ref->into() + "_reg"));
    load->ref_by_into(ref->into() + "_reg")->mut().dir = stripe::RefDir::Out;
    load->set_tag("subgroup_read");
    load->set_tag("subgroup_inline");
    accum->stmts.push_front(load);
  }
  for (const stripe::Refinement *ref : block->ref_outs()) {
    auto store = xfer_blocks[ref->into()];
    store->stmts.push_back(
        std::make_shared<stripe::Load>(ref->into() + "_reg", "$x"));
    store->stmts.push_back(std::make_shared<stripe::Store>("$x", ref->into()));
    store->ref_by_into(ref->into() + "_reg")->mut().dir = stripe::RefDir::In;
    store->idx_by_name(plan.thread_idx + "_i")->affine =
        stripe::Affine(plan.thread_idx);
    store->set_tag("subgroup_write");
    store->set_tag("subgroup_inline");
    thread->stmts.push_back(store);
  }
  for (auto &stmt : inner->stmts) {
    auto load = stripe::Load::Downcast(stmt);
    if (!load) {
      continue;
    }
    std::string sub_idx = plan.ref_idx[load->from];
    if (sub_idx != "" && sub_idx != plan.thread_idx) {
      load->add_tags({"subgroup_broadcast"});
    }
  }

  // Add some tags!
  outer->remove_tag("contraction");
  outer->set_tag("subgroup_outer");
  outer->set_attr("subgroup_size", static_cast<int64_t>(plan.subgroup_size));
  thread->set_tag("subgroup_thread");
  thread->set_tag("gpu_thread");
  accum->set_tag("subgroup_accum");
  inner->set_tag("subgroup_inner");
  inner->set_tag("subgroup_inline");
}

static void TagTx(stripe::Block* block, const std::set<std::string>& elems) {
  IVLOG(1, "TagTX: " << elems);
  for (auto& stmt : block->stmts) {
    switch (stmt->kind()) {
      case stripe::StmtKind::Load: {
        auto load = stripe::Load::Downcast(stmt);
        if (elems.count(load->from)) {
          load->set_tag("vector_tx");
        }
      } break;
      case stripe::StmtKind::Store: {
        auto store = stripe::Store::Downcast(stmt);
        if (elems.count(store->into)) {
          store->set_tag("vector_tx");
        }
      } break;
      case stripe::StmtKind::Block: {
        auto inner = stripe::Block::Downcast(stmt);
        std::set<std::string> inner_elems;
        for (const auto& ref : inner->refs) {
          if (elems.count(ref.from)) {
            inner_elems.insert(ref.into());
          }
        }
        TagTx(inner.get(), inner_elems);
      } break;
      default:
        break;
    }
  }
}

void VectorizeTx(stripe::Block* block, const AliasMap& map) {
  std::string the_idx;
  for (const auto& idx : block->idxs) {
    if (idx.affine == stripe::Affine() && idx.range != 1) {
      if (!the_idx.empty()) {
        IVLOG(1, *block);
        throw std::runtime_error("Multiple indexes for vectorize_tx, " + the_idx + " vs " + idx.name);
      }
      the_idx = idx.name;
    }
  }
  if (the_idx.empty()) {
    throw std::runtime_error("No real indexes for vectorize_tx, invalid");
  }
  std::set<std::string> elems;
  for (auto& ref : block->refs) {
    if (ref.FlatAccess()[the_idx] == 1) {
      elems.insert(ref.into());
      for (auto& aff : ref.mut().access) {
        aff.mutateMap().erase(the_idx);
      }
    }
  }
  TagTx(block, elems);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
