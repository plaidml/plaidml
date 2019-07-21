// Copyright 2019, Intel Corp.

#include "tile/codegen/subgroup.h"

#include "tile/codegen/tile.h"
#include "tile/codegen/cache.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

struct SubgroupPlan {
  size_t subgroup_size;
  size_t max_mem;
  std::vector<std::string> idxs;
  std::map<std::string, size_t> subgroup_tile;
  std::map<std::string, size_t> extra_tile;
  std::map<std::string, std::string> ref_idx;
  std::string thread_idx;
};

class SubgroupCostModel {
 public:
  SubgroupCostModel(stripe::Block* block, const SubgroupPlan& plan, const proto::SubgroupPass& options)
      : block_(block), plan_(plan), options_(options) {}

  size_t num_accesses(const std::map<std::string, size_t>& tot_tile, const Refinement& ref) {
    auto flat = ref.FlatAccess();
    const auto& acc_map = flat.getMap();
    size_t num = 1;
    for (const auto& it : tot_tile) {
      if (acc_map.find(it.first) != acc_map.end()) {
        num *= it.second;
      }
    }
    return num;
  }

  size_t num_work_items(const std::map<std::string, size_t>& tot_tile) {
    stripe::Affine out_flat = block_->ref_outs(true)[0]->FlatAccess();
    size_t num = 1;
    for (const auto& idx : block_->idxs) {
      size_t prod = 1;
      if (out_flat[idx.name] == 0) {
        prod = idx.range;
      } else {
        prod = (idx.name == plan_.thread_idx ? 1 : plan_.subgroup_tile[idx.name]);
        prod *= plan_.extra_tile[idx.name];
      }
      num *= (idx.range / prod);
    }
    return num;
  }

  int SetRefIdxStrideOne(Refinement* ref) {
    int subgroup = 0;
    for (size_t i = 0; i < ref->access.size(); i++) {
      if (ref->interior_shape.dims[i].stride != 1) {
        continue;
      }
      for (const auto& kvp : ref->access[i].getMap()) {
        if (kvp.first != "" && kvp.second == 1 &&
            plan_.subgroup_tile.at(kvp.first) == static_cast<size_t>(plan_.subgroup_size)) {
          subgroup++;
          plan_.ref_idx[ref->into()] = kvp.first;
        }
      }
    }
    return subgroup;
  }

  int SetRefIdxThreadIdx(Refinement* ref) {
    int subgroup = 0;
    for (size_t i = 0; i < ref->access.size(); i++) {
      auto acc_map = ref->access[i].getMap();
      if (acc_map.size() == 1 && acc_map.begin()->first == plan_.thread_idx && acc_map.begin()->second == 1) {
        subgroup++;
        plan_.ref_idx[ref->into()] = acc_map.begin()->first;
      }
    }
    return subgroup;
  }

  void ComputeCost() {
    // Compute total tile size for each index
    std::map<std::string, size_t> tot_tile;
    for (const auto& idx : plan_.idxs) {
      tot_tile[idx] = plan_.subgroup_tile[idx] * plan_.extra_tile[idx];
    }
    // Compute the amount of memory transfer for each refinement
    size_t tot_accesses = 0;
    size_t tot_mem = 0;
    double tot_mem_io = 0;
    plan_.ref_idx.clear();
    plan_.thread_idx = "";
    // Process outputs first to determine the thread_idx first
    std::vector<Refinement*> ref_list;
    for (const auto ref : block_->ref_outs(true)) {
      ref_list.push_back(ref);
    }
    for (const auto ref : block_->ref_ins()) {
      ref_list.push_back(ref);
    }
    for (const auto& ref : ref_list) {
      TensorShape tile_shape = ref->ApplyTile(tot_tile);
      // Get overall memory cost
      int subgroup = 0;
      size_t mem = tile_shape.sizes_product_bytes();
      double cache_miss = tile_shape.memory_io(options_.cache_width());
      size_t accesses = num_accesses(tot_tile, *ref);
      double cache_hit = accesses - cache_miss;
      double mem_io = cache_miss * options_.mem_latency() + cache_hit * options_.cache_latency();
      // Figure out if we have a single unique stride 1 subgroup block
      if (IsWriteDir(ref->dir)) {
        subgroup = SetRefIdxStrideOne(ref);
        // To determine out ref's idx as well as thread_idx
        if (subgroup == 1) {
          plan_.thread_idx = plan_.ref_idx[ref->into()];
        } else {
          return;
        }
      } else {
        // For in refs
        if (plan_.thread_idx == "") {
          return;
        }
        auto flat_access = ref->FlatAccess().getMap();
        subgroup = (flat_access.find(plan_.thread_idx) == flat_access.end()) ? SetRefIdxStrideOne(ref)
                                                                             : SetRefIdxThreadIdx(ref);
      }
      if (subgroup > 1) {
        subgroup = 0;
        plan_.ref_idx.erase(ref->into());
      }
      // If it works, set thread_idx, otherwise increase memory
      if (subgroup == 1) {
        if (ref->dir == stripe::RefDir::Out) {
          plan_.thread_idx = plan_.ref_idx[ref->into()];
        }
        accesses /= plan_.subgroup_size;
        mem /= plan_.subgroup_size;
        mem_io /= plan_.subgroup_size;
      }
      // More than one subgroup index per refinment = madness
      if (subgroup > 1) {
        return;
      }
      tot_accesses += accesses;
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
    if (tot_mem > static_cast<size_t>(plan_.max_mem)) {
      IVLOG(2, "subgroup: " << plan_.subgroup_tile << ", extra: " << plan_.extra_tile << ", mem: " << tot_mem
                            << ", mem_io: " << tot_mem_io << ", cost: INF");
      return;
    }
    // Compute compute
    size_t tot_ops = 1;
    for (const auto& kvp : tot_tile) {
      tot_ops *= kvp.second;
    }

    size_t tot_stmts = tot_accesses * 2 + tot_ops * block_->stmts.size() / plan_.subgroup_size;

    // Unrolling too many inner stmts may slow down
    if (tot_stmts > options_.inner_stmts_limit()) {
      return;
    }

    size_t num_wis = num_work_items(tot_tile);
    bool is_mem_bounded =
        (tot_mem * num_wis > options_.cache_size()) ||
        (static_cast<double>(tot_ops) / static_cast<double>(tot_mem) <= options_.mem_bounded_threshold());
    // Cost = mem / compute
    IVLOG(2, "subgroup: " << plan_.subgroup_tile << ", extra: " << plan_.extra_tile << ", mem: " << tot_mem
                          << ", mem_io: " << tot_mem_io << ", cost: " << tot_mem_io / static_cast<double>(tot_ops));
    double cost = tot_mem_io / static_cast<double>(tot_ops);

    bool replace = !is_mem_bounded && best_is_mem_bounded_;
    if (is_mem_bounded == best_is_mem_bounded_) {
      replace = is_mem_bounded ? (cost < best_cost_) : (num_wis / tot_mem_io > best_work_items_ / best_tot_mem_io_);
    }
    if (replace) {
      best_cost_ = cost;
      best_plan_ = plan_;
      best_is_mem_bounded_ = is_mem_bounded;
      best_work_items_ = num_wis;
      best_tot_mem_io_ = tot_mem_io;
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
        continue;  // Skip the subgroup if not even divison by subgroup size
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

  double BestCost(SubgroupPlan* best_plan) {
    best_cost_ = std::numeric_limits<double>::infinity();
    best_is_mem_bounded_ = true;
    best_work_items_ = 0;
    best_tot_mem_io_ = 1.0;
    for (int i = 0; i < options_.subgroup_sizes().size(); ++i) {
      plan_.subgroup_size = options_.subgroup_sizes()[i];
      plan_.max_mem = options_.max_mem()[i];
      BestCostRecursive(0);
    }
    *best_plan = best_plan_;
    return best_cost_;
  }

 private:
  stripe::Block* block_;
  SubgroupPlan plan_;
  double best_cost_;
  size_t best_work_items_;
  double best_tot_mem_io_;
  bool best_is_mem_bounded_;
  SubgroupPlan best_plan_;
  proto::SubgroupPass options_;
};

// Add a dim with access idx_name
void AddSubgroupDim(const Refinement& ref, const std::string& idx_name, size_t size) {
  ref.mut().access.push_back(idx_name == "" ? Affine(0) : Affine(idx_name));
  ref.mut().interior_shape.dims.emplace_back(0, size);
  ref.mut().bank_dim = stripe::BankDimension{ref.access.size() - 1};
}

// Move the dim access with idx_name to the lowest dim
void MoveSubgroupDim(const Refinement& ref, const std::string& idx_name, size_t size) {
  bool found = false;
  for (size_t i = 0; i < ref.access.size(); ++i) {
    if (ref.access[i] == Affine(idx_name)) {
      ref.mut().access.erase(ref.mut().access.begin() + i);
      ref.mut().interior_shape.dims.erase(ref.mut().interior_shape.dims.begin() + i);
      found = true;
      break;
    }
  }
  if (found) {
    ref.mut().access.push_back(idx_name == "" ? Affine(0) : Affine(idx_name));
    ref.mut().interior_shape.dims.emplace_back(0, size);
    ref.mut().bank_dim = stripe::BankDimension{ref.access.size() - 1};
  }
}

void FixRefInCacheBlock(Block* outer, const std::string& tag, RefDir dir, 
                        const SubgroupPlan& plan, bool use_ref_idx) {
  for (const auto& stmt : outer->stmts) {
    auto cache_block = Block::Downcast(stmt);
    if (cache_block && cache_block->has_tag(tag)) {
      std::string ridx;
      for (auto& ref : cache_block->refs) {
        if (ref.dir == dir && plan.ref_idx.find(ref.from) != plan.ref_idx.end()) {
          ridx = plan.ref_idx.at(ref.from);
          break;
        }
      }
      if (ridx.size() == 0) {
        continue;
      }
      std::string tidx_name = ridx + "_i";
      Index* idx = cache_block->idx_by_name(tidx_name);
      if (idx) {
        idx->range = 1;
        idx->affine = Affine(use_ref_idx ? ridx : "thread_idx");
      }
      else {
        cache_block->idxs.push_back({tidx_name, 1, Affine(use_ref_idx ? ridx : "thread_idx")});
      }
      for (auto& ref : cache_block->refs) {
        if (ref.dir == dir) {
          // insert new subgroup dim
          if (ridx == plan.thread_idx) {
            AddSubgroupDim(ref, tidx_name, 1);
          }
          else {
            MoveSubgroupDim(ref, tidx_name, 1);
          }
        }
      }
    }
  }
}

void Subgroup(stripe::Block* block, const AliasMap& map, const proto::SubgroupPass& options) {
  if (block->constraints.size()) {
    IVLOG(1, "Failed due to constraints");
    return;
  }
  if (block->ref_outs(true).size() != 1) {
    IVLOG(1, "Giving up due to outputs != 1");
    return;
  }
  // Setup an empty plan
  SubgroupPlan plan;
  for (const auto& idx : block->idxs) {
    if (idx.affine == stripe::Affine()) {
      plan.idxs.push_back(idx.name);
    } else {
      IVLOG(1, "Failed due to passthrus");
      return;  // Right now we don't handle this case
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
  for (const auto& idx : plan.idxs) {
    inner_tile[idx] = plan.subgroup_tile[idx] * plan.extra_tile[idx];
    inner_idxs.emplace_back(idx + "_e", plan.extra_tile[idx]);
    if (plan.subgroup_tile[idx] == size_t(plan.subgroup_size)) {
      inner_idxs.emplace_back(idx + "_i", plan.subgroup_tile[idx]);
      replace[idx] = stripe::Affine(idx + "_e") * plan.subgroup_tile[idx] + stripe::Affine(idx + "_i");
    } else {
      replace[idx] = stripe::Affine(idx + "_e");
    }
  }

  // Now, prepare to tile the block
  TileShape threaded_ts, accum_ts, inner_ts;
  stripe::Affine out_flat = block->ref_outs(true)[0]->FlatAccess();
  for (const auto& idx : plan.idxs) {
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
  stripe::Block* outer = block;
  ApplyTile(outer, threaded_ts, false);
  stripe::Block* thread = block->SubBlock(0).get();
  ApplyTile(thread, accum_ts, false, false, true);
  stripe::Block* accum = thread->SubBlock(0).get();
  ApplyTile(accum, inner_ts, false);
  stripe::Block* inner = accum->SubBlock(0).get();

  // Add some tags!
  outer->remove_tag("contraction");
  outer->set_tag("subgroup_outer");
  outer->set_attr("subgroup_size", static_cast<int64_t>(plan.subgroup_size));
  thread->set_tag("subgroup_thread");
  thread->set_tag("gpu_thread");
  accum->set_tag("subgroup_accum");
  inner->set_tag("subgroup_inner");
  inner->set_tag("subgroup_inline");

  // Pass the thread_id through
  inner->idxs = inner_idxs;
  accum->idxs.emplace_back("thread_idx", 1, stripe::Affine(plan.thread_idx));
  inner->idx_by_name(plan.thread_idx + "_i")->range = 1;
  inner->idx_by_name(plan.thread_idx + "_i")->affine = stripe::Affine("thread_idx");

  // Change index names in inner block
  for (const auto& ref : inner->refs) {
    auto ref_idx = plan.ref_idx[ref.into()];
    auto ref_repl = replace;
    if (ref_idx.size()) {
      ref_repl[ref_idx] = (ref_idx == plan.thread_idx) ? stripe::Affine(ref_idx + "_e") :
        (stripe::Affine(ref_idx + "_e", plan.subgroup_size) + stripe::Affine(ref_idx + "_i"));
    }
    std::vector<Affine> new_access;
    for (auto& aff : ref.mut().access) {
      new_access.push_back(aff.sym_eval(ref_repl));
    }
    ref.mut().access = new_access;
  }

  // Out index at accum level should be all zero
  // Check the fact and clear the access of the out ref at accum level
  for (auto ref : accum->ref_outs(true)) {
    for (auto& acc : ref->mut().access) {
      auto& acc_map = acc.mutateMap();
      for (auto& kvp : acc_map) {
        if (kvp.first == "") {
          if (kvp.second != 0) {
            throw std::runtime_error("Wrong refinement access at accum level.");
          }
          continue;
        }
        Index* idx = accum->idx_by_name(kvp.first);
        if (idx->range != 1) {
          throw std::runtime_error("Non-zero access of output refinement at accum level.");
        }
      }
      acc_map.clear();
    }
  }

  AliasMap outer_map(*(map.parent_alias_map()), outer);
  AliasMap thread_map(outer_map, thread);
  AliasMap accum_map(thread_map, accum);
  Location reg_loc = {{{"REGISTER", {0}}}};

  // We have to separately process the In and Out refinements.
  // When an out ref is cached, we modify the refs in thread block 
  // and affect its alias map. So the accum_map is invalid then.
  for (auto ref : inner->refs) {
    if (ref.dir == RefDir::In) {
      ApplyCache(accum_map,                              // alias_map
                 RefDir::In,                             // dir
                 inner,                                  // ref_block
                 accum,                                  // outer_block
                 ref.into(),                             // var_name
                 reg_loc,                                // mem_loc
                 {},                                     // xfer_loc
                 {"subgroup_read", "subgroup_inline"},   // load_tags
                 {"subgroup_write", "subgroup_inline"},  // store_tags
                 true,                                   // add_constraints
                 true,                                   // reorder_idx
                 false                                   // odd_size
      );
    }
  }

  for (auto ref : inner->refs) {
    if (IsWriteDir(ref.dir)) {
      ApplyCache(thread_map,                             // alias_map
                 RefDir::Out,                            // dir
                 inner,                                  // ref_block
                 thread,                                 // outer_block
                 ref.into(),                             // var_name
                 reg_loc,                                // mem_loc
                 {},                                     // xfer_loc
                 {"subgroup_read", "subgroup_inline"},   // load_tags
                 {"subgroup_write", "subgroup_inline"},  // store_tags
                 true,                                   // add_constraints
                 true,                                   // reorder_idx
                 false                                   // odd_size
      );
    }
  }

  // Add subgroup dims in refs in inner block
  for (auto& ref : inner->refs) {
    auto ridx = plan.ref_idx[ref.into()];
    if (ridx.size()) {
      if (ridx == plan.thread_idx) {
        AddSubgroupDim(ref, ridx + "_i", plan.subgroup_tile[ridx]);
      }
      else {
        MoveSubgroupDim(ref, ridx + "_i", plan.subgroup_tile[ridx]);
      }
    }
  }

  // Fix thread block refs
  for (auto& ref : thread->refs) {
    if (ref.dir == RefDir::None) {
      auto ridx = plan.ref_idx[ref.into()];
      if (ridx.size()) {
        if (ridx == plan.thread_idx) {
          AddSubgroupDim(ref, "", plan.subgroup_tile[ridx]);
        }
        else {
          MoveSubgroupDim(ref, "", plan.subgroup_tile[ridx]);
        }
      }
    }
  }

  // Fix accum block refs
  for (auto& ref : accum->refs) {
    if (ref.dir == RefDir::None || IsWriteDir(ref.dir)) {
      auto ridx = plan.ref_idx[ref.into()];
      if (ridx.size()) {
        if (ridx == plan.thread_idx) {
          AddSubgroupDim(ref, "", plan.subgroup_tile[ridx]);
        }
        else {
          MoveSubgroupDim(ref, "", plan.subgroup_tile[ridx]);
        }
      }
    }
  }

  // Fix refs in read cache block
  FixRefInCacheBlock(accum, "subgroup_read", RefDir::Out, plan, false);

  // Fix refs in write cache block
  FixRefInCacheBlock(thread, "subgroup_write", RefDir::In, plan, true);

  // Set broadcast directives
  for (auto& stmt : inner->stmts) {
    auto load = stripe::Load::Downcast(stmt);
    if (!load) {
      continue;
    }
    std::string sub_idx = plan.ref_idx[load->from];
    // If sub_idx is merged, the last dim is not subgroup dim.
    // Then ignore the broadcast
    if (sub_idx != "" && sub_idx != plan.thread_idx) {
      auto ref_it = inner->ref_by_into(load->from);
      if (ref_it->bank_dim) {
        load->add_tags({"subgroup_broadcast"});
      }
    }
  }
}

static void TagTx(stripe::Block* block, const std::set<std::string>& elems) {
  IVLOG(2, "TagTX: " << elems);
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

void VectorizeTx(stripe::Block* block, const AliasMap& map, size_t read_align_bytes, size_t write_align_bytes) {
  std::string the_idx;
  for (const auto& idx : block->idxs) {
    if (idx.affine == stripe::Affine() && idx.range != 1) {
      if (!the_idx.empty()) {
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
    auto ai = map.at(ref.into());
    auto access = ai.flat();
    auto global_idx = "d" + std::to_string(map.depth()) + ":" + the_idx;
    if (access[global_idx] == 1) {
      bool aligned = true;
      access.mutateMap().erase(global_idx);
      size_t data_size = byte_width(ref.interior_shape.type);
      if (data_size == 0) {
        throw std::runtime_error("Refinement has data type with zero size");
      }
      size_t read_align;
      if (read_align_bytes >= data_size) {
        if (read_align_bytes % data_size == 0) {
          read_align = read_align_bytes / data_size;
        } else {
          continue;
        }
      } else {
        if (data_size % read_align_bytes == 0) {
          read_align = 1;
        } else {
          continue;
        }
      }
      size_t write_align;
      if (write_align_bytes >= data_size) {
        if (write_align_bytes % data_size == 0) {
          write_align = write_align_bytes / data_size;
        } else {
          continue;
        }
      } else {
        if (data_size % write_align_bytes == 0) {
          write_align = 1;
        } else {
          continue;
        }
      }
      for (const auto& kvp : access.getMap()) {
        if (IsReadDir(ref.dir) && (kvp.second % read_align > 0)) {
          aligned = false;
          break;
        }
        if (IsWriteDir(ref.dir) && (kvp.second % write_align > 0)) {
          aligned = false;
          break;
        }
      }
      if (aligned) {
        elems.insert(ref.into());
        for (auto& aff : ref.mut().access) {
          aff.mutateMap().erase(the_idx);
        }
      }
    }
  }
  TagTx(block, elems);
}

void SubgroupPass::Apply(CompilerState* state) const {
  auto reqs = stripe::FromProto(options_.reqs());
  RunOnBlocks(state->entry(), reqs, [&](const AliasMap& map, stripe::Block* block) {  //
    Subgroup(block, map, options_);
  });
}

void VectorizePass::Apply(CompilerState* state) const {
  auto reqs = stripe::FromProto(options_.reqs());
  RunOnBlocks(state->entry(), reqs, [this](const AliasMap& map, stripe::Block* block) {  //
    VectorizeTx(block, map, options_.read_align_bytes(), options_.write_align_bytes());
  });
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<SubgroupPass, proto::SubgroupPass>::Register();
  CompilePassFactory<VectorizePass, proto::VectorizePass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
