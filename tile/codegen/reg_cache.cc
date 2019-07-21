// Copyright 2018, Intel Corporation

#include "tile/codegen/reg_cache.h"

#include "base/util/throw.h"
#include "tile/math/bignum.h"
#include "tile/stripe/stripe.h"
#include "tile/codegen/alias.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace math;    // NOLINT
using namespace stripe;  // NOLINT

struct RegisterPassOptions {
  RefDir dir;
  Location local_loc;
  Location reg_loc;
  size_t align_size;
  size_t reg_size;
  size_t gmem_lat;
  size_t lmem_lat;
  size_t reg_lat;
  std::string comp_parent_tag;
  bool cache_index_order;
};

bool ParseGlobalIndex(const std::string& idx, size_t* depth, std::string* name) {
  size_t pos = idx.find(':');
  if (pos == std::string::npos) {
    return false;
  }
  *depth = std::atoi(idx.substr(1, pos - 1).c_str());
  *name = idx.substr(pos + 1);
  return true;
}

// Get the outer and inner (outer's first sub-block) loop count
static bool OuterInnerLoopCount(Block* outer, size_t* outer_loop, size_t* inner_loop) {
  *outer_loop = outer->idxs_product();
  Block* inner = outer->SubBlock(0).get();
  *inner_loop = inner->idxs_product();
  return true;
}

void FixRefLoc(const Refinement& src, const Refinement& dst) {
  dst.mut().location = src.location;
  dst.mut().offset = src.offset;
  for (size_t i = 0; i < dst.interior_shape.dims.size(); i++) {
    dst.mut().interior_shape.dims[i].stride = src.interior_shape.dims[i].stride;
  }
}

// Propagate the location, offset, and stride information recursively
static void PropagateRefLoc(Block* block, const Refinement& outer_ref) {
  for (auto stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      for (auto& ref : inner->refs) {
        if (ref.from == outer_ref.into()) {
          FixRefLoc(outer_ref, ref);
          PropagateRefLoc(inner.get(), ref);
        }
      }
    }
  }
}

static Index* IndexByDim(Block* block, const Refinement& ref, size_t dim) {
  const auto& aff = ref.access[dim];
  if (aff == Affine()) {
    return nullptr;
  }
  const auto& aff_map = aff.getMap();
  if (aff_map.size() != 1) {
    throw std::runtime_error("Try to find the index with complex affine.");
  }
  return block->idx_by_name(aff_map.begin()->first);
}

static void AdjustRefAccessHelper(Block* outer, Refinement* outer_ref, Block* inner, //
                                  const std::map<std::string, Rational>& multiple,   //
                                  bool adjust_all) {

  // Adjust the refinement in inner
  // For the cached refinement, the access must be the corresponding index.
  // Only if adjust_all == true, adjust other refinements.
  // In this case, the access are the corresponding index with the coefficient
  // same as the outer index range
  for (auto& inner_ref : inner->refs) {
    if (adjust_all || inner_ref.from == outer_ref->into()) {
      for (auto& aff : inner_ref.mut().access) {
        auto& aff_map = aff.mutateMap();
        for (auto& it : aff_map) {
          it.second = (inner_ref.from == outer_ref->into()) ?
                      1 : ToInteger(it.second / multiple.at(it.first));
        }
      }
    }
  }
  // Adjust the refinement in outer block
  // Clear the accesses in block index to zero
  for (auto& aff : outer_ref->mut().access) {
    auto& aff_map = aff.mutateMap();
    for (auto& aff_it : aff_map) {
      if (outer->idx_by_name(aff_it.first)) {
        aff_map.clear();
        break;
      }
    }
  }
}

static void AdjustRefAccess(Block* outer, const std::string& ref_name,        //
                            const std::map<std::string, Rational>& multiple,  //
                            bool adjust_all) {
  auto it = outer->ref_by_from(ref_name);
  for (auto stmt : outer->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      AdjustRefAccessHelper(outer, &it->mut(), inner.get(), multiple, adjust_all);
    }
  }
}

// Reorder block's index according to base
static void ReorderIndex(Block* base, Block* block, const std::string& ref_name) {
  // Maps base index to block index for the same refinement
  auto base_ref_it = base->ref_by_from(ref_name);
  auto block_ref_it = block->ref_by_from(ref_name);
  size_t n_dims = base_ref_it->access.size();
  std::map<std::string, std::string> base2block;
  std::map<std::string, std::string> block2base;
  for (size_t i = 0; i < n_dims; ++i) {
    auto base_idx = IndexByDim(base, *base_ref_it, i);
    auto block_idx = IndexByDim(block, *block_ref_it, i);
    if (base_idx == nullptr && block_idx == nullptr) {
      continue;
    }
    if (base_idx == nullptr || block_idx == nullptr) {
      throw std::runtime_error("Base refinement and block refinement don't match.");
    }
    base2block.emplace(base_idx->name, block_idx->name);
    block2base.emplace(block_idx->name, base_idx->name);
  }
  // Remove all index in block, which also exist in base
  block->idxs.erase(std::remove_if(block->idxs.begin(), block->idxs.end(),  //
                                   [&block2base](const Index& idx) -> bool {
                                     return block2base.find(idx.name) != block2base.end();
                                   }),  //
                    block->idxs.end());
  for (const auto& idx : base->idxs) {
    if (idx.affine == Affine()) {
      auto it = base2block.find(idx.name);
      if (it != base2block.end()) {
        block->idxs.push_back({it->second, idx.range, Affine()});
      } else {
        block->idxs.push_back({idx.name + "_0", idx.range, Affine()});
      }
    }
  }
}

// Check if cache's ref idx and comp's ref idx are in the same order from the end
static bool ConsistentIdxOrder(Block* cache, const Refinement& cache_ref,  //
                               Block* comp, const Refinement& comp_ref) {
  size_t n_dims = cache_ref.access.size();
  if (comp_ref.access.size() != n_dims) {
    return false;
  }
  for (size_t i = 0; i < n_dims; ++i) {
    if (cache_ref.access[i] == Affine() && comp_ref.access[i] == Affine()) {
      continue;
    }
    if (cache_ref.access[i] == Affine() || comp_ref.access[i] == Affine()) {
      return false;
    }
    const auto& cache_map = cache_ref.access[i].getMap();
    const auto& comp_map = comp_ref.access[i].getMap();
    if (cache_map.size() != 1 || comp_map.size() != 1) {
      return false;
    }
    const auto& cache_idx = cache_map.begin()->first;
    const auto& comp_idx = comp_map.begin()->first;
    size_t cache_order = 0;
    size_t cache_idx_range = 0;
    for (auto it = cache->idxs.rbegin(); it != cache->idxs.rend(); ++it) {
      if (it->range > 1) {
        ++cache_order;
      }
      if (it->name == cache_idx) {
        cache_idx_range = it->range;
        break;
      }
    }
    size_t comp_order = 0;
    size_t comp_idx_range = 0;
    for (auto it = comp->idxs.rbegin(); it != comp->idxs.rend(); ++it) {
      if (it->range > 1) {
        ++comp_order;
      }
      if (it->name == comp_idx) {
        comp_idx_range = it->range;
        break;
      }
    }
    if (cache_order != comp_order) {
      // If both index ranges are 1, the index can be removed later. It does not matter.
      if (cache_idx_range != 1 || comp_idx_range != 1) {
        return false;
      }
    }
  }
  return true;
}

// Transform the cache block and computational block as well as their refinements.
static void PartialCacheInRegister(const AliasMap& parent_map,          //
                                   Block* parent, Block* comp_parent,   //
                                   Block* cache, Block* comp,           //
                                   Block* orig_cache,                   //
                                   bool cache_index_order,              //
                                   bool keep_local,                     //
                                   const std::string& ref_name,         //
                                   const std::map<std::string, Rational>& multiple) {

  // Note that parent is cache block's parent, probably not be computational block's parent
  auto parent_ref_it = parent->ref_by_into(ref_name);
  auto cache_inner = cache->SubBlock(0);

  // Conver the constraints in global index sytle
  std::vector<Affine> constraints;
  // Only global->register needs the original constraints
  if (!keep_local) {
    AliasMap old_cache_map(parent_map, orig_cache);
    AliasMap old_inner_map(old_cache_map, orig_cache->SubBlock(0).get());
    for (const auto& cons : cache_inner->constraints) {
      auto global_cons = cons.sym_eval(old_inner_map.idx_sources());
      constraints.push_back(global_cons);
    }
  }

  if (keep_local) {
    std::string local_ref_name = parent_ref_it->into() + "_local";
    Refinement local_ref = parent_ref_it->WithInto(local_ref_name);
    parent->refs.insert(local_ref);
    auto ref_it = orig_cache->ref_by_from(parent_ref_it->into());
    ref_it->mut().from = local_ref_name;
  }
  parent_ref_it->mut().location.devs[0].name = "REGISTER";

  // Transform the cache block's index to be same as computational block's index
  for (const auto& it : multiple) {
    bool found = false;
    for (auto& idx : cache->idxs) {
      if (it.first == idx.name) {
        found = true;
        idx.range = ToInteger(idx.range / it.second);
        break;
      }
    }
    if (!found) {
      // The index doesn't exist. It means the range is 1.
      cache->idxs.push_back({it.first, static_cast<uint64_t>(ToInteger(1 / it.second)), Affine()});
    }
    found = false;
    for (auto& idx : cache_inner->idxs) {
      if (it.first == idx.name) {
        found = true;
        idx.range = ToInteger(it.second * idx.range);
        break;
      }
    }
    if (!found) {
      // The index doesn't exist. It means the range is 1.
      cache_inner->idxs.push_back({it.first, static_cast<uint64_t>(ToInteger(it.second)), Affine()});
    }
  }

  // Reorder the idx in cache block or comp block
  if (cache_index_order) {
    // Adjust comp blocks' index order according to cache block
    ReorderIndex(cache, comp, ref_name);
      // In this case we need to mark as "reg_cache" in order to
      // tell the code emitter
      cache->set_tag("reg_cache");
      cache_inner->set_tag("reg_cache");
  } else {
    // Adjust cache block's index order according to comp block
    ReorderIndex(comp, cache, ref_name);
    // Do not set reg_cache for the cache/cache_inner
    // because the index of cache is same as the index of comp
    // and it needs conditions.
  }
  comp->set_tag("ordered_idx");

  AdjustRefAccess(cache, ref_name, multiple, true);
  AdjustRefAccess(comp, ref_name, multiple, false);

  // Stride changed. So need to fix all ref's interior size
  for (auto& cache_ref : cache->refs) {
    const auto& inner_ref = cache_inner->ref_by_from(cache_ref.into());
    cache_ref.mut().interior_shape = cache_inner->exterior_shape(inner_ref->into());
    // For the cached refinement, we should change both interior
    // and exterior shapes. So we need to change the stripe as well.
    // For other refinement, we just change the interior shape.
    if (cache_ref.from == ref_name) {
      size_t n_dims = cache_ref.interior_shape.dims.size();
      std::vector<size_t> sizes(n_dims);
      for (size_t i = 0; i < n_dims; ++i) {
        sizes[i] = cache_ref.interior_shape.dims[i].size;
      }
      parent_ref_it->mut().interior_shape =                //
          SimpleShape(parent_ref_it->interior_shape.type,  //
                      sizes, parent_ref_it->interior_shape.layout);
    }
  }
  auto comp_inner = comp->SubBlock(0);
  auto comp_ref_it = comp->ref_by_from(ref_name);
  const auto& inner_ref = comp_inner->ref_by_from(comp_ref_it->into());
  comp_ref_it->mut().interior_shape = comp_inner->exterior_shape(inner_ref->into());
  if (comp_parent != parent) {
    auto comp_parent_ref_it = comp_parent->ref_by_into(ref_name);
    comp_parent_ref_it->mut().interior_shape = comp->exterior_shape(comp_ref_it->into());
  }

  PropagateRefLoc(parent, *parent_ref_it);

  // Adjust constraints
  cache_inner->constraints.clear();
  // Build alias maps from parent to cache_inner
  AliasMap new_parent_map(*(parent_map.parent_alias_map()), parent);
  AliasMap new_cache_map(new_parent_map, cache);
  AliasMap new_inner_map(new_cache_map, cache_inner.get());
  // Adjust constraints
  if (keep_local) {
    // Local -> Register, the original constraints don't work.
    // So check the ref extents only
    Refinement* cache_inner_ref = (comp_ref_it->dir == RefDir::In) ?
                                  cache_inner->ref_ins()[0] : cache_inner->ref_outs(true)[0];
    const auto& ai = new_inner_map.at(cache_inner_ref->into());
    for (size_t i = 0; i < cache_inner_ref->interior_shape.dims.size(); ++i) {
      new_inner_map.AddConstraintForIndex(cache_inner.get(), ai, i,
        "", cache_inner_ref->interior_shape.dims[i].size <= 1);
    }
  }
  else {
    // Global -> Register, modify the original constraints
    for (auto& cons : constraints) {
      for (auto& kvp : cons.mutateMap()) {
        if (kvp.first == "") {
          continue;
        }
        std::string idx_name;
        size_t depth;
        if (!ParseGlobalIndex(kvp.first, &depth, &idx_name)) {
          throw std::runtime_error("Incorrect global index " + kvp.first);
        }
        if (depth == new_parent_map.depth() + 1) {
          kvp.second = ToInteger(kvp.second * multiple.at(idx_name));
        }
        else if (depth == new_parent_map.depth() + 2) {
          kvp.second = ToInteger(kvp.second / multiple.at(idx_name));
        }
      }
      // Translate back to local index
      auto local_cons = new_inner_map.translate(cons);
      cache_inner->constraints.push_back(local_cons);
    }
  }
}

// Try to decide if we should load the ref in cache into registers
static bool CacheRefInRegister(const AliasMap& parent_map,                       //
                               Block* parent, Block* comp_parent, Block* cache,  //
                               Block* comp, const RegisterPassOptions& opt) {
  std::set<Refinement>::const_iterator cache_ref_it;
  // Determine the refinement in cache block
  if (opt.dir == RefDir::In) {
    cache_ref_it = cache->ref_by_into("dst");
  } else if (opt.dir == RefDir::Out) {
    cache_ref_it = cache->ref_by_into("src");
  } else {
    throw std::runtime_error("Invalid direction for caching into registers.");
  }

  // The candidate must be the ref marked as LOCAL
  if (cache_ref_it->location.devs[0].name != "LOCAL") {
    return false;
  }

  const auto& cache_inner = cache->SubBlock(0);
  const std::string ref_name = cache_ref_it->from;

  // Make sure each access of the refinement is single index, not complex affine
  const auto& comp_ref_it = comp->ref_by_from(ref_name);
  for (const auto& aff : comp_ref_it->access) {
    if (aff != Affine() && aff.getMap().size() > 1) {
      return false;
    }
  }

  // The ref dim in all computation block should be same
  if (comp_ref_it->access.size() != cache_ref_it->access.size()) {
    return false;
  }

  // Get the inner and outer loop counts for the cache block
  size_t outer_loop;
  size_t inner_loop;
  if (!OuterInnerLoopCount(cache, &outer_loop, &inner_loop)) {
    return false;
  }

  // multiple is the vector of cache block index / computational block index
  std::map<std::string, Rational> multiple;
  // mul_prod is the product of multiple
  double mul_prod = 1.0;
  const auto& cache_inner_ref_it = cache_inner->ref_by_from(cache_ref_it->into());
  for (size_t i = 0; i < cache_ref_it->access.size(); ++i) {
    const auto& cache_ref_aff = cache_ref_it->access[i];
    const auto& comp_ref_aff = comp_ref_it->access[i];
    if (cache_ref_aff == Affine()) {
      if (comp_ref_aff == Affine()) {
        continue;
      } else {
        return false;
      }
    }
    if (cache_ref_aff.getMap().size() != 1 || comp_ref_aff.getMap().size() != 1) {
      return false;
    }

    const auto& cache_inner_aff = cache_inner_ref_it->access[i];
    Index* cache_idx = cache->idx_by_name(cache_ref_aff.getMap().begin()->first);
    Index* comp_idx = comp->idx_by_name(comp_ref_aff.getMap().begin()->first);
    Index* cache_inner_idx = cache_inner->idx_by_name(cache_inner_aff.getMap().begin()->first);
    // The index must be real loop, not affine
    if (cache_idx->affine != Affine() || comp_idx->affine != Affine() || cache_inner_idx->affine != Affine()) {
      return false;
    }

    // For each dim, make sure the total range (inner*outer) of cache block is
    // divisible by comp_idx's range
    size_t cache_total_range = cache_idx->range * cache_inner_idx->range;
    // The first condition is for cache_total_range == 1
    if (comp_idx->range > cache_total_range || cache_total_range % comp_idx->range > 0) {
      return false;
    }

    Rational times = cache_idx->range;
    // If not divisible, it's hard to transform, then give up.
    if (cache_idx->range % comp_idx->range > 0 && comp_idx->range % cache_idx->range > 0) {
      return false;
    }
    times /= comp_idx->range;
    mul_prod *= static_cast<double>(times);
    multiple.emplace(cache_idx->name, times);
  }

  // Now we compute the load count of the refinement elements in the computational block
  size_t comp_load_count = 0;
  size_t iloop;
  size_t oloop;
  if (!OuterInnerLoopCount(comp, &oloop, &iloop)) {
    return false;
  }
  // If the block is threaded, only count the inner loop
  if (comp->has_tag("gpu_thread")) {
    comp_load_count += iloop;
  } else {
    comp_load_count += (iloop * oloop);
  }

  // Don't use interior shape size here because the refinement may be transformed
  double load_size = mul_prod * cache_inner->idxs_product() *  //
                     byte_width(cache_ref_it->interior_shape.type);
  if (load_size > static_cast<double>(opt.reg_size)) {
    return false;
  }

  // the cost could be negative because mul_prod is probably less than 1.0
  double cache_load_cost = (inner_loop * mul_prod) * (opt.gmem_lat + opt.reg_lat)  //
                           - inner_loop * (opt.gmem_lat + opt.lmem_lat);
  double comp_load_benefit = comp_load_count * (opt.lmem_lat - opt.reg_lat);

  // If the index order is based on the cache block,
  // and the index order in comp block can't be change,
  // we need to check if the the index order between cache block
  // and comp block are consistent. If not consistent, copy 
  // data from local to register
  Block* to_cache = cache;
  bool cache_index_order = opt.cache_index_order;
  bool keep_local = comp_load_benefit <= cache_load_cost;
  if (opt.cache_index_order && comp->has_tag("ordered_idx")) {
    if (!ConsistentIdxOrder(cache, *cache_ref_it, comp, *comp_ref_it)) {
      keep_local = true;
    }
  }

  if (keep_local) {
    // New cache block from local to register
    auto new_cache = CloneBlock(*cache);
    auto new_inner = new_cache->SubBlock(0);
    auto global_outer_ref_it = new_cache->ref_by_from(ref_name + "_raw");
    auto global_inner_ref_it = new_inner->ref_by_from(global_outer_ref_it->into());
    new_cache->refs.erase(*global_outer_ref_it);
    new_inner->refs.erase(*global_inner_ref_it);

    // outer ref for new_cache
    auto local_outer_ref_it = cache->ref_by_from(ref_name);
    std::string local_into = (local_outer_ref_it->into() == "src") ? "dst" : "src";
    Refinement local_outer_ref = local_outer_ref_it->WithInto(local_into);
    local_outer_ref.dir = (local_outer_ref.dir == RefDir::In) ? RefDir::Out : RefDir ::In;
    local_outer_ref.from = ref_name + "_local";
    new_cache->refs.insert(local_outer_ref);

    // inner ref for new_cache
    auto local_inner_ref_it = cache_inner->ref_by_from(local_outer_ref_it->into());
    Refinement local_inner_ref = local_inner_ref_it->WithInto(local_into);
    local_inner_ref.dir = (local_inner_ref.dir == RefDir::In) ? RefDir::Out : RefDir ::In;
    local_inner_ref.from = local_inner_ref.into();
    new_inner->refs.insert(local_inner_ref);

    InsertAfterBlock(parent, cache, new_cache);
    to_cache = new_cache.get();
    cache_index_order = false;
  }

  // Now it's better to load the refinement into registers
  PartialCacheInRegister(parent_map, parent, comp_parent, to_cache, comp, cache, 
                         cache_index_order, keep_local, ref_name, multiple);
  return true;
}

static void CacheWholeRefInRegister(Block* parent, Block* cache, Block* comp,  //
                                    const RegisterPassOptions& opt) {
  std::set<Refinement>::const_iterator cache_ref_it;
  // Determine the refinement in cache block
  if (opt.dir == RefDir::In) {
    cache_ref_it = cache->ref_by_into("dst");
  } else if (opt.dir == RefDir::Out) {
    cache_ref_it = cache->ref_by_into("src");
  } else {
    throw std::runtime_error("Invalid direction for caching into registers.");
  }

  // The candidate must be the ref marked as LOCAL
  if (cache_ref_it->location.devs[0].name != "LOCAL") {
    return;
  }

  // Get the inner and outer loop counts for the cache block
  size_t outer_loop;
  size_t inner_loop;
  if (!OuterInnerLoopCount(cache, &outer_loop, &inner_loop)) {
    return;
  }

  auto cache_inner = cache->SubBlock(0);
  size_t type_size = byte_width(cache_ref_it->interior_shape.type);
  size_t load_size = cache_inner->idxs_product() * type_size;
  if (load_size > opt.reg_size) {
    return;
  }

  // Now we compute the load count of the refinement elements in the computational block
  size_t comp_load_count = 0;
  size_t iloop;
  size_t oloop;
  if (!OuterInnerLoopCount(comp, &oloop, &iloop)) {
    return;
  }

  // If the block is threaded, only count the inner loop
  if (comp->has_tag("gpu_thread")) {
    comp_load_count += iloop;
  } else {
    comp_load_count += (iloop * oloop);
  }

  double times = 1.0 * opt.align_size / type_size;
  double glmem_lat = (opt.gmem_lat + (times - 1) * opt.lmem_lat) / times;
  double cost = inner_loop * outer_loop * (glmem_lat + opt.reg_lat) -  //
                inner_loop * (glmem_lat + opt.lmem_lat);
  double benefit = comp_load_count * (opt.lmem_lat - opt.reg_lat);

  if (benefit > cost) {
    // Load the whole buffer into register
    cache->remove_tag("gpu_thread");
    cache->set_tag("reg_cache");
    cache_inner->set_tag("reg_cache");
    auto parent_ref_it = parent->ref_by_into(cache_ref_it->from);
    parent_ref_it->mut().location.devs[0].name = "REGISTER";
    PropagateRefLoc(parent, *parent_ref_it);
  }
}

static void BlocksForRegisterCache(const AliasMap& parent_map,
                                   Block* parent, Block* cache,
                                   const RegisterPassOptions& opt) {
  std::string ref_name;
  if (opt.dir == RefDir::In) {
    ref_name = cache->ref_by_into("dst")->from;
  } else if (opt.dir == RefDir::Out) {
    ref_name = cache->ref_by_into("src")->from;
  } else {
    throw std::runtime_error("Invalid direction for caching into registers.");
  }

  Block* comp = nullptr;
  Block* comp_parent;
  if (parent->has_tag(opt.comp_parent_tag)) {
    comp_parent = parent;
  } else if (parent->SubBlock(0)->has_tag(opt.comp_parent_tag)) {
    comp_parent = parent->SubBlock(0).get();
  } else {
    return;
  }

  for (auto stmt : comp_parent->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner && !inner->has_tag("cache")) {
      // Sometimes inner doesn't have ref, ignore it
      if (inner->ref_by_from(ref_name, false) != inner->refs.end()) {
        comp = inner.get();
        break;
      }
    }
  }

  if (parent != comp_parent) {
    // In this case we need to check if there is any use of the refinement
    // in parent, which may not be safe
    for (auto stmt : parent->stmts) {
      auto inner = Block::Downcast(stmt);
      if (inner && inner.get() != cache && !inner->has_tag(opt.comp_parent_tag)) {
        if (inner->ref_by_from(ref_name, false) != inner->refs.end()) {
          return;
        }
      }
    }
  }

  if (comp) {
    bool ret = CacheRefInRegister(parent_map, parent, comp_parent, cache, comp, opt);
    // It is not safe to let cache store be in registers
    if (!ret && opt.dir == RefDir::In) {
      CacheWholeRefInRegister(parent, cache, comp, opt);
    }
  }
}

static void RegisterCacheRecurse(const AliasMap& parent_map,   //
                                 Block* parent, Block* block,  //
                                 const Tags& reqs,             //
                                 const RegisterPassOptions& opt) {
  if (block->has_tags(reqs)) {
    BlocksForRegisterCache(parent_map, parent, block, opt);
  } else {
    for (auto& stmt : block->stmts) {
      auto inner = stripe::Block::Downcast(stmt);
      if (inner) {
        AliasMap alias_map(parent_map, block);
        RegisterCacheRecurse(alias_map, block, inner.get(), reqs, opt);
      }
    }
  }
}

void RegisterCachePass::Apply(CompilerState* state) const {
  RegisterPassOptions opt;
  auto reqs = FromProto(options_.reqs());
  opt.local_loc = stripe::FromProto(options_.local_loc());
  opt.reg_loc = stripe::FromProto(options_.register_loc());
  opt.reg_size = options_.register_size();
  opt.gmem_lat = options_.global_memory_latency();
  opt.lmem_lat = options_.local_memory_latency();
  opt.reg_lat = options_.register_latency();
  opt.dir = stripe::FromProto(static_cast<stripe::proto::Refinement::Dir>(options_.dir()));
  opt.comp_parent_tag = options_.comp_parent_tag();
  opt.cache_index_order = options_.index_order() == "cache";
  opt.align_size = options_.align_size();

  AliasMap base;
  RegisterCacheRecurse(base, nullptr, state->entry(), reqs, opt);
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<RegisterCachePass, proto::RegisterCachePass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
