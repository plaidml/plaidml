// Copyright 2018, Intel Corporation

#include "tile/codegen/reg_cache.h"

#include "base/util/throw.h"
#include "tile/math/bignum.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace math;    // NOLINT
using namespace stripe;  // NOLINT

// Get the outer and inner (outer's first sub-block) loop count
static bool OuterInnerLoopCount(Block* outer, size_t* outer_loop, size_t* inner_loop) {
  *outer_loop = outer->idxs_product();
  Block* inner = outer->SubBlock(0).get();
  *inner_loop = inner->idxs_product();
  return true;
}

static void PropagateRefLoc(Block* block, const Refinement& outer_ref) {
  for (auto stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      for (auto& ref : inner->refs) {
        if (ref.from == outer_ref.into) {
          ref.location = outer_ref.location;
          ref.offset = outer_ref.offset;
          for (size_t i = 0; i < ref.interior_shape.dims.size(); i++) {
            ref.interior_shape.dims[i].stride = outer_ref.interior_shape.dims[i].stride;
          }
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

static void RemoveThreadCoefficientRef(Block* outer, Refinement* outer_ref,  //
                                       Block* inner, bool adjust_all) {
  for (auto& inner_ref : inner->refs) {
    if (adjust_all || inner_ref.from == outer_ref->into) {
      for (auto& aff : inner_ref.access) {
        auto& aff_map = aff.mutateMap();
        if (aff_map.size() == 1) {
          auto it = aff_map.begin();
          for (const auto& idx : outer->idxs) {
            if (idx.name == it->first) {
              it->second = (inner_ref.from == outer_ref->into) ? 1 : idx.range;
              break;
            }
          }
        }
      }
    }
  }
  for (auto& aff : outer_ref->access) {
    auto& aff_map = aff.mutateMap();
    if (aff_map.size() == 1) {
      auto aff_it = aff_map.begin();
      for (const auto& idx : outer->idxs) {
        if (idx.name == aff_it->first) {
          aff_map.clear();
          break;
        }
      }
    }
  }
}

static void RemoveThreadCoefficientBlock(Block* outer, const std::string& ref_name, bool adjust_all) {
  auto it = outer->ref_by_from(ref_name);
  for (auto stmt : outer->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      for (auto& ref : inner->refs) {
        if (ref.from == it->into) {
          RemoveThreadCoefficientRef(outer, &(*it), inner.get(), adjust_all);
        }
      }
    }
  }
}

static void PartialCacheInRegister(Block* parent, Block* cache,                                    //
                                   const std::vector<Block*>& comps, const std::string& ref_name,  //
                                   const std::map<std::string, Rational>& multiple) {
  auto parent_it = parent->ref_by_into(ref_name);
  auto cache_it = cache->ref_by_from(ref_name);

  auto cache_inner = cache->SubBlock(0);
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
      throw std::runtime_error("Cannot find idx " + it.first);
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
      cache_inner->idxs.push_back({it.first, static_cast<uint64_t>(ToInteger(it.second)), Affine()});
    }
  }

  // Reorder the idx in computation blocks
  // Put the idx used by the cahce block at the end with the same order
  size_t n_dims = cache_it->access.size();
  for (auto& comp : comps) {
    auto comp_it = comp->ref_by_from(ref_name);
    // Maps cache index to comp index
    std::map<std::string, std::string> cache2comp;
    std::map<std::string, std::string> comp2cache;
    for (size_t i = 0; i < n_dims; ++i) {
      auto cache_idx = IndexByDim(cache, *cache_it, i);
      auto comp_idx = IndexByDim(comp, *comp_it, i);
      if (cache_idx == nullptr && comp_idx == nullptr) {
        continue;
      }
      if (cache_idx == nullptr || comp_idx == nullptr) {
        throw std::runtime_error("Cache refinement and computation refinement don't match.");
      }
      cache2comp.emplace(cache_idx->name, comp_idx->name);
      comp2cache.emplace(comp_idx->name, cache_idx->name);
    }
    comp->idxs.erase(std::remove_if(comp->idxs.begin(), comp->idxs.end(),  //
                                    [&](const Index& idx) {                //
                                      return comp2cache.find(idx.name) != comp2cache.end();
                                    }),  //
                     comp->idxs.end());
    for (const auto& idx : cache->idxs) {
      comp->idxs.push_back({cache2comp[idx.name], idx.range, Affine()});
    }
    comp->set_tag("ordered_idx");
  }

  RemoveThreadCoefficientBlock(cache, ref_name, true);
  for (const auto& comp : comps) {
    RemoveThreadCoefficientBlock(comp, ref_name, false);
  }

  // Fix ref's interior size
  for (auto& cache_ref : cache->refs) {
    const auto& inner_ref = cache_inner->ref_by_from(cache_ref.into);
    cache_ref.interior_shape = cache_inner->exterior_shape(inner_ref->into);
    // For the cached refinement, we should change the both interior
    // and exterior shapes. So we need to change the stripe as well.
    // For other refinement, we just change the interior shape.
    if (cache_ref.from == ref_name) {
      size_t n_dims = cache_ref.interior_shape.dims.size();
      std::vector<size_t> sizes(n_dims);
      for (size_t i = 0; i < n_dims; ++i) {
        sizes[i] = cache_ref.interior_shape.dims[i].size;
      }
      parent_it->interior_shape = SimpleShape(parent_it->interior_shape.type,  //
                                              sizes, parent_it->interior_shape.layout);
    }
  }

  parent_it->location.devs[0].name = "REGISTER";
  PropagateRefLoc(parent, *parent_it);
  cache->set_tag("reg_cache");
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
    int cache_order = 0;
    for (auto it = cache->idxs.rbegin(); it != cache->idxs.rend(); ++it, ++cache_order) {
      if (it->name == cache_idx) {
        break;
      }
    }
    int comp_order = 0;
    for (auto it = comp->idxs.rbegin(); it != comp->idxs.rend(); ++it, ++comp_order) {
      if (it->name == comp_idx) {
        break;
      }
    }
    if (cache_order != comp_order) {
      return false;
    }
  }
  return true;
}

static void CacheRefInRegister(Block* parent, Block* cache, const std::vector<Block*>& comps, size_t reg_size,
                               size_t gmem_lat, size_t lmem_lat, size_t reg_lat) {
  // if we can load only a part of the refinement
  const auto& cache_it = cache->ref_by_into("dst");
  if (cache_it->location.devs[0].name != "LOCAL") {
    return;
  }

  // Currently we don't process the cache load with constraints.
  const auto& cache_inner = cache->SubBlock(0);
  if (cache_inner->constraints.size() > 0) {
    return;
  }

  // Make sure each access of the refinement is single index
  for (const auto& comp : comps) {
    const auto& comp_it = comp->ref_by_from(cache_it->from);
    for (const auto& aff : comp_it->access) {
      if (aff != Affine() && aff.getMap().size() > 1) {
        return;
      }
    }
    if (comp->has_tag("ordered_idx")) {
      if (!ConsistentIdxOrder(cache, *cache_it, comp, *comp_it)) {
        return;
      }
    }
  }

  // The ref dim in all computation block should be same
  const auto& comp0_it = comps[0]->ref_by_from(cache_it->from);
  for (size_t i = 1; i < comps.size(); ++i) {
    const auto& comp_it = comps[i]->ref_by_from(cache_it->from);
    if (comp_it->access != comp0_it->access) {
      return;
    }
  }
  if (comp0_it->access.size() != cache_it->access.size()) {
    return;
  }

  size_t outer_loop;
  size_t inner_loop;
  if (!OuterInnerLoopCount(cache, &outer_loop, &inner_loop)) {
    return;
  }

  std::map<std::string, Rational> multiple;
  double mul_prod = 1.0;
  for (size_t i = 0; i < cache_it->access.size(); ++i) {
    const auto& cache_aff = cache_it->access[i];
    const auto& comp0_aff = comp0_it->access[i];
    if (cache_aff == Affine()) {
      if (comp0_aff == Affine()) {
        continue;
      } else {
        return;
      }
    }
    if (cache_aff.getMap().size() != 1 || comp0_aff.getMap().size() != 1) {
      return;
    }
    Index* cache_idx = cache->idx_by_name(cache_aff.getMap().begin()->first);
    Index* comp0_idx = comps[0]->idx_by_name(comp0_aff.getMap().begin()->first);
    if (cache_idx->affine != Affine() || comp0_idx->affine != Affine()) {
      return;
    }
    Rational times = cache_idx->range;
    if (cache_idx->range % comp0_idx->range > 0 && comp0_idx->range % cache_idx->range > 0) {
      return;
    }
    times /= comp0_idx->range;
    mul_prod *= static_cast<double>(times);
    multiple.emplace(cache_idx->name, times);
  }

  size_t comp_load_count = 0;
  for (const auto& comp : comps) {
    size_t iloop;
    size_t oloop;
    if (!OuterInnerLoopCount(comp, &oloop, &iloop)) {
      return;
    }
    if (comp->has_tag("gpu_thread")) {
      comp_load_count += iloop;
    } else {
      comp_load_count += (iloop * oloop);
    }
  }

  double load_size = mul_prod * cache_it->interior_shape.sizes_product_bytes();
  if (load_size > static_cast<double>(reg_size)) {
    return;
  }

  // the cost could be negative
  double cache_load_cost = (inner_loop * mul_prod) * (gmem_lat + reg_lat)  //
                           - inner_loop * (gmem_lat + lmem_lat);
  double comp_load_benefit = comp_load_count * (lmem_lat - reg_lat);
  if (comp_load_benefit > cache_load_cost) {
    PartialCacheInRegister(parent, cache, comps, cache_it->from, multiple);
  }
}

static void BlocksForRegisterCache(Block* parent, Block* cache, const Location& local_loc,  //
                                   const Location& reg_log, size_t reg_size, size_t gmem_lat, size_t lmem_lat,
                                   size_t reg_lat) {
  std::vector<Block*> comps;
  for (auto stmt : parent->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner && !inner->has_tag("cache")) {
      comps.push_back(inner.get());
    }
  }
  CacheRefInRegister(parent, cache, comps, reg_size, gmem_lat, lmem_lat, reg_lat);
}

static void RegisterCacheRecurse(Block* parent, Block* block, const Tags& reqs,       //
                                 const Location& local_loc, const Location& reg_loc,  //
                                 size_t reg_size, size_t gmem_lat, size_t lmem_lat, size_t reg_lat) {
  if (block->has_tags(reqs)) {
    BlocksForRegisterCache(parent, block, local_loc, reg_loc, reg_size, gmem_lat, lmem_lat, reg_lat);
  } else {
    for (auto& stmt : block->stmts) {
      auto inner = stripe::Block::Downcast(stmt);
      if (inner) {
        RegisterCacheRecurse(block, inner.get(), reqs, local_loc, reg_loc,  //
                             reg_size, gmem_lat, lmem_lat, reg_lat);
      }
    }
  }
}

void RegisterCachePass(Block* root, const proto::RegisterPass& options) {
  auto reqs = FromProto(options.reqs());
  auto local_loc = stripe::FromProto(options.local_loc());
  auto reg_loc = stripe::FromProto(options.register_loc());
  auto reg_size = options.register_size();
  auto gmem_lat = options.global_memory_latency();
  auto lmem_lat = options.local_memory_latency();
  auto reg_lat = options.register_latency();
  RegisterCacheRecurse(nullptr, root, reqs, local_loc, reg_loc,  //
                       reg_size, gmem_lat, lmem_lat, reg_lat);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
