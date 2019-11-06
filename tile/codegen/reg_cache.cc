// Copyright 2018, Intel Corporation

#include "tile/codegen/reg_cache.h"

#include "base/util/throw.h"
#include "tile/math/bignum.h"
#include "tile/stripe/stripe.h"
#include "tile/codegen/alias.h"
#include "tile/codegen/idx_order.h"

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

// Get the outer and inner (outer's first sub-block) loop count
void OuterInnerLoopCount(Block* outer, size_t* outer_loop, size_t* inner_loop) {
  *outer_loop = outer->idxs_product();
  Block* inner = outer->SubBlock(0).get();
  *inner_loop = inner->idxs_product();
}

void ClearAccesses(const Refinement& ref) {
  for (auto& access : ref.mut().access) {
    access = Affine();
  }
}

std::map<std::string, size_t> RefinementIndex(Block* block, const Refinement& ref) {
  std::map<std::string, size_t> result;
  for (auto& access : ref.access) {
    auto& acc_map = access.getMap();
    for (auto& kvp : acc_map) {
      if (kvp.first.size() > 0) {
        Index* idx = block->idx_by_name(kvp.first);
        result.emplace(kvp.first, idx->range);
      }
    }
  }
  return result;
}

size_t IndexProduct(const std::map<std::string, size_t>& idxs) {
  size_t result = 1;
  for (auto& idx : idxs) {
    result *= idx.second;
  }
  return result;
}

void FixRefLoc(const Refinement& src, const Refinement& dst) {
  dst.mut().location = src.location;
  dst.mut().offset = src.offset;
  for (size_t i = 0; i < dst.interior_shape.dims.size(); i++) {
    dst.mut().interior_shape.dims[i].stride = src.interior_shape.dims[i].stride;
  }
}

// Propagate the location, offset, and stride information recursively
void PropagateRefLoc(Block* block, const Refinement& outer_ref) {
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

// Copy index from src to dst except for wo
std::set<std::string> CopyIndexWithout(Block* dst, Block* src,
                                       const std::map<std::string, size_t>& valid_idxs,
                                       const std::set<std::string>& invalid_idxs) {
  dst->idxs.clear();
  std::set<std::string> removed;
  for (auto& idx : src->idxs) {
    if (idx.affine == Affine()) {
      if (valid_idxs.find(idx.name) != valid_idxs.end()) {
        dst->idxs.push_back(idx);
      }
    }
    else {
      bool has_invalid_idx = false;
      auto& acc_map = idx.affine.getMap();
      for (auto& kvp : acc_map) {
        if (kvp.first.size() > 0) {
          if  (invalid_idxs.find(kvp.first) != invalid_idxs.end()) {
            has_invalid_idx = true;
            break;
          }
        }
      }
      if (has_invalid_idx) {
        removed.insert(idx.name);
      }
      else {
        dst->idxs.push_back(idx);
      }
    }
  }
  return removed;
}

void CopyConstraints(Block* dst, Block* src) {
  dst->constraints.clear();
  for (auto& cons : src->constraints) {
    auto& acc_map = cons.getMap();
    bool valid = true;
    for (auto& kvp : acc_map) {
      if (kvp.first.size() > 0) {
        Index *idx = dst->idx_by_name(kvp.first);
        if (!idx) {
          valid = false;
          break;
        }
      }
    }
    if (valid) {
      dst->constraints.push_back(cons);
    }
  }
}

void CopyIndexConstraints(Block* dst, Block* dst_inner, Block * src,
                          Block* src_inner, Block* src_parent,
                          const std::map<std::string, size_t>& outer_idxs,
                          const std::map<std::string, size_t>& inner_idxs) {
  // Correct the affine index in cache block from computing block
  std::set<std::string> to_prune_idxs;
  for (auto& idx : src_parent->idxs) {
    if (idx.affine == Affine()) {
      to_prune_idxs.insert(idx.name);
    }
  }
  std::map<std::string, size_t> dummy_map;
  for (auto& idx : src->idxs) {
    dummy_map.emplace(idx.name, idx.range);
  }
  std::set<std::string> invalid_idxs = CopyIndexWithout(dst, src, dummy_map, to_prune_idxs);
  CopyIndexWithout(dst_inner, src_inner, inner_idxs, invalid_idxs);

  for (auto& idx : dst->idxs) {
    if (idx.affine == Affine()) {
      continue;
    }
    std::map<std::string, Affine> replacement;
    const auto& acc_map = idx.affine.getMap();
    for (const auto& kvp : acc_map) {
      if (kvp.first.size() > 0) {
        Index* to_prune_var = src_parent->idx_by_name(kvp.first);
        if (to_prune_var->affine != Affine()) {
          // The index should not in to_prune block
          replacement.emplace(kvp.first, to_prune_var->affine);
        }
      }
    }
    idx.affine.substitute(replacement);
  }

  // Remove the invalid constraints
  CopyConstraints(dst_inner, src_inner);
}

std::vector<Affine> AccessCoefficients(const std::vector<Affine>& ref_access) {
  std::vector<Affine> updated_access;
  for (auto& access : ref_access) {
    auto& acc_map = access.getMap();
    size_t num_idxs = 0;
    Affine new_access;
    for (auto& kvp : acc_map) { 
      if (kvp.first.size() > 0) {
        ++num_idxs;
        new_access = new_access + Affine(kvp.first);
      }
      else {
        new_access = new_access + Affine(kvp.second);
      }
    }
    // If there is no index or are more than one index in the access
    // so that we just keep this access
    updated_access.push_back(num_idxs == 1 ? new_access : access);
  }
  return updated_access;
}

bool CheckEltwiseAccess(Block* block, size_t n_dim, const Refinement& except) {
  for (size_t i = 0; i < n_dim; ++i) {
    Affine this_access;
    for (const auto& ref : block->refs) {
      if (ref.into() == except.into()) {
        continue;
      }
      if (i < n_dim - ref.access.size()) {
        continue;
      }
      size_t this_dim = i + ref.access.size() - n_dim;
      if (this_access == Affine()) {
        this_access = ref.access[this_dim];
      }
      else {
        if (ref.access[this_dim] != Affine() && this_access != ref.access[this_dim]) {
          return false;
        }
      }
    }
  }
  return true;
}

// Replace the cache block (global <-> local) with a new block (global <-> register).
// The new block's index and access reference the computing block.
bool ReferenceComputingBlock(const AliasMap& parent_map,          //
                             Block* parent, Block* comp_parent,   //
                             Block* cache, Block* comp,           //
                             const std::string& ref_name,         //
                             const RegisterPassOptions& opt) {
  auto cache_inner = cache->SubBlock(0);
  auto comp_inner = comp->SubBlock(0);

  const auto& comp_outer_local_ref = comp->ref_by_into(ref_name);
  const auto& comp_inner_local_ref = comp_inner->ref_by_into(comp_outer_local_ref->into());
  const auto& cache_outer_local_ref = cache->ref_by_from(ref_name);
  const auto& cache_inner_local_ref = cache_inner->ref_by_from(cache_outer_local_ref->into());
  size_t n_dim = comp_outer_local_ref->access.size();

  // Build the target register refinement shape
  // according to local refinement shape in computing block
  std::vector<Affine> comp_reg_access = AccessCoefficients(comp_inner_local_ref->access);
  Refinement tmp_reg_ref = *comp_inner_local_ref;
  tmp_reg_ref.access = comp_reg_access;
  std::vector<Extent> comp_reg_extents = tmp_reg_ref.Extents(comp_inner->idxs);
  std::vector<size_t> comp_reg_sizes;
  size_t load_size = byte_width(tmp_reg_ref.interior_shape.type);
  for (const auto& reg_ext : comp_reg_extents) {
    comp_reg_sizes.push_back(reg_ext.max);
    load_size *= (reg_ext.max - reg_ext.min);
  }
  auto comp_reg_shape = SimpleShape(tmp_reg_ref.interior_shape.type,
                                    comp_reg_sizes, tmp_reg_ref.interior_shape.layout);

  if (load_size > opt.reg_size) {
    return false;
  }

  if (cache->has_tag("eltwise") &&
     !(CheckEltwiseAccess(cache, n_dim, *cache_outer_local_ref) &&
       CheckEltwiseAccess(cache_inner.get(), n_dim, *cache_inner_local_ref))) {
    return false;
  }

  std::map<std::string, size_t> inner_used_idxs = RefinementIndex(comp_inner.get(), *comp_inner_local_ref);
  std::map<std::string, size_t> outer_used_idxs = RefinementIndex(comp, *comp_outer_local_ref);

  size_t cache_oloop;
  size_t cache_iloop;
  OuterInnerLoopCount(cache, &cache_oloop, &cache_iloop);
  size_t comp_oloop;
  size_t comp_iloop;
  OuterInnerLoopCount(comp, &comp_oloop, &comp_iloop);
  size_t new_cache_iloop = IndexProduct(inner_used_idxs);
  double benefit = opt.lmem_lat * comp_iloop * comp_oloop + (opt.lmem_lat + opt.gmem_lat) * cache_iloop * cache_oloop;
  double cost = opt.reg_lat * comp_iloop * comp_oloop + (opt.reg_lat + opt.gmem_lat) * new_cache_iloop * comp_oloop;
  if (benefit < cost) {
    return false;
  }

  // Determine the new shapes and accesses
  std::vector<Affine> new_inner_access = comp_inner_local_ref->access;
  std::vector<Affine> new_outer_access = comp_outer_local_ref->access;
  std::vector<size_t> new_cache_sizes(n_dim);
  for (size_t i = 0; i < n_dim; ++i) {
    new_cache_sizes[i] = comp_outer_local_ref->interior_shape.dims[i].size;
  }

  if (parent == comp_parent) {
    cache->idxs = comp->idxs;
    std::set<std::string> dummy;
    CopyIndexWithout(cache_inner.get(), comp_inner.get(), inner_used_idxs, dummy);
    CopyConstraints(cache_inner.get(), comp_inner.get());
  }
  else {
    CopyIndexConstraints(cache, cache_inner.get(), comp, comp_inner.get(),
        comp_parent, outer_used_idxs, inner_used_idxs);
  }

  // New register refinements in cache block
  std::string rref_short_name = opt.dir == RefDir::In ? "dst" : "src";
  RefDir rref_dir = opt.dir == RefDir::In ? RefDir::Out : RefDir::In;
  Refinement cache_inner_reg_ref = comp_inner_local_ref->WithInto(rref_short_name);
  Refinement cache_outer_reg_ref = comp_outer_local_ref->WithInto(rref_short_name);
  cache_inner_reg_ref.interior_shape = comp_reg_shape;
  cache_inner_reg_ref.location.devs[0].name = "REGISTER";
  cache_inner_reg_ref.access = comp_reg_access;
  cache_inner_reg_ref.from = rref_short_name;
  cache_inner_reg_ref.dir = rref_dir;
  cache_inner_reg_ref.agg_op = "";
  for (auto& dim : cache_inner_reg_ref.interior_shape.dims) {
    dim.size = 1;
  }
  cache_inner->refs.erase(cache_inner_local_ref);
  cache_inner->refs.insert(cache_inner_reg_ref);
  cache_outer_reg_ref.interior_shape = comp_reg_shape;
  ClearAccesses(cache_outer_reg_ref);
  cache_outer_reg_ref.location.devs[0].name = "REGISTER";
  cache_outer_reg_ref.dir = rref_dir;
  cache_outer_reg_ref.agg_op = "";
  cache->refs.erase(cache_outer_local_ref);
  cache->refs.insert(cache_outer_reg_ref);
  cache->set_tag("reg_cache_partial");

  // Correct the refinements except for the register refinement
  // Note that there may be multiple refinements except for the register refinement,
  // especially for the store cache merged with the following element-wise operations
  // Assign the shapes and accesses
  for (const auto& inner_ref : cache_inner->refs) {
    if (IsRegisterRef(inner_ref)) {
      continue;
    }
    const auto& outer_ref = cache->ref_by_into(inner_ref.from);
    // Change the accesses
    auto& inner_access = inner_ref.mut().access;
    size_t inner_access_dim = inner_access.size();
    for (size_t i = 0; i < inner_access_dim; ++i) {
      if (inner_access[i] != Affine()) {
        inner_access[i] = new_inner_access[n_dim - inner_access_dim + i];
      }
    }
    auto& outer_access = outer_ref->mut().access;
    size_t outer_access_dim = outer_access.size();
    for (size_t i = 0; i < outer_access_dim; ++i) {
      if (outer_access[i] != Affine()) {
        outer_access[i] = new_outer_access[n_dim - outer_access_dim + i];
      }
    }
    // Change the sizes, do not have to change the strides
    auto& inner_dims = inner_ref.mut().interior_shape.dims;
    size_t inner_dims_size = inner_dims.size();
    for (size_t i = 0; i < inner_dims_size; ++i) {
      inner_dims[i].size = 1;
    }
    auto& outer_dims = outer_ref->mut().interior_shape.dims;
    size_t outer_dims_size = outer_dims.size();
    for (size_t i = 0; i < outer_dims_size; ++i) {
      outer_dims[i].size = outer_access[i] == Affine() ?
          1 : new_cache_sizes[n_dim - outer_dims_size + i];
    }
  }

  // New register refinements in computing block
  Refinement comp_inner_reg_ref = *comp_inner_local_ref;
  Refinement comp_outer_reg_ref = *comp_outer_local_ref;
  comp_inner_reg_ref.interior_shape = comp_reg_shape;
  for (auto& dim : comp_inner_reg_ref.interior_shape.dims) {
    dim.size = 1;
  }
  comp_inner_reg_ref.location.devs[0].name = "REGISTER";
  comp_inner_reg_ref.access = comp_reg_access;
  comp_inner->refs.erase(comp_inner_local_ref);
  comp_inner->refs.insert(comp_inner_reg_ref);
  comp_outer_reg_ref.interior_shape = comp_reg_shape;
  ClearAccesses(comp_outer_reg_ref);
  comp_outer_reg_ref.location.devs[0].name = "REGISTER";
  comp->refs.erase(comp_outer_local_ref);
  comp->refs.insert(comp_outer_reg_ref);

  // Add the register refinement at parent level
  const auto& parent_local_ref = parent->ref_by_into(ref_name);
  Refinement parent_reg_ref = *parent_local_ref;
  parent_reg_ref.location.devs[0].name = "REGISTER";
  parent_reg_ref.interior_shape = comp_reg_shape;
  parent->refs.erase(parent_local_ref);
  parent->refs.insert(parent_reg_ref);
  PropagateRefLoc(parent, parent_reg_ref);

  return true;
}

// Replace the local refinements in place with register refinements
// in cache block and computing block
bool ReplaceLocalRefinement(const AliasMap& parent_map,          //
                            Block* parent, Block* comp_parent,   //
                            Block* cache, Block* comp,           //
                            const std::string& ref_name,         //
                            const RegisterPassOptions& opt) {
  // Here the target refinement has same_access tag, which means
  // the index in cache block and access of the refinement are
  // consisitent with that in computing block

  auto cache_inner = cache->SubBlock(0);
  auto comp_inner = comp->SubBlock(0);

  const auto& comp_outer_local_ref = comp->ref_by_into(ref_name);
  const auto& comp_inner_local_ref = comp_inner->ref_by_into(comp_outer_local_ref->into());
  const auto& cache_outer_local_ref = cache->ref_by_from(ref_name);
  const auto& cache_inner_local_ref = cache_inner->ref_by_from(cache_outer_local_ref->into());
  size_t n_dim = cache_outer_local_ref->access.size();

  std::map<std::string, size_t> inner_used_idxs = RefinementIndex(comp_inner.get(), *comp_inner_local_ref);
  std::map<std::string, size_t> outer_used_idxs = RefinementIndex(comp, *comp_outer_local_ref);

  // Build the target register refinement shape
  // according to local refinement shape in computing block
  std::vector<Affine> comp_reg_access = AccessCoefficients(comp_inner_local_ref->access);
  Refinement tmp_reg_ref = *comp_inner_local_ref;
  tmp_reg_ref.access = comp_reg_access;
  std::vector<Extent> comp_reg_extents = tmp_reg_ref.Extents(comp_inner->idxs);
  std::vector<size_t> comp_reg_sizes;
  size_t load_size = byte_width(tmp_reg_ref.interior_shape.type);
  for (const auto& reg_ext : comp_reg_extents) {
    comp_reg_sizes.push_back(reg_ext.max);
    load_size *= (reg_ext.max - reg_ext.min);
  }
  auto comp_reg_shape = SimpleShape(tmp_reg_ref.interior_shape.type,
                                    comp_reg_sizes, tmp_reg_ref.interior_shape.layout);

  if (load_size > opt.reg_size) {
    return false;
  }

  size_t cache_oloop;
  size_t cache_iloop;
  OuterInnerLoopCount(cache, &cache_oloop, &cache_iloop);
  size_t comp_oloop;
  size_t comp_iloop;
  OuterInnerLoopCount(comp, &comp_oloop, &comp_iloop);
  size_t new_cache_iloop = IndexProduct(inner_used_idxs);
  double benefit = opt.lmem_lat * comp_iloop * comp_oloop + (opt.lmem_lat + opt.gmem_lat) * cache_iloop * cache_oloop;
  double cost = opt.reg_lat * comp_iloop * comp_oloop + (opt.reg_lat + opt.gmem_lat) * new_cache_iloop * comp_oloop;
  if (benefit < cost) {
    return false;
  }

  if (cache->has_tag("eltwise") &&
     !(CheckEltwiseAccess(cache, n_dim, *cache_outer_local_ref) &&
       CheckEltwiseAccess(cache_inner.get(), n_dim, *cache_inner_local_ref))) {
    return false;
  }

  // Before index conversion, compute the index multiple before and after conversion
  std::map<std::string, Rational> multiple;
  for (auto& old_idx : cache->idxs) {
    Index* new_idx = comp->idx_by_name(old_idx.name);
    Rational new_idx_range = new_idx ? new_idx->range : 1;
    multiple.emplace(old_idx.name, new_idx_range / old_idx.range);
  }

  if (parent == comp_parent) {
    cache->idxs = comp->idxs;
    std::set<std::string> dummy;
    CopyIndexWithout(cache_inner.get(), comp_inner.get(), inner_used_idxs, dummy);
    CopyConstraints(cache_inner.get(), comp_inner.get());
  }
  else {
    CopyIndexConstraints(cache, cache_inner.get(), comp, comp_inner.get(),
        comp_parent, outer_used_idxs, inner_used_idxs);
  }

  // New register refinements in cache block
  std::string rref_short_name = opt.dir == RefDir::In ? "dst" : "src";
  RefDir rref_dir = opt.dir == RefDir::In ? RefDir::Out : RefDir::In;
  Refinement cache_inner_reg_ref = comp_inner_local_ref->WithInto(rref_short_name);
  Refinement cache_outer_reg_ref = comp_outer_local_ref->WithInto(rref_short_name);
  cache_inner_reg_ref.interior_shape = comp_reg_shape;
  cache_inner_reg_ref.access = comp_reg_access;
  cache_inner_reg_ref.location.devs[0].name = "REGISTER";
  cache_inner_reg_ref.from = rref_short_name;
  cache_inner_reg_ref.dir = rref_dir;
  for (auto& dim : cache_inner_reg_ref.interior_shape.dims) {
    dim.size = 1;
  }   
  cache_inner->refs.erase(cache_inner_local_ref);
  cache_inner->refs.insert(cache_inner_reg_ref);
  cache_outer_reg_ref.interior_shape = comp_reg_shape;
  ClearAccesses(cache_outer_reg_ref); 
  cache_outer_reg_ref.location.devs[0].name = "REGISTER";
  cache_outer_reg_ref.dir = rref_dir;
  cache->refs.erase(cache_outer_local_ref);
  cache->refs.insert(cache_outer_reg_ref);
  cache->set_tag("reg_cache_partial");

  // Correct refinements' accesses and shapes except for the register refinement
  for (const auto& inner_ref : cache_inner->refs) {
    if (IsRegisterRef(inner_ref)) {
      continue;
    }
    const auto& outer_ref = cache->ref_by_into(inner_ref.from);
    for (auto& access : inner_ref.mut().access) {
      auto& acc_map = access.mutateMap();
      for (auto& kvp : acc_map) {
        if (kvp.first.size() > 0) {
          kvp.second = ToInteger(kvp.second * multiple.at(kvp.first));
        }
      }
    }
    std::vector<Extent> extents = inner_ref.Extents(cache_inner->idxs);
    for (size_t i = 0; i < extents.size(); ++i) {
      outer_ref->mut().interior_shape.dims[i].size = extents[i].max;
    }
  }

  // New register refinements in computing block
  Refinement comp_inner_reg_ref = *comp_inner_local_ref;
  Refinement comp_outer_reg_ref = *comp_outer_local_ref;
  comp_inner_reg_ref.interior_shape = comp_reg_shape;
  for (auto& dim : comp_inner_reg_ref.interior_shape.dims) {
    dim.size = 1;
  }
  comp_inner_reg_ref.location.devs[0].name = "REGISTER";
  comp_inner_reg_ref.access = comp_reg_access;
  comp_inner->refs.erase(comp_inner_local_ref);
  comp_inner->refs.insert(comp_inner_reg_ref);
  comp_outer_reg_ref.interior_shape = comp_reg_shape;
  ClearAccesses(comp_outer_reg_ref);
  comp_outer_reg_ref.location.devs[0].name = "REGISTER";
  comp->refs.erase(comp_outer_local_ref);
  comp->refs.insert(comp_outer_reg_ref);

  // Add the register refinement at parent level
  const auto& parent_local_ref = parent->ref_by_into(ref_name);
  Refinement parent_reg_ref = *parent_local_ref;
  parent_reg_ref.location.devs[0].name = "REGISTER";
  parent_reg_ref.interior_shape = comp_reg_shape;
  parent->refs.erase(parent_local_ref);
  parent->refs.insert(parent_reg_ref);
  PropagateRefLoc(parent, parent_reg_ref);

  return true;
}

// Append a new block of register<->global after the original block of local<->global
bool AppendRegCacheBlock(const AliasMap& parent_map,          //
                         Block* parent, Block* comp_parent,   //
                         Block* orig_cache, Block* comp,      //
                         const std::string& ref_name,         //
                         const RegisterPassOptions& opt) {
  auto comp_inner = comp->SubBlock(0);

  const auto& comp_outer_local_ref = comp->ref_by_into(ref_name);
  const auto& comp_inner_local_ref = comp_inner->ref_by_into(comp_outer_local_ref->into());
  size_t n_dim = comp_outer_local_ref->access.size();

  // Build the target register refinement shape
  // according to local refinement shape in computing block
  std::vector<Affine> comp_reg_access = AccessCoefficients(comp_inner_local_ref->access);
  Refinement tmp_reg_ref = *comp_inner_local_ref;
  tmp_reg_ref.access = comp_reg_access;
  std::vector<Extent> comp_reg_extents = tmp_reg_ref.Extents(comp_inner->idxs);
  std::vector<size_t> comp_reg_sizes;
  size_t load_size = byte_width(tmp_reg_ref.interior_shape.type);
  for (const auto& reg_ext : comp_reg_extents) {
    comp_reg_sizes.push_back(reg_ext.max);
    load_size *= (reg_ext.max - reg_ext.min);
  }
  auto comp_reg_shape = SimpleShape(tmp_reg_ref.interior_shape.type,
                                    comp_reg_sizes, tmp_reg_ref.interior_shape.layout);

  if (load_size > opt.reg_size) {
    return false;
  }

  // Build a new cache block
  auto cache = std::make_shared<Block>();
  auto cache_inner = std::make_shared<Block>();
  cache->set_tags({"cache", "cache_outer", "gpu_thread", "reg_cache_partial"});
  cache->set_tag(opt.dir == RefDir::In ? "cache_load" : "cache_store");
  cache_inner->set_tags({"cache_threads", "inline"});

  std::map<std::string, size_t> inner_used_idxs = RefinementIndex(comp_inner.get(), *comp_inner_local_ref);
  std::map<std::string, size_t> outer_used_idxs = RefinementIndex(comp, *comp_outer_local_ref);

  // Determine the new shapes and accesses
  std::vector<Affine> new_inner_access = comp_inner_local_ref->access;
  std::vector<Affine> new_outer_access = comp_outer_local_ref->access;
  std::vector<size_t> new_cache_sizes(n_dim);
  for (size_t i = 0; i < n_dim; ++i) {
    new_cache_sizes[i] = comp_outer_local_ref->interior_shape.dims[i].size;
  }

  if (parent == comp_parent) {
    cache->idxs = comp->idxs;
    std::set<std::string> dummy;
    CopyIndexWithout(cache_inner.get(), comp_inner.get(), inner_used_idxs, dummy);
    CopyConstraints(cache_inner.get(), comp_inner.get());
  }
  else {
    CopyIndexConstraints(cache.get(), cache_inner.get(), comp, comp_inner.get(),
        comp_parent, outer_used_idxs, inner_used_idxs);
  }

  // Register refinements in new cache blocks
  std::string rref_short_name = opt.dir == RefDir::In ? "dst" : "src";
  RefDir rref_dir = opt.dir == RefDir::In ? RefDir::Out : RefDir::In;
  Refinement cache_inner_reg_ref = comp_inner_local_ref->WithInto(rref_short_name);
  Refinement cache_outer_reg_ref = comp_outer_local_ref->WithInto(rref_short_name);
  cache_inner_reg_ref.interior_shape = comp_reg_shape;
  cache_inner_reg_ref.location.devs[0].name = "REGISTER";
  cache_inner_reg_ref.access = comp_reg_access;
  cache_inner_reg_ref.from = rref_short_name;
  cache_inner_reg_ref.dir = rref_dir;
  cache_inner_reg_ref.agg_op = "";
  for (auto& dim : cache_inner_reg_ref.interior_shape.dims) {
    dim.size = 1;
  }
  cache_inner->refs.insert(cache_inner_reg_ref);
  cache_outer_reg_ref.interior_shape = comp_reg_shape;
  ClearAccesses(cache_outer_reg_ref);
  cache_outer_reg_ref.location.devs[0].name = "REGISTER";
  cache_outer_reg_ref.dir = rref_dir;
  cache_outer_reg_ref.agg_op = "";
  cache->refs.insert(cache_outer_reg_ref);

  // Change the local refinement in the original cache block
  std::string local_ref_name = ref_name + "_local";
  auto orig_cache_outer_local_ref = orig_cache->ref_by_from(ref_name);
  orig_cache_outer_local_ref->mut().from = local_ref_name;

  // Local refinements in new cache blocks
  std::string local_ref_dir = opt.dir == RefDir::In ? "src" : "dst";
  Refinement cache_inner_local_ref = comp_inner_local_ref->WithInto(local_ref_dir);
  Refinement cache_outer_local_ref = comp_outer_local_ref->WithInto(local_ref_dir);
  cache_inner_local_ref.from = cache_outer_local_ref.into();
  cache_outer_local_ref.from = local_ref_name;
  cache_inner->refs.insert(cache_inner_local_ref);
  cache->refs.insert(cache_outer_local_ref);

  // Insert the load and store statements
  auto load = std::make_shared<Load>("src", "$X");
  cache_inner->stmts.push_back(load);
  auto store = std::make_shared<Store>("$X", "dst");
  cache_inner->stmts.push_back(store);

  // Insert the new cache block
  cache->stmts.push_back(cache_inner);
  InsertAfterBlock(parent, orig_cache, cache);

  // New register refinements in computing block
  Refinement comp_inner_reg_ref = *comp_inner_local_ref;
  Refinement comp_outer_reg_ref = *comp_outer_local_ref;
  comp_inner_reg_ref.interior_shape = comp_reg_shape;
  for (auto& dim : comp_inner_reg_ref.interior_shape.dims) {
    dim.size = 1;
  }
  comp_inner_reg_ref.location.devs[0].name = "REGISTER";
  comp_inner_reg_ref.access = comp_reg_access;
  comp_inner->refs.erase(comp_inner_local_ref);
  comp_inner->refs.insert(comp_inner_reg_ref);
  comp_outer_reg_ref.interior_shape = comp_reg_shape;
  ClearAccesses(comp_outer_reg_ref);
  comp_outer_reg_ref.location.devs[0].name = "REGISTER";
  comp->refs.erase(comp_outer_local_ref);
  comp->refs.insert(comp_outer_reg_ref);

  // Add the register refinement at parent level
  const auto& parent_local_ref = parent->ref_by_into(ref_name);
  Refinement parent_reg_ref = *parent_local_ref;
  Refinement new_parent_local_ref = parent_local_ref->WithInto(local_ref_name);
  parent->refs.erase(parent_local_ref);
  parent->refs.insert(new_parent_local_ref);
  parent_reg_ref.location.devs[0].name = "REGISTER";
  parent_reg_ref.interior_shape = comp_reg_shape;
  parent->refs.insert(parent_reg_ref);
  PropagateRefLoc(parent, parent_reg_ref);

  return true;
}

// Replace the block, loading from global memory and store into registers
bool CacheRefInRegister(const AliasMap& parent_map,          //
                        Block* parent, Block* comp_parent,   //
                        Block* cache, Block* comp,           //
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
  std::string ref_name = cache_ref_it->from;

  auto cache_inner = cache->SubBlock(0);
  const auto& cache_outer_local_ref = cache->ref_by_from(ref_name);
  const auto& cache_inner_local_ref = cache_inner->ref_by_from(cache_outer_local_ref->into());
  const auto cache_outer_global_ref = opt.dir == RefDir::In ? cache->ref_ins()[0] : cache->ref_outs(true)[0];
  const auto cache_inner_global_ref = cache_inner->ref_by_from(cache_outer_global_ref->into());

  bool optimized = false;

  if (cache_outer_local_ref->access == cache_outer_global_ref->access &&
      cache_inner_local_ref->access == cache_inner_global_ref->access) {
    // If global access and local access are same in the cache block,
    // replace the cache block referencing the computing block.
    // The global access and register access in the new cache block are
    // same as the original local access in the computing block.
    optimized = ReferenceComputingBlock(parent_map, parent, comp_parent, cache, comp, ref_name, opt);
  }

  if (!optimized && cache_outer_local_ref->has_tag("same_access")) {
    // If the cache block's local access is same as that in the computing block,
    // replace local refinement with register refinement and keep global access
    optimized = ReplaceLocalRefinement(parent_map, parent, comp_parent, cache, comp, ref_name, opt);
  }
/*
  if (!optimized) {
    optimized = AppendRegCacheBlock(parent_map, parent, comp_parent, cache, comp, ref_name, opt);
  }
*/
  return optimized;
}

void CacheWholeRefInRegister(Block* parent, Block* cache, Block* comp,  //
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
  OuterInnerLoopCount(cache, &outer_loop, &inner_loop);

  auto cache_inner = cache->SubBlock(0);
  size_t load_size = cache->exterior_shape(cache_ref_it->into()).byte_size();
  if (load_size > opt.reg_size) {
    return;
  }

  // Now we compute the load count of the refinement elements in the computational block
  size_t comp_load_count = 0;
  size_t iloop;
  size_t oloop;
  OuterInnerLoopCount(comp, &oloop, &iloop);

  // If the block is threaded, only count the inner loop
  if (comp->has_tag("gpu_thread")) {
    comp_load_count += iloop;
  } else {
    comp_load_count += (iloop * oloop);
  }

  double cost = inner_loop * outer_loop * (opt.lmem_lat + opt.reg_lat);
  double benefit = comp_load_count * (opt.lmem_lat - opt.reg_lat);

  if (benefit > cost) {
    // Add a block loading from local and caching into registers
    auto reg_cache = CloneBlock(*cache);
    reg_cache->remove_tag("gpu_thread");
    reg_cache->set_tag("reg_cache_whole");
    auto reg_cache_inner = reg_cache->SubBlock(0);
    reg_cache_inner->set_tag("reg_cache_whole");
    InsertAfterBlock(parent, cache, reg_cache);
    // Create a local refinement
    auto parent_ref_it = parent->ref_by_into(cache_ref_it->from);
    std::string local_ref = cache_ref_it->from + "_local";
    Refinement parent_local_ref = parent_ref_it->WithInto(local_ref);
    parent->refs.insert(parent_local_ref);
    // Rename the local refinement in the original cache block
    cache_ref_it->mut().from = local_ref;
    // Replace the source refinement from raw to local
    auto src_ref_it = reg_cache->ref_by_into("src");
    reg_cache->refs.erase(src_ref_it);
    Refinement new_src_ref = cache_ref_it->WithInto("src");
    reg_cache->refs.insert(new_src_ref);
    // Change the register refinement in the parent block
    parent_ref_it->mut().location.devs[0].name = "REGISTER";
    PropagateRefLoc(parent, *parent_ref_it);
  }
}

void BlocksForRegisterCache(const AliasMap& parent_map,
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

  // Find the parent of the computing block with comp_parent_tag
  if (parent->has_tag(opt.comp_parent_tag)) {
    comp_parent = parent;
  } else if (parent->SubBlock(0)->has_tag(opt.comp_parent_tag)) {
    comp_parent = parent->SubBlock(0).get();
  } else {
    IVLOG(1, "Cannot find the computing block.");
    return;
  }

  // Find the computing block inside comp_parent
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

  // Now we can start the caching if computing block and its parent exist
  if (comp) {
    CacheRefInRegister(parent_map, parent, comp_parent, cache, comp, opt);
  }
}

static void RegisterCacheRecurse(const AliasMap& parent_map,   //
                                 Block* parent, Block* block,  //
                                 const Tags& reqs,             //
                                 const RegisterPassOptions& opt) {
  if (block->has_tags(reqs)) {
    if (!block->has_tag("reg_cache_partial") && !block->has_tag("reg_cache_whole")) {
      BlocksForRegisterCache(parent_map, parent, block, opt);
    }
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
