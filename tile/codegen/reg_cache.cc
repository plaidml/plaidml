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
static bool OuterInnerLoopCount(Block* outer, size_t* outer_loop, size_t* inner_loop) {
  *outer_loop = outer->idxs_product();
  Block* inner = outer->SubBlock(0).get();
  *inner_loop = inner->idxs_product();
  return true;
}

// Propagate the location, offset, and stride information recursively
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

static void AdjustRefAccessHelper(Block* outer, Refinement* outer_ref,  //
                                  Block* inner, bool adjust_all) {
  // Adjust the refinement in inner
  // For the cached refinement, the access must be exact the corresponding index.
  // Only the cache block set adjust_all == true, and then adjust other refinements.
  // In this case, the access are the corresponding index with the coefficient
  // same as the outer index range
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
  // Adjust the refinement in outer
  // Clear the accesses in block index to zero
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

static void AdjustRefAccess(Block* outer, const std::string& ref_name, bool adjust_all) {
  auto it = outer->ref_by_from(ref_name);
  for (auto stmt : outer->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      AdjustRefAccessHelper(outer, &(*it), inner.get(), adjust_all);
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
static void PartialCacheInRegister(Block* parent, Block* comp_parent, Block* cache,  //
                                   const std::vector<Block*>& comps,                 //
                                   bool cache_index_order,                           //
                                   const std::string& ref_name,                      //
                                   const std::map<std::string, Rational>& multiple) {
  // Note that parent is cache block's parent, probably not be computational block's parent
  auto parent_ref_it = parent->ref_by_into(ref_name);
  auto cache_inner = cache->SubBlock(0);

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

  // Adjust the coffeicents in the constraints of cache inner block
  for (auto& cons : cache_inner->constraints) {
    for (auto& kvp : cons.mutateMap()) {
      if (kvp.first == "") {
        continue;
      }
      Index* inner_idx = cache_inner->idx_by_name(kvp.first);
      if (inner_idx->affine == Affine()) {
        kvp.second = ToInteger(kvp.second / multiple.at(inner_idx->name));
      } else {
        const auto& imap = inner_idx->affine.getMap();
        if (imap.size() > 1) {
          throw std::runtime_error("Complex affine in cache block");
        }
        if (imap.begin()->first != "") {
          Index* outer_idx = cache->idx_by_name(imap.begin()->first);
          if (outer_idx->affine == Affine()) {
            Rational m = multiple.at(outer_idx->name);
            if (m >= 1) {
              kvp.second = ToInteger(kvp.second * m);
            } else {
              // The multiple m == 1/n. So other coefficients times n.
              kvp.second = (kvp.second > 0) ? 1 : ((kvp.second < 0) ? -1 : 0);
              for (auto& k : cons.mutateMap()) {
                if (kvp.first != k.first) {
                  k.second = ToInteger(k.second / m);
                }
              }
            }
          }
        }
      }
    }
  }

  // Reorder the idx in cache block or comp block
  if (cache_index_order) {
    // Adjust comp blocks' index order according to cache block
    for (auto& comp : comps) {
      ReorderIndex(cache, comp, ref_name);
      comp->set_tag("ordered_idx");
    }
    // In this case we need to mark as "reg_cache" in order to
    // tell the code emitter
    cache->set_tag("reg_cache");
  } else {
    // Adjust cache block's index order according to comp block
    ReorderIndex(comps[0], cache, ref_name);
  }

  AdjustRefAccess(cache, ref_name, true);
  for (const auto& comp : comps) {
    AdjustRefAccess(comp, ref_name, false);
  }

  // Stride changed. So need to fix all ref's interior size
  for (auto& cache_ref : cache->refs) {
    const auto& inner_ref = cache_inner->ref_by_from(cache_ref.into);
    cache_ref.interior_shape = cache_inner->exterior_shape(inner_ref->into);
    // For the cached refinement, we should change both interior
    // and exterior shapes. So we need to change the stripe as well.
    // For other refinement, we just change the interior shape.
    if (cache_ref.from == ref_name) {
      size_t n_dims = cache_ref.interior_shape.dims.size();
      std::vector<size_t> sizes(n_dims);
      for (size_t i = 0; i < n_dims; ++i) {
        sizes[i] = cache_ref.interior_shape.dims[i].size;
      }
      parent_ref_it->interior_shape =                      //
          SimpleShape(parent_ref_it->interior_shape.type,  //
                      sizes, parent_ref_it->interior_shape.layout);
    }
  }
  bool first_comp = true;
  for (auto& comp : comps) {
    auto comp_inner = comp->SubBlock(0);
    auto comp_ref_it = comp->ref_by_from(ref_name);
    const auto& inner_ref = comp_inner->ref_by_from(comp_ref_it->into);
    comp_ref_it->interior_shape = comp_inner->exterior_shape(inner_ref->into);
    if (first_comp && comp_parent != parent) {
      first_comp = false;
      auto comp_parent_ref_it = comp_parent->ref_by_into(ref_name);
      comp_parent_ref_it->interior_shape = comp->exterior_shape(comp_ref_it->into);
    }
  }

  parent_ref_it->location.devs[0].name = "REGISTER";
  PropagateRefLoc(parent, *parent_ref_it);
}

// Try to decide if we should load the ref in cache into registers
static bool CacheRefInRegister(Block* parent, Block* comp_parent, Block* cache,  //
                               const std::vector<Block*>& comps, const RegisterPassOptions& opt) {
  std::vector<Refinement>::const_iterator cache_ref_it;
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
  const std::string& ref_name = cache_ref_it->from;

  // Make sure each access of the refinement is single index, not affine
  for (const auto& comp : comps) {
    const auto& comp_ref_it = comp->ref_by_from(ref_name);
    for (const auto& aff : comp_ref_it->access) {
      if (aff != Affine() && aff.getMap().size() > 1) {
        return false;
      }
    }
    // If the index order is based on the cache block,
    // and the index order in comp block can't be change,
    // we need to check if the the index order between cache block
    // and comp block are consistent. If not consistent, give up
    if (opt.cache_index_order && comp->has_tag("ordered_idx")) {
      if (!ConsistentIdxOrder(cache, *cache_ref_it, comp, *comp_ref_it)) {
        return false;
      }
    }
  }

  // The ref dim in all computation block should be same
  const auto& comp0_ref_it = comps[0]->ref_by_from(ref_name);
  if (comp0_ref_it->access.size() != cache_ref_it->access.size()) {
    return false;
  }
  for (size_t i = 1; i < comps.size(); ++i) {
    const auto& comp_ref_it = comps[i]->ref_by_from(ref_name);
    if (comp_ref_it->access != comp0_ref_it->access) {
      return false;
    }
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
  const auto& cache_inner_ref_it = cache_inner->ref_by_from(cache_ref_it->into);
  for (size_t i = 0; i < cache_ref_it->access.size(); ++i) {
    const auto& cache_ref_aff = cache_ref_it->access[i];
    const auto& comp0_ref_aff = comp0_ref_it->access[i];
    if (cache_ref_aff == Affine()) {
      if (comp0_ref_aff == Affine()) {
        continue;
      } else {
        return false;
      }
    }
    if (cache_ref_aff.getMap().size() != 1 || comp0_ref_aff.getMap().size() != 1) {
      return false;
    }

    const auto& cache_inner_aff = cache_inner_ref_it->access[i];
    Index* cache_idx = cache->idx_by_name(cache_ref_aff.getMap().begin()->first);
    Index* comp0_idx = comps[0]->idx_by_name(comp0_ref_aff.getMap().begin()->first);
    Index* cache_inner_idx = cache_inner->idx_by_name(cache_inner_aff.getMap().begin()->first);
    // The index must be real loop, not affine
    if (cache_idx->affine != Affine() || comp0_idx->affine != Affine() || cache_inner_idx->affine != Affine()) {
      return false;
    }

    // For each dim, make sure the total range (inner*outer) of cache block is
    // divisible by comp0_idx's range
    size_t cache_total_range = cache_idx->range * cache_inner_idx->range;
    // The first condition is for cache_total_range == 1
    if (comp0_idx->range > cache_total_range || cache_total_range % comp0_idx->range > 0) {
      return false;
    }

    Rational times = cache_idx->range;
    // If not divisible, it's hard to transform, then give up.
    if (cache_idx->range % comp0_idx->range > 0 && comp0_idx->range % cache_idx->range > 0) {
      return false;
    }
    times /= comp0_idx->range;
    mul_prod *= static_cast<double>(times);
    multiple.emplace(cache_idx->name, times);
  }

  // Now we compute the load count of the refinement elements in the computational block
  size_t comp_load_count = 0;
  for (const auto& comp : comps) {
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

  if (comp_load_benefit <= cache_load_cost) {
    return false;
  }

  // Now it's better to load the refinement into registers
  PartialCacheInRegister(parent, comp_parent, cache, comps, opt.cache_index_order, ref_name, multiple);
  return true;
}

static void CacheWholeRefInRegister(Block* parent, Block* cache, const std::vector<Block*>& comps,  //
                                    const RegisterPassOptions& opt) {
  std::vector<Refinement>::const_iterator cache_ref_it;
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
  for (const auto& comp : comps) {
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
  }

  double times = 1.0 * opt.align_size / type_size;
  double glmem_lat = (opt.gmem_lat + (times - 1) * opt.lmem_lat) / times;
  double cost = inner_loop * outer_loop * (glmem_lat + opt.reg_lat) -  //
                inner_loop * (glmem_lat + opt.lmem_lat);
  double benefit = comp_load_count * (opt.lmem_lat - opt.reg_lat);

  if (benefit > cost) {
    // Load the whole buffer into register
    cache->tags.erase("gpu_thread");
    auto parent_ref_it = parent->ref_by_into(cache_ref_it->from);
    parent_ref_it->location.devs[0].name = "REGISTER";
    PropagateRefLoc(parent, *parent_ref_it);
  }
}

static void BlocksForRegisterCache(Block* parent, Block* cache, const RegisterPassOptions& opt) {
  std::vector<Block*> comps;
  Block* comp_parent;
  if (parent->has_tag(opt.comp_parent_tag)) {
    comp_parent = parent;
  } else if (parent->SubBlock(0)->has_tag(opt.comp_parent_tag)) {
    comp_parent = parent->SubBlock(0).get();
  } else {
    return;
  }

  std::string ref_name;
  if (opt.dir == RefDir::In) {
    ref_name = cache->ref_by_into("dst")->from;
  } else if (opt.dir == RefDir::Out) {
    ref_name = cache->ref_by_into("src")->from;
  } else {
    throw std::runtime_error("Invalid direction for caching into registers.");
  }

  for (auto stmt : comp_parent->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner && !inner->has_tag("cache")) {
      // Sometimes inner doesn't have ref, ignore it
      if (inner->ref_by_from(ref_name, false) != inner->refs.end()) {
        comps.push_back(inner.get());
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

  if (comps.size() > 0) {
    bool ret = CacheRefInRegister(parent, comp_parent, cache, comps, opt);
    // It is not safe to let cache store be in registers
    if (!ret && opt.dir == RefDir::In) {
      CacheWholeRefInRegister(parent, cache, comps, opt);
    }
  }
}

static void RegisterCacheRecurse(Block* parent, Block* block,  //
                                 const Tags& reqs,             //
                                 const RegisterPassOptions& opt) {
  if (block->has_tags(reqs)) {
    BlocksForRegisterCache(parent, block, opt);
  } else {
    for (auto& stmt : block->stmts) {
      auto inner = stripe::Block::Downcast(stmt);
      if (inner) {
        RegisterCacheRecurse(block, inner.get(), reqs, opt);
      }
    }
  }
}

void RegisterCachePass(Block* root, const proto::RegisterPass& options) {
  RegisterPassOptions opt;
  auto reqs = FromProto(options.reqs());
  opt.local_loc = stripe::FromProto(options.local_loc());
  opt.reg_loc = stripe::FromProto(options.register_loc());
  opt.reg_size = options.register_size();
  opt.gmem_lat = options.global_memory_latency();
  opt.lmem_lat = options.local_memory_latency();
  opt.reg_lat = options.register_latency();
  opt.dir = stripe::FromProto(static_cast<stripe::proto::Refinement::Dir>(options.dir()));
  opt.comp_parent_tag = options.comp_parent_tag();
  opt.cache_index_order = options.index_order() == "cache";
  opt.align_size = options.align_size();
  RegisterCacheRecurse(nullptr, root, reqs, opt);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
