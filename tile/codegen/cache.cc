// Copyright 2018, Intel Corporation

#include "tile/codegen/cache.h"
#include "tile/codegen/idx_order.h"

#include <algorithm>

#include <boost/format.hpp>

#include "base/util/stream_container.h"
#include "tile/codegen/localize.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

struct CacheVar {
  std::string idx;
  int32_t sign;   // 1/-1 for positive/negative coefficient
  int64_t coeff;  // index coefficient for global access, always positive
  size_t range;   // index range
  int64_t low;    // minimal value of sign*coeff*idx
  int64_t high;   // maximal value of sign*coeff*idx
  Affine access;  // new acess for local memory for idx
};

bool comp_coeff(const CacheVar& a, const CacheVar& b) {
  return a.coeff > b.coeff;
}

bool AllZeroAccess(const Refinement& ref) {
  for (const auto& access : ref.access) {
    if (access != Affine()) {
      return false;
    }
  }
  return true;
}

// Find out the minimal odd >= n
size_t NextOdd(size_t n) {
  return (n & 0x1) ? n : (n + 1);
}

// Test if outer contains inner
bool ContainBlock(Block* outer, Block* inner) {
  if (outer == inner) {
    return true;
  }
  for (const auto& stmt : outer->stmts) {
    auto block = Block::Downcast(stmt);
    if (block && ContainBlock(block.get(), inner)) {
      return true;
    }
  }
  return false;
}

void NewIndexIntoBlock(Block* block, const std::string& idx_name, size_t range,
                       std::map<std::string, size_t>* used_idxs) {
  if (used_idxs->find(idx_name) == used_idxs->end()) {
    block->idxs.push_back({idx_name, range});
    used_idxs->emplace(idx_name, range);
  }
  else {
    if (used_idxs->at(idx_name) != range) {
      throw std::runtime_error("Try to add duplicated index with different ranges.");
    }
  }
}

// Find the insertion position for the cache block
// The position is the sub-block of outer, which contains inner
// In/Out ref is inserted before/after the position
StatementIt InsertPosition(Block* outer, Block* inner) {
  for (auto stmt_it = outer->stmts.begin(); stmt_it != outer->stmts.end(); ++stmt_it) {
    auto block = Block::Downcast(*stmt_it);
    if (block && ContainBlock(block.get(), inner)) {
      return stmt_it;
    }
  }
  throw std::runtime_error("Cannot find the sub-block containing the inner block.");
}

// Fixup the refs between the outer block and the inner block
void FixupMiddleBlockRefs(Block* outer, Block* inner,
                          const std::string& var_name,
                          const std::string& raw_name,
                          const Tags& end_tags) {
  auto it = outer->ref_by_into(var_name, false);
  if (it == outer->refs.end()) {
    return;
  }
  for (auto stmt : outer->stmts) {
    auto sub = Block::Downcast(stmt);
    if (sub && sub.get() != inner && !sub->has_any_tags(end_tags)) {
      for (auto& ref : sub->refs) {
        if (ref.from == var_name) {
          if (AllZeroAccess(ref)) {
            ref.mut().location = it->location;
            ref.mut().offset = it->offset;
            ref.mut().access = it->access;
            ref.mut().interior_shape = it->interior_shape;
            FixupMiddleBlockRefs(sub.get(), inner, ref.into(), raw_name, end_tags);
          }
          else {
            ref.mut().from = raw_name;
          }
        }
      }
    }
  }
}

// Apply the cache pass if the reference block exists.
// The reference block, ref_block, indicates the access pattern
// that can be leveraged for optimizing the cache.
void ApplyCache(const AliasMap& alias_map,    //
                RefDir dir,                   //
                Block* ref_block,             //
                Block* outer_block,           //
                const std::string& var_name,  //
                const Location& mem_loc,      //
                const Location& xfer_loc,     //
                const Tags load_tags,         //
                const Tags store_tags,        //
                bool add_constraints,         //
                bool reorder_idx,             //
                bool odd_size,                //
                double odd_limit) {

  auto ref_it = ref_block->ref_by_from(var_name, false);
  if (ref_it == ref_block->refs.end()) {
    return;
  }

  size_t n_dim = ref_it->interior_shape.dims.size();
  // Collect the index in constraints. We can't merge these
  // index because they can't be eliminated
  std::set<std::string> cstr_vars;
  for (const auto& cons : ref_block->constraints) {
    for (auto& kvp : cons.getMap()) {
      if (kvp.first != "") {
        cstr_vars.insert(kvp.first);
      }
    }
  }

  std::vector<std::vector<CacheVar>> vars;
  std::set<size_t> keep_dims;
  // remain_const is the access' constant minus all index lows
  // i.e., the low of the access, usually zero, but could be positive
  std::vector<int64_t> remain_const;
  for (size_t k = 0; k < n_dim; ++k) {
    auto& access = ref_it->access[k];
    int64_t cst = 0;
    // Set up the initial index according to ref_block 
    std::vector<CacheVar> dim_vars;
    for (const auto& it : access.getMap()) {
      if (it.first == "") {
        cst += it.second;
        continue;
      }
      // If any index in this dim is used by the constraints,
      // do not change the dim
      if (cstr_vars.find(it.first) != cstr_vars.end()) {
        keep_dims.insert(k);
      }
      Index* idx = ref_block->idx_by_name(it.first);
      int64_t low = (it.second >= 0) ? 0 : (it.second * (idx->range - 1));
      int64_t high = (it.second >= 0) ? (it.second * (idx->range - 1)) : 0;
      CacheVar var = {it.first, (it.second >= 0) ? 1 : -1, std::abs(it.second), idx->range, low, high, Affine(it.first)};
      dim_vars.push_back(var);
      cst += low;
    }
    remain_const.push_back(cst);
    if (keep_dims.find(k) != keep_dims.end()) {
      vars.push_back(dim_vars);
      continue;
    }
    // Sort the index by their coefficients
    std::sort(dim_vars.begin(), dim_vars.end(), comp_coeff);

    // Traverse the variables to determine if they should be merged
    // dim_vars[i] and dim_vars[i + 1] should be merged if,
    // 1) one coefficient is the multiple of the other; and
    // 2) dim_vars[i] and dim_vars[i + 1] are overlapped for non-zero coefficients
    int i = static_cast<int>(dim_vars.size()) - 1;
    while (i > 0) {
      if (cstr_vars.find(dim_vars[i - 1].idx) == cstr_vars.end() &&
          cstr_vars.find(dim_vars[i].idx) == cstr_vars.end() &&
          dim_vars[i - 1].coeff % dim_vars[i].coeff == 0)
      {
        // This works for negative coefficients
        if (dim_vars[i].coeff * static_cast<int64_t>(dim_vars[i].range - 1) >= dim_vars[i - 1].coeff) {
          // Merge dim_vars[i] and dim_vars[i + 1]
          CacheVar new_var;
          new_var.idx = dim_vars[i - 1].idx + "_" + dim_vars[i].idx;
          // The new access includes the constant. So low is always zero
          new_var.low = 0;
          new_var.high = dim_vars[i - 1].high + dim_vars[i].high - dim_vars[i - 1].low - dim_vars[i].low;
          // Here dim_vars[i + 1].coeff is multiple of dim_vars[i].coeff
          new_var.coeff = dim_vars[i].coeff;
          new_var.sign = 1;
          new_var.range = (new_var.high - new_var.low) / new_var.coeff + 1;
          new_var.access = dim_vars[i - 1].access * (dim_vars[i - 1].sign * dim_vars[i - 1].coeff / new_var.coeff) +
                           dim_vars[i].access * (dim_vars[i].sign * dim_vars[i].coeff / new_var.coeff) +
                           Affine(- (dim_vars[i - 1].low + dim_vars[i].low) / new_var.coeff);
          dim_vars.erase(dim_vars.begin() + i);
          dim_vars[i - 1] = new_var;
        }
      }
      // Cannot merge
      --i;
    }
    vars.push_back(dim_vars);
  }

  // Make a base block for loading/storing
  Block xfer_block;
  xfer_block.location = xfer_loc;
  std::string raw_name = outer_block->unique_ref_name(var_name + "_raw");
  // Replace the old refinement to rename it.
  auto outer_ref_it = outer_block->ref_by_into(var_name);
  Refinement replacement_ref = outer_ref_it->WithInto(raw_name);
  outer_block->refs.erase(outer_ref_it);
  outer_ref_it = outer_block->refs.emplace(std::move(replacement_ref)).first;

  std::vector<size_t> local_sizes;
  std::vector<Affine> local_access;
  std::vector<Affine> global_access(n_dim);
  std::map<std::string, size_t> used_idxs;

  for (size_t k = 0; k < n_dim; ++k) {
    const auto& dim_vars = vars[k];
    // Insert the new index, and build the access for the local ref of cache block
    if (dim_vars.size() > 0) {
      if (keep_dims.find(k) == keep_dims.end()) {
        // Transform this dim
        for (size_t i = 0; i < dim_vars.size(); ++i) {
          const auto& var = dim_vars[i];
          NewIndexIntoBlock(&xfer_block, var.idx, var.range, &used_idxs);
          local_access.push_back(Affine(var.idx));
          local_sizes.push_back(odd_size ? NextOdd(var.range) : var.range);
        }
      }
      else {
        // keep this dim
        Affine exp;
        int64_t high = 0;
        int64_t low = 0;
        for (size_t i = 0; i < dim_vars.size(); ++i) {
          const auto& var = dim_vars[i];
          NewIndexIntoBlock(&xfer_block, var.idx, var.range, &used_idxs);
          exp = exp + Affine(var.idx, var.coeff * var.sign);
          high += var.high;
          low += var.low;
        }
        exp = exp + Affine(remain_const[k] - low);
        size_t range = high - low + remain_const[k] + 1;
        local_access.push_back(exp);
        local_sizes.push_back(odd_size ? NextOdd(range) : range);
      }
    }
    else {
      local_access.push_back(Affine());
      local_sizes.push_back(1);
    }

    // Build the access for global ref of cache block
    if (keep_dims.find(k) == keep_dims.end()) {
      Affine dim_access;
      for (size_t i = 0; i < dim_vars.size(); ++i) {
        global_access[k] = global_access[k]
                           + Affine(dim_vars[i].idx, dim_vars[i].sign * dim_vars[i].coeff)
                           + Affine(-dim_vars[i].low);
      }
      global_access[k] = global_access[k] + remain_const[k];
    }
    else {
      global_access[k] = local_access.back();
    }
  }
  
  TensorShape cached_exterior_ts = SimpleShape(outer_ref_it->interior_shape.type, local_sizes);
  TensorShape cached_interior_ts = cached_exterior_ts;
  for (auto& dim : cached_interior_ts.dims) {
    dim.size = 1;
  }

  // Build the local/global refs for the cache block
  Refinement global_ref(
      RefDir::In,         // dir
      var_name,           // from
      "src",              // into
      local_access,       // access
      cached_interior_ts, // interior_shape
      "",                 // agg_op
      ref_it->location,   // location
      ref_it->offset,     // offset
      ref_it->bank_dim    // bank_dim
  );
  Refinement local_ref(
      RefDir::Out,        // dir
      var_name,           // from
      "dst",              // into
      local_access,       // access
      cached_interior_ts, // interior_shape
      "",                 // agg_op
      ref_it->location,   // location
      ref_it->offset,     // offset
      ref_it->bank_dim    // bank_dim
  );

  xfer_block.refs.emplace(global_ref);
  xfer_block.refs.emplace(local_ref);
  xfer_block.stmts.emplace_back(std::make_shared<Load>("src", "$X"));
  xfer_block.stmts.emplace_back(std::make_shared<Store>("$X", "dst"));

  // Build the cache block
  std::shared_ptr<Block> cache_block;
  // If original refinement was input, load into cache
  if (IsReadDir(dir)) {
    cache_block = std::make_shared<Block>(xfer_block);
    cache_block->name = str(boost::format("load_%s") % var_name);
    cache_block->set_tags(load_tags);
    auto& src = cache_block->refs.find("src")->mut();
    auto& dst = cache_block->refs.find("dst")->mut();
    src.from = raw_name;
    src.interior_shape = ref_it->interior_shape;
    src.access = global_access;
    dst.location = mem_loc;
    if (reorder_idx) {
      ReorderIndex(cache_block.get(), true, false);
    }
    StatementIt pos = InsertPosition(outer_block, ref_block);
    outer_block->stmts.insert(pos, cache_block);
  }

  // If original refinement was output, flush from cache
  if (IsWriteDir(dir)) {
    cache_block = std::make_shared<Block>(xfer_block);
    cache_block->name = str(boost::format("store_%s") % var_name);
    cache_block->set_tags(store_tags);
    auto& src = cache_block->refs.find("src")->mut();
    auto& dst = cache_block->refs.find("dst")->mut();
    dst.from = raw_name;
    dst.interior_shape = ref_it->interior_shape;
    dst.access = global_access;
    src.location = mem_loc;
    if (reorder_idx) {
      ReorderIndex(cache_block.get(), true, false);
    }
    StatementIt pos = InsertPosition(outer_block, ref_block);
    ++pos;
    outer_block->stmts.insert(pos, cache_block);
  }

  // Add the new declaration (replacing the original)
  auto decl = outer_block->refs
                  .emplace(Refinement{
                      RefDir::None,        // dir
                      "",                  // from
                      var_name,            // into
                      {},                  // access
                      cached_exterior_ts,  // interior_shape
                      ref_it->agg_op,      // agg_op
                      mem_loc,             // location
                  })
                  .first;
  decl->mut().access.resize(cached_exterior_ts.dims.size());

  // Build the local map that translates local access from
  // cache block to ref block
  std::map<std::string, Affine> local_map;
  for (const auto& dim_vars : vars) {
    for (const auto& var : dim_vars) {
      local_map.emplace(var.idx, var.access);
    }
  }
  // Change the global ref in ref_block to local ref
  ref_it->mut().location = mem_loc;
  ref_it->mut().interior_shape = cached_interior_ts;
  std::vector<Affine> ref_local_access;
  for (size_t k = 0; k < local_access.size(); ++k) {
    auto& aff = local_access[k];
    if (keep_dims.find(k) == keep_dims.end()) {
      ref_local_access.push_back(aff.sym_eval(local_map));
    }
    else {
      ref_local_access.push_back(aff);
    }
  }
  ref_it->mut().access = ref_local_access;

  if (local_access == ref_local_access) {
    const auto& ref = cache_block->ref_by_into(dir == RefDir::In ? "dst" : "src");
    // To determine if all index in cache block
    // are same as that in the reference block
    bool same_idxs = true;
    for (auto& idx : cache_block->idxs) {
      Index* ref_idx = ref_block->idx_by_name(idx.name);
      if (!ref_idx || ref_idx->range != idx.range) {
        same_idxs = false;
        break;
      }
    }
    if (same_idxs) {
      // If index and access are both consistent,
      // set same_access tag for the reg_cache pass
      ref->mut().set_tag("same_access");
    }
  }

  // Update inner blocks strides + locations
  FixupMiddleBlockRefs(outer_block, ref_block, var_name, raw_name,
    IsReadDir(dir) ? load_tags : store_tags);

  // Add constraints
  if (add_constraints) {
    // Add constraints according to refinement access ranges
    AliasMap outer_map(*(alias_map.parent_alias_map()), outer_block);
    AliasMap cache_map(outer_map, cache_block.get());
    auto cache_ref_it = IsReadDir(dir) ?
        cache_block->ref_by_into("src") : cache_block->ref_by_into("dst");
    cache_map.AddConstraintsForRef(*cache_ref_it);
    // Add possible original constraints in ref_block
    for (const auto cons : ref_block->constraints) {
      bool all_exist = true;
      for (auto& kvp : cons.getMap()) {
        if (kvp.first != "" && cache_block->idx_by_name(kvp.first) == nullptr) {
          all_exist = false;
          break;
        }
      }
      if (all_exist) {
        cache_block->constraints.push_back(cons);
      }
    }
  }
}

// If we don't have a reference block that shows the ref access pattern.
// We can only use the interior shape of the outer block and do the simple
// cache pass.
void ApplySimpleCache(const AliasMap& map,          //
                      RefDir dir,                   //
                      Block* block,                 //
                      const std::string& var_name,  //
                      const Location& mem_loc,      //
                      const Location& xfer_loc,     //
                      const Tags load_tags,         //
                      const Tags store_tags,        //
                      bool add_constraints,         //
                      bool reorder_idx,             //
                      bool odd_size,                //
                      double odd_limit) {
  auto it = block->ref_by_into(var_name, false);
  if (it == block->refs.end()) {
    throw std::runtime_error("ApplySimpleCache: Invalid var_name");
  }
  // Get the alias info
  const auto& ai = map.at(var_name);
  // Get the shape
  TensorShape raw_ts = it->interior_shape;
  std::vector<size_t> sizes = raw_ts.sizes();
  if (odd_size) {
    for (int i = sizes.size() - 1; i >= 0; --i) {
      sizes[i] = NextOdd(sizes[i]);
    }
  }
  TensorShape cached_ts = SimpleShape(raw_ts.type, sizes);
  // Make a new name for the raw variable
  std::string raw_name = block->unique_ref_name(var_name + "_raw");
  // Replace the old refinement to rename it.
  Refinement replacement_ref = it->WithInto(raw_name);
  block->refs.erase(it);
  it = block->refs.emplace(std::move(replacement_ref)).first;
  // Make a base block for loading/storing
  // Set both from refinements to the cached version, we will replace
  // one of them with the 'raw' version based on transfer direction
  Block xfer_block;
  xfer_block.location = xfer_loc;
  std::vector<Affine> xfer_access;
  for (size_t i = 0; i < sizes.size(); i++) {
    std::string iname = str(boost::format("i%zu") % i);
    if (sizes[i] > 1) {
      xfer_block.idxs.emplace_back(Index{iname, sizes[i]});
      xfer_access.emplace_back(Affine(iname));
    } else {
      xfer_access.emplace_back(Affine());
    }
    if (add_constraints) {
      map.AddConstraintForIndex(&xfer_block, ai, i, iname, sizes[i] <= 1);
    }
  }
  TensorShape raw_xfer_shape = raw_ts;
  TensorShape cached_xfer_shape = cached_ts;
  for (size_t i = 0; i < sizes.size(); i++) {
    raw_xfer_shape.dims[i].size = 1;
    cached_xfer_shape.dims[i].size = 1;
  }
  xfer_block.refs.emplace(Refinement{
      RefDir::In,         // dir
      var_name,           // from
      "src",              // into
      xfer_access,        // access
      cached_xfer_shape,  // interior_shape
      "",                 // agg_op
      it->location,       // location
      it->offset,         // offset
      it->bank_dim,       // bank_dim
  });
  xfer_block.refs.emplace(Refinement{
      RefDir::Out,        // dir
      var_name,           // from
      "dst",              // into
      xfer_access,        // access
      cached_xfer_shape,  // interior_shape
      "",                 // agg_op
      it->location,       // location
      it->offset,         // offset
      it->bank_dim,       // bank_dim
  });
  xfer_block.stmts.emplace_back(std::make_shared<Load>("src", "$X"));
  xfer_block.stmts.emplace_back(std::make_shared<Store>("$X", "dst"));
  // If original refinement was input, load into cache
  if (IsReadDir(dir)) {
    auto cache_load = std::make_shared<Block>(xfer_block);
    cache_load->name = str(boost::format("load_%s") % var_name);
    cache_load->set_tags(load_tags);
    auto& src = cache_load->refs.find("src")->mut();
    auto& dst = cache_load->refs.find("dst")->mut();
    src.from = raw_name;
    src.interior_shape = raw_xfer_shape;
    dst.location = mem_loc;
    if (reorder_idx) {
      ReorderIndex(cache_load.get(), true, false);
    }
    block->stmts.emplace_front(cache_load);
  }
  // If original refinement was output, flush from cache
  if (IsWriteDir(dir)) {
    auto cache_store = std::make_shared<Block>(xfer_block);
    cache_store->name = str(boost::format("store_%s") % var_name);
    cache_store->set_tags(store_tags);
    auto& src = cache_store->refs.find("src")->mut();
    auto& dst = cache_store->refs.find("dst")->mut();
    dst.from = raw_name;
    dst.interior_shape = raw_xfer_shape;
    src.location = mem_loc;
    if (reorder_idx) {
      ReorderIndex(cache_store.get(), true, false);
    }
    block->stmts.emplace_back(cache_store);
  }
  // Add the new declaration (replacing the original)
  auto decl = block->refs
                  .emplace(Refinement{
                      RefDir::None,  // dir
                      "",            // from
                      var_name,      // into
                      {},            // access
                      cached_ts,     // interior_shape
                      it->agg_op,    // agg_op
                      mem_loc,       // location
                  })
                  .first;
  decl->mut().access.resize(cached_ts.dims.size());
  // Update inner blocks strides + locations
  FixupRefs(block, var_name);
}

static Block* LookupRefBlock(Block* block, const std::string& ref_tag) {
  for (const auto stmt : block->stmts) {
    auto sub = Block::Downcast(stmt);
    if (sub) {
      if (sub->has_tag(ref_tag)) {
        return sub.get();
      }
      Block* ret = LookupRefBlock(sub.get(), ref_tag);
      if (ret) {
        return ret;
      }
    }
  }
  return nullptr;
}

static void CacheBlock(const AliasMap& map, Block* block, const proto::CachePass& options) {
  std::set<RefDir> dirs;
  for (const auto& dir : options.dirs()) {
    dirs.emplace(stripe::FromProto(static_cast<stripe::proto::Refinement::Dir>(dir)));
  }
  // This indicates how to deal with InOut ref
  RefDir inout;
  if (dirs.count(RefDir::In) && dirs.count(RefDir::Out)) {
    // We don't support cache In and Out refs in the same pass
    throw std::runtime_error("Incorrect dir for cache pass.");
  }
  else if (dirs.count(RefDir::In)) {
    inout = RefDir::In;     // process In and InOut as read cache
  }
  else if (dirs.count(RefDir::Out)) {
    inout = RefDir::Out;    // process Out and InOut as write cache
  }
  else if (dirs.count(RefDir::InOut)) {
    inout = RefDir::InOut;  // process InOut as both read and write cache
  }
  else {
    throw std::runtime_error("Incorrect dir for cache pass.");
  }
  std::string ref_tag = options.ref();
  auto mem_loc = stripe::FromProto(options.mem_loc());
  auto xfer_loc = stripe::FromProto(options.xfer_loc());
  auto refs = block->refs;
  if (ref_tag.size() > 0) {
    // If we have the reference block, do the advanced cahce pass
    Block* ref_block = LookupRefBlock(block, ref_tag);
    if (ref_block == nullptr) {
      throw std::runtime_error("No block has tag " + ref_tag);
    }
    for (const auto& ref : refs) {
      if (dirs.count(ref.dir)) {
        codegen::ApplyCache(map, inout, ref_block, block, ref.into(), mem_loc, xfer_loc, 
          {"cache", "cache_load"}, {"cache", "cache_store"}, options.add_constraints(),
          options.reorder_idx(), options.odd_size(), options.odd_limit());
      }
    }
  }
  else {
    // If we don't have the reference block, do the simple cache pass according to 
    // the outer block's interior shape
    for (const auto& ref : refs) {
      if (dirs.count(ref.dir)) {
        codegen::ApplySimpleCache(map, inout, block, ref.into(), mem_loc, xfer_loc,
          {"cache", "cache_load"}, {"cache", "cache_store"}, options.add_constraints(),
          options.reorder_idx(), options.odd_size());
      }
    }
  }
}

void CachePass::Apply(CompilerState* state) const {
  auto reqs = FromProto(options_.reqs());
  RunOnBlocks(state->entry(), reqs, [&](const AliasMap& map, Block* block) {  //
    CacheBlock(map, block, options_);
  });
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<CachePass, proto::CachePass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
