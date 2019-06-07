// Copyright 2018, Intel Corporation

#include "tile/codegen/cache.h"

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

// Make a passthru index if it doesn't exist
std::string MakePassthruIdx(Block* block, const std::string& idx_name) {
  for (const auto idx : block->idxs) {
    if (idx.affine == Affine(idx_name)) {
      // The passthru index exists, return the existing index
      return idx.name;
    }
  }
  // doesn't exist
  std::string new_idx = block->unique_idx_name(idx_name);
  block->idxs.push_back({new_idx, 1, Affine(idx_name)});
  return new_idx;
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

// When we want to add a constraint in the inner block, it may use some outer blocks' 
// index. So we have to pass the outer index to the inner block and generate the 
// corresponding passthru index along the path from outer to inner.
void FixPassthruIdx(const AliasMap& inner_map, Block* outer, Affine exp) {
  // Find a path from outer to inner
  std::vector<Block*> path;
  const AliasMap* am = &inner_map; 
  while (am && am->this_block() != outer) {
    path.push_back(am->this_block());
    am = am->parent_alias_map();
  }
  if (am == nullptr) {
    throw std::runtime_error("Cannot find outer block.");
  }
  path.push_back(outer);
  // The idxs to be built in this level or inner
  std::map<std::string, size_t> to_build;
  // The idxs to be built in the inner level or inner
  std::map<std::string, size_t> to_build_next;
  // Resolve the initialze idxs
  for (const auto& it : exp.getMap()) {
    if (it.first != "") {
      size_t pos = it.first.find(':');
      std::string idx_name = it.first.substr(pos + 1);
      size_t idx_depth = std::stoi(it.first.substr(1, pos - 1));
      to_build.emplace(idx_name, idx_depth); 
    }
  }
  // Traverse from outer to inner
  for (int i = path.size() - 1; i >= 0; --i) {
    Block* this_block = path[i];
    size_t depth = inner_map.depth() - i;
    for (auto& it : to_build) {
      if (it.second == depth - 1) {
        std::string new_idx = MakePassthruIdx(this_block, it.first);
        to_build_next.emplace(new_idx, depth);
      }
      else {
        to_build_next.emplace(it.first, it.second);
      }
    }
    to_build = to_build_next;
    to_build_next.clear();
  } 
}

// Fixup the refs between the outer block and the inner block
void FixupMiddleBlockRefs(Block* outer, Block* inner,
                          const std::string& var_name, const std::string& raw_name) {
  auto it = outer->ref_by_into(var_name, false);
  if (it == outer->refs.end()) {
    return;
  }
  for (auto stmt : outer->stmts) {
    auto sub = Block::Downcast(stmt);
    if (sub && sub.get() != inner && !sub->has_tag("cache")) {
      for (auto& ref : sub->refs) {
        if (ref.from == var_name) {
          if (AllZeroAccess(ref)) {
            ref.mut().location = it->location;
            ref.mut().offset = it->offset;
            ref.mut().access = it->access;
            ref.mut().interior_shape = it->interior_shape;
            FixupMiddleBlockRefs(sub.get(), inner, ref.into(), raw_name);
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
                bool add_constraints) {
  auto ref_it = ref_block->ref_by_from(var_name, false);
  if (ref_it == ref_block->refs.end()) {
    return;
  }
  auto access = ref_it->FlatAccess();

  // Set up the initial index according to ref_block 
  std::vector<CacheVar> vars;
  // remain_const is the access' constant minus all index lows
  // i.e., the low of the access, usually zero, but could be positive
  int64_t remain_const = 0;
  for (const auto& it : access.getMap()) {
    if (it.first == "") {
      remain_const += it.second;
      continue;
    }
    Index* idx = ref_block->idx_by_name(it.first);
    int64_t low = (it.second >= 0) ? 0 : (it.second * (idx->range - 1));
    int64_t high = (it.second >= 0) ? (it.second * (idx->range - 1)) : 0;
    CacheVar var = {it.first, (it.second >= 0) ? 1 : -1, std::abs(it.second), idx->range, low, high, Affine(it.first)};
    vars.push_back(var);
    remain_const += low;
  }
  // Sort the index by their coefficients
  std::sort(vars.begin(), vars.end(), comp_coeff);

  // Traverse the variables to determine if they should be merged
  // vars[i] and vars[i + 1] should be merged if,
  // 1) one coefficient is the multiple of the other; and
  // 2) vars[i] and vars[i + 1] are overlapped for non-zero coefficients
  int i = 0;
  while (i < static_cast<int>(vars.size()) - 1) {
    if (vars[i + 1].coeff % vars[i].coeff == 0) {
      // This works for negative coefficients
      if (vars[i].coeff * static_cast<int64_t>(vars[i].range - 1) >= vars[i + 1].coeff) {
        // Merge vars[i] and vars[i + 1]
        CacheVar new_var;
        new_var.idx = vars[i].idx + "_" + vars[i + 1].idx;
        // The new access includes the constant. So low is always zero
        new_var.low = 0;
        new_var.high = vars[i].high + vars[i + 1].high - vars[i].low - vars[i + 1].low;
        // Here vars[i + 1].coeff is multiple of vars[i].coeff
        new_var.coeff = vars[i].coeff;
        new_var.sign = 1;
        new_var.range = (new_var.high - new_var.low) / new_var.coeff + 1;
        new_var.access = vars[i].access * (vars[i].sign * vars[i].coeff / new_var.coeff) +
                         vars[i + 1].access * (vars[i + 1].sign * vars[i + 1].coeff / new_var.coeff) +
                         Affine(- (vars[i].low + vars[i + 1].low) / new_var.coeff);
        vars.erase(vars.begin() + i);
        vars[i] = new_var;
        continue;
      }
    }
    // Cannot merge
    ++i;
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
  std::vector<Affine> global_access(ref_it->interior_shape.dims.size());
  // Insert the new index, and build the access for the local ref of cache block
  if (vars.size() > 0) {
    for (size_t i = 0; i < vars.size(); ++i) {
      const auto& var = vars[i];
      xfer_block.idxs.push_back({var.idx, var.range});
      local_access.push_back(Affine(var.idx));
      local_sizes.push_back(var.range);
    }
  }
  else {
    local_access = {Affine()};
    local_sizes = {1};
  }

  // Build the access for global ref of cache block
  std::vector<CacheVar> vars_tmp = vars;
  for (size_t i = 0; i < ref_it->interior_shape.dims.size(); ++i) {
    Affine dim_access;
    int64_t this_stride = static_cast<int>(ref_it->interior_shape.dims[i].stride);
    for (size_t j = 0; j < vars_tmp.size(); ++j) {
      if (vars_tmp[j].coeff >= this_stride) {
        global_access[i] = global_access[i]
                         + Affine(vars_tmp[j].idx, vars_tmp[j].sign * vars_tmp[j].coeff / this_stride)
                         + Affine(-vars_tmp[j].low / this_stride);
        vars_tmp[j].coeff %= this_stride;
        vars_tmp[j].low = -((-vars_tmp[j].low) % this_stride);
      }
    }
    if (remain_const >= this_stride) {
      global_access[i] = global_access[i] + Affine(remain_const / this_stride);
      remain_const %= this_stride;
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
  for (const auto& var : vars) {
    local_map.emplace(var.idx, var.access);
  }
  // Change the global ref in ref_block to local ref
  ref_it->mut().location = mem_loc;
  ref_it->mut().interior_shape = cached_interior_ts;
  std::vector<Affine> ref_local_access;
  for (auto& aff : local_access) {
    ref_local_access.push_back(aff.sym_eval(local_map));
  }
  ref_it->mut().access = ref_local_access;

  // Update inner blocks strides + locations
  FixupMiddleBlockRefs(outer_block, ref_block, var_name, raw_name);

  // Add constraints
  if (add_constraints) {
    // Build the alias maps first for the new local ref's alias info
    AliasMap old_outer_map(*(alias_map.parent_alias_map()), outer_block);
    AliasMap old_cache_map(old_outer_map, cache_block.get());
    auto ref_it = cache_block->ref_by_from(raw_name);
    const auto& old_ai = old_cache_map.at(ref_it->into());
    // Add the passthru index that are used by the constraints
    for (size_t i = 0; i < ref_it->interior_shape.dims.size(); ++i) {
      FixPassthruIdx(old_cache_map, outer_block, old_ai.access[i]);
    }
    // Build the alias map agian due to new passthru index
    AliasMap outer_map(*(alias_map.parent_alias_map()), outer_block);
    AliasMap cache_map(outer_map, cache_block.get());
    const auto& ai = cache_map.at(ref_it->into());
    // We have everything for adding constraints
    for (size_t i = 0; i < ref_it->interior_shape.dims.size(); ++i) {
      cache_map.AddConstraintForIndex(cache_block.get(), ai, i,
          "", ref_it->interior_shape.dims[i].size <= 1);
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
                      bool add_constraints) {
  auto it = block->ref_by_into(var_name, false);
  if (it == block->refs.end()) {
    throw std::runtime_error("ApplySimpleCache: Invalid var_name");
  }
  // Get the alias info
  const auto& ai = map.at(var_name);
  // Get the shape
  TensorShape raw_ts = it->interior_shape;
  std::vector<size_t> sizes = raw_ts.sizes();
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

static void CacheBlock(const AliasMap& map, Block* block, 
                       const std::string& ref_tag, const std::set<RefDir>& dirs,
                       const Location& mem_loc, const Location& xfer_loc, 
                       bool add_constraints) {
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
          {"cache", "cache_load"}, {"cache", "cache_store"}, add_constraints);
      }
    }
  }
  else {
    // If we don't have the reference block, do the simple cache pass according to 
    // the outer block's interior shape
    for (const auto& ref : refs) {
      if (dirs.count(ref.dir)) {
        codegen::ApplySimpleCache(map, inout, block, ref.into(), mem_loc, xfer_loc,
          {"cache", "cache_load"}, {"cache", "cache_store"}, add_constraints);
      }
    }
  }
}

void CachePass::Apply(CompilerState* state) const {
  auto reqs = FromProto(options_.reqs());
  std::set<RefDir> dirs;
  for (const auto& dir : options_.dirs()) {
    dirs.emplace(stripe::FromProto(static_cast<stripe::proto::Refinement::Dir>(dir)));
  }
  auto mem_loc = stripe::FromProto(options_.mem_loc());
  auto xfer_loc = stripe::FromProto(options_.xfer_loc());
  RunOnBlocks(state->entry(), reqs, [&](const AliasMap& map, Block* block) {  //
    CacheBlock(map, block, options_.ref(), dirs,
               mem_loc, xfer_loc, options_.add_constraints());
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
