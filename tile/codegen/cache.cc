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

void ApplyCache(const AliasMap& map,                     //
                Block* block,                            //
                const std::string& var_name,             //
                const Location& mem_loc,                 //
                const Location& xfer_loc,                //
                const std::set<std::string> load_tags,   //
                const std::set<std::string> store_tags,  //
                bool add_constraints) {
  auto it = block->ref_by_into(var_name, false);
  if (it == block->refs.end()) {
    throw std::runtime_error("ApplyCache: Invalid var_name");
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
    if (sizes[i] > 1) {
      std::string iname = str(boost::format("i%zu") % i);
      xfer_block.idxs.emplace_back(Index{iname, sizes[i]});
      xfer_access.emplace_back(Affine(iname));
      if (add_constraints) {
        map.AddConstraintForIndex(&xfer_block, ai, i, iname);
      }
    } else {
      xfer_access.emplace_back(Affine());
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
  if (IsReadDir(it->dir)) {
    auto cache_load = std::make_shared<Block>(xfer_block);
    cache_load->name = str(boost::format("load_%s") % var_name);
    cache_load->tags = load_tags;
    auto& src = cache_load->refs.find("src")->mut();
    auto& dst = cache_load->refs.find("dst")->mut();
    src.from = raw_name;
    src.interior_shape = raw_xfer_shape;
    dst.location = mem_loc;
    block->stmts.emplace_front(cache_load);
  }
  // If original refinement was output, flush from cache
  if (IsWriteDir(it->dir)) {
    auto cache_store = std::make_shared<Block>(xfer_block);
    cache_store->name = str(boost::format("store_%s") % var_name);
    cache_store->tags = store_tags;
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

static void CacheBlock(const AliasMap& map, Block* block, const std::set<RefDir>& dirs, const Location& mem_loc,
                       const Location& xfer_loc, bool add_constraints) {
  auto refs = block->refs;
  for (const auto& ref : refs) {
    if (dirs.count(ref.dir)) {
      codegen::ApplyCache(map, block, ref.into(), mem_loc, xfer_loc, {"cache", "cache_load"}, {"cache", "cache_store"},
                          add_constraints);
    }
  }
}

void CachePass(Block* root, const proto::CachePass& options) {
  auto reqs = FromProto(options.reqs());
  std::set<RefDir> dirs;
  for (const auto& dir : options.dirs()) {
    dirs.emplace(stripe::FromProto(static_cast<stripe::proto::Refinement::Dir>(dir)));
  }
  auto mem_loc = stripe::FromProto(options.mem_loc());
  auto xfer_loc = stripe::FromProto(options.xfer_loc());
  RunOnBlocks(root, reqs, [&](const AliasMap& map, Block* block) {  //
    CacheBlock(map, block, dirs, mem_loc, xfer_loc, options.add_constraints());
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
