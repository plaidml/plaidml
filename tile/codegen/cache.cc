// Copyright 2018, Intel Corporation

#include "tile/codegen/cache.h"

#include <algorithm>

#include "base/util/stream_container.h"
#include "tile/codegen/localize.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

void ApplyCache(Block* block,                 //
                const std::string& var_name,  //
                const Location& mem_loc,      //
                const Location& xfer_loc) {
  auto it = block->ref_by_into(var_name, false);
  if (it == block->refs.end()) {
    throw std::runtime_error("ApplyCache: Invalid var_name");
  }
  // Get the shape
  TensorShape raw_ts = it->shape;
  std::vector<size_t> sizes;
  for (const auto& dim : raw_ts.dims) {
    sizes.push_back(dim.size);
  }
  TensorShape cached_ts = SimpleShape(raw_ts.type, sizes);
  // Make a new name for the raw variable
  std::string raw_name = block->unique_ref_name(var_name + "_raw");
  // Update the old refinement to rename
  it->into = raw_name;
  // Make a base block for loading/storing
  // Set both from refinements to the cached version, we will replace
  // one of them with the 'raw' version based on transfer direction
  Block xfer_block;
  xfer_block.location = xfer_loc;
  std::vector<Affine> xfer_access;
  for (size_t i = 0; i < sizes.size(); i++) {
    std::string iname = printstring("i%zu", i);
    xfer_block.idxs.emplace_back(Index{iname, sizes[i]});
    xfer_access.emplace_back(Affine(iname));
  }
  TensorShape raw_xfer_shape = raw_ts;
  TensorShape cached_xfer_shape = cached_ts;
  for (size_t i = 0; i < sizes.size(); i++) {
    raw_xfer_shape.dims[i].size = 1;
    cached_xfer_shape.dims[i].size = 1;
  }
  xfer_block.refs.push_back(Refinement{
      RefDir::In,         // dir
      var_name,           // from
      "src",              // into
      xfer_access,        // access
      cached_xfer_shape,  // shape
      "",                 // agg_op
      it->location,       // location
      it->is_const        // is_const
  });
  xfer_block.refs.push_back(Refinement{
      RefDir::Out,        // dir
      var_name,           // from
      "dst",              // into
      xfer_access,        // access
      cached_xfer_shape,  // shape
      "",                 // agg_op
      it->location,       // location
      it->is_const        // is_const
  });
  xfer_block.stmts.push_back(std::make_shared<Load>("src", "$X"));
  xfer_block.stmts.push_back(std::make_shared<Store>("$X", "dst"));
  // If original refinement was input, load into cache
  if (IsReadDir(it->dir)) {
    auto cache_load = std::make_shared<Block>(xfer_block);
    cache_load->name = printstring("load_%s", var_name.c_str());
    cache_load->refs[0].from = raw_name;
    cache_load->refs[0].shape = raw_xfer_shape;
    cache_load->refs[1].location = mem_loc;
    block->stmts.push_front(cache_load);
  }
  // If original refinement was output, flush from cache
  if (IsWriteDir(it->dir)) {
    auto cache_store = std::make_shared<Block>(xfer_block);
    cache_store->name = printstring("store_%s", var_name.c_str());
    cache_store->refs[1].from = raw_name;
    cache_store->refs[1].shape = raw_xfer_shape;
    cache_store->refs[0].location = mem_loc;
    block->stmts.push_back(cache_store);
  }
  // Add the new declaration (replacing the original)
  block->refs.push_back(Refinement{
      RefDir::None,  // dir
      "",            // from
      var_name,      // into
      {},            // access
      cached_ts,     // shape
      "",            // agg_op
      mem_loc        // location
  });
  block->refs.back().access.resize(cached_ts.dims.size());
  // Update inner blocks strides + locations
  FixupRefs(block, var_name);
}

void CacheBlock(Block* block, const std::set<RefDir>& dirs, const Location& mem_loc, const Location& xfer_loc) {
  auto refs = block->refs;
  for (const auto& ref : refs) {
    if (dirs.count(ref.dir)) {
      codegen::ApplyCache(block, ref.into, mem_loc, xfer_loc);
    }
  }
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
