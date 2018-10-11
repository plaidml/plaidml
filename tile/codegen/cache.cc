// Copyright 2018, Intel Corp.

#include "tile/codegen/cache.h"

#include <algorithm>

#include "base/util/stream_container.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void ApplyCache(stripe::Block* block, const std::string& var_name) {
  // Find the appropriate refinement
  auto ref_it = block->ref_by_into(var_name);
  if (ref_it == block->refs.end()) {
    throw std::runtime_error("ApplyCache: Invalid var_name");
  }
  // Get the shape
  TensorShape raw_ts = ref_it->shape;
  std::vector<size_t> sizes;
  for (const auto& d : raw_ts.dims) {
    sizes.push_back(d.size);
  }
  TensorShape cached_ts = SimpleShape(raw_ts.type, sizes);
  // Make a new name for the raw variable
  std::string raw_name = block->unique_ref_name(var_name + "_raw");
  // Update the old refinement to rename
  ref_it->into = raw_name;
  // Add the new declaration (replacing the original)
  block->refs.push_back({
      stripe::RefDir::None,  // dir
      "",                    // from
      var_name,              // into
      {},                    // access
      cached_ts,             // shape
      "",
  });
  // Make a base block for loading/storing
  // Set both from refinements to the cached version, we will replace
  // one of them with the 'raw' version based on transfer direction
  stripe::Block xfer_block;
  std::vector<stripe::Affine> xfer_access;
  for (size_t i = 0; i < sizes.size(); i++) {
    std::string iname = std::string("i") + std::to_string(i);
    xfer_block.idxs.emplace_back(iname, "", sizes[i], 0);
    xfer_access.push_back(stripe::Affine(iname));
  }
  TensorShape raw_xfer_shape = raw_ts;
  TensorShape cached_xfer_shape = cached_ts;
  for (size_t i = 0; i < sizes.size(); i++) {
    raw_xfer_shape.dims[i].size = 1;
    cached_xfer_shape.dims[i].size = 1;
  }
  xfer_block.refs.push_back({stripe::RefDir::In, var_name, "src", xfer_access, cached_xfer_shape});
  xfer_block.refs.push_back({stripe::RefDir::Out, var_name, "dst", xfer_access, cached_xfer_shape});
  xfer_block.stmts.push_back(std::make_shared<stripe::Load>("src", "$X"));
  xfer_block.stmts.push_back(std::make_shared<stripe::Store>("$X", "dst"));
  // If original refinement was input, load into cache
  if (ref_it->dir == stripe::RefDir::In || ref_it->dir == stripe::RefDir::InOut) {
    auto cache_load = std::make_shared<stripe::Block>(xfer_block);
    cache_load->refs[0].from = raw_name;
    cache_load->refs[0].shape = raw_xfer_shape;
    block->stmts.push_front(cache_load);
  }
  // If original refinement was output, flush from cache
  if (ref_it->dir == stripe::RefDir::Out || ref_it->dir == stripe::RefDir::InOut) {
    auto cache_store = std::make_shared<stripe::Block>(xfer_block);
    cache_store->refs[1].from = raw_name;
    cache_store->refs[1].shape = raw_xfer_shape;
    block->stmts.push_back(cache_store);
  }
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
