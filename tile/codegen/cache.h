// Copyright 2018, Intel Corp.

#pragma once

#include <set>
#include <string>
#include <vector>

#include "tile/codegen/tags.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void ApplyCache(stripe::Block* block,             //
                const std::string& var_name,      //
                const stripe::Location& mem_loc,  //
                const stripe::Location& xfer_loc);

void CacheBlock(stripe::Block* block, const std::set<stripe::RefDir>& dirs, const stripe::Location& mem_loc,
                const stripe::Location& xfer_loc);

struct CachePassOptions {
  Tags reqs;
  std::set<stripe::RefDir> dirs;
  stripe::Location mem_loc;
  stripe::Location xfer_loc;
};

inline void CachePass(stripe::Block* root, const CachePassOptions& options) {
  RunOnBlocks(root, options.reqs, [&options](const AliasMap& map, stripe::Block* block) {
    CacheBlock(block, options.dirs, options.mem_loc, options.xfer_loc);
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
