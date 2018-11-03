// Copyright 2018, Intel Corporation

#pragma once

#include <set>
#include <string>

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/tags.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void ApplyCache(stripe::Block* block,             //
                const std::string& var_name,      //
                const stripe::Location& mem_loc,  //
                const stripe::Location& xfer_loc);

void CacheBlock(stripe::Block* block,                  //
                const std::set<stripe::RefDir>& dirs,  //
                const stripe::Location& mem_loc,       //
                const stripe::Location& xfer_loc);

inline void CachePass(stripe::Block* root, const proto::CachePass& options) {
  auto reqs = FromProto(options.reqs());
  std::set<stripe::RefDir> dirs;
  for (const auto& dir : options.dirs()) {
    dirs.emplace(stripe::FromProto(static_cast<stripe::proto::Refinement::Dir>(dir)));
  }
  auto mem_loc = stripe::FromProto(options.mem_loc());
  auto xfer_loc = stripe::FromProto(options.xfer_loc());
  RunOnBlocks(root, reqs, [&](const AliasMap& map, stripe::Block* block) {  //
    CacheBlock(block, dirs, mem_loc, xfer_loc);
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
