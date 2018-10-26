// Copyright 2018, Intel Corp.

#pragma once

#include <map>
#include <string>

#include <boost/optional.hpp>

#include "tile/codegen/alias.h"

namespace vertexai {
namespace tile {
namespace codegen {

struct FusionPlan {
  TileShape tile_a;
  std::map<std::string, std::string> remap_a;
  TileShape tile_b;
  std::map<std::string, std::string> remap_b;
};

// Given a shared buffer between two blocks, compute a possible fusion
boost::optional<FusionPlan> ComputeFusionPlan(const stripe::Block& a, const stripe::Block& b,
                                              const std::string& buf_name);

// A transform that flattens trivial indexes.  TODO: move to a utility header
void FlattenTrivial(stripe::Block* block);

// Prepare each block for fusion by renaming / moving indexes
std::shared_ptr<stripe::Block> FusionRefactor(const stripe::Block& block,                         //
                                              const std::map<std::string, std::string>& mapping,  //
                                              const TileShape& tile,                              //
                                              const std::string& location);

// Attempt to fuse b into a.  Return true on success, in which case blocks have been
// destructively modified.  Otherwise returns false and leave blocks unaltered.
bool FuseBlocks(const AliasMap& scope, stripe::Block* a, stripe::Block* b);

class FusionStrategy {
 public:
  // Called when candidate blocks for fusion are located, returns whether to attempt a fusion
  virtual bool attempt_fuse(const stripe::Block& a, const stripe::Block& b) = 0;
  // Called when an attempted fusion fails
  virtual void fusion_failed() = 0;
  // Called when a fusion succeeds, with the new fused block (which can be edited)
  virtual void on_fused(const AliasMap& outer, stripe::Block* block) = 0;
};

void FusionPass(const AliasMap& scope, stripe::Block* block, FusionStrategy* strategy);

class AlwaysFuseRecursive : public FusionStrategy {
 public:
  bool attempt_fuse(const stripe::Block& a, const stripe::Block& b) { return true; }
  void fusion_failed() {}
  void on_fused(const AliasMap& outer, stripe::Block* block) {
    AliasMap inner(outer, *block);
    FusionPass(inner, block, this);
  }
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
