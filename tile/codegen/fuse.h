// Copyright 2018, Intel Corporation

#pragma once

#include <map>
#include <string>

#include <boost/optional.hpp>

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/tile.h"

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
                                              const TileShape& tile);

// Attempt to fuse b into a.  Return true on success, in which case blocks have been
// destructively modified.  Otherwise returns false and leave blocks unaltered.
bool FuseBlocks(const AliasMap& scope, stripe::Block* a, stripe::Block* b);

class FusionStrategy {
 public:
  // Called when candidate blocks for fusion are located, returns whether to attempt a fusion
  virtual bool AttemptFuse(const stripe::Block& parent, const stripe::Block& a, const stripe::Block& b) = 0;
  // Called when an attempted fusion fails
  virtual void OnFailed() = 0;
  // Called when a fusion succeeds, with the new fused block (which can be edited)
  virtual void OnFused(const AliasMap& outer, stripe::Block* block, const stripe::Block& a, const stripe::Block& b) = 0;
};

void FusionInner(const AliasMap& scope, stripe::Block* block, FusionStrategy* strategy);

class AlwaysFuseRecursive : public FusionStrategy {
 public:
  bool AttemptFuse(const stripe::Block& parent, const stripe::Block& a, const stripe::Block& b) { return true; }
  void OnFailed() {}
  void OnFused(const AliasMap& outer, stripe::Block* block, const stripe::Block& a, const stripe::Block& b) {
    block->location = a.location;
    AliasMap inner(outer, block);
    FusionInner(inner, block, this);
  }
};

void FusionPass(stripe::Block* block, const proto::FusionPass& options);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
