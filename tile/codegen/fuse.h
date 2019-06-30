// Copyright 2018, Intel Corporation

#pragma once

#include <map>
#include <memory>
#include <string>

#include <boost/optional.hpp>

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/codegen/tile.h"

namespace vertexai {
namespace tile {
namespace codegen {

struct FusionPlan {
  TileShape tile_a;
  bool a_interleave;
  std::map<std::string, std::string> remap_a;
  TileShape tile_b;
  bool b_interleave;
  std::map<std::string, std::string> remap_b;
};

// Given a shared buffer between two blocks, compute a possible fusion
boost::optional<FusionPlan> ComputeFusionPlan(const AliasMap& scope, const stripe::Block& a, const stripe::Block& b,
                                              const std::string& buf_name);

// A transform that flattens trivial indexes.  TODO: move to a utility header
void FlattenTrivial(stripe::Block* block);

// Prepare each block for fusion by renaming / moving indexes
std::shared_ptr<stripe::Block> FusionRefactor(const stripe::Block& block,                         //
                                              const std::map<std::string, std::string>& mapping,  //
                                              const TileShape& tile,                              //
                                              bool interleave = false);

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

class TagFusionStrategy : public FusionStrategy {
 public:
  TagFusionStrategy() {}
  explicit TagFusionStrategy(const proto::FusionPass& options) : options_(options) {}
  bool AttemptFuse(const stripe::Block& parent, const stripe::Block& a, const stripe::Block& b) {
    bool tag_match = parent.has_tags(stripe::FromProto(options_.parent_reqs())) &&  //
                     a.has_tags(stripe::FromProto(options_.a_reqs())) &&      //
                     b.has_tags(stripe::FromProto(options_.b_reqs())) &&      //
                     !a.has_any_tags(stripe::FromProto(options_.exclude())) &&      //
                     !b.has_any_tags(stripe::FromProto(options_.exclude()));
    if (!tag_match) {
      return false;
    }
    if (options_.output_match() && a.idxs.size() != b.idxs.size()) {
      return false;
    }
    return true;
  }
  void OnFailed() {}
  void OnFused(const AliasMap& outer, stripe::Block* block, const stripe::Block& a, const stripe::Block& b) {
    block->add_tags(stripe::FromProto(options_.fused_set()));
    for (auto stmt : block->stmts) {
       auto sub = stripe::Block::Downcast(stmt);
       if (sub) {
         sub->remove_tags(stripe::FromProto(options_.inner_remove_set()));
       }
    }
  }
  bool NoInner() { return options_.no_inner(); }
  const proto::FusionPass& Options() { return options_; }

 private:
  const proto::FusionPass options_;
};

void FusionInner(const AliasMap& scope, stripe::Block* block, TagFusionStrategy* strategy, bool no_inner = false);

class AlwaysFuseRecursive : public TagFusionStrategy {
 public:
  bool AttemptFuse(const stripe::Block& parent, const stripe::Block& a, const stripe::Block& b) { return true; }
  void OnFailed() {}
  void OnFused(const AliasMap& outer, stripe::Block* block, const stripe::Block& a, const stripe::Block& b) {
    block->location = a.location;
    AliasMap inner(outer, block);
    FusionInner(inner, block, this, false);
  }
};

class FusionPass final : public CompilePass {
 public:
  explicit FusionPass(const proto::FusionPass& options) : options_{options} {}
  void Apply(CompilerState* state) const final;

 private:
  proto::FusionPass options_;
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
