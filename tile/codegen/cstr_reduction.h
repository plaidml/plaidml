// Copyright 2018, Intel Corporation

#pragma once

#include <map>
#include <set>
#include <vector>

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void LightCstrReduction(const AliasMap& alias_map, stripe::Block* block, const proto::ConstraintReductionPass& options);

inline void LightCstrReductionPass(stripe::Block* root, const proto::ConstraintReductionPass& options) {
  auto reqs = stripe::FromProto(options.reqs());
  RunOnBlocks(
      root, reqs,
      [options](const AliasMap& alias_map, stripe::Block* block) {  //
        LightCstrReduction(alias_map, block, options);
      },
      true);
}

void IlpCstrReduction(const AliasMap& alias_map, stripe::Block* block, const proto::ConstraintReductionPass& options);

inline void IlpCstrReductionPass(stripe::Block* root, const proto::ConstraintReductionPass& options) {
  auto reqs = stripe::FromProto(options.reqs());
  RunOnBlocks(
      root, reqs,
      [options](const AliasMap& alias_map, stripe::Block* block) {  //
        IlpCstrReduction(alias_map, block, options);
      },
      true);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
