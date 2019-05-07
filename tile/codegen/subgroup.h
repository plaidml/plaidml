// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

void Subgroup(stripe::Block* block, const AliasMap& map, const proto::SubgroupPass& options);

void VectorizeTx(stripe::Block* block, const AliasMap& map, size_t read_align_bytes, size_t write_align_bytes);

inline void SubgroupPass(stripe::Block* root, const proto::SubgroupPass& options) {
  auto reqs = stripe::FromProto(options.reqs());
  RunOnBlocks(root, reqs, [&](const AliasMap& map, stripe::Block* block) {  //
    Subgroup(block, map, options);
  });
}

inline void VectorizeTxPass(stripe::Block* root, const proto::VectorizePass& options) {
  auto reqs = stripe::FromProto(options.reqs());
  RunOnBlocks(root, reqs, [options](const AliasMap& map, stripe::Block* block) {  //
    VectorizeTx(block, map, options.read_align_bytes(), options.write_align_bytes());
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
