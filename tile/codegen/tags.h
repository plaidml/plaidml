// Copyright 2018, Intel Corporation

#pragma once

#include <string>

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

template <typename F>
void RunOnBlocksRecurse(const AliasMap& map, stripe::Block* block, const stripe::Tags& reqs, const F& func) {
  if (block->has_tags(reqs)) {
    func(map, block);
  } else {
    for (auto& stmt : block->stmts) {
      auto inner = stripe::Block::Downcast(stmt);
      if (inner) {
        AliasMap inner_map(map, inner.get());
        RunOnBlocksRecurse(inner_map, inner.get(), reqs, func);
      }
    }
  }
}

template <typename F>
void RunOnBlocks(stripe::Block* root, const stripe::Tags& reqs, const F& func) {
  AliasMap base;
  AliasMap root_map(base, root);
  RunOnBlocksRecurse(root_map, root, reqs, func);
}

inline stripe::Tags FromProto(const google::protobuf::RepeatedPtrField<std::string>& pb_tags) {
  stripe::Tags tags;
  for (const auto& tag : pb_tags) {
    tags.emplace(tag);
  }
  return tags;
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
