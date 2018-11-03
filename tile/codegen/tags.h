// Copyright 2018, Intel Corporation

#pragma once

#include <set>
#include <string>

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using Tags = std::set<std::string>;

inline bool HasTags(const stripe::Statement& stmt, const Tags& tags) {
  for (const auto& tag : tags) {
    if (stmt.tags.count(tag) == 0) {
      return false;
    }
  }
  return true;
}

inline void AddTags(stripe::Statement* stmt, const Tags& tags) {
  for (const auto& tag : tags) {
    stmt->tags.emplace(tag);
  }
}

template <typename F>
void RunOnBlocksRecurse(const AliasMap& map, stripe::Block* block, const Tags& reqs, const F& func) {
  if (HasTags(*block, reqs)) {
    func(map, block);
  } else {
    for (const auto& stmt : block->stmts) {
      auto inner = stripe::Block::Downcast(stmt);
      if (inner) {
        AliasMap inner_map(map, *inner);
        RunOnBlocksRecurse(inner_map, inner.get(), reqs, func);
      }
    }
  }
}

template <typename F>
void RunOnBlocks(stripe::Block* root, const Tags& reqs, const F& func) {
  AliasMap base;
  AliasMap root_map(base, *root);
  RunOnBlocksRecurse(root_map, root, reqs, func);
}

inline Tags FromProto(const google::protobuf::RepeatedPtrField<std::string>& pb_tags) {
  Tags tags;
  for (const auto& tag : pb_tags) {
    tags.emplace(tag);
  }
  return tags;
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
