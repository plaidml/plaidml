// Copyright 2018, Intel Corporation

#pragma once

#include "tile/codegen/alias.h"
#include "tile/codegen/codegen.pb.h"
#include "tile/stripe/stripe.h"

#include <map>
#include <string>

namespace vertexai {
namespace tile {
namespace codegen {

struct RefDefine {
  std::string ref_name;
  stripe::Block *block;
  stripe::StatementIt stmt_iter;
};
typedef std::map<std::string, RefDefine> RefDefineMap;

void Pad(stripe::Block *block, const AliasMap &map,
         const RefDefineMap &ref_def_map);
void CollectRefDefine(stripe::Block *block, RefDefineMap *ref_def_map);

inline void PadPass(stripe::Block *root, const proto::GenericPass &options) {
  auto reqs = stripe::FromProto(options.reqs());
  RefDefineMap ref_def_map;
  CollectRefDefine(root->SubBlock(0).get(), &ref_def_map);
  RunOnBlocks(root, reqs, [&](const AliasMap &map, stripe::Block *block) { //
    Pad(block, map, ref_def_map);
  });
}

} // namespace codegen
} // namespace tile
} // namespace vertexai
