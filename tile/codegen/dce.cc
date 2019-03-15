// Copyright 2018, Intel Corporation

#include "tile/codegen/dce.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT
using namespace math;    // NOLINT

void DeadCodeElimination(const AliasMap& alias_map, Block* block) {
  // Make sure this is not the root block which hasn't parent
  if (alias_map.parent_block() == nullptr) {
    return;
  }

  // First, if the block hasn't output, remove it
  if (std::none_of(block->refs.begin(), block->refs.end(),
                   [](const Refinement& ref) { return ref.dir == RefDir::Out; })) {
    IVLOG(1, "Remove block " << block->name);
    block->set_tag("removed");
    return;
  }
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
