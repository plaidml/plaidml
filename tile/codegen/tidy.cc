// Copyright 2018, Intel Corporation

#include "tile/codegen/tidy.h"

#include "base/util/logging.h"
#include "tile/codegen/tags.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

void PruneIndexes(stripe::Block* block) {
  // Find all the indexes to remove
  std::set<const Index*> to_remove;
  for (const auto& idx : block->idxs) {
    if (idx.range == 1 && idx.affine == 0) {
      to_remove.emplace(&idx);
    }
  }
  // Remove from refinements
  for (auto& refs : block->refs) {
    for (auto& aff : refs.access) {
      for (const auto& idx : to_remove) {
        aff.mutateMap().erase(idx->name);
      }
    }
  }
  // Remove from constraints
  for (auto& con : block->constraints) {
    for (const auto& idx : to_remove) {
      con.mutateMap().erase(idx->name);
    }
  }
  // Remove from index list
  block->idxs.erase(std::remove_if(block->idxs.begin(), block->idxs.end(),
                                   [&to_remove](const Index& idx) { return to_remove.count(&idx); }),
                    block->idxs.end());
  // Remove from inner blocks
  for (auto& stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      for (auto& oidx : inner->idxs) {
        for (const auto& idx : to_remove) {
          oidx.affine.mutateMap().erase(idx->name);
        }
      }
    }
  }
}

void PruneIndexesPass(Block* root, const proto::GenericPass& options) {
  auto reqs = FromProto(options.reqs());
  RunOnBlocks(root, reqs, [](const AliasMap& map, Block* block) {  //
    PruneIndexes(block);
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
