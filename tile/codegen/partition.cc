// Copyright 2018, Intel Corporation

#include "tile/codegen/partition.h"

#include "base/util/logging.h"
#include "base/util/printstring.h"
#include "base/util/stream_container.h"
#include "tile/codegen/localize.h"
#include "tile/codegen/tags.h"
#include "tile/codegen/tile.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

void PartitionPass(Block* root, const proto::PartitionPass& options) {
  auto reqs = FromProto(options.reqs());
  RunOnBlocks(root, reqs, [&options](const AliasMap& map, Block* block) {
    if (block->ref_outs().size() != 1) {
      IVLOG(1, "Skipped partition due to multiple outputs");
      return;  // Only work on blocks with a single output
    }
    // Get the (only) output
    const Refinement* out_ref = block->ref_outs()[0];
    // Find the largest input
    size_t biggest = 0;
    const Refinement* big_ref = nullptr;
    for (const auto& ref : block->ref_ins()) {
      // Get the source buffer size
      size_t src_size = map.at(ref->into).base_ref->shape.elem_size();
      if (src_size > biggest) {
        biggest = src_size;
        big_ref = ref;
      }
    }
    if (big_ref == nullptr) {
      IVLOG(1, "Skipped partition due to no inputs");
      return;  // No inputs?  Skip this block
    }
    // Find the evenest index that has a non-zero stride both on the large
    // input and on the output refinment
    double best_ratio = 0;
    size_t best_tile = 0;
    size_t idx_id = 0;
    TileShape tile;
    for (size_t i = 0; i < block->idxs.size(); i++) {
      const auto& idx = block->idxs[i];
      tile.push_back(idx.range);
      if (big_ref->FlatAccess().get(idx.name) == 0) {
        continue;
      }
      if (out_ref->FlatAccess().get(idx.name) == 0) {
        continue;
      }
      size_t tile_size = (idx.range + options.num_parts() - 1) / options.num_parts();
      size_t tile_count = (idx.range + tile_size - 1) / tile_size;
      size_t rounded_size = tile_size * tile_count;
      double ratio = static_cast<double>(idx.range) / static_cast<double>(rounded_size);
      if (ratio > best_ratio) {
        best_ratio = idx.range;
        best_tile = tile_size;
        idx_id = i;
      }
    }
    if (best_tile == 0) {
      IVLOG(1, "Skipped partition due to no valid indexes");
      return;  // No valid indexes?  Skip this block
    }
    // Determine the bank_dim for the split memory
    int bank_dim = -1;
    for (size_t i = 0; i < big_ref->shape.dims.size(); i++) {
      if (big_ref->access[i].get(block->idxs[idx_id].name)) {
        if (bank_dim != -1) {
          IVLOG(1, "Skipped partition due to complex banking");
          return;
        }
        bank_dim = i;
      }
    }
    // Update the bank_dim for the memory in question
    const auto& ai = map.at(big_ref->into);
    ai.base_ref->bank_dim = bank_dim;
    FixupRefs(ai.base_block, ai.base_ref->into);
    // Now, tile on that index
    tile[idx_id] = best_tile;
    // Do the tiling
    ApplyTile(block, tile, false);
    // Add the new tags
    AddTags(block, FromProto(options.outer_set()));
    auto inner = Block::Downcast(*block->stmts.begin());
    AddTags(inner.get(), FromProto(options.inner_set()));
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
