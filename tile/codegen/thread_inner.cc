// Copyright 2018, Intel Corporation

#include "tile/codegen/thread_inner.h"

#include <algorithm>

#include <boost/format.hpp>

#include "base/util/throw.h"
#include "tile/codegen/tile.h"
#include "tile/math/util.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT
using math::NearestPo2;
using math::RoundUp;

void DoThreadInnerPass(const AliasMap& scope, Block* block, const proto::ThreadInnerPass& options) {
  if (block->has_any_tags(FromProto(options.exclude()))) {
    return;
  }
  if (block->ref_outs(true).size() != 1) {
    if (block->ref_outs(true).size() == 0) {
      // We may remove the output refinements in the contracts during optimizations,
      // so here is not definitely wrong.
      return;
    }
    // If there is "eltwise" tag, the refinements must have aligned accesses
    // though the accesses may not exactly same
    if (!block->has_tag("eltwise")) {
      const auto refs = block->ref_outs(true);
      for (size_t i = 1; i < refs.size(); ++i) {
        if (refs[0]->access != refs[i]->access) {
          throw std::runtime_error("Thread inner pass only works with a single output");
        }
      }
    }
  }
  const Refinement* out_ref = block->ref_outs(true)[0];
  const auto& idxs = block->idxs;
  Affine flat = out_ref->FlatAccess();
  std::vector<size_t> sorted_idxs;
  // Pull across indexes that are part of the output
  for (size_t i = 0; i < idxs.size(); i++) {
    // TODO: Rollup
    if (flat.get(idxs[i].name) != 0) {
      sorted_idxs.push_back(i);
    }
  }
  IVLOG(3, "Output indexes: " << sorted_idxs);
  // Sort indexes by is_out, power-of-twoness, then size
  auto sort_func = [&](size_t i, size_t j) {
    bool out_i = flat.get(idxs[i].name) == 0;
    bool out_j = flat.get(idxs[j].name) == 0;
    if (out_i != out_j) {
      return out_j;
    }
    double ratio_i = static_cast<double>(idxs[i].range) / NearestPo2(idxs[i].range);
    double ratio_j = static_cast<double>(idxs[j].range) / NearestPo2(idxs[j].range);
    if (ratio_i != ratio_j) {
      return ratio_i > ratio_j;
    }
    return idxs[i].range > idxs[j].range;
  };
  TileShape tile(block->idxs.size(), 1);
  std::sort(sorted_idxs.begin(), sorted_idxs.end(), sort_func);
  IVLOG(3, "Sorted indexes: " << sorted_idxs);
  size_t cur = 0;
  size_t threads = options.threads();
  while (threads > 1 && cur < sorted_idxs.size()) {
    size_t ci = sorted_idxs[cur];
    size_t split = std::min(size_t(threads), size_t(NearestPo2(idxs[ci].range)));
    tile[ci] = split;
    threads /= split;
    cur++;
  }
  for (size_t i = 0; i < tile.size(); i++) {
    tile[i] = RoundUp(block->idxs[i].range, tile[i]);
  }
  ApplyTile(block, tile, false, false, true);
  block->add_tags(FromProto(options.outer_set()));
  block->SubBlock(0)->add_tags(FromProto(options.inner_set()));
}

// Localize starting from root for things that match reqs
void ThreadInnerPass::Apply(CompilerState* state) const {
  auto reqs = stripe::FromProto(options_.reqs());
  RunOnBlocks(state->entry(), reqs, [this](const AliasMap& map, stripe::Block* block) {  //
    DoThreadInnerPass(map, block, options_);
  });
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<ThreadInnerPass, proto::ThreadInnerPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
