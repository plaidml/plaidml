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

void ThreadInnerPass(const AliasMap& scope, Block* block, int64_t threads) {
  if (block->ref_outs().size() != 1) {
    throw std::runtime_error("Thread inner pass only works with a single output");
  }
  const Refinement* out_ref = block->ref_outs()[0];
  const auto& idxs = block->idxs;
  Affine flat = out_ref->FlatAccess();
  std::vector<size_t> sorted_idxs;
  // Pull across indexes that are part of the output
  for (size_t i = 0; i < idxs.size(); i++) {
    sorted_idxs.push_back(i);
  }
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
  size_t cur = 0;

  while (threads > 1 && cur < idxs.size()) {
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
  block->set_tag("gpu_thread");
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
