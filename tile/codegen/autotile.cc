// Copyright 2018, Intel Corporation

#include "tile/codegen/autotile.h"

#include "base/util/logging.h"
#include "base/util/printstring.h"
#include "base/util/stream_container.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

static size_t ComputeSize(const std::map<std::string, size_t>& tile_by_name, const Refinement& ref,
                          const proto::AutotilePass& options) {
  if (options.skip_1d() && ref.shape.dims.size() == 1) {
    return 0;
  }
  size_t total_size = 1;
  if (options.use_bytes()) {
    for (const auto& d : ref.shape.dims) {
      total_size *= d.size;
    }
  }
  Affine flat = ref.FlatAccess();
  for (const auto& kvp : flat.getMap()) {
    if (!kvp.first.empty()) {
      total_size *= tile_by_name.at(kvp.first);
    }
  }
  return total_size;
}

static std::pair<size_t, size_t> ComputeSizes(const std::map<std::string, size_t>& tile_by_name, const Block& block,
                                              const proto::AutotilePass& options) {
  std::pair<size_t, size_t> out;
  for (const auto& ref : block.refs) {
    if (ref.dir == RefDir::None) {
      continue;
    }
    if (ref.dir == RefDir::In) {
      out.second += ComputeSize(tile_by_name, ref, options);
    } else if (ref.dir == RefDir::Out) {
      out.first += ComputeSize(tile_by_name, ref, options);
    }
  }
  return out;
}

static double TileCost(const Block& block, const proto::AutotilePass& options, const TileShape& tile) {
  std::map<std::string, size_t> tile_by_name;
  for (size_t i = 0; i < block.idxs.size(); i++) {
    tile_by_name[block.idxs[i].name] = tile[i];
  }
  auto sizes = ComputeSizes(tile_by_name, block, options);
  if (static_cast<int64_t>(sizes.first) > options.max_output_size() ||
      static_cast<int64_t>(sizes.second) > options.max_input_size()) {
    return std::numeric_limits<double>::infinity();
  }
  double tot_compute = 1;
  double tile_expand = 1;
  for (size_t i = 0; i < tile.size(); i++) {
    tot_compute *= tile[i];
    size_t padded_size = (block.idxs[i].range + tile[i] - 1) / tile[i] * tile[i];
    tile_expand *= static_cast<double>(padded_size) / static_cast<double>(block.idxs[i].range);
  }
  double cost = tile_expand * (options.output_cost() * sizes.first + options.input_cost() * sizes.second) / tot_compute;
  return cost;
}

static TileShape PickBestShape(const Block& block, const proto::AutotilePass& options) {
  size_t sz = block.idxs.size();
  std::multimap<double, TileShape> by_cost;
  std::map<TileShape, double> by_tile;
  std::set<std::pair<double, TileShape>> to_do;
  TileShape tile(sz, 1);
  double cost = TileCost(block, options, tile);
  double base_cost = cost;
  by_tile.emplace(tile, cost);
  by_cost.emplace(cost, tile);
  to_do.emplace(cost, tile);
  while (!to_do.empty()) {
    auto it = to_do.begin();
    if (it->first > cost && options.fast()) {
      break;
    }
    cost = it->first;
    tile = it->second;
    to_do.erase(*it);
    for (size_t i = 0; i < sz; i++) {
      uint64_t prev = tile[i];
      if (options.only_po2()) {
        tile[i] = std::min(2 * tile[i], block.idxs[i].range);
      } else {
        tile[i] = std::min(tile[i] + 1, block.idxs[i].range);
      }
      if (!by_tile.count(tile)) {
        cost = TileCost(block, options, tile);
        by_tile.emplace(tile, cost);
        by_cost.emplace(cost, tile);
        if (!std::isinf(cost)) {
          to_do.emplace(cost, tile);
        }
      }
      tile[i] = prev;
    }
  }
  IVLOG(2, "Autotile Result: " << by_cost.begin()->second << ", Cost: " << by_cost.begin()->first / base_cost);
  return by_cost.begin()->second;
}

void AutotilePass(Block* root, const proto::AutotilePass& options) {
  auto reqs = FromProto(options.reqs());
  RunOnBlocks(root, reqs, [&options](const AliasMap& map, Block* block) {
    TileShape tile = PickBestShape(*block, options);
    if (ApplyTile(block, tile)) {
      AddTags(block, FromProto(options.outer_set()));
      auto inner = Block::Downcast(*block->stmts.begin());
      AddTags(inner.get(), FromProto(options.inner_set()));
    }
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
