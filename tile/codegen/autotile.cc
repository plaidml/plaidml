// Copyright 2018, Intel Corporation

#include "tile/codegen/autotile.h"

#include "base/util/logging.h"
#include "base/util/stream_container.h"
#include "base/util/throw.h"
#include "tile/codegen/math.h"
#include "tile/codegen/tags.h"
#include "tile/codegen/tile.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

namespace {

struct TileMetrics {
  int64_t inputs = 0;
  int64_t outputs = 0;
  int64_t total = 0;
};

struct TileDimension {
  size_t range = 0;
  size_t size = 0;
};

struct Tile {
  explicit Tile(const Block& block) : dims(block.idxs.size()) {
    for (size_t i = 0; i < block.idxs.size(); i++) {
      set(i, 1, block.idxs[i].range);
    }
  }

  std::vector<TileDimension> dims;
  TileMetrics metrics;

  void set(size_t i, size_t value, size_t range) {
    value = std::min(value, range);
    dims[i].range = value;
    dims[i].size = IntDivCeil(range, value);
  }

  size_t product() const {
    size_t ret = 1;
    for (const auto& dim : dims) {
      ret *= dim.range;
    }
    return ret;
  }

  TileShape ranges() const {
    TileShape ret(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
      ret[i] = dims[i].range;
    }
    return ret;
  }

  TileShape sizes() const {
    TileShape ret(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
      ret[i] = dims[i].size;
    }
    return ret;
  }

  bool IsValid(const proto::AutotilePass& options) const {
    return !((options.max_output_size() && metrics.outputs > options.max_output_size()) ||
             (options.max_input_size() && metrics.inputs > options.max_input_size()) ||
             (options.max_total_size() && metrics.total > options.max_total_size()));
  }
};

bool operator<(const TileDimension& lhs, const TileDimension& rhs) {
  return std::tie(lhs.range, lhs.size) < std::tie(rhs.range, rhs.size);
}

bool operator<(const Tile& lhs, const Tile& rhs) {  //
  return lhs.dims < rhs.dims;
}

std::ostream& operator<<(std::ostream& os, const TileMetrics& metrics) {
  os << "(" << metrics.inputs << ", " << metrics.outputs << ", " << metrics.total << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const TileDimension& dim) {
  os << dim.range << ":" << dim.size;
  return os;
}

std::ostream& operator<<(std::ostream& os, const Tile& tile) {
  os << StreamContainer(tile.dims);
  return os;
}

size_t ComputeSize(const std::map<std::string, size_t>& tile_by_name,  //
                   const Refinement& ref) {
  size_t total_size = 1;
  Affine flat = ref.FlatAccess();
  for (const auto& kvp : flat.getMap()) {
    if (!kvp.first.empty()) {
      total_size *= tile_by_name.at(kvp.first);
    }
  }
  IVLOG(4, "    ComputeSize> ref: " << ref.into << ", total_size: " << total_size);
  return total_size;
}

size_t ComputeBytes(const std::map<std::string, size_t>& tile_by_name,  //
                    const Refinement& ref) {
  auto tiled = ref.ApplyTile(tile_by_name);
  auto bytes = tiled.byte_size();
  IVLOG(4, "    ComputeBytes> ref: " << ref);
  IVLOG(4, "                tiled: " << tiled);
  IVLOG(4, "                bytes: " << bytes);
  return bytes;
}

TileMetrics ComputeSizes(const std::map<std::string, size_t>& tile_by_name,  //
                         const Block& block,                                 //
                         const proto::AutotilePass& options) {
  TileMetrics ret;
  for (const auto& ref : block.refs) {
    if (ref.dir == RefDir::None) {
      continue;
    }
    if (options.skip_1d() && ref.interior_shape.dims.size() == 1) {
      continue;
    }
    auto size = options.use_bytes() ? ComputeBytes(tile_by_name, ref)  //
                                    : ComputeSize(tile_by_name, ref);
    ret.total += size;
    if (ref.dir == RefDir::In) {
      ret.inputs += size;
    } else if (ref.dir == RefDir::Out) {
      ret.outputs += size;
    }
  }
  return ret;
}

double TileCost(const Block& block, const proto::AutotilePass& options, Tile* tile) {
  std::map<std::string, size_t> tile_by_name;
  for (size_t i = 0; i < block.idxs.size(); i++) {
    tile_by_name[block.idxs[i].name] = options.use_bytes() ? tile->dims[i].size : tile->dims[i].range;
  }
  tile->metrics = ComputeSizes(tile_by_name, block, options);
  IVLOG(3, "    TileCost> tile_by_name: " << tile_by_name << ", metrics: " << tile->metrics);
  if (!tile->IsValid(options)) {
    return std::numeric_limits<double>::infinity();
  }
  double total_compute = 1;
  double tile_expand = 1;
  for (size_t i = 0; i < block.idxs.size(); i++) {
    const auto& tile_dim = tile->dims[i];
    total_compute *= tile_by_name[block.idxs[i].name];
    size_t padded_size = tile_dim.range * tile_dim.size;
    tile_expand *= static_cast<double>(padded_size) / static_cast<double>(block.idxs[i].range);
  }
  auto input_cost = options.input_cost() * tile->metrics.inputs;
  auto output_cost = options.output_cost() * tile->metrics.outputs;
  double cost = tile_expand * (output_cost + input_cost) / total_compute;
  IVLOG(3, "        cost: " << cost);
  return cost;
}

Tile PickBestTile(const Block& block, const proto::AutotilePass& options) {
  IVLOG(3, "Autotile> PickBestTile> block: " << block.name);
  std::multimap<double, Tile> by_cost;
  std::map<Tile, double> by_tile;
  std::set<std::pair<double, Tile>> to_do;
  Tile tile(block);
  double cost = TileCost(block, options, &tile);
  double base_cost = cost;
  bool any_valid = tile.IsValid(options);
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
    for (size_t i = 0; i < block.idxs.size(); i++) {
      if (block.idxs[i].has_tag("bank")) {
        continue;
      }
      auto prev = tile.dims[i];
      if (options.only_po2()) {
        tile.set(i, 2 * tile.dims[i].range, block.idxs[i].range);
      } else {
        tile.set(i, tile.dims[i].range + 1, block.idxs[i].range);
      }
      if (!by_tile.count(tile)) {
        cost = TileCost(block, options, &tile);
        any_valid |= tile.IsValid(options);
        by_tile.emplace(tile, cost);
        by_cost.emplace(cost, tile);
        if (!any_valid || !std::isinf(cost)) {
          to_do.emplace(cost, tile);
        }
      }
      tile.dims[i] = prev;
    }
  }
  auto winner = by_cost.begin();
  IVLOG(4, "Tiles: ");
  for (const auto& kvp : by_cost) {
    IVLOG(4, "    " << kvp.first << ": " << kvp.second);
  }
  IVLOG(3, "    best: " << winner->second << ", cost: " << winner->first / base_cost);
  return winner->second;
}

bool TileViaStrategy(Block* block, const Tile& tile, const proto::AutotilePass& options) {
  IVLOG(2, "Autotile> block: " << block->name         //
                               << ", tile: " << tile  //
                               << ", metrics: " << tile.metrics);
  switch (options.strategy()) {
    case proto::TileStrategy::TileRanges:
      return ApplyTile(block, tile.ranges());
    case proto::TileStrategy::TileSizes:
      return ApplyTile(block, tile.sizes());
    case proto::TileStrategy::TileSplitIndex:
      // TODO
      break;
  }
  throw_with_trace(std::runtime_error("Unsupported tiling strategy"));
}

}  // namespace

void AutotilePass(Block* root, const proto::AutotilePass& options) {
  auto reqs = FromProto(options.reqs());
  RunOnBlocks(root, reqs, [&options](const AliasMap& map, Block* block) {
    auto tile = PickBestTile(*block, options);
    if (tile.product() > 1 && TileViaStrategy(block, tile, options)) {
      auto inner = block->SubBlock(0);
      if (options.copy_tags()) {
        inner->tags = block->tags;
      }
      if (options.drop_outer_tags()) {
        block->tags.clear();
      }
      block->add_tags(FromProto(options.outer_set()));
      inner->add_tags(FromProto(options.inner_set()));
      if (!options.idx_tag().empty()) {
        for (auto& idx : block->idxs) {
          if (idx.range > 1) {
            idx.set_tag(options.idx_tag());
          }
          // HACK: remove this somehow
          idx.tags.erase("bank");
        }
      }
    } else if (!tile.IsValid(options)) {
      LOG(WARNING) << "Autotile> block: " << block->name << " was NOT split: " << tile.metrics;
    }
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
