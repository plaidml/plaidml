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

  bool IsValid(const proto::AutotilePass& options) const {
    return !((options.max_output_size() && outputs > options.max_output_size()) ||
             (options.max_input_size() && inputs > options.max_input_size()) ||
             (options.max_total_size() && total > options.max_total_size()));
  }
};

struct TileDimension {
  size_t size = 0;
  size_t count = 0;
};

struct Tile {
  std::vector<TileDimension> dims;

  explicit Tile(const Block& block) : dims(block.idxs.size()) {
    for (size_t i = 0; i < block.idxs.size(); i++) {
      set(i, 1, block.idxs[i].range);
    }
  }

  void set(size_t i, size_t size, size_t range) {
    size = std::min(size, range);
    dims[i].size = size;
    dims[i].count = IntDivCeil(range, size);
  }

  size_t counts_product() const {
    size_t ret = 1;
    for (const auto& dim : dims) {
      ret *= dim.count;
    }
    return ret;
  }

  size_t sizes_product() const {
    size_t ret = 1;
    for (const auto& dim : dims) {
      ret *= dim.size;
    }
    return ret;
  }

  TileShape counts() const {
    TileShape ret(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
      ret[i] = dims[i].count;
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
};

bool operator<(const TileDimension& lhs, const TileDimension& rhs) {
  return std::tie(lhs.size, lhs.count) < std::tie(rhs.size, rhs.count);
}

bool operator<(const Tile& lhs, const Tile& rhs) {  //
  return lhs.dims < rhs.dims;
}

std::ostream& operator<<(std::ostream& os, const TileMetrics& metrics) {
  os << "(" << metrics.inputs << ", " << metrics.outputs << ", " << metrics.total << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const TileDimension& dim) {
  os << dim.size << ":" << dim.count;
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
  auto bytes = tiled.sizes_product_bytes();
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

struct ComputeDensityCostModel {
  const proto::AutotilePass& options;

  explicit ComputeDensityCostModel(const proto::AutotilePass& options) : options(options) {}

  bool IndexFilter(const Block& block, const Index& idx) const {  //
    return !idx.has_tag("bank");
  }

  double ComputeCost(const Block& block, const Tile& tile) const {
    std::map<std::string, size_t> tile_by_name;
    for (size_t i = 0; i < block.idxs.size(); i++) {
      tile_by_name[block.idxs[i].name] = tile.dims[i].size;
    }
    auto metrics = ComputeSizes(tile_by_name, block, options);
    IVLOG(4, "    TileCost> tile_by_name: " << tile_by_name << ", metrics: " << metrics);
    if (!metrics.IsValid(options)) {
      return std::numeric_limits<double>::infinity();
    }
    double total_compute = 1;
    double tile_expand = 1;
    for (size_t i = 0; i < block.idxs.size(); i++) {
      const auto& tile_dim = tile.dims[i];
      total_compute *= tile_by_name[block.idxs[i].name];
      size_t padded_size = tile_dim.size * tile_dim.count;
      tile_expand *= static_cast<double>(padded_size) / static_cast<double>(block.idxs[i].range);
    }
    auto input_cost = options.input_cost() * metrics.inputs;
    auto output_cost = options.output_cost() * metrics.outputs;
    double cost = (tile_expand * (output_cost + input_cost) / total_compute) + tile.counts_product();
    IVLOG(4, "        cost: " << cost);
    return cost;
  }
};

struct PartitionComputeCostModel {
  double ideal_cost;
  std::set<const Index*> acc_idxs;

  PartitionComputeCostModel(const Block& block, const proto::PartitionPass& options)
      : ideal_cost(block.idxs_product() / (1.0 * options.num_parts())),  //
        acc_idxs(block.accumulation_idxs())                              //
  {}

  bool IndexFilter(const Block& block, const Index& idx) const {  //
    return !idx.has_tag("bank") && !acc_idxs.count(&idx);
  }

  double ComputeCost(const Block& block, const Tile& tile) const {
    double cost = tile.counts_product();
    if (cost < ideal_cost) {
      return std::numeric_limits<double>::infinity();
    }
    return cost;
  }
};

struct TileSearchState {
  std::multimap<double, Tile> by_cost;
  std::map<Tile, double> by_tile;
  std::set<std::pair<double, Tile>> todo;

  void insert(const Tile& tile, double cost) {
    by_tile.emplace(tile, cost);
    by_cost.emplace(cost, tile);
    if (!std::isinf(cost)) {
      todo.emplace(cost, tile);
    }
  }
};

struct TileResult {
  Tile tile;
  double cost;
};

template <typename CostModel>
TileResult PickBestTile(const Block& block, bool only_po2, bool is_fast, const CostModel& model) {
  IVLOG(3, "Autotile> PickBestTile> block: " << block.name);
  TileSearchState state;
  Tile tile(block);
  double cost = model.ComputeCost(block, tile);
  double base_cost = cost;
  state.insert(tile, base_cost);
  while (!state.todo.empty()) {
    auto it = state.todo.begin();
    if (it->first > cost && is_fast) {
      break;
    }
    cost = it->first;
    tile = it->second;
    state.todo.erase(*it);
    for (size_t i = 0; i < block.idxs.size(); i++) {
      if (!model.IndexFilter(block, block.idxs[i])) {
        continue;
      }
      auto prev = tile.dims[i];
      if (only_po2) {
        tile.set(i, 2 * prev.size, block.idxs[i].range);
      } else {
        tile.set(i, prev.size + 1, block.idxs[i].range);
      }
      if (!state.by_tile.count(tile)) {
        cost = model.ComputeCost(block, tile);
        state.insert(tile, cost);
      }
      tile.dims[i] = prev;
    }
  }
  auto winner = state.by_cost.begin();
  IVLOG(4, "Tiles: ");
  for (const auto& kvp : state.by_cost) {
    IVLOG(4, "    " << kvp.first << ": " << kvp.second);
  }
  IVLOG(3, "    best: " << winner->second << ", cost: " << winner->first / base_cost);
  return TileResult{winner->second, winner->first};
}

}  // namespace

void AutotilePass(Block* root, const proto::AutotilePass& options) {
  auto reqs = FromProto(options.reqs());
  RunOnBlocks(root, reqs, [&options](const AliasMap& map, Block* block) {
    ComputeDensityCostModel model(options);
    auto result = PickBestTile(*block, options.only_po2(), options.fast(), model);
    IVLOG(2, "Autotile> block: " << block->name << ", tile: " << result.tile << ", cost: " << result.cost);
    if (!std::isinf(result.cost)) {
      if (ApplyTile(block, result.tile.sizes(), false)) {
        auto inner = block->SubBlock(0);
        if (options.copy_tags()) {
          inner->tags = block->tags;
        }
        block->add_tags(FromProto(options.outer_set()));
        inner->add_tags(FromProto(options.inner_set()));
      }
    } else {
      LOG(WARNING) << "Autotile> block: " << block->name << " was NOT split: " << result.tile;
    }
  });
}

void PartitionComputePass(stripe::Block* root, const proto::PartitionPass& options) {
  auto reqs = FromProto(options.reqs());
  RunOnBlocks(root, reqs, [&options](const AliasMap& map, Block* block) {
    PartitionComputeCostModel model(*block, options);
    auto result = PickBestTile(*block, false, false, model);
    IVLOG(2, "PartitionCompute> block: " << block->name                           //
                                         << ", tile: " << result.tile             //
                                         << ", cost: " << result.cost             //
                                         << ", ideal_cost: " << model.ideal_cost  //
                                         << ", ratio: " << model.ideal_cost / result.cost);
    if (ApplyTile(block, result.tile.counts(), false)) {
      auto inner = block->SubBlock(0);
      inner->tags = block->tags;
      block->tags.clear();
      block->add_tags(FromProto(options.set_tags()));
      if (!options.idx_tag().empty()) {
        for (auto& idx : block->idxs) {
          if (idx.range > 1) {
            idx.set_tag(options.idx_tag());
          }
          // HACK: remove this somehow
          idx.tags.erase("bank");
        }
      }
    }
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
