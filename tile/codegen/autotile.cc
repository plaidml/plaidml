// Copyright 2018, Intel Corporation

#include "tile/codegen/autotile.h"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <boost/math/special_functions/prime.hpp>

#include "base/util/logging.h"
#include "base/util/stream_container.h"
#include "base/util/throw.h"
#include "tile/codegen/alias.h"
#include "tile/codegen/tile.h"
#include "tile/math/util.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

namespace {

struct TileMetrics {
  int64_t input_bytes = 0;
  int64_t max_input_bytes = 0;
  double input_bandwidth = 0;
  int64_t output_bytes = 0;
  int64_t max_output_bytes = 0;
  double output_bandwidth = 0;
  int64_t total_bytes = 0;
  double total_bandwidth = 0;

  bool IsValid(const proto::AutotilePass& options) const {
    return !((options.max_output_size() && output_bytes > options.max_output_size()) ||
             (options.max_per_output_size() && max_output_bytes > options.max_per_output_size()) ||
             (options.max_input_size() && input_bytes > options.max_input_size()) ||
             (options.max_per_input_size() && max_input_bytes > options.max_per_input_size()) ||
             (options.max_total_size() && total_bytes > options.max_total_size()));
  }
};

struct TileDimension {
  size_t size = 0;
  size_t count = 0;
};

struct Tile {
  std::vector<TileDimension> dims;

  Tile(const Block& block, size_t initial_tile_size) : dims(block.idxs.size()) {
    for (size_t i = 0; i < block.idxs.size(); i++) {
      set(i, initial_tile_size, block.idxs[i].range);
    }
  }

  void set(size_t i, size_t size, size_t range) {
    size = std::min(size, range);
    dims[i].size = size;
    dims[i].count = math::RoundUp(range, size);
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
  os << "(" << metrics.input_bytes        //
     << ", " << metrics.max_input_bytes   //
     << ", " << metrics.input_bandwidth   //
     << ", " << metrics.output_bytes      //
     << ", " << metrics.max_output_bytes  //
     << ", " << metrics.output_bandwidth  //
     << ", " << metrics.total_bytes       //
     << ", " << metrics.total_bandwidth << ")";
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

TensorShape MakeOddTile(const TensorShape& tile) {
  TensorShape odd_tile = tile;
  for (size_t i = 0; i < odd_tile.dims.size(); ++i) {
    if ((odd_tile.dims[i].size & 0x1) == 0) {
      ++odd_tile.dims[i].size;
    }
  }
  return odd_tile;
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
    if (!options.loc_name().empty() && ref.location != options.loc_name()) {
      continue;
    }
    auto tiled = ref.ApplyTile(tile_by_name);
    int64_t bytes =
        options.odd_size() ? Codec::Resolve(MakeOddTile(tiled))->byte_size() : Codec::Resolve(tiled)->byte_size();
    double bandwidth = tiled.memory_io(options.cache_width());
    ret.total_bytes += bytes;
    ret.total_bandwidth += bandwidth;
    if (ref.dir == RefDir::In) {
      ret.input_bytes += bytes;
      ret.max_input_bytes = std::max(ret.max_input_bytes, bytes);
      ret.input_bandwidth += bandwidth;
    } else if (ref.dir == RefDir::Out) {
      ret.output_bytes += bytes;
      ret.max_output_bytes = std::max(ret.max_output_bytes, bytes);
      ret.output_bandwidth += bandwidth;
    }
    IVLOG(4, "    ComputeSizes> ref: " << ref);
    IVLOG(4, "                tiled: " << tiled);
    IVLOG(4, "                bytes: " << bytes);
    IVLOG(4, "            bandwidth: " << bandwidth);
    IVLOG(4, "          cache_width: " << options.cache_width());
  }
  return ret;
}

struct Cost {
  enum Outcome {
    Valid,    // A valid cost
    Stop,     // Unusable, and growing the tile size will not help; cease exploration
    Continue  // Unusable, but growing the tile size may help; continue exploration
  };

  Cost(Outcome outcome_) : outcome{outcome_}, value{std::numeric_limits<double>::infinity()} {}  // NOLINT
  Cost(double value_) : outcome{Valid}, value{value_} {}                                         // NOLINT

  Outcome outcome;
  double value;
};

std::ostream& operator<<(std::ostream& os, Cost cost) {
  switch (cost.outcome) {
    case Cost::Stop:
      os << "Stop";
      break;
    case Cost::Continue:
      os << "Continue";
      break;
    default:
    case Cost::Valid:
      os << cost.value;
      break;
  }
  return os;
}

struct ComputeDensityCostModel {
  const proto::AutotilePass& options;
  std::set<const Index*> acc_idxs;
  std::map<size_t, size_t> small_factors;

  explicit ComputeDensityCostModel(const Block& block, const proto::AutotilePass& options)
      : options(options),  //
        acc_idxs(block.accumulation_idxs(true)) {}

  bool IndexFilter(const Block& block, const Index& idx) const {  //
    return options.acc_idxs() || !acc_idxs.count(&idx);
  }

  size_t CountSmallFactors(size_t num) const {
    if (small_factors.find(num) != small_factors.end()) {
      return small_factors.at(num);
    }
    size_t n = num;
    size_t count = 0;
    size_t i = 0;
    while (true) {
      size_t prime = boost::math::prime(i);
      if (prime > options.small_factor_upbound()) {
        break;
      }
      while (n > 1 && n % prime == 0) {
        n /= prime;
        ++count;
      }
      ++i;
    }
    const_cast<std::map<size_t, size_t>&>(small_factors).emplace(num, count);
    return count;
  }

  Cost ComputeCost(const Block& block, const Tile& tile) const {
    std::map<std::string, size_t> tile_by_name;
    for (size_t i = 0; i < block.idxs.size(); i++) {
      tile_by_name[block.idxs[i].name] = tile.dims[i].size;
    }
    auto metrics = ComputeSizes(tile_by_name, block, options);
    IVLOG(4, "    TileCost> tile_by_name: " << tile_by_name << ", metrics: " << metrics);
    if (!metrics.IsValid(options)) {
      return Cost::Stop;
    }
    if (options.max_sizes_product() && tile.sizes_product() > size_t(options.max_sizes_product())) {
      return Cost::Stop;
    }
    if (options.max_po2_product()) {
      size_t tot_po2 = 1;
      for (const auto& d : tile.dims) {
        tot_po2 *= math::NearestPo2(d.size);
      }
      if (tot_po2 > static_cast<size_t>(options.max_po2_product())) {
        return Cost::Stop;
      }
    }

    double total_compute = 1;
    double tile_expand = 1;
    int64_t tot_size = tile.sizes_product();
    int64_t tot_count = tile.counts_product();
    if (options.max_count() && tot_count > options.max_count()) {
      return Cost::Continue;
    }
    int64_t tot_out_size = 1;
    int64_t tot_out_count = 1;
    for (size_t i = 0; i < block.idxs.size(); i++) {
      const auto& tile_dim = tile.dims[i];
      total_compute *= tile_by_name[block.idxs[i].name];
      size_t padded_size = tile_dim.size * tile_dim.count;
      tile_expand *= static_cast<double>(padded_size) / static_cast<double>(block.idxs[i].range);
      if (!acc_idxs.count(&block.idxs[i])) {
        tot_out_size *= tile_dim.size;
        tot_out_count *= tile_dim.count;
      }
    }

    size_t n_factors = 1;
    if (options.small_factor_upbound() > 0) {
      for (auto& iter : tile_by_name) {
        const Index* idx = block.idx_by_name(iter.first);
        if (!acc_idxs.count(idx)) {
          n_factors += CountSmallFactors(iter.second);
        }
      }
    }

    double inv_size_util = static_cast<double>(options.min_size()) / std::min(tot_size, options.min_size());
    double inv_out_size_util =
        static_cast<double>(options.min_out_size()) / std::min(tot_out_size, options.min_out_size());
    double inv_count_util = static_cast<double>(options.min_count()) / std::min(tot_count, options.min_count());
    double inv_out_count_util =
        static_cast<double>(options.min_out_count()) / std::min(tot_out_count, options.min_out_count());
    double ineff = inv_size_util * inv_out_size_util * inv_count_util * inv_out_count_util * tile_expand;
    auto input_cost = options.input_cost() * metrics.input_bandwidth;
    auto output_cost = options.output_cost() * metrics.output_bandwidth;
    auto io_cost = 1.0 + input_cost + output_cost;  // Add 1 to make sure ineff still gets counted if in/out cost == 0
    double cost = ((ineff * io_cost / total_compute) + options.split_factor() * log2(tile.counts_product())) / n_factors;
    IVLOG(4, "        cost: " << cost);
    return cost;
  }
};

struct PartitionComputeCostModel {
  size_t num_parts;
  std::set<const Index*> acc_idxs;

  PartitionComputeCostModel(const Block& block, const proto::PartitionComputePass& options)
      : num_parts(options.num_parts()),  //
        acc_idxs(block.accumulation_idxs()) {}

  bool IndexFilter(const Block& block, const Index& idx) const {  //
    return !acc_idxs.count(&idx);
  }

  Cost ComputeCost(const Block& block, const Tile& tile) const {
    auto count = tile.counts_product();
    if (count > num_parts) {
      return (num_parts + 1) * (count - num_parts);
    }
    return num_parts - count;
  }
};

struct TileResult {
  Tile tile;
  double cost;
};

struct TileSearchState {
  std::set<Tile> found_tiles;
  std::optional<TileResult> best_so_far;
  std::set<std::pair<double, Tile>> todo;

  void AddTile(const Tile& tile, Cost cost) {
    IVLOG(4, "    Found " << cost << ": " << tile);
    found_tiles.emplace(tile);
    if (cost.outcome == Cost::Valid && (!best_so_far || cost.value < best_so_far->cost)) {
      best_so_far = TileResult{tile, cost.value};
    }
    if (cost.outcome != Cost::Stop) {
      todo.emplace(cost.outcome == Cost::Valid ? cost.value : 0, tile);
    }
  }
};

template <typename CostModel>
std::optional<TileResult> PickBestTile(const Block& block, bool only_po2, bool only_even, bool only_multiple_of_32,
                                       bool is_fast, const CostModel& model) {
  IVLOG(3, "Autotile> PickBestTile> block: " << block.name);
  TileSearchState state;
  Tile tile(block, only_multiple_of_32 ? 32 : 1);

  for (size_t i = 0; i < block.idxs.size(); i++) {
    if (!model.IndexFilter(block, block.idxs[i])) {
      tile.dims[i].size = block.idxs[i].range;
      tile.dims[i].count = 1;
    }
  }
  Cost cost = model.ComputeCost(block, tile);
  Cost base_cost = cost;
  state.AddTile(tile, base_cost);
  while (!state.todo.empty()) {
    auto it = state.todo.begin();
    if (cost.outcome == Cost::Valid && it->first > cost.value && is_fast) {
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
      if (prev.size == block.idxs[i].range) {
        continue;  // Already at max size
      }
      if (only_po2) {
        tile.set(i, 2 * prev.size, block.idxs[i].range);
      } else if (only_even) {
        // Find the next even divisor of range
        for (size_t j = prev.size + 1; j <= block.idxs[i].range; j++) {
          if (block.idxs[i].range % j == 0) {
            tile.set(i, j, block.idxs[i].range);
            break;
          }
        }
      } else if (only_multiple_of_32) {
        tile.set(i, 32 + prev.size, block.idxs[i].range);
      } else {
        tile.set(i, prev.size + 1, block.idxs[i].range);
      }
      if (!state.found_tiles.count(tile)) {
        cost = model.ComputeCost(block, tile);
        state.AddTile(tile, cost);
      }
      tile.dims[i] = prev;
    }
  }
  return state.best_so_far;
}

}  // namespace

void AutotilePass::Apply(CompilerState* state) const {
  auto reqs = FromProto(options_.reqs());
  RunOnBlocks(state->entry(), reqs, [this](const AliasMap& map, Block* block) {
    if (block->has_any_tags(FromProto(options_.exclude()))) {
      return;
    }
    ComputeDensityCostModel model(*block, options_);
    auto result = PickBestTile(*block, options_.only_po2(), options_.only_even(), options_.only_multiple_of_32(),
                               options_.fast(), model);
    if (result) {
      IVLOG(2, "Autotile> block: " << block->name << ", tile: " << result->tile << ", cost: " << result->cost);
      const TileShape& tiling_shape = options_.flip() ? result->tile.counts() : result->tile.sizes();
      if (ApplyTile(block, tiling_shape, false, false, options_.flip() || options_.interleave(),
                    options_.location_idx_tag())) {
        auto inner = block->SubBlock(0);
        if (options_.copy_tags()) {
          inner->set_attrs(*block);
        }
        if (options_.clear_outer()) {
          block->clear_tags();
        }
        block->add_tags(FromProto(options_.outer_set()));
        inner->add_tags(FromProto(options_.inner_set()));
        if (options_.clear_location()) {
          block->location = Location{};
        }
      }
    } else {
      auto fail_inner_set = FromProto(options_.fail_inner_set());
      auto fail_outer_set = FromProto(options_.fail_outer_set());
      if (fail_inner_set.size() > 0 || fail_outer_set.size() > 0) {
        TileShape tiling_shape;
        for (const auto idx : block->idxs) {
          if (idx.affine == Affine()) {
            tiling_shape.push_back(idx.range);
          }
        }
        ApplyTile(block, tiling_shape, false, false, options_.flip(), options_.location_idx_tag());
        auto inner = block->SubBlock(0);
        inner->add_tags(FromProto(options_.fail_inner_set()));
        block->add_tags(FromProto(options_.fail_outer_set()));
      }
      LOG(WARNING) << "Autotile> block: " << block->name << " was NOT split; unable to find a valid tiling";
    }
  });
}

void PartitionComputePass::Apply(CompilerState* state) const {
  auto reqs = FromProto(options_.reqs());
  RunOnBlocks(state->entry(), reqs, [this](const AliasMap& map, Block* block) {
    PartitionComputeCostModel model(*block, options_);
    auto result = PickBestTile(*block, false, false, options_.only_multiple_of_32(), false, model);
    if (result) {
      IVLOG(2, "PartitionCompute> block: " << block->name                 //
                                           << ", tile: " << result->tile  //
                                           << ", cost: " << result->cost);
      if (ApplyTile(block, result->tile.sizes(), false)) {
        auto inner = block->SubBlock(0);
        inner->set_attrs(*block);
        block->clear_tags();
        block->add_tags(FromProto(options_.set_tags()));
        if (!options_.idx_tag().empty()) {
          for (auto& idx : block->idxs) {
            if (idx.range > 1) {
              idx.set_tag(options_.idx_tag());
            }
            // HACK: remove this somehow
            idx.remove_tag("bank");
          }
        }
      }
    }
  });
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<AutotilePass, proto::AutotilePass>::Register();
  CompilePassFactory<PartitionComputePass, proto::PartitionComputePass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
