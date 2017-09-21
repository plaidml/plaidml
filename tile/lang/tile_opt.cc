
#include "tile/lang/tile_opt.h"

#include <algorithm>
#include <set>
#include <utility>

#include "tile/lang/mutil.h"
#include "tile/lang/out_plan.h"
#include "tile/lang/read_plan.h"

namespace vertexai {
namespace tile {
namespace lang {

FlatContraction Vectorize(const FlatContraction& iop, uint64_t vec_size) {  // NOLINT(runtime/references)
  size_t sz = iop.ranges.size();
  IVLOG(3, "Attempting vectorization");
  FlatContraction op = iop;
  // For now we require that the output is vectorized
  // This implies that all stride 1 indexes are vectorized
  std::set<size_t> to_vec;
  for (size_t i = 0; i < sz; i++) {
    if (op.access[0].strides[i] == 1) {
      to_vec.insert(i);
    }
  }
  // Given that, see if we are valid for all accesses
  bool valid = true;
  for (auto& a : op.access) {
    if (a.offset % vec_size != 0) {
      valid = false;
      break;
    }
    if (a.global_index_limit % vec_size != 0) {
      valid = false;
      break;
    }
    for (size_t i = 0; i < sz; i++) {
      if (to_vec.find(i) != to_vec.end()) {
        if (a.strides[i] == 0) {
          continue;
        }
        if (a.strides[i] == 1 && op.ranges[i] % vec_size == 0) {
          continue;
        }
      } else {
        if (a.strides[i] % vec_size == 0) {
          continue;
        }
      }
      valid = false;
    }
  }
  // Nope?  Forget it
  if (!valid) {
    IVLOG(1, "Unable to vectorize");
    return op;
  }
  // Adjust ranges
  for (size_t i : to_vec) {
    op.ranges[i] /= vec_size;
  }
  // Adjust access
  for (auto& a : op.access) {
    bool do_vec = false;
    for (size_t i : to_vec) {
      if (a.strides[i] == 1) {
        do_vec = true;
      }
    }
    if (!do_vec) {
      continue;
    }
    a.vector = vec_size;
    a.offset /= vec_size;
    a.global_index_limit /= vec_size;
    for (size_t i = 0; i < a.strides.size(); i++) {
      if (!to_vec.count(i)) {
        a.strides[i] /= vec_size;
      }
    }
  }
  op.agg_vec = op.access[0].vector;
  IVLOG(2, "Vectorized: \n" << op.toString());
  return op;
}

PerfStats ComputeTileStats(const DirectSettings& settings, const FlatContraction& op,
                           const std::vector<uint64_t>& tile) {
  PerfStats r;
  IVLOG(4, "Computing cost for tile size: " << tile);
  uint64_t sz = op.ranges.size();

  OutPlan pout(op, tile, settings.threads, settings.mem_width / op.access[0].elem_size());
  r.out_regs = pout.localSize() * op.access[0].elem_size();
  r.mem_write = pout.outputs() * settings.mem_width;
  r.shared_mem = 0;
  r.mem_read = 0;
  r.true_ops = 2;
  for (size_t i = 1; i < op.access.size(); i++) {
    const auto& a = op.access[i];
    uint64_t mem_width = settings.mem_width / a.elem_size();
    if (mem_width == 0) {
      throw std::runtime_error("Memory width smaller than vector size");
    }
    ReadPlan mi(op.names, a.strides, tile, mem_width);
    r.mem_read += mi.numLoads() * settings.mem_width;
    if (!settings.use_global) {
      r.shared_mem += mi.localSize() * a.elem_size();
    }
  }

  uint64_t out_tiles = 1;
  uint64_t all_tiles = 1;
  uint64_t out_max_threads = 1;
  uint64_t all_max_threads = 1;
  for (size_t i = 0; i < sz; i++) {
    r.true_ops *= op.ranges[i];
    all_max_threads *= tile[i];
    all_tiles *= RoundUp(op.ranges[i], tile[i]);
    if (op.access[0].strides[i] != 0) {
      out_max_threads *= tile[i];
      out_tiles *= RoundUp(op.ranges[i], tile[i]);
    }
  }

  r.work_groups = out_tiles;
  r.inner_loops = all_tiles / out_tiles;
  r.operations = all_max_threads;
  r.true_ops *= op.agg_vec;
  r.rollups = 0;
  uint64_t comp_threads = std::min(settings.threads, all_max_threads);
  if (out_max_threads < comp_threads) {
    r.shared_mem += settings.threads * op.access[0].elem_size();
  }
  while (out_max_threads < comp_threads) {
    r.rollups++;
    out_max_threads *= 2;
  }
  return r;
}

// Compute score from PerfStats
double ComputeScore(const HardwareSettings& settings, const PerfStats& perf) {
  IVLOG(4, "Compute score:"
               << " wg=" << perf.work_groups << " sm=" << perf.shared_mem << " or=" << perf.out_regs
               << " mr=" << perf.mem_read << " mw=" << perf.mem_write << " op=" << perf.operations);
  if (perf.shared_mem > settings.max_mem) {
    IVLOG(4, "  over memory");
    return -1;
  }
  if (perf.out_regs > settings.max_regs) {
    IVLOG(4, "  over regs");
    return -1;
  }
  // Compute the logical amount memory io (ignoring OOB)
  double bytes = perf.work_groups * (perf.inner_loops * perf.mem_read + perf.mem_write);
  double flops_per_byte = perf.true_ops / bytes;
  double roof = std::min(flops_per_byte, static_cast<double>(settings.goal_flops_per_byte));
  double occupancy = std::min(perf.work_groups, settings.goal_groups);
  double roof_ratio = roof / static_cast<double>(settings.goal_flops_per_byte);
  double occ_ratio = occupancy / static_cast<double>(settings.goal_groups);
  double score = roof_ratio * occ_ratio;
  IVLOG(4, "  flops_per_byte=" << flops_per_byte << " occupancy=" << occupancy);
  IVLOG(4, "  roof_ratio=" << roof_ratio << " occ_ration=" << occ_ratio << " score=" << score);
  return score;
}

std::multimap<double, std::vector<uint64_t>> TileOptimize(const HardwareSettings& settings, const FlatContraction& op,
                                                          bool fast) {
  std::multimap<double, std::vector<uint64_t>> by_score;
  size_t sz = op.ranges.size();

  std::map<std::vector<uint64_t>, double> by_tile;
  std::set<std::pair<double, std::vector<uint64_t>>> to_do;
  IVLOG(3, "Computing optimal tile cost");
  std::vector<uint64_t> tile(sz, 1);
  double score = ComputeScore(settings, ComputeTileStats(settings, op, tile));
  by_tile.emplace(tile, score);
  by_score.emplace(score, tile);
  to_do.emplace(score, tile);
  while (!to_do.empty()) {
    auto it = to_do.rbegin();
    if (it->first < score && fast) {
      break;
    }
    score = it->first;
    tile = it->second;
    to_do.erase(*it);
    for (size_t i = 0; i < sz; i++) {
      uint64_t prev = tile[i];
      tile[i] = std::min(2 * tile[i], op.ranges[i]);
      if (!by_tile.count(tile)) {
        score = ComputeScore(settings, ComputeTileStats(settings, op, tile));
        by_tile.emplace(tile, score);
        by_score.emplace(score, tile);
        if (score > 0) {
          to_do.emplace(score, tile);
        }
      }
      tile[i] = prev;
    }
  }
  IVLOG(3, "  Final Tile: " << by_score.rbegin()->second);
  IVLOG(3, "  Final Score: " << by_score.rbegin()->first);
  return by_score;
}

// Performs vectorization + tile size optimization
std::vector<uint64_t> TileVecOptimize(const HardwareSettings& settings,
                                      FlatContraction& op) {  // NOLINT(runtime/references)
  if (settings.vec_size > 1) {
    op = Vectorize(op, settings.vec_size);
  }
  std::multimap<double, std::vector<uint64_t>> by_score = TileOptimize(settings, op, true);
  return by_score.rbegin()->second;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
