
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

PerfStats ComputeTileStats(const DirectSettings& settings, const FlatContraction& op, const std::vector<uint64_t>& tile,
                           const Bindings& vars) {
  PerfStats r;
  IVLOG(4, "Computing cost for tile size: " << tile);
  uint64_t sz = op.ranges.size();

  OutPlan pout(op, tile, settings.threads, settings.mem_width / op.access[0].elem_size());
  r.out_regs = pout.localSize() * op.access[0].elem_size();
  r.mem_write = pout.outputs() * settings.mem_width * op.kernel_outputs.size();
  r.shared_mem = 0;
  r.mem_read = 0;
  r.true_ops = 1;
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
  for (const auto& input : op.post_op_inputs) {
    // We read the post-op inputs during the output phase.
    r.mem_read += pout.outputs() * byte_width(vars.at(input).shape.type);
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
  r.true_ops *= (op.post_ops.size() + (op.generate_contraction ? 2 : 0));

  r.work_groups = out_tiles;
  r.inner_loops = all_tiles / out_tiles;
  r.operations = std::min(settings.threads, all_max_threads);
  r.true_ops *= op.agg_vec;
  r.rollups = 0;
  if (out_max_threads < r.operations) {
    r.shared_mem += settings.threads * op.access[0].elem_size();
  }
  while (out_max_threads < r.operations) {
    r.rollups++;
    out_max_threads *= 2;
  }
  std::uint64_t output_threads = 1;
  for (const auto& idx : pout.indexes()) {
    output_threads *= idx.threads;
  }
  r.threads_used = std::max(r.operations, output_threads);
  return r;
}

// Compute score from PerfStats
double ComputeScore(const HardwareSettings& settings, const PerfStats& perf) {
  IVLOG(4, "Compute score:"
               << " to=" << perf.true_ops << " wg=" << perf.work_groups << " il=" << perf.inner_loops << " sm="
               << perf.shared_mem << " or=" << perf.out_regs << " mr=" << perf.mem_read << " mw=" << perf.mem_write
               << " op=" << perf.operations << " rp=" << perf.rollups << " tu=" << perf.threads_used);
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
  double thread_ratio = perf.threads_used / static_cast<double>(settings.threads);
  double roof_ratio = roof / static_cast<double>(settings.goal_flops_per_byte);
  double occ_ratio = occupancy / static_cast<double>(settings.goal_groups);
  double score = roof_ratio * occ_ratio * thread_ratio;
  IVLOG(4, "  flops_per_byte=" << flops_per_byte << " occupancy=" << occupancy);
  IVLOG(4, "  roof_ratio=" << roof_ratio << " occ_ratio=" << occ_ratio << " thread_ratio=" << thread_ratio
                           << " score=" << score);
  return score;
}

std::multimap<double, std::vector<uint64_t>> TileOptimize(const HardwareSettings& settings, const FlatContraction& op,
                                                          bool fast, const Bindings& vars) {
  std::multimap<double, std::vector<uint64_t>> by_score;
  size_t sz = op.ranges.size();

  std::map<std::vector<uint64_t>, double> by_tile;
  std::set<std::pair<double, std::vector<uint64_t>>> to_do;
  IVLOG(3, "Computing optimal tile cost");
  std::vector<uint64_t> tile(sz, 1);
  double score = ComputeScore(settings, ComputeTileStats(settings, op, tile, vars));
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
        score = ComputeScore(settings, ComputeTileStats(settings, op, tile, vars));
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
                                      FlatContraction& op,  // NOLINT(runtime/references)
                                      const Bindings& vars) {
  if (settings.vec_size > 1) {
    op = Vectorize(op, settings.vec_size);
  }
  std::multimap<double, std::vector<uint64_t>> by_score = TileOptimize(settings, op, true, vars);
  return by_score.rbegin()->second;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
