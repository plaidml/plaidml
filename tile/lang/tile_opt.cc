
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

  // Give up early if we have nothing to vectorize on
  if (to_vec.size() == 0) {
    IVLOG(1, "Unable to vectorize: no stride 1 outputs");
    return op;
  }

  // Give up early if we have any constraints since we don't currently handle them properly
  if (iop.constraints.size()) {
    IVLOG(1, "Unable to vectorize do due constraints");
    return op;
  }

  // Given that, see if we are valid for all accesses
  auto is_vectorizable = [&](const FlatTensorAccess& a) {
    if (a.offset % vec_size != 0) {
      IVLOG(1, "Unable to vectorize: Offset not valid, " << a.offset);
      return false;
    }
    if (a.global_index_limit % vec_size != 0) {
      IVLOG(1, "Unable to vectorize: Size not not valid, " << a.global_index_limit);
      return false;
    }
    bool has_stride_1 = false;
    bool has_uneven = false;
    for (size_t i = 0; i < sz; i++) {
      if (to_vec.find(i) != to_vec.end()) {
        if (a.strides[i] == 0) {
          continue;
        }
        if (a.strides[i] == 1 && op.ranges[i] % vec_size == 0) {
          has_stride_1 = true;
          continue;
        }
      } else {
        if (a.strides[i] % vec_size != 0) {
          has_uneven = true;
        }
        continue;
      }
      IVLOG(1, "Unable to vectorize: Strides not valid, " << a.strides << ", case " << i);
      return false;
    }
    if (has_stride_1 && has_uneven) {
      IVLOG(1, "Unable to vectorize: Vector stride uneven, " << a.strides);
      return false;
    }
    return true;
  };

  bool valid = true;
  for (auto& a : op.access) {
    valid = is_vectorizable(a);
    if (!valid) {
      break;
    }
  }
  if (valid) {
    for (auto& op_input : op.post_op_inputs) {
      valid = is_vectorizable(op_input.access);
      if (!valid) {
        break;
      }
    }
  }
  // Nope?  Forget it
  if (!valid) {
    return op;
  }

  IVLOG(1, "Vectorizing on " << to_vec);

  // Adjust ranges
  for (size_t i : to_vec) {
    op.ranges[i] /= vec_size;
  }

  // Adjust access
  auto adjust_access = [&](FlatTensorAccess* a) {
    bool do_vec = false;
    for (size_t i : to_vec) {
      if (a->strides[i] == 1) {
        do_vec = true;
      }
    }
    if (!do_vec) {
      return;
    }
    a->vector = vec_size;
    a->offset /= vec_size;
    a->global_index_limit /= vec_size;
    for (size_t i = 0; i < a->strides.size(); i++) {
      if (!to_vec.count(i)) {
        a->strides[i] /= vec_size;
      }
    }
  };
  for (auto& a : op.access) {
    adjust_access(&a);
  }
  for (auto& op_input : op.post_op_inputs) {
    adjust_access(&op_input.access);
  }
  op.agg_vec = op.access[0].vector;
  IVLOG(2, "Vectorized: \n" << op.toString());
  return op;
}

proto::PerfStats ComputeTileStats(const DirectSettings& settings, const FlatContraction& op,
                                  const std::vector<uint64_t>& tile) {
  proto::PerfStats r;
  IVLOG(4, "Computing cost for tile size: " << tile);
  uint64_t sz = op.ranges.size();

  OutPlan pout(op, tile, settings.threads, settings.mem_width / op.access[0].elem_size());
  r.set_out_regs(pout.localSize() * op.access[0].elem_size());
  r.set_mem_write(pout.outputs() * settings.mem_width * op.kernel_outputs.size());

  std::uint64_t mem_read = 0;
  std::uint64_t shared_mem = 0;

  for (size_t i = 1; i < op.access.size(); i++) {
    const auto& a = op.access[i];
    uint64_t mem_width = settings.mem_width / a.elem_size();
    if (mem_width == 0) {
      throw std::runtime_error("Memory width smaller than vector size");
    }
    ReadPlan mi(op.names, a.strides, tile, mem_width);
    mem_read += mi.numLoads() * settings.mem_width;
    if (!settings.use_global) {
      shared_mem += mi.localSize() * a.elem_size();
    }
  }
  for (const auto& op_input : op.post_op_inputs) {
    // We read the post-op inputs during the output phase.
    mem_read += pout.outputs() * byte_width(op_input.binding.shape.type);
  }

  std::uint64_t rollups = 0;
  std::uint64_t true_ops = 1;
  std::uint64_t out_tiles = 1;
  std::uint64_t all_tiles = 1;
  std::uint64_t out_max_threads = 1;
  std::uint64_t all_max_threads = 1;
  for (size_t i = 0; i < sz; i++) {
    true_ops *= op.ranges[i];
    all_max_threads *= tile[i];
    all_tiles *= RoundUp(op.ranges[i], tile[i]);
    if (op.access[0].strides[i] != 0) {
      out_max_threads *= tile[i];
      out_tiles *= RoundUp(op.ranges[i], tile[i]);
    }
  }
  true_ops *= (op.post_ops.size() + (op.generate_contraction ? 2 : 0));
  r.set_work_groups(out_tiles);
  r.set_inner_loops(all_tiles / out_tiles);
  r.set_operations(std::min(settings.threads, all_max_threads));
  true_ops *= op.agg_vec;
  if (out_max_threads < r.operations()) {
    shared_mem += settings.threads * op.access[0].elem_size();
  }
  while (out_max_threads < r.operations()) {
    rollups++;
    out_max_threads *= 2;
  }
  std::uint64_t output_threads = 1;
  for (const auto& idx : pout.indexes()) {
    output_threads *= idx.threads;
  }
  r.set_threads_used(std::max(r.operations(), output_threads));
  r.set_mem_read(mem_read);
  r.set_shared_mem(shared_mem);
  r.set_true_ops(true_ops);
  r.set_rollups(rollups);

  return r;
}

// Compute score from PerfStats
double ComputeScore(const HardwareSettings& settings, const proto::PerfStats& perf) {
  IVLOG(4, "Compute score:"
               << " to=" << perf.true_ops() << " wg=" << perf.work_groups() << " il=" << perf.inner_loops()
               << " sm=" << perf.shared_mem() << " or=" << perf.out_regs() << " mr=" << perf.mem_read()
               << " mw=" << perf.mem_write() << " op=" << perf.operations() << " rp=" << perf.rollups()
               << " tu=" << perf.threads_used());
  if (perf.shared_mem() > settings.max_mem) {
    IVLOG(4, "  over memory");
    return -1;
  }
  if (perf.out_regs() > settings.max_regs) {
    IVLOG(4, "  over regs");
    return -1;
  }
  // Compute the logical amount memory io (ignoring OOB)
  double bytes = perf.work_groups() * (perf.inner_loops() * perf.mem_read() + perf.mem_write());
  double flops_per_byte = perf.true_ops() / bytes;
  double roof = std::min(flops_per_byte, static_cast<double>(settings.goal_flops_per_byte));
  double occupancy = std::min(perf.work_groups(), settings.goal_groups);
  double thread_ratio = perf.threads_used() / static_cast<double>(settings.threads);
  double roof_ratio = roof / static_cast<double>(settings.goal_flops_per_byte);
  double occ_ratio = occupancy / static_cast<double>(settings.goal_groups);
  double score = roof_ratio * occ_ratio * thread_ratio;
  IVLOG(4, "  flops_per_byte=" << flops_per_byte << " occupancy=" << occupancy);
  IVLOG(4, "  roof_ratio=" << roof_ratio << " occ_ratio=" << occ_ratio << " thread_ratio=" << thread_ratio
                           << " score=" << score);
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

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
