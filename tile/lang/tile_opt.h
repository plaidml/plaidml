
#pragma once

#include <map>
#include <vector>

#include "tile/lang/flat.h"
#include "tile/lang/gen_contract.h"
#include "tile/lang/generate.h"

namespace vertexai {
namespace tile {
namespace lang {

struct PerfStats {
  uint64_t true_ops;      // How many useful ops for everything
  uint64_t work_groups;   // How many work groups
  uint64_t inner_loops;   // How many read/compute loops per WG
  uint64_t shared_mem;    // How much shared memory per WG
  uint64_t out_regs;      // How much register usage for outputs (in bytes)
  uint64_t mem_read;      // How much memory does each WG read
  uint64_t mem_write;     // How much memory does each WG write
  uint64_t operations;    // How many primary operations per WG
  uint64_t rollups;       // How many rollups per WG
  uint64_t threads_used;  // How many useful threads we're using per WG
};

// Do vectorization if it's easy, otherwise punt
FlatContraction Vectorize(const FlatContraction& op, uint64_t vector_size);

// Compute the stats for a given tile size
PerfStats ComputeTileStats(const DirectSettings& settings, const FlatContraction& op, const std::vector<uint64_t>& tile,
                           const Bindings& vars);

// Compute score from PerfStats
double ComputeScore(const HardwareSettings& settings, const PerfStats& perf);

// Performs tile size optimization
std::multimap<double, std::vector<uint64_t>> TileOptimize(const HardwareSettings& settings, const FlatContraction& op,
                                                          bool fast, const Bindings& vars);

// Performs vectorization + tile size optimization
std::vector<uint64_t> TileVecOptimize(const HardwareSettings& settings,
                                      FlatContraction& op,  // NOLINT(runtime/references)
                                      const Bindings& vars);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
