
#pragma once

#include <map>
#include <vector>

#include "tile/lang/flat.h"
#include "tile/lang/gen_contract.h"
#include "tile/lang/generate.h"
#include "tile/lang/lang.pb.h"

namespace vertexai {
namespace tile {
namespace lang {

// Do vectorization if it's easy, otherwise punt
FlatContraction Vectorize(const FlatContraction& op, uint64_t vector_size);

// Compute the stats for a given tile size
proto::PerfStats ComputeTileStats(const DirectSettings& settings, const FlatContraction& op,
                                  const std::vector<uint64_t>& tile);

// Compute score from PerfStats
double ComputeScore(const HardwareSettings& settings, const proto::PerfStats& perf);

// Performs tile size optimization
std::multimap<double, std::vector<uint64_t>> TileOptimize(const HardwareSettings& settings, const FlatContraction& op,
                                                          bool fast);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
