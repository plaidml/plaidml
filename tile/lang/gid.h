#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "tile/lang/generate.h"
#include "tile/lang/semtree.h"

namespace vertexai {
namespace tile {
namespace lang {
namespace gid {

// When a kernel is launched, each instance receives a unique index in a global index space, allowing the instance to
// determine the data it should operate on.  The range of the global index space -- the number of dimensions and the
// maximum size available on each dimension -- is determined by the hardware the code is running on.
//
// This bit of code determines, for a set of logical dimensions of various sizes and a set of suggested global index
// space limits (i.e. limits which will allow all of the kernel instances to run as a single work group), the size of
// the global index space to use (per-dimension), and the local mapping from a point in the global index space back to
// the per-logical-dimension indicies needed to perform the desired computation.

// DimInfo describes how to calculate a given logical dimension index.
// For a given dimension:
//
//    dim_index = (((gid[gid_index] >> right_shift) & mask) / divisor) % modulus
//
// Each operation may be omitted if the corresponding has_ variable is false.
struct DimInfo {
  std::size_t gid_index = 0;
  bool has_right_shift = false;
  bool has_mask = false;
  bool has_divisor = false;
  bool has_modulus = false;
  std::size_t right_shift = 0;
  std::size_t mask = static_cast<std::size_t>(-1);
  std::size_t divisor = 1;
  std::size_t modulus = 0;
};

// Map describes the allocated global index space and mappings to logical indicies.
struct Map {
  std::vector<DimInfo> dims;
  std::vector<std::size_t> gid_sizes;
  std::vector<std::size_t> lid_limits;
};

// MakeMap builds a Map from the provided GID space suggestions and logical dimension sizes.
Map MakeMap(const std::vector<std::size_t>& lid_limits, const std::vector<std::size_t>& logical_dims,
            bool do_early_exit = true);

// Builds the expression computing a logical index.
std::shared_ptr<sem::Expression> LogicalIndex(const std::vector<std::shared_ptr<sem::Expression>>& gids,
                                              const DimInfo& info);

// Make the global and local grid sizes from the map
std::pair<GridSize, GridSize> ComputeGrids(const Map& m, size_t threads);

}  // namespace gid
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
