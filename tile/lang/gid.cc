#include "tile/lang/gid.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "tile/lang/sembuilder.h"

namespace vertexai {
namespace tile {
namespace lang {
namespace gid {

// The goal of this code is to determine a set of global index space (i.e. hardware) dimensions and efficient mappings
// to logical dimension indicies for points within that global index space.
//
// Modulus is expensive, so we avoid it if at all possible, by spreading out the non-power-of-two logical dimensions
// among the global index dimensions.  If a logical dimension's size is a power of two, we instead use the low bits of a
// global dimension to track where a thread is in that logical dimension with some simple bit-wise operations, while
// continuing to use the high bits for other dimensions.
Map MakeMap(const std::vector<std::size_t>& lid_limits, const std::vector<std::size_t>& logical_dims,
            bool do_early_exit) {
  // Early exit: if all of the logical dimensions are size 1, return an empty map.
  bool early_exit = true;
  for (auto size : logical_dims) {
    if (size != 1) {
      early_exit = false;
      break;
    }
  }
  if (do_early_exit && early_exit) {
    return Map{};
  }

  auto logical_dim_count = logical_dims.size();
  auto global_dim_count = lid_limits.size();
  Map result = {std::vector<DimInfo>(logical_dim_count), std::vector<std::size_t>(global_dim_count, 1), lid_limits};
  std::vector<std::uint_fast8_t> gid_bits_available;
  std::vector<std::uint_fast8_t> gid_pow2_shift(global_dim_count, 0);
  std::vector<bool> gid_mod(global_dim_count, false);
  std::vector<bool> gid_used(global_dim_count, false);

  auto is_pow2 = [](std::size_t val) { return (val & (val - 1)) == 0; };

  // Determine the suggested bits available for each global dimension.
  gid_bits_available.reserve(global_dim_count);
  for (std::size_t gidx = 0; gidx < global_dim_count; ++gidx) {
    gid_bits_available.push_back(std::floor(std::log2(lid_limits[gidx])));
  }

  // Allocate the non-power-of-two logical dimensions.
  for (std::size_t lidx = 0; lidx < logical_dim_count; ++lidx) {
    if (is_pow2(logical_dims[lidx])) {
      continue;
    }

    uint_fast8_t bits = std::ceil(std::log2(logical_dims[lidx]));

    auto use_gidx_for_lidx = [&](std::size_t gidx) {
      result.dims[lidx].gid_index = gidx;
      if (result.gid_sizes[gidx] > 1) {
        result.dims[lidx].divisor = result.gid_sizes[gidx];
        result.dims[lidx].has_divisor = true;
      }
      result.gid_sizes[gidx] *= logical_dims[lidx];
      gid_bits_available[gidx] -= std::min(gid_bits_available[gidx], bits);
      gid_used[gidx] = true;
    };

    // Attempt to find a large-enough global dimension with no allocations.
    for (std::size_t gidx = 0; gidx < global_dim_count; ++gidx) {
      if (result.gid_sizes[gidx] == 1 && bits <= gid_bits_available[gidx]) {
        use_gidx_for_lidx(gidx);
        goto allocated_gidx_for_non_pow2_lidx;
      }
    }

    // Attempt to find any global dimension with no allocations, because modulus is awful.
    for (std::size_t gidx = 0; gidx < global_dim_count; ++gidx) {
      if (result.gid_sizes[gidx] == 1) {
        use_gidx_for_lidx(gidx);
        goto allocated_gidx_for_non_pow2_lidx;
      }
    }

    // Since we have to combine anyway, attempt to find a large-enough global dimension.
    for (std::size_t gidx = 0; gidx < global_dim_count; ++gidx) {
      if (bits <= gid_bits_available[gidx]) {
        use_gidx_for_lidx(gidx);
        goto allocated_gidx_for_non_pow2_lidx;
      }
    }

    // Fall back to using dimension 0.
    use_gidx_for_lidx(0);

  allocated_gidx_for_non_pow2_lidx : {}
  }

  // Allocate the power-of-two logical dimensions:
  for (std::size_t lidx = 0; lidx < logical_dim_count; ++lidx) {
    if (!is_pow2(logical_dims[lidx])) {
      continue;
    }

    uint_fast8_t bits = std::floor(std::log2(logical_dims[lidx]));

    auto use_gidx_for_lidx = [&](std::size_t gidx) {
      result.dims[lidx].gid_index = gidx;
      auto shift = gid_pow2_shift[gidx];
      gid_pow2_shift[gidx] += bits;
      if (shift) {
        result.dims[lidx].has_right_shift = true;
        result.dims[lidx].right_shift = shift;
      }
      result.dims[lidx].has_mask = true;
      result.dims[lidx].mask = (1 << bits) - 1;
      result.gid_sizes[gidx] *= logical_dims[lidx];
      gid_bits_available[gidx] -= std::min(gid_bits_available[gidx], bits);
    };

    // Attempt to find a global dimension with enough bits available.
    for (std::size_t gidx = 0; gidx < global_dim_count; ++gidx) {
      if (bits <= gid_bits_available[gidx]) {
        use_gidx_for_lidx(gidx);
        goto allocated_gidx_for_pow2_lidx;
      }
    }

    // Fall back to using dimension 0
    use_gidx_for_lidx(0);

  allocated_gidx_for_pow2_lidx : {}
  }

  // Fill in the non-power-of-two logical dimensions with shift and modulus info.
  for (std::size_t nlidx = 0; nlidx < logical_dim_count; ++nlidx) {
    std::size_t lidx = logical_dim_count - 1 - nlidx;
    if (is_pow2(logical_dims[lidx])) {
      continue;
    }
    auto gidx = result.dims[lidx].gid_index;
    auto shift = gid_pow2_shift[gidx];
    if (shift) {
      result.dims[lidx].has_right_shift = true;
      result.dims[lidx].right_shift = shift;
    }
    if (gid_mod[gidx]) {
      result.dims[lidx].has_modulus = true;
      result.dims[lidx].modulus = logical_dims[lidx];
    }
    gid_mod[gidx] = true;
  }

  // Remove masks from power-of-two logical dimensions when they're the last logical dimension in a global dimension.
  for (std::size_t nlidx = 0; nlidx < logical_dim_count; ++nlidx) {
    std::size_t lidx = logical_dim_count - 1 - nlidx;
    if (is_pow2(logical_dims[lidx]) && !gid_used[result.dims[lidx].gid_index]) {
      result.dims[lidx].has_mask = false;
      result.dims[lidx].mask = static_cast<std::size_t>(-1);
    }
    gid_used[result.dims[lidx].gid_index] = true;
  }

  // Trim trailing global dimensions.
  while (1 < result.gid_sizes.size() && result.gid_sizes.back() == 1) {
    result.gid_sizes.pop_back();
  }

  return result;
}

std::shared_ptr<sem::Expression> LogicalIndex(const std::vector<std::shared_ptr<sem::Expression>>& gids,
                                              const DimInfo& info) {
  using namespace sem::builder;  // NOLINT
  auto expr = gids[info.gid_index];
  if (info.has_right_shift) {
    expr = expr >> info.right_shift;
  }
  if (info.has_mask) {
    expr = expr & info.mask;
  }
  if (info.has_divisor) {
    expr = expr / info.divisor;
  }
  if (info.has_modulus) {
    expr = expr % info.modulus;
  }
  return expr;
}

std::pair<GridSize, GridSize> ComputeGrids(const Map& gids, size_t threads) {
  GridSize gwork;
  GridSize lwork;
  auto gid_count = gids.gid_sizes.size();
  gwork[0] = 0 < gid_count ? gids.gid_sizes[0] : 1;
  gwork[1] = 1 < gid_count ? gids.gid_sizes[1] : 1;
  gwork[2] = 2 < gid_count ? gids.gid_sizes[2] : 1;
  lwork[0] = lwork[1] = lwork[2] = 1;
  // TODO: Refactor this duplication from gen_func
  size_t threads_per = 1;
  // Peel off easy powers of two
  while (threads_per * 2 <= threads) {
    bool got_one = false;
    for (int i = 2; i >= 0; i--) {
      if (gwork[i] % 2 == 0 && lwork[i] * 2 <= gids.lid_limits[i]) {
        gwork[i] /= 2;
        lwork[i] *= 2;
        threads_per *= 2;
        got_one = true;
        break;
      }
    }
    if (!got_one) break;
  }
  for (size_t i = 0; i < 3; i++) {
    gwork[i] *= lwork[i];
  }
  return std::make_pair(gwork, lwork);
}

}  // namespace gid
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
