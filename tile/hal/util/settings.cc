// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/util/settings.h"

#include <utility>
#include <vector>

#include "base/util/error.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace settings {

void Validate(const proto::HardwareSettings& settings) {
  if (settings.threads() < 1 || settings.threads() > 512 || ((settings.threads() - 1) & settings.threads()) != 0) {
    throw error::InvalidArgument("Threads must be >= 1 and <= 512 and a power of two: " +
                                 std::to_string(settings.threads()));
  }

  // TODO: Pre-refactor of settings override logic, the OpenCL backend
  // had this check, but the CPU backend did not, and is moreover
  // incompatible with this check.  It's unclear what the correct
  // check should be.
  //
  // if (settings.vec_size() < 1 || settings.vec_size() > 16 || settings.threads() %
  // settings.vec_size() != 0) {
  //   throw error::InvalidArgument("Vector size must be >= 1 and <= 16 and divide threads evenly: " +
  //                                std::to_string(settings.vec_size()));
  // }

  if (settings.mem_width() < 8 || settings.mem_width() > 4096 ||
      ((settings.mem_width() - 1) & settings.mem_width()) != 0) {
    throw error::InvalidArgument("Memory width must be >= 8 and <= 4096 and a power of two: " +
                                 std::to_string(settings.mem_width()));
  }
  if (settings.max_mem() < 1024) {
    throw error::InvalidArgument("Max mem must be >= 1024: " + std::to_string(settings.max_mem()));
  }
  if (settings.goal_groups() < 1) {
    throw error::InvalidArgument("goal groups must be >= 1: " + std::to_string(settings.goal_groups()));
  }
  if (settings.goal_flops_per_byte() < 1) {
    throw error::InvalidArgument("goal flops per byte must be >= 1: " + std::to_string(settings.goal_flops_per_byte()));
  }
}

lang::HardwareSettings ToHardwareSettings(const proto::HardwareSettings& settings) {
  std::vector<std::size_t> dim_sizes;
  for (auto size : settings.dim_sizes()) {
    dim_sizes.push_back(size);
  }

  // TODO: Support more than three dimensions.
  if (3 < dim_sizes.size()) {
    dim_sizes.resize(3);
  }

  lang::HardwareSettings result;
  result.threads = settings.threads();
  result.vec_size = settings.vec_size();
  result.use_global = settings.use_global();
  result.mem_width = settings.mem_width();
  result.max_mem = settings.max_mem();
  result.max_regs = settings.max_regs();
  result.goal_groups = settings.goal_groups();
  result.goal_flops_per_byte = settings.goal_flops_per_byte();
  result.goal_dimension_sizes = std::move(dim_sizes);
  result.disable_io_aliasing = settings.disable_io_aliasing();

  return result;
}

}  // namespace settings
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
