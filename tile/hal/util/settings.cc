// Copyright 2017, Vertex.AI.

#include "tile/hal/util/settings.h"

#include <utility>
#include <vector>

#include "base/util/error.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace settings {

void Validate(const proto::HardwareSettings& settings) {
  if (settings.threads().value() < 1 || settings.threads().value() > 512 ||
      ((settings.threads().value() - 1) & settings.threads().value()) != 0) {
    throw error::InvalidArgument("Threads must be >= 1 and <= 512 and a power of two: " +
                                 std::to_string(settings.threads().value()));
  }

  // TODO: Pre-refactor of settings override logic, the OpenCL backend
  // had this check, but the CPU backend did not, and is moreover
  // incompatible with this check.  It's unclear what the correct
  // check should be.
  //
  // if (settings.vec_size().value() < 1 || settings.vec_size().value() > 16 || settings.threads().value() %
  // settings.vec_size().value() != 0) {
  //   throw error::InvalidArgument("Vector size must be >= 1 and <= 16 and divide threads evenly: " +
  //                                std::to_string(settings.vec_size().value()));
  // }

  if (settings.mem_width().value() < 8 || settings.mem_width().value() > 4096 ||
      ((settings.mem_width().value() - 1) & settings.mem_width().value()) != 0) {
    throw error::InvalidArgument("Memory width must be >= 8 and <= 4096 and a power of two: " +
                                 std::to_string(settings.mem_width().value()));
  }
  if (settings.max_mem().value() < 1024) {
    throw error::InvalidArgument("Max mem must be >= 1024: " + std::to_string(settings.max_mem().value()));
  }
  if (settings.goal_groups().value() < 1) {
    throw error::InvalidArgument("goal groups must be >= 1: " + std::to_string(settings.goal_groups().value()));
  }
  if (settings.goal_flops_per_byte().value() < 1) {
    throw error::InvalidArgument("goal flops per byte must be >= 1: " +
                                 std::to_string(settings.goal_flops_per_byte().value()));
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
  result.threads = settings.threads().value();
  result.vec_size = settings.vec_size().value();
  result.use_global = settings.use_global().value();
  result.mem_width = settings.mem_width().value();
  result.max_mem = settings.max_mem().value();
  result.max_regs = settings.max_regs().value();
  result.goal_groups = settings.goal_groups().value();
  result.goal_flops_per_byte = settings.goal_flops_per_byte().value();
  result.goal_dimension_sizes = std::move(dim_sizes);
  result.enable_half = settings.enable_half().value();

  return result;
}

}  // namespace settings
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
