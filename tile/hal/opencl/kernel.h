// Copyright 2018, Intel Corporation.

#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "tile/base/hal.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

class Kernel {
 public:
  virtual ~Kernel() {}

  virtual std::shared_ptr<hal::Event> Run(const context::Context& ctx,
                                          const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                          const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                          bool enable_profiling) = 0;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
