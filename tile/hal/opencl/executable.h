// Copyright 2018, Intel Corporation.

#pragma once

#include <memory>
#include <vector>

#include "tile/base/hal.h"
#include "tile/hal/opencl/kernel.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

class Executable final : public hal::Executable {
 public:
  explicit Executable(std::vector<std::unique_ptr<Kernel>> kernels);

  std::shared_ptr<hal::Event> Run(const context::Context& ctx, std::size_t kernel_index,
                                  const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                  const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                  bool enable_profiling = false) final;

 private:
  std::vector<std::unique_ptr<Kernel>> kernels_;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
