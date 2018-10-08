// Copyright 2018, Intel Corporation.

#include "tile/hal/opencl/executable.h"

#include <utility>

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

Executable::Executable(std::vector<std::unique_ptr<Kernel>> kernels) : kernels_{std::move(kernels)} {}

std::shared_ptr<hal::Event> Executable::Run(const context::Context& ctx, std::size_t kernel_index,
                                            const std::vector<std::shared_ptr<hal::Buffer>>& params,
                                            const std::vector<std::shared_ptr<hal::Event>>& dependencies,
                                            bool enable_profiling) {
  return kernels_[kernel_index]->Run(ctx, params, dependencies, enable_profiling);
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
