#pragma once

#include <vector>

#include "tile/base/hal.h"
#include "tile/lang/generate.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

void OptimizeKernel(const lang::KernelInfo& ki, bool cl_khr_fp16, const hal::proto::HardwareSettings& settings);

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
