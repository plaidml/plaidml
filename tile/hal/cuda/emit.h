// Copyright 2018, Intel Corporation.

#pragma once

#include <string>
#include <vector>

#include "tile/lang/generate.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cuda {

std::string EmitCudaC(const std::vector<lang::KernelInfo>& kernels);

}  // namespace cuda
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
