// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cpu/device.h"

#include <utility>

#include "tile/hal/cpu/compiler.h"
#include "tile/hal/cpu/executor.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

Device::Device() : compiler_{new Compiler}, executor_{new Executor} {}

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
