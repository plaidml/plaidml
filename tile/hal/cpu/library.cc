// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cpu/library.h"

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <utility>

#include "base/util/error.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

Library* Library::Downcast(hal::Library* library) {
  Library* exe = dynamic_cast<Library*>(library);
  return exe;
}

Library::Library(const std::vector<std::shared_ptr<llvm::ExecutionEngine>>& engines,
                 const std::vector<lang::KernelInfo>& kernels)
    : engines_{engines}, kernels_{kernels} {}

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
