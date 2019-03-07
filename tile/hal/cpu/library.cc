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

Library::Library(std::shared_ptr<llvm::LLVMContext> context,
                 const std::vector<std::shared_ptr<llvm::ExecutionEngine>>& engines,
                 const std::vector<lang::KernelInfo>& kernels)
    : context_{context}, engines_{engines}, kernels_{kernels} {}

Library::~Library() {
  // release all of the ExecutionEngine instances first, before the LLVMContext,
  // because each ExecutionEngine is associated with an llvm::Module, and
  // modules must not outlive their LLVMContext.
  engines_.clear();
}

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
