// Copyright 2020 Intel Corporation

#include "pmlc/rt/llvm_cpu/device.h"

#include "pmlc/rt/jit_executable.h"

namespace pmlc::rt::llvm_cpu {

std::unique_ptr<Executable>
Device::compile(const std::shared_ptr<pmlc::compiler::Program> &program,
                llvm::ArrayRef<void *> bufptrs) {
  return makeJitExecutable(program, shared_from_this(),
                           llvm::ArrayRef<void *>{}, bufptrs);
}

} // namespace pmlc::rt::llvm_cpu
