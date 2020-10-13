// Copyright 2020 Intel Corporation

#include "pmlc/rt/llvm/device.h"

#include "pmlc/rt/jit_executable.h"

namespace pmlc::rt::llvm {

std::unique_ptr<Executable>
Device::compile(const std::shared_ptr<pmlc::compiler::Program> &program) {
  return makeJitExecutable(program, shared_from_this(),
                           mlir::ArrayRef<void *>{});
}

} // namespace pmlc::rt::llvm
