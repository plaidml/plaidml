// Copyright 2020 Intel Corporation

#include "pmlc/rt/executable.h"

#include "pmlc/rt/jit_executable.h"

using pmlc::compiler::Program;

namespace pmlc::rt {

std::unique_ptr<Executable>
Executable::fromProgram(const std::shared_ptr<Program> &program,
                        llvm::StringRef deviceID,
                        llvm::ArrayRef<void *> bufptrs) {
  return makeJitExecutable(program, deviceID, bufptrs);
}

} // namespace pmlc::rt
