// Copyright 2020 Intel Corporation

#pragma once

#include <memory>

#include "pmlc/rt/executable.h"

namespace pmlc::rt {

// Returns an Executable implementation that treats the program entry point as
// host-native code which, when compiled for the system CPU and called directly,
// has the effect of running the program.
std::unique_ptr<Executable>
makeJitExecutable(const std::shared_ptr<pmlc::compiler::Program> &program,
                  llvm::StringRef deviceID, llvm::ArrayRef<void *> bufptrs);

} // namespace pmlc::rt
