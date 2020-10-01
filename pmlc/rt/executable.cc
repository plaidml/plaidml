// Copyright 2020 Intel Corporation

#include "pmlc/rt/executable.h"

#include "pmlc/rt/device_id.h"

using pmlc::compiler::Program;

namespace pmlc::rt {

std::unique_ptr<Executable>
Executable::fromProgram(const std::shared_ptr<Program> &program,
                        mlir::StringRef deviceID,
                        mlir::ArrayRef<void *> bufptrs) {
  return getDevice(deviceID)->compile(program, bufptrs);
}

} // namespace pmlc::rt
