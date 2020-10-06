// Copyright 2020 Intel Corporation

#include "pmlc/rt/executable.h"

#include "pmlc/rt/device_id.h"
#include "pmlc/util/buffer.h"

using pmlc::compiler::Program;

namespace pmlc::rt {

std::unique_ptr<Executable>
Executable::fromProgram(const std::shared_ptr<Program> &program,
                        mlir::StringRef deviceID,
                        mlir::ArrayRef<util::BufferPtr> inputBuffers,
                        mlir::ArrayRef<util::BufferPtr> outputBuffers) {
  return getDevice(deviceID)->compile(program, inputBuffers, outputBuffers);
}

} // namespace pmlc::rt
