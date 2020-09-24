// Copyright 2020 Intel Corporation

#include "pmlc/rt/executable.h"

#include <utility>

#include "pmlc/rt/device_id.h"
#include "pmlc/rt/timed_executable.h"
#include "pmlc/util/logging.h"

using pmlc::compiler::Program;

namespace pmlc::rt {

std::unique_ptr<Executable>
Executable::fromProgram(const std::shared_ptr<Program> &program,
                        mlir::StringRef deviceID,
                        mlir::ArrayRef<void *> bufptrs) {
  auto executable = getDevice(deviceID)->compile(program, bufptrs);
  if (VLOG_IS_ON(1)) {
    executable = makeTimedExecutable(std::move(executable));
  }
  return executable;
}

} // namespace pmlc::rt
