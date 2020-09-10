// Copyright 2020 Intel Corporation

#pragma once

#include <memory>
#include <vector>

#include "pmlc/compiler/program.h"
#include "pmlc/rt/runtime.h"

namespace pmlc::rt {

enum class EngineKind {
  MCJIT,
  OrcJIT,
};

class Executable {
public:
  static std::unique_ptr<Executable>
  fromProgram(const std::shared_ptr<pmlc::compiler::Program> &program,
              llvm::StringRef deviceID, mlir::ArrayRef<void *> bufptrs,
              EngineKind kind = EngineKind::OrcJIT);

  virtual ~Executable() {}

  virtual void invoke() = 0;
};

} // namespace pmlc::rt
