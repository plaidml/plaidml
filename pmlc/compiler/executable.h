// Copyright 2020 Intel Corporation

#pragma once

#include <memory>
#include <vector>

#include "pmlc/compiler/program.h"

namespace pmlc::compiler {

enum class EngineKind {
  MCJIT,
  OrcJIT,
};

struct ExecutableImpl;
class Executable {
public:
  Executable(const std::shared_ptr<Program> &program,
             mlir::ArrayRef<void *> bufptrs,
             EngineKind kind = EngineKind::OrcJIT);
  ~Executable();

  void invoke();

private:
  std::unique_ptr<ExecutableImpl> impl;
};

} // namespace pmlc::compiler
