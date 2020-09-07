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

namespace detail {
struct ExecutableImpl;
} // namespace detail

class Executable {
public:
  Executable(const std::shared_ptr<pmlc::compiler::Program> &program,
             llvm::StringRef deviceID, mlir::ArrayRef<void *> bufptrs,
             EngineKind kind = EngineKind::OrcJIT);
  ~Executable();

  void invoke();

private:
  std::unique_ptr<detail::ExecutableImpl> impl;
};

} // namespace pmlc::rt
