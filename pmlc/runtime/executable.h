// Copyright 2020 Intel Corporation

#pragma once

#include <memory>
#include <vector>

#include "pmlc/compiler/program.h"
#include "pmlc/runtime/runtime.h"

namespace pmlc::runtime {

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
             std::shared_ptr<Device> device, mlir::ArrayRef<void *> bufptrs,
             EngineKind kind = EngineKind::OrcJIT);
  ~Executable();

  void invoke();

private:
  std::unique_ptr<detail::ExecutableImpl> impl;
};

} // namespace pmlc::runtime
