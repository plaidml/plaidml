// Copyright 2020 Intel Corporation

#pragma once

#include <memory>
#include <vector>

#include "pmlc/compiler/program.h"

namespace llvm {
class ExecutionEngine;
};

namespace pmlc::compiler {

class MemRefDescriptor;

class Executable {
public:
  Executable(const std::shared_ptr<Program> &program,
             mlir::ArrayRef<void *> bufptrs);
  ~Executable();

  void invoke();

private:
  std::shared_ptr<Program> program;
  std::unique_ptr<llvm::ExecutionEngine> engine;
  std::vector<MemRefDescriptor> descriptors;
  std::vector<void *> ptrs;

  using Function = void (*)(void **);
  Function jitEntry;
};

} // namespace pmlc::compiler
