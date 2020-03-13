// Copyright 2020 Intel Corporation

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"

#include "pmlc/util/buffer.h"

namespace mlir {
class ExecutionEngine;
} // namespace mlir

namespace pmlc::compiler {

class MemRefDescriptor;

struct ProgramArgument {
  bool isInput;
  mlir::Value value;
  mlir::RankedTensorType shape;
  pmlc::util::BufferPtr buffer;
};

struct PassInfo {
  std::string name;
  std::string ir;
};

struct Program {
  std::string entry;
  std::string tileIR;
  mlir::OwningModuleRef module;
  std::vector<ProgramArgument> arguments;
  std::vector<PassInfo> passes;
  mlir::MLIRContext context;
  std::string targetValue;

  explicit Program(mlir::ModuleOp module);
  explicit Program(mlir::StringRef source);

  void compile(mlir::StringRef target, bool collectPasses = false);

  static void initialize();
};

class Executable {
public:
  Executable(const std::shared_ptr<Program> &program,
             mlir::ArrayRef<void *> bufptrs);
  ~Executable();

  void invoke();

  static void initialize();

private:
  std::shared_ptr<Program> program;
  std::unique_ptr<mlir::ExecutionEngine> engine;
  std::vector<MemRefDescriptor> descriptors;
  std::vector<void *> ptrs;
};

} // namespace pmlc::compiler
