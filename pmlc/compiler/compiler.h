// Copyright 2019, Intel Corporation

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

struct Program {
  std::string entry;
  std::string tileIR;
  mlir::OwningModuleRef module;
  std::vector<mlir::Value> outputs;
  std::vector<ProgramArgument> arguments;
  std::vector<mlir::MemRefType> memRefTypes;

  explicit Program(mlir::ModuleOp module) : module(module) {}

  void compile(mlir::StringRef target);
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
  std::vector<void *> args;
  std::vector<void *> ptrs;
};

} // namespace pmlc::compiler
