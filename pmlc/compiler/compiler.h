// Copyright 2019, Intel Corporation

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
class ExecutionEngine;
}  // namespace mlir

namespace pmlc::compiler {

class MemRefDescriptor;

class Executable {
 public:
  Executable(mlir::StringRef entry, mlir::StringRef target, mlir::ModuleOp module, mlir::ArrayRef<void*> bufptrs,
             mlir::FloatType floatx, mlir::IntegerType intx);
  ~Executable();

  void invoke();

  static void initialize();

 private:
  std::string entry;
  std::unique_ptr<mlir::ExecutionEngine> engine;
  std::vector<MemRefDescriptor> descriptors;
  std::vector<void*> args;
  std::vector<void*> ptrs;
  mlir::FloatType floatx;
  mlir::IntegerType intx;
};

}  // namespace pmlc::compiler
