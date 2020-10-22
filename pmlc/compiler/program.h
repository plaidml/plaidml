// Copyright 2020 Intel Corporation

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"

#include "pmlc/util/buffer.h"

namespace pmlc::compiler {

struct PassInfo {
  std::string name;
  std::string ir;
};

struct ConstantArgument {
  mlir::Type type;
  util::BufferPtr buffer;
};

struct Program {
  std::string entry;
  std::string tileIR;
  mlir::MLIRContext context;
  mlir::OwningModuleRef module;
  std::vector<mlir::Type> inputs;
  std::vector<mlir::Type> outputs;
  std::vector<ConstantArgument> constants;
  std::vector<PassInfo> passes;

  explicit Program(llvm::StringRef name = "");
  explicit Program(mlir::ModuleOp module);
  explicit Program(std::unique_ptr<llvm::MemoryBuffer> buffer);

  static Program fromSource(llvm::StringRef source);

  void compile(mlir::StringRef target, bool collectPasses = false,
               mlir::StringRef dumpDir = "");
};

} // namespace pmlc::compiler
