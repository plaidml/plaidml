// Copyright 2020 Intel Corporation

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "pmlc/util/buffer.h"

namespace mlir {
class OpPassManager;
} // namespace mlir

namespace pmlc::compiler {

struct PassInfo {
  std::string name;
  std::string ir;
};

struct ConstantArgument {
  mlir::Type type;
  util::BufferPtr buffer;
};

struct Program;

class Target {
public:
  virtual ~Target() = default;

  virtual void buildPipeline(mlir::OpPassManager &pm,
                             llvm::StringRef targetOptions) = 0;

  virtual util::BufferPtr
  save(Program &program,
       const std::unordered_map<std::string, std::string> &config) = 0;
};

using TargetPtr = std::shared_ptr<Target>;

struct Program {
  std::string entry;
  std::string tileIR;
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningModuleRef module;
  std::vector<mlir::Type> inputs;
  std::vector<mlir::Type> outputs;
  std::vector<ConstantArgument> constants;
  std::vector<PassInfo> passes;
  TargetPtr target;

  explicit Program(llvm::StringRef name = "");
  explicit Program(mlir::ModuleOp module);
  explicit Program(std::unique_ptr<mlir::MLIRContext> context,
                   std::unique_ptr<llvm::MemoryBuffer> buffer);

  static std::unique_ptr<Program>
  fromSource(std::unique_ptr<mlir::MLIRContext> context,
             llvm::StringRef source);

  void compile(mlir::StringRef targetName, bool collectPasses = false,
               mlir::StringRef dumpDir = "");

  util::BufferPtr
  save(const std::unordered_map<std::string, std::string> &config = {});
};

void registerTargets();

} // namespace pmlc::compiler
