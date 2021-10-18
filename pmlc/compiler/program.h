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
  std::string entry;  // TODO: Temp note: Needed (as `"main"` ... may come from
                      // caller as such?)
  std::string tileIR; // TODO: Temp note: May Ignore
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningModuleRef module;            // TODO: Temp note: Needed
  std::vector<mlir::Type> inputs;          // TODO: Temp note: Needed
  std::vector<mlir::Type> outputs;         // TODO: Temp note: Needed
  std::vector<ConstantArgument> constants; // TODO: Temp note: May Delay
  std::vector<PassInfo> passes;            // TODO: Temp note: May Ignore
  TargetPtr target;

  explicit Program(llvm::StringRef name = "");
  explicit Program(mlir::ModuleOp module);
  explicit Program(std::unique_ptr<mlir::MLIRContext> context,
                   std::unique_ptr<llvm::MemoryBuffer> buffer,
                   llvm::StringRef entry);

  void compile(mlir::StringRef targetName, bool collectPasses = false,
               mlir::StringRef dumpDir = "");

  util::BufferPtr
  save(const std::unordered_map<std::string, std::string> &config = {});

  void parseIOTypes(std::unique_ptr<llvm::MemoryBuffer> buffer);
};

std::shared_ptr<Program> loadProgram(llvm::StringRef code, llvm::StringRef name,
                                     llvm::StringRef entry);
void registerTargets();

} // namespace pmlc::compiler
