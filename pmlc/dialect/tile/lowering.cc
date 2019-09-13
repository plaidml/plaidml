// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/lowering.h"

#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/tile/internal.h"

namespace pmlc {
namespace dialect {
namespace tile {

struct LoweringPass : public mlir::ModulePass<LoweringPass> {
  void runOnModule() override {  //
  }
};

struct StripeProgram {
  mlir::OwningModuleRef module;
  mlir::MLIRContext context;

  explicit StripeProgram(mlir::ModuleOp module) : module(module) {}
};

mlir::OwningModuleRef LowerIntoStripe(mlir::MLIRContext* context, TileProgram* program) {
  mlir::OwningModuleRef module(llvm::cast<mlir::ModuleOp>(program->module->getOperation()->clone()));
  return module;
}

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
