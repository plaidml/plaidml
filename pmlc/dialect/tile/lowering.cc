// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/lowering.h"

#include "pmlc/dialect/tile/internal.h"

namespace pmlc {
namespace dialect {
namespace tile {

struct StripeProgram {
  mlir::OwningModuleRef module;
  mlir::MLIRContext context;

  explicit StripeProgram(mlir::ModuleOp module) : module(module) {}
};

std::shared_ptr<StripeProgram> LowerIntoStripe(TileProgram* program) {
  auto module = llvm::cast<mlir::ModuleOp>(program->module->getOperation()->clone());
  return std::make_shared<StripeProgram>(module);
}

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
