// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"

namespace pmlc::dialect::tile {

void AddStripeLoweringPasses(mlir::PassManager* pm);

mlir::OwningModuleRef LowerIntoStripe(mlir::ModuleOp module);

}  // namespace pmlc::dialect::tile
