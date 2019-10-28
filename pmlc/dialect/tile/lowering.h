// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/Module.h"

namespace pmlc::dialect::tile {

mlir::OwningModuleRef LowerIntoStripe(mlir::ModuleOp module);

}  // namespace pmlc::dialect::tile
