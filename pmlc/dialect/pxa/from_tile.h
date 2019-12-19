// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/Module.h"

namespace pmlc::dialect::pxa {

mlir::OwningModuleRef ConvertTileToPXA(mlir::ModuleOp module);

}  // namespace pmlc::dialect::pxa
