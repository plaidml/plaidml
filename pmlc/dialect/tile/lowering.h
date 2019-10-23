// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/Module.h"

namespace pmlc {
namespace dialect {
namespace tile {

mlir::OwningModuleRef LowerIntoStripe(mlir::ModuleOp module);

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
