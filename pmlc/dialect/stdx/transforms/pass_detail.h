#pragma once

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/stdx/ir/ops.h"

namespace mlir {
namespace func {
class FuncOp;
} // end namespace func
} // end namesapce mlir

namespace mlir {
class ModuleOp;
} // end namespace mlir

namespace pmlc::dialect::stdx {

#define GEN_PASS_CLASSES
#include "pmlc/dialect/stdx/transforms/passes.h.inc"

} // namespace pmlc::dialect::stdx
