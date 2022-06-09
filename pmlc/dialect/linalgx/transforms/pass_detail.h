#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace func {
class FuncOp;
} // end namespace func
} // end namespace mlir

namespace pmlc::dialect::linalgx {

#define GEN_PASS_CLASSES
#include "pmlc/dialect/linalgx/transforms/passes.h.inc"

} // namespace pmlc::dialect::linalgx
