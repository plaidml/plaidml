#pragma once

#include "mlir/Pass/Pass.h"
#include "pmlc/dialect/stdx/ir/ops.h"

namespace mlir {
namespace func {
class FuncOp;
} // end namespace func
} // end namespace mlir

namespace pmlc::dialect::tile {

#define GEN_PASS_CLASSES
#include "pmlc/dialect/tile/transforms/passes.h.inc"

} // namespace pmlc::dialect::tile
