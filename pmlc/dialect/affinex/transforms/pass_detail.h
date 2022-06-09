#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace func {
class FuncOp;
} // end namespace func
} // end namespace mlir

namespace pmlc::dialect::affinex {

#define GEN_PASS_CLASSES
#include "pmlc/dialect/affinex/transforms/passes.h.inc"

} // namespace pmlc::dialect::affinex
