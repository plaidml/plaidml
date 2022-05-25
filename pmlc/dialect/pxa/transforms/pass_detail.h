#pragma once

#include <thread>

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace func {
class FuncOp;
} // end namespace func
} // end namespace mlir

namespace pmlc::dialect::pxa {

#define GEN_PASS_CLASSES
#include "pmlc/dialect/pxa/transforms/passes.h.inc"

} // namespace pmlc::dialect::pxa
