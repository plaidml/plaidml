#pragma once

// TODO: Lorenzo check if fw decl.
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/linalgx/ir/ops.h"
#include "pmlc/dialect/xsmm/ir/ops.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
} // end namespace mlir

namespace pmlc::target::x86 {

#define GEN_PASS_CLASSES
#include "pmlc/target/x86/passes.h.inc"

} // namespace pmlc::target::x86
