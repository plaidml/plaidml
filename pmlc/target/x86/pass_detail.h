#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/xsmm/ir/ops.h"

namespace pmlc::target::x86 {

#define GEN_PASS_CLASSES
#include "pmlc/target/x86/passes.h.inc"

} // namespace pmlc::target::x86
