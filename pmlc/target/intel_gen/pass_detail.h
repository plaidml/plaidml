#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"

#include "mlir/Pass/Pass.h"

namespace pmlc::target::intel_gen {

#define GEN_PASS_CLASSES
#include "pmlc/target/intel_gen/passes.h.inc"

} // namespace pmlc::target::intel_gen
