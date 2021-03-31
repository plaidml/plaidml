#pragma once

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace pmlc::conversion::scf_to_omp {

#define GEN_PASS_CLASSES
#include "pmlc/conversion/scf_to_omp/passes.h.inc"

} // namespace pmlc::conversion::scf_to_omp
