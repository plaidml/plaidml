// Copyright 2020, Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class OwningRewritePatternList;
class Pass;
} // namespace mlir

namespace pmlc::conversion::stdx_to_llvm {

void populateStdXToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter,
    mlir::OwningRewritePatternList &patterns);

std::unique_ptr<mlir::Pass> createLowerToLLVMPass();

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "pmlc/conversion/stdx_to_llvm/passes.h.inc"

} // namespace pmlc::conversion::stdx_to_llvm
