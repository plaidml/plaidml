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

} // namespace pmlc::conversion::stdx_to_llvm
