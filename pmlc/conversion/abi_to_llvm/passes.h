// Copyright 2020, Intel Corporation

#pragma once

#include <memory>

namespace mlir {
class LLVMTypeConverter;
class OwningRewritePatternList;
} // namespace mlir

namespace pmlc::conversion::abi_to_llvm {

void populateABIToLLVMConversionPatterns(
    mlir::LLVMTypeConverter &converter,
    mlir::OwningRewritePatternList &patterns);

} // namespace pmlc::conversion::abi_to_llvm
