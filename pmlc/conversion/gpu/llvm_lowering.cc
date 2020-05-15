// Copyright 2020, Intel Corporation

#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/LoopsToGPU/LoopsToGPUPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/StandardTypes.h"

#include "pmlc/conversion/gpu/pass_detail.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::conversion::gpu {

static LLVM::LLVMType unwrap(Type type) {
  if (!type)
    return nullptr;
  auto *mlirContext = type.getContext();
  auto wrappedLLVMType = type.dyn_cast<LLVM::LLVMType>();
  if (!wrappedLLVMType)
    emitError(UnknownLoc::get(mlirContext),
              "conversion resulted in a non-LLVM type");
  return wrappedLLVMType;
}

/// Convert a MemRef type to a bare pointer to the MemRef element type.
static Type convertMemRefTypeToBarePtr(LLVMTypeConverter &converter,
                                       MemRefType type) {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  if (failed(getStridesAndOffset(type, strides, offset)))
    return {};

  LLVM::LLVMType elementType =
      unwrap(converter.convertType(type.getElementType()));
  if (!elementType)
    return {};
  return elementType.getPointerTo(type.getMemorySpace());
}

/// Callback to convert function argument types. It converts MemRef function
/// arguments to bare pointers to the MemRef element type.
LogicalResult mixedPtrFuncArgTypeConverter(LLVMTypeConverter &converter,
                                           Type type,
                                           SmallVectorImpl<Type> &result) {
  if (auto memrefTy = type.dyn_cast<MemRefType>()) {
    auto llvmTy = convertMemRefTypeToBarePtr(converter, memrefTy);
    if (!llvmTy)
      return failure();

    result.push_back(llvmTy);
    return success();
  }
  return structFuncArgTypeConverter(converter, type, result);
}

struct ConvertToStdPass
    : public mlir::PassWrapper<ConvertToStdPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    auto *context = module.getContext();

    OwningRewritePatternList patterns;
    populateAffineToStdConversionPatterns(patterns, context);
    populateLoopToStdConversionPatterns(patterns, context);

    ConversionTarget target(*context);
    target.addLegalDialect<StandardOpsDialect>();
    if (failed(applyPartialConversion(module, target, patterns))) {
      signalPassFailure();
    }
  }

  static std::unique_ptr<OperationPass<ModuleOp>> create() {
    return std::make_unique<ConvertToStdPass>();
  }
};

struct ConvertToLLVMPass
    : public mlir::PassWrapper<ConvertToLLVMPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    auto *context = module.getContext();

    LLVMTypeConverterCustomization customs;
    customs.funcArgConverter = mixedPtrFuncArgTypeConverter;
    LLVMTypeConverter typeConverter(&getContext(), customs);

    OwningRewritePatternList patterns;
    populateStdToLLVMBarePtrConversionPatterns(typeConverter, patterns,
                                               /*useAlloca=*/true);

    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    if (failed(
            applyPartialConversion(module, target, patterns, &typeConverter))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createLLVMLoweringPass() {
  return std::make_unique<ConvertToLLVMPass>();
}
} // namespace pmlc::conversion::gpu
