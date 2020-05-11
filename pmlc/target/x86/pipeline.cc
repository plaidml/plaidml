// Copyright 2020 Intel Corporation

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/conversion/pxa_to_affine/passes.h"
#include "pmlc/conversion/stdx_to_llvm/passes.h"
#include "pmlc/conversion/tile_to_pxa/passes.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/stdx/transforms/passes.h"
#include "pmlc/dialect/tile/transforms/passes.h"
#include "pmlc/target/x86/heatmap.h"
#include "pmlc/target/x86/pass_detail.h"
#include "pmlc/target/x86/passes.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

std::unique_ptr<Pass> createXSMMStencilPass() {
  auto numThreads = std::thread::hardware_concurrency();
  return pmlc::dialect::pxa::createXSMMStencilPass(numThreads, heatmapCost);
}

namespace {

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
                                               /*useAlignedAlloc=*/false);
    conversion::stdx_to_llvm::populateStdXToLLVMConversionPatterns(
        typeConverter, patterns);

    ConversionTarget target(*context);
    target.addLegalDialect<LLVM::LLVMDialect>();
    if (failed(
            applyPartialConversion(module, target, patterns, &typeConverter))) {
      signalPassFailure();
    }
  }

  static std::unique_ptr<OperationPass<ModuleOp>> create() {
    return std::make_unique<ConvertToLLVMPass>();
  }
};

void addToPipeline(OpPassManager &pm) {
  pm.addPass(pmlc::dialect::tile::createComputeBoundsPass());
  pm.addPass(pmlc::dialect::tile::createPadPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(pmlc::conversion::tile_to_pxa::createLowerTileToPXAPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(pmlc::dialect::pxa::createXSMMStencilPass(1, heatmapCost));
  pm.addPass(createXSMMLoweringPass());

  pm.addPass(conversion::pxa_to_affine::createLowerPXAToAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(ConvertToStdPass::create());
  if (pmlc::util::getEnvVar("PLAIDML_BOUNDS_CHECK") == "1") {
    pm.addPass(pmlc::dialect::stdx::createBoundsCheckPass());
  }
  pm.addPass(ConvertToLLVMPass::create());
  pm.addPass(createTraceLinkingPass());
}

static PassPipelineRegistration<>
    passPipelineReg("target-cpu", "Target pipeline for CPU", addToPipeline);
static compiler::TargetRegistration targetReg("llvm_cpu", addToPipeline);

} // namespace

} // namespace pmlc::target::x86
