// Copyright 2020 Intel Corporation
#include "pmlc/target/x86/pipeline.h"

#include <algorithm>
#include <memory>

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/conversion/pxa_to_affine/passes.h"
#include "pmlc/conversion/scf_to_omp/passes.h"
#include "pmlc/conversion/stdx_to_llvm/passes.h"
#include "pmlc/conversion/tile_to_pxa/passes.h"
#include "pmlc/dialect/layer/transforms/passes.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "pmlc/dialect/stdx/transforms/passes.h"
#include "pmlc/dialect/tile/transforms/passes.h"
#include "pmlc/dialect/xsmm/ir/ops.h"
#include "pmlc/target/x86/heatmap.h"
#include "pmlc/target/x86/pass_detail.h"
#include "pmlc/target/x86/passes.h"
#include "pmlc/transforms/passes.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"

#include "omp.h" // NOLINT

#include "pmlc/target/x86/utils.h"

using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace layer = dialect::layer;
namespace pxa = dialect::pxa;
namespace stdx = dialect::stdx;
namespace tile = dialect::tile;
namespace xsmm = dialect::xsmm;

namespace {

struct LowerPXAToAffinePass
    : public ConvertPXAToAffineBase<LowerPXAToAffinePass> {
  void runOnOperation() final {
    auto &ctx = getContext();
    conversion::pxa_to_affine::PXAToAffineConversionTarget target(ctx);
    target.addLegalDialect<xsmm::XSMMDialect>();

    OwningRewritePatternList patterns;
    populatePXAGemmToXSMMConversionPatterns(patterns, &ctx);
    populatePXAPrngToAffineConversionPatterns(patterns, &ctx);
    conversion::pxa_to_affine::populatePXAToAffineConversionPatterns(patterns,
                                                                     &ctx);

    if (failed(applyPartialConversion(getOperation(), target, patterns,
                                      nullptr))) {
      getOperation().dump();
      emitError(UnknownLoc::get(&ctx), "Error lowering pxa -> affine\n");
      signalPassFailure();
    }
  }
};

struct ConvertStandardToLLVMPass
    : public ConvertStandardToLLVMBase<ConvertStandardToLLVMPass> {
  void runOnOperation() override {
    auto module = getOperation();
    auto *context = module.getContext();

    LowerToLLVMOptions options = {
        /*useBarePtrCallConv=*/false,
        /*emitCWrappers=*/true,
        /*indexBitwidth=*/kDeriveIndexBitwidthFromDataLayout,
        /*useAlignedAlloc=*/false,
    };
    LLVMTypeConverter typeConverter(context, options);

    OwningRewritePatternList patterns;
    populateExpandTanhPattern(patterns, context);
    populateXSMMToLLVMConversionPatterns(typeConverter, patterns);
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    conversion::stdx_to_llvm::populateStdXToLLVMConversionPatterns(
        typeConverter, patterns);
    populateOpenMPToLLVMConversionPatterns(context, typeConverter, patterns);

    LLVMConversionTarget target(*context);
    target.addDynamicallyLegalOp<omp::ParallelOp>([&](omp::ParallelOp op) {
      return typeConverter.isLegal(&op.getRegion());
    });
    target.addLegalOp<omp::TerminatorOp, omp::TaskyieldOp, omp::FlushOp,
                      omp::BarrierOp, omp::TaskwaitOp>();
    if (failed(applyPartialConversion(module, target, patterns))) {
      signalPassFailure();
    }
  }
};

// OpenMP has issues passing values through to OpenMP blocks.  As a workaround,
// we have a simple pass to smuggle values that cross the boundary via an
// alloca()'d struct.
struct OpenMPWorkaroundPass final
    : public OpenMPWorkaroundBase<OpenMPWorkaroundPass> {
  void runOnOperation() final {
    LLVM::LLVMFuncOp funcOp = getOperation();
    OpBuilder builder{&getContext()};
    funcOp.walk([&](omp::ParallelOp parOp) {
      llvm::SetVector<Value> values;

      visitUsedValuesDefinedAbove({parOp.region()}, [&](OpOperand *opOperand) {
        auto value = opOperand->get();

        // If it's not an LLVM type, or if it's an LLVM pointer type, we
        // don't need or want to smuggle this value in via a struct.
        auto llvmType = value.getType().dyn_cast<LLVM::LLVMType>();
        if (!llvmType || llvmType.isPointerTy()) {
          return;
        }

        // Otherwise, we need to smuggle the value through an alloca'd
        // struct.
        values.insert(value);
      });

      if (!values.size()) {
        return; // Nothing to do.
      }

      // Build the structure.
      builder.setInsertionPoint(parOp);
      LLVM::LLVMType structTy;
      {
        SmallVector<LLVM::LLVMType, 8> types;
        for (auto val : values) {
          types.push_back(val.getType().cast<LLVM::LLVMType>());
        }
        structTy = LLVM::LLVMType::getStructTy(&getContext(), types);
      }
      auto structPtrTy = structTy.getPointerTo();
      auto numElements = builder.create<LLVM::ConstantOp>(
          parOp.getLoc(), LLVM::LLVMType::getInt64Ty(&getContext()),
          builder.getIndexAttr(1));
      auto structPtr = builder.create<LLVM::AllocaOp>(
          parOp.getLoc(), structPtrTy, numElements, 0);
      Value srcStructVal =
          builder.create<LLVM::UndefOp>(parOp.getLoc(), structTy);
      for (auto srcIdx : llvm::enumerate(values)) {
        srcStructVal = builder.create<LLVM::InsertValueOp>(
            parOp.getLoc(), srcStructVal, srcIdx.value(),
            builder.getI64ArrayAttr(srcIdx.index()));
      }
      builder.create<LLVM::StoreOp>(parOp.getLoc(), srcStructVal, structPtr);

      // Unpack the structure, rewriting the affected values.
      builder.setInsertionPointToStart(&parOp.region().front());
      auto dstStructVal =
          builder.create<LLVM::LoadOp>(parOp.getLoc(), structPtr);
      for (auto srcIdx : llvm::enumerate(values)) {
        auto smuggledValue = builder.create<LLVM::ExtractValueOp>(
            parOp.getLoc(), srcIdx.value().getType(), dstStructVal,
            builder.getI64ArrayAttr(srcIdx.index()));
        replaceAllUsesInRegionWith(srcIdx.value(), smuggledValue,
                                   parOp.region());
      }
    });
  }
};

} // namespace

// NOTE: the stencil pass uses row-major ordering, the heatmap is
// specified in column-major ordering.
static pxa::StencilCost heatmapCostTransposed(ArrayRef<int64_t> tile,
                                              ArrayRef<Type> types) {
  // Only f32 is supported currently.
  if (llvm::any_of(types, [](Type type) {
        if (auto shapedType = type.dyn_cast<ShapedType>()) {
          type = shapedType.getElementType();
        }
        return !type.isF32();
      })) {
    return pxa::StencilCost{/*throughput=*/0.0, /*startupCost=*/0};
  }
  return heatmapCost(ArrayRef<int64_t>{tile[1], tile[0], tile[2]});
}

std::unique_ptr<Pass> createXSMMStencilPass() {
  auto numThreads = std::thread::hardware_concurrency();
  return pxa::createStencilGEMMPass(numThreads, /*doBatch=*/true,
                                    heatmapCostTransposed);
}

std::unique_ptr<Pass> createLowerPXAToAffinePass() {
  return std::make_unique<LowerPXAToAffinePass>();
}

std::unique_ptr<Pass> createLowerToLLVMPass() {
  return std::make_unique<ConvertStandardToLLVMPass>();
}

std::unique_ptr<Pass> createOpenMPWorkaroundPass() {
  return std::make_unique<OpenMPWorkaroundPass>();
}

void pipelineBuilder(OpPassManager &pm) {
  pm.addPass(layer::createInlineLayersPass());
  pm.addPass(tile::createComputeBoundsPass());
  pm.addPass(tile::createSplitMainPass());
  pm.addPass(transforms::createHoistingPass());
  pm.addPass(tile::createPadConstraintsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(pmlc::conversion::tile_to_pxa::createLowerTileToPXAPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(layer::createInlineLayersPass());

  pm.addPass(pxa::createStencilGEMMPass(/*numThreads=*/1, /*doBatch=*/true,
                                        heatmapCostTransposed));
  pm.addPass(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());

  pm.addPass(pxa::createTileAccumulatePass());
  pm.addPass(pxa::createAffineNormalizePass(/*promote=*/false));
  pm.addPass(createCanonicalizerPass());

  // Use OMP thread count
  unsigned maxThreads = omp_get_max_threads();
  unsigned physCores = getPhysicalCoreNumber();
  if (0 != physCores) {
    maxThreads = std::min(physCores, maxThreads);
  }

  pm.addPass(pxa::createCPUThreadPass(maxThreads));

  pm.addPass(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());

  pm.addPass(pxa::createFusionPass());
  pm.addPass(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());

  pm.addPass(pxa::createMemRefDataFlowOptPass());
  pm.addPass(createCanonicalizerPass());

  pm.addPass(pxa::createLocalizePass());
  pm.addPass(pxa::createResizeTmpsPass());
  pm.addPass(pxa::createDeallocPlacementPass());
  pm.addPass(pxa::createAffineNormalizePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerPXAToAffinePass());

  // Unroll affine.for loops.
  pm.addPass(createLoopUnrollPass(
      /*unrollFactor=*/32,
      /*unrollUpToFactor=*/true));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLoopInvariantCodeMotionPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(pmlc::conversion::scf_to_omp::createLowerSCFToOpenMPPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(createLowerToCFGPass());
  if (pmlc::util::getEnvVar("PLAIDML_BOUNDS_CHECK") == "1") {
    pm.addPass(stdx::createBoundsCheckPass());
  }

  pm.addPass(createLowerToLLVMPass());
  pm.addPass(createTraceLinkingPass());
  pm.addPass(createOpenMPWorkaroundPass());
}

} // namespace pmlc::target::x86
